"""Opt-in OpenAI embeddings and a local Chroma index.

This module is independent of the offline lexical baseline. Importing it does not
import either optional SDK or make a request. The byte budget is a conservative
UTF-8 token upper bound, not a tokenizer measurement or a monetary cost estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence


class EmbeddingError(RuntimeError):
    """An error whose message never includes the source text or API credentials."""


@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool = False
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    max_input_bytes: int = 8192
    max_request_token_upper_bound: int = 250000
    max_total_token_upper_bound: int = 250000
    max_inputs_per_operation: int = 50000
    collection_name: str = "rfp_phase3_openai"

    def __post_init__(self):
        capacity = {"text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072}
        if self.model not in capacity:
            raise ValueError("Only the verified text-embedding-3 model names are supported")
        numeric = ("dimensions", "batch_size", "max_input_bytes",
                   "max_request_token_upper_bound", "max_total_token_upper_bound",
                   "max_inputs_per_operation")
        for name in numeric:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dimensions > capacity[self.model]:
            raise ValueError("Dimensions exceed the selected model capacity")
        if self.max_input_bytes > 8192 or self.max_request_token_upper_bound > 300000:
            raise ValueError("Configured upper bounds exceed the verified API limits")
        if self.batch_size > 2048:
            raise ValueError("batch_size cannot exceed 2048 inputs")
        if self.max_input_bytes > self.max_request_token_upper_bound:
            raise ValueError("The per-input bound cannot exceed the request bound")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,60}[A-Za-z0-9]", self.collection_name):
            raise ValueError("Use a simple 3 to 63 character collection name")


def _config(value):
    return value if isinstance(value, EmbeddingConfig) else EmbeddingConfig(**(value or {}))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: str):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _vector(value, dimensions):
    if not isinstance(value, (list, tuple)) or len(value) != dimensions:
        raise EmbeddingError("Embedding vector has an unexpected dimension")
    result = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise EmbeddingError("Embedding vector contains a nonfinite or invalid value")
        result.append(float(number))
    if not any(result):
        raise EmbeddingError("Zero embedding vectors cannot be indexed with cosine distance")
    return result


class OpenAIEmbeddingClient:
    """Exact-input cache and bounded SDK calls; a fake client can be injected.

    Even an injected client requires explicit enablement and OPENAI_API_KEY.
    Failed API attempts consume their reserved upper-bound budget; automatic SDK
    retries are disabled. Start a new client only after consciously resetting the
    operation budget. Cached inputs do not consume the API budget.
    """

    def __init__(self, cache_dir, config=None, client=None):
        self.config = _config(config)
        self.cache_dir = Path(cache_dir)
        self._client = client
        self._reserved = 0

    def _guard(self):
        if self.config.enabled is not True:
            raise EmbeddingError("OpenAI embeddings are disabled; explicit opt-in is required")
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise EmbeddingError("OPENAI_API_KEY must be set in the local environment")

    def _sdk(self):
        self._guard()
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise EmbeddingError("Install the optional openai package before enabling this adapter") from None
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        return self._client

    def _db(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.cache_dir / "embeddings.sqlite3")
        connection.execute("CREATE TABLE IF NOT EXISTS vectors "
                           "(cache_key TEXT PRIMARY KEY, model TEXT, dimensions INTEGER, "
                           "input_sha256 TEXT, vector_json TEXT, vector_sha256 TEXT)")
        connection.commit()
        return connection

    def _key(self, text):
        return _sha(_json([self.config.model, self.config.dimensions, _sha(text)]))

    def embed(self, texts: Sequence[str]):
        self._guard()
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence of strings")
        texts = list(texts)
        if len(texts) > self.config.max_inputs_per_operation:
            raise EmbeddingError("Embedding input count exceeds the configured operation bound")
        for item in texts:
            if not isinstance(item, str) or not item.strip():
                raise EmbeddingError("Embedding inputs must be nonempty strings")
            if len(item.encode("utf-8")) > self.config.max_input_bytes:
                raise EmbeddingError("An input exceeds the conservative UTF-8 byte bound; rechunk it first")
        unique = list(dict.fromkeys(texts))
        vectors = {}
        pending = []
        report = {"model": self.config.model, "dimensions": self.config.dimensions,
                  "input_count": len(texts), "unique_input_count": len(unique),
                  "cache_hits": 0, "api_calls": 0, "reserved_token_upper_bound": 0,
                  "budget_unit": "utf8_bytes_token_upper_bound",
                  "actual_prompt_tokens": 0, "actual_total_tokens": 0,
                  "actual_usage_available": True}
        if not texts:
            return [], report
        connection = self._db()
        try:
            # Inspect every cache row and every input before the first network call.
            for item in unique:
                key = self._key(item)
                row = connection.execute("SELECT model, dimensions, input_sha256, vector_json, vector_sha256 "
                                         "FROM vectors WHERE cache_key=?", (key,)).fetchone()
                if row is None:
                    pending.append(item)
                    continue
                if (row[:3] != (self.config.model, self.config.dimensions, _sha(item))
                        or _sha(row[3]) != row[4]):
                    raise EmbeddingError("The local embedding cache failed its integrity check")
                try:
                    vectors[item] = _vector(json.loads(row[3]), self.config.dimensions)
                except (ValueError, TypeError):
                    raise EmbeddingError("The local embedding cache contains invalid vector data") from None
                report["cache_hits"] += 1
            upper_bound = sum(len(item.encode("utf-8")) for item in pending)
            if self._reserved + upper_bound > self.config.max_total_token_upper_bound:
                raise EmbeddingError("The uncached inputs exceed the remaining API token upper-bound budget")
            batches, batch, batch_bytes = [], [], 0
            for item in pending:
                size = len(item.encode("utf-8"))
                if batch and (len(batch) >= self.config.batch_size or
                              batch_bytes + size > self.config.max_request_token_upper_bound):
                    batches.append(batch)
                    batch, batch_bytes = [], 0
                batch.append(item)
                batch_bytes += size
            if batch:
                batches.append(batch)
            for batch in batches:
                reserved = sum(len(item.encode("utf-8")) for item in batch)
                self._reserved += reserved
                report["reserved_token_upper_bound"] += reserved
                report["api_calls"] += 1
                try:
                    response = self._sdk().embeddings.create(
                        model=self.config.model, dimensions=self.config.dimensions,
                        input=batch, encoding_format="float")
                except EmbeddingError:
                    raise
                except Exception:
                    # SDK messages can include request bodies or credentials.
                    raise EmbeddingError("The embedding request failed; no vectors from that batch were cached") from None
                if _field(response, "model", self.config.model) != self.config.model:
                    raise EmbeddingError("The embedding response model does not match the request")
                data = _field(response, "data")
                if not isinstance(data, (list, tuple)) or len(data) != len(batch):
                    raise EmbeddingError("The embedding response has an unexpected item count")
                ordered = {}
                for item in data:
                    index = _field(item, "index")
                    if (isinstance(index, bool) or not isinstance(index, int) or
                            index < 0 or index >= len(batch) or index in ordered):
                        raise EmbeddingError("The embedding response has invalid item indexes")
                    ordered[index] = _vector(_field(item, "embedding"), self.config.dimensions)
                rows = []
                for index, item in enumerate(batch):
                    vector = ordered[index]
                    encoded = _json(vector)
                    rows.append((self._key(item), self.config.model, self.config.dimensions,
                                 _sha(item), encoded, _sha(encoded)))
                # Validate the whole response before an atomic cache transaction.
                with connection:
                    connection.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?, ?)", rows)
                vectors.update({item: ordered[index] for index, item in enumerate(batch)})
                usage = _field(response, "usage")
                for name, target in (("prompt_tokens", "actual_prompt_tokens"),
                                     ("total_tokens", "actual_total_tokens")):
                    value = _field(usage, name)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        report[target] += value
                    else:
                        report["actual_usage_available"] = False
            report["session_reserved_token_upper_bound"] = self._reserved
            return [vectors[item][:] for item in texts], report
        finally:
            connection.close()


class OpenAIChromaIndex:
    """Persistent *local* Chroma with explicit vectors and source allowlisting.

    Provide a reviewed canonical-doc-ID -> raw SHA256 allowlist to build(). A
    different corpus, model, dimension, or source version requires a new index
    directory/collection; an existing collection is never silently repurposed.
    """

    def __init__(self, index_dir, config=None, embedding_client=None, chroma_client=None):
        self.index_dir = Path(index_dir)
        self.config = _config(config)
        self.embeddings = embedding_client or OpenAIEmbeddingClient(self.index_dir / "cache", self.config)
        if self.embeddings.config != self.config:
            raise ValueError("Embedding client configuration does not match the index")
        self._chroma = chroma_client
        self._collection = None
        self._chunks = None
        self._aliases = None
        self._manifest = None

    def _guard(self):
        self.embeddings._guard()

    def _client(self):
        self._guard()
        if self._chroma is None:
            try:
                import chromadb
            except ImportError:
                raise EmbeddingError("Install the optional chromadb package before enabling this adapter") from None
            self._chroma = chromadb.PersistentClient(path=str(self.index_dir / "chroma"))
        return self._chroma

    def _validate_chunks(self, chunks, allowed_sources):
        if not isinstance(allowed_sources, Mapping) or not allowed_sources:
            raise ValueError("An explicit canonical source SHA256 allowlist is required")
        approved = dict(allowed_sources)
        for doc_id, digest in approved.items():
            if not isinstance(doc_id, str) or not doc_id or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("Invalid canonical source allowlist entry")
        chunks = json.loads(_json(list(chunks)))  # Local detached JSON payload.
        if not chunks:
            raise ValueError("An empty corpus cannot be built")
        if len(chunks) > self.config.max_inputs_per_operation:
            raise EmbeddingError("Corpus exceeds the configured input-count bound")
        aliases, ids = {}, set()
        for chunk in chunks:
            doc_id = chunk.get("canonical_doc_id")
            if doc_id not in approved or chunk.get("sha256") != approved[doc_id]:
                raise EmbeddingError("A chunk is outside the approved canonical source version")
            if chunk.get("doc_id") != doc_id:
                raise EmbeddingError("Chunk doc_id must be its canonical document ID")
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in ids:
                raise ValueError("Chunk IDs must be nonempty and unique")
            ids.add(chunk_id)
            text = chunk.get("search_text", chunk.get("text"))
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise EmbeddingError("A chunk has no original source text")
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError("A chunk has no nonempty embedding input")
            if len(text.encode("utf-8")) > self.config.max_input_bytes:
                raise EmbeddingError("A chunk exceeds the configured embedding input byte bound")
            method = chunk.get("source_method")
            if method not in ("raw_parser", "reviewed_visual_transcription"):
                raise EmbeddingError("Only raw-parser or reviewed-transcription sources can be embedded")
            spans = chunk.get("source_spans")
            if not isinstance(spans, list) or not spans or any(not isinstance(s, dict) for s in spans):
                raise EmbeddingError("Every chunk needs original source-span metadata")
            if any(span.get("collection") not in ("blocks", "supplemental_blocks")
                   for span in spans):
                raise EmbeddingError("Reference views cannot be embedded as independent evidence")
            if any(span.get("collection") == "supplemental_blocks"
                   for span in chunk.get("source_spans", [])) and method != "reviewed_visual_transcription":
                raise EmbeddingError("Supplemental transcription must be explicitly reviewed")
            if any("unreviewed" in str(flag).lower() for flag in chunk.get("quality_flags", [])):
                raise EmbeddingError("Unreviewed OCR/transcription cannot be embedded")
            for alias in [doc_id] + chunk.get("alias_doc_ids", []):
                if not isinstance(alias, str) or not alias:
                    raise ValueError("Document aliases must be nonempty strings")
                if alias in aliases and aliases[alias] != doc_id:
                    raise EmbeddingError("A document alias resolves to conflicting canonical sources")
                if alias in approved and alias != doc_id:
                    raise EmbeddingError("An alias conflicts with a canonical allowlist ID")
                aliases[alias] = doc_id
        return chunks, approved, aliases

    def _metadata(self, chunks, approved):
        return {"schema_version": 1, "embedding_model": self.config.model,
                "embedding_dimensions": self.config.dimensions,
                "distance_metric": "cosine",
                "corpus_fingerprint": _sha(_json({"chunks": chunks, "sources": approved}))}

    def _open_collection(self, metadata):
        collection = self._client().get_or_create_collection(
            name=self.config.collection_name, metadata=metadata,
            embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
        actual = getattr(collection, "metadata", None) or {}
        if any(actual.get(key) != value for key, value in metadata.items()):
            raise EmbeddingError("Existing Chroma collection metadata does not match this corpus/model/dimensions")
        return collection

    def build(self, chunks, allowed_sources):
        self._guard()
        chunks, approved, aliases = self._validate_chunks(chunks, allowed_sources)
        metadata = self._metadata(chunks, approved)
        manifest_path = self.index_dir / "manifest.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raise EmbeddingError("Existing index manifest cannot be validated") from None
            if existing.get("metadata") != metadata:
                raise EmbeddingError("The existing local index belongs to a different corpus/model/dimensions")
        collection = self._open_collection(metadata)
        # embed() validates the total uncached budget before its first API call.
        vectors, usage = self.embeddings.embed([c.get("search_text", c["text"]) for c in chunks])
        batch_size = self.config.batch_size
        client = self._client()
        if callable(getattr(client, "get_max_batch_size", None)):
            batch_size = min(batch_size, int(client.get_max_batch_size()))
        if batch_size < 1:
            raise EmbeddingError("Chroma reported an invalid batch bound")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            try:
                collection.upsert(ids=[c["chunk_id"] for c in batch],
                                  embeddings=vectors[start:start + batch_size],
                                  metadatas=[{"doc_id": c["doc_id"], "sha256": c["sha256"],
                                              "chunk_id": c["chunk_id"]} for c in batch])
            except Exception:
                raise EmbeddingError("Local Chroma upsert failed; this build was not marked complete") from None
        if collection.count() != len(chunks):
            raise EmbeddingError("Collection item count differs from the approved corpus; use a fresh collection")
        manifest = {"schema_version": 1, "metadata": metadata, "allowed_sources": approved,
                    "aliases": aliases, "chunks": chunks}
        self.index_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.index_dir / "manifest.json.tmp"
        temporary.write_text(_json(manifest), encoding="utf-8")
        temporary.replace(manifest_path)
        self._collection, self._chunks = collection, {c["chunk_id"]: c for c in chunks}
        self._aliases, self._manifest = aliases, manifest
        return {"chunk_count": len(chunks), "canonical_document_count": len({c["doc_id"] for c in chunks}),
                "embedding_usage": usage, "metadata": metadata,
                "live_service_integration_measured": False}

    def load(self):
        self._guard()
        try:
            manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise EmbeddingError("A complete local index manifest is required") from None
        chunks, approved, aliases = self._validate_chunks(manifest.get("chunks", []), manifest.get("allowed_sources"))
        metadata = self._metadata(chunks, approved)
        if manifest.get("schema_version") != 1 or manifest.get("metadata") != metadata or manifest.get("aliases") != aliases:
            raise EmbeddingError("Local index manifest integrity/model validation failed")
        collection = self._open_collection(metadata)
        if collection.count() != len(chunks):
            raise EmbeddingError("Persistent Chroma count does not match the complete manifest")
        self._collection, self._chunks = collection, {c["chunk_id"]: c for c in chunks}
        self._aliases, self._manifest = aliases, manifest
        return self

    def resolve_doc_ids(self, doc_ids):
        if self._chunks is None:
            self.load()
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        result = set()
        for doc_id in doc_ids:
            if doc_id not in self._aliases:
                raise ValueError("Unknown document scope; global search fallback is disabled")
            result.add(self._aliases[doc_id])
        return sorted(result)

    def search(self, query, top_k=5, allowed_doc_ids=None):
        self._guard()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty query is required")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self._chunks is None:
            self.load()
        scope = None if allowed_doc_ids is None else self.resolve_doc_ids(allowed_doc_ids)
        if scope == []:
            return []
        count = sum(1 for c in self._chunks.values() if scope is None or c["doc_id"] in scope)
        if not count:
            return []
        vectors, _ = self.embeddings.embed([query])
        kwargs = {"query_embeddings": vectors, "n_results": min(top_k, count),
                  "include": ["metadatas", "distances"]}
        if scope is not None:
            kwargs["where"] = {"doc_id": scope[0]} if len(scope) == 1 else {"doc_id": {"$in": scope}}
        try:
            response = self._collection.query(**kwargs)
        except Exception:
            raise EmbeddingError("Local Chroma query failed") from None
        try:
            ids, distances, metadatas = response["ids"][0], response["distances"][0], response["metadatas"][0]
            if len(ids) != len(distances) or len(ids) != len(metadatas):
                raise ValueError()
            hits = []
            seen = set()
            for chunk_id, distance, meta in zip(ids, distances, metadatas):
                if chunk_id in seen or chunk_id not in self._chunks:
                    raise ValueError()
                seen.add(chunk_id)
                chunk = self._chunks[chunk_id]
                if (scope is not None and chunk["doc_id"] not in scope) or any(
                        meta.get(k) != chunk[k] for k in ("doc_id", "sha256", "chunk_id")):
                    raise ValueError()
                if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
                    raise ValueError()
                hit = json.loads(_json(chunk))
                hit.update({"score": -float(distance), "distance": float(distance),
                            "rank": len(hits) + 1, "method": "openai_chroma",
                            "score_kind": "negative_cosine_distance_not_confidence"})
                hits.append(hit)
            return hits
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            raise EmbeddingError("Chroma returned invalid or out-of-scope source references") from None
