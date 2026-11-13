"""Network-free contract tests; the real optional SDKs are not required."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_rag.embeddings import (
    EmbeddingConfig, EmbeddingError, OpenAIChromaIndex, OpenAIEmbeddingClient,
)


class FakeSDK:
    def __init__(self, failure=None):
        self.embeddings = self
        self.calls = []
        self.failure = failure

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure == "exception":
            raise RuntimeError("DO_NOT_LEAK_KEY_OR_SOURCE")
        data = [{"index": i, "embedding": [float(i + 1)] + [0.5] * (kwargs["dimensions"] - 1)}
                for i in range(len(kwargs["input"]))]
        if self.failure == "nan":
            data[-1]["embedding"][0] = float("nan")
        elif self.failure == "dimension":
            data[-1]["embedding"].append(1.0)
        elif self.failure == "index":
            data[-1]["index"] = 0
        return {"model": kwargs["model"], "data": list(reversed(data)),
                "usage": {"prompt_tokens": len(data), "total_tokens": len(data)}}


class FakeCollection:
    def __init__(self, metadata):
        self.metadata = metadata.copy()
        self.rows = {}
        self.queries = []
        self.scope_violation = False

    def upsert(self, ids, embeddings, metadatas):
        for row_id, embedding, metadata in zip(ids, embeddings, metadatas):
            self.rows[row_id] = {"embedding": embedding, "metadata": metadata}

    def count(self):
        return len(self.rows)

    def query(self, **kwargs):
        self.queries.append(kwargs)
        scope = kwargs.get("where", {}).get("doc_id")
        if isinstance(scope, dict):
            scope = scope["$in"]
        elif scope is not None:
            scope = [scope]
        rows = [(key, value) for key, value in sorted(self.rows.items())
                if self.scope_violation or scope is None or value["metadata"]["doc_id"] in scope]
        rows = rows[:kwargs["n_results"]]
        return {"ids": [[key for key, _ in rows]],
                "distances": [[i * 0.1 + 0.1 for i in range(len(rows))]],
                "metadatas": [[value["metadata"] for _, value in rows]]}


class FakeChroma:
    def __init__(self):
        self.collection = None
        self.calls = []

    def get_or_create_collection(self, **kwargs):
        self.calls.append(kwargs)
        if self.collection is None:
            self.collection = FakeCollection(kwargs["metadata"])
        return self.collection

    def get_max_batch_size(self):
        return 2


def chunk(chunk_id="c1", doc_id="doc1", text="예산은 1억원입니다."):
    digest = hashlib.sha256(doc_id.encode()).hexdigest()
    return {"chunk_id": chunk_id, "doc_id": doc_id, "canonical_doc_id": doc_id,
            "alias_doc_ids": ["alias_" + doc_id], "sha256": digest,
            "text": text, "search_text": text, "file_name": doc_id + ".pdf",
            "source_method": "raw_parser", "kind": "prose", "quality_flags": [],
            "locator": "p1", "page": 1,
            "source_spans": [{"collection": "blocks", "block_id": "p1"}]}


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-not-a-real-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = EmbeddingConfig(enabled=True, dimensions=3)

    def client(self, config=None, sdk=None):
        sdk = sdk or FakeSDK()
        return OpenAIEmbeddingClient(self.root / "cache", config or self.config, sdk), sdk

    def index(self, config=None, chroma=None, sdk=None):
        config = config or self.config
        embedding, sdk = self.client(config, sdk)
        chroma = chroma or FakeChroma()
        return OpenAIChromaIndex(self.root / "index", config, embedding, chroma), sdk, chroma

    def build(self, index, chunks=None):
        chunks = chunks or [chunk()]
        sources = {c["doc_id"]: c["sha256"] for c in chunks}
        return index.build(chunks, allowed_sources=sources)

    def cache_count(self):
        with sqlite3.connect(self.root / "cache" / "embeddings.sqlite3") as connection:
            return connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def test_explicit_enable_and_environment_key_required(self):
        client, sdk = self.client(replace(self.config, enabled=False))
        with self.assertRaisesRegex(EmbeddingError, "disabled"):
            client.embed(["hello"])
        self.assertEqual(sdk.calls, [])
        client, sdk = self.client()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EmbeddingError, "OPENAI_API_KEY"):
                client.embed(["hello"])
        self.assertEqual(sdk.calls, [])

    def test_module_import_does_not_import_optional_sdks(self):
        script = ("import sys; import local_rag.embeddings; "
                  "assert 'openai' not in sys.modules; assert 'chromadb' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cached_inputs_do_not_spend_api_budget(self):
        client, sdk = self.client(replace(self.config, max_total_token_upper_bound=4))
        client.embed(["four"])
        _, report = client.embed(["four", "four"])
        self.assertEqual(report["reserved_token_upper_bound"], 0)
        self.assertEqual(report["cache_hits"], 1)
        self.assertEqual(len(sdk.calls), 1)

    def test_every_input_preflight_precedes_first_call(self):
        client, sdk = self.client(replace(self.config, max_input_bytes=8, batch_size=1))
        for texts in (["ok", "too long input"], ["ok", "   "], ["ok", None]):
            with self.assertRaises(EmbeddingError):
                client.embed(texts)
            self.assertEqual(sdk.calls, [])

    def test_budget_preflight_does_not_partially_send(self):
        client, sdk = self.client(replace(self.config, max_total_token_upper_bound=5, batch_size=1))
        with self.assertRaisesRegex(EmbeddingError, "budget"):
            client.embed(["abc", "def"])
        self.assertEqual(sdk.calls, [])

    def test_request_batches_respect_count_and_utf8_bounds(self):
        config = replace(self.config, max_input_bytes=8, max_request_token_upper_bound=8, batch_size=2)
        client, sdk = self.client(config)
        client.embed(["가나", "abc", "12345", "xy"])
        self.assertEqual([len(c["input"]) for c in sdk.calls], [1, 2, 1])
        for request in sdk.calls:
            self.assertLessEqual(sum(len(s.encode("utf-8")) for s in request["input"]), 8)
            self.assertEqual(request["encoding_format"], "float")

    def test_response_indexes_restore_order_and_exact_input_cache(self):
        client, sdk = self.client()
        vectors, report = client.embed(["first", "second", "first"])
        self.assertEqual([v[0] for v in vectors], [1., 2., 1.])
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(report["actual_prompt_tokens"], 2)
        _, report = client.embed(["first", "second"])
        self.assertEqual(report["cache_hits"], 2)
        self.assertEqual(len(sdk.calls), 1)
        client.embed(["first "])
        self.assertEqual(len(sdk.calls), 2)
        self.assertEqual(report["budget_unit"], "utf8_bytes_token_upper_bound")

    def test_dimensions_and_model_are_part_of_cache_key(self):
        client, _ = self.client()
        client.embed(["same"])
        other, sdk = self.client(replace(self.config, dimensions=2))
        self.assertEqual(len(other.embed(["same"])[0][0]), 2)
        self.assertEqual(len(sdk.calls), 1)
        third, sdk = self.client(replace(self.config, model="text-embedding-3-large"))
        third.embed(["same"])
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(self.cache_count(), 3)

    def test_invalid_response_does_not_commit_partial_batch(self):
        for failure in ("nan", "dimension", "index"):
            client, sdk = self.client(sdk=FakeSDK(failure))
            with self.assertRaises(EmbeddingError):
                client.embed(["one", "two"])
            self.assertEqual(self.cache_count(), 0)

    def test_api_errors_are_sanitized_and_failed_attempt_reserves_budget(self):
        client, sdk = self.client(replace(self.config, max_total_token_upper_bound=4), FakeSDK("exception"))
        with self.assertRaises(EmbeddingError) as error:
            client.embed(["four"])
        self.assertNotIn("DO_NOT_LEAK", str(error.exception))
        with self.assertRaisesRegex(EmbeddingError, "budget"):
            client.embed(["four"])
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(self.cache_count(), 0)

    def test_corrupt_cache_rejected_before_network_call(self):
        client, sdk = self.client()
        client.embed(["cached"])
        with sqlite3.connect(self.root / "cache" / "embeddings.sqlite3") as connection:
            connection.execute("UPDATE vectors SET vector_json='[999,999,999]'")
        with self.assertRaisesRegex(EmbeddingError, "integrity"):
            client.embed(["new input", "cached"])
        self.assertEqual(len(sdk.calls), 1)

    def test_config_rejects_unverified_limits(self):
        for kwargs in ({"batch_size": 2049}, {"max_input_bytes": 8193},
                       {"max_request_token_upper_bound": 300001}, {"dimensions": 1537},
                       {"model": "unknown"}, {"dimensions": True}):
            with self.assertRaises(ValueError):
                EmbeddingConfig(**kwargs)

    def test_source_sha_allowlist_failure_precedes_api_and_chroma(self):
        index, sdk, chroma = self.index()
        with self.assertRaises(EmbeddingError):
            index.build([chunk()], {"doc1": "0" * 64})
        self.assertEqual(sdk.calls, [])
        self.assertEqual(chroma.calls, [])

    def test_reference_views_and_unreviewed_transcription_rejected(self):
        index, sdk, chroma = self.index()
        for invalid in (dict(chunk(), source_method="unreviewed_ocr"),
                        dict(chunk(), source_spans=[{"collection": "reference_blocks"}]),
                        dict(chunk(), source_spans=[{"collection": "supplemental_blocks"}]),
                        dict(chunk(), quality_flags=["unreviewed_ocr"])):
            with self.assertRaises(EmbeddingError):
                self.build(index, [invalid])
        self.assertEqual(sdk.calls, [])

    def test_existing_collection_fingerprint_mismatch_rejected(self):
        chroma = FakeChroma()
        chroma.collection = FakeCollection({"embedding_model": "different"})
        index, sdk, _ = self.index(chroma=chroma)
        with self.assertRaisesRegex(EmbeddingError, "metadata"):
            self.build(index)
        self.assertEqual(sdk.calls, [])

    def test_build_uses_explicit_embeddings_and_cosine_metadata(self):
        index, sdk, chroma = self.index()
        report = self.build(index, [chunk("c1"), chunk("c2", "doc2")])
        self.assertEqual(report["chunk_count"], 2)
        self.assertIsNone(chroma.calls[0]["embedding_function"])
        self.assertEqual(chroma.calls[0]["configuration"], {"hnsw": {"space": "cosine"}})
        self.assertNotIn("documents", chroma.collection.rows["c1"])
        self.assertTrue((self.root / "index" / "manifest.json").exists())

    def test_alias_scope_applied_before_query_embedding(self):
        index, sdk, chroma = self.index()
        self.build(index, [chunk("c1"), chunk("c2", "doc2")])
        initial = len(sdk.calls)
        self.assertEqual(index.search("budget?", allowed_doc_ids=[]), [])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            index.search("budget?", allowed_doc_ids=["not_present"])
        self.assertEqual(len(sdk.calls), initial)
        hits = index.search("budget?", allowed_doc_ids=["alias_doc2"])
        self.assertEqual([h["doc_id"] for h in hits], ["doc2"])
        self.assertEqual(chroma.collection.queries[-1]["where"], {"doc_id": "doc2"})
        self.assertEqual(hits[0]["score"], -hits[0]["distance"])
        self.assertIn("not_confidence", hits[0]["score_kind"])
        self.assertEqual(hits[0]["locator"], "p1")

    def test_out_of_scope_collection_response_rejected(self):
        index, sdk, chroma = self.index()
        self.build(index, [chunk("c1"), chunk("c2", "doc2")])
        chroma.collection.scope_violation = True
        with self.assertRaisesRegex(EmbeddingError, "out-of-scope"):
            index.search("question", allowed_doc_ids=["doc2"])

    def test_load_and_payload_integrity(self):
        index, sdk, chroma = self.index()
        self.build(index)
        reloaded = OpenAIChromaIndex(self.root / "index", self.config,
                                    index.embeddings, chroma).load()
        self.assertEqual(reloaded.resolve_doc_ids(["alias_doc1"]), ["doc1"])
        manifest_path = self.root / "index" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["chunks"][0]["text"] = "tampered"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(EmbeddingError, "integrity"):
            OpenAIChromaIndex(self.root / "index", self.config, index.embeddings, chroma).load()

    def test_corpus_or_model_cannot_silently_reuse_local_index(self):
        index, sdk, chroma = self.index()
        self.build(index)
        before = len(sdk.calls)
        with self.assertRaisesRegex(EmbeddingError, "different corpus"):
            self.build(index, [chunk(text="source version changed")])
        self.assertEqual(len(sdk.calls), before)
        changed_config = replace(self.config, dimensions=2)
        embedding = OpenAIEmbeddingClient(self.root / "cache", changed_config, sdk)
        with self.assertRaisesRegex(EmbeddingError, "integrity/model"):
            OpenAIChromaIndex(self.root / "index", changed_config, embedding, chroma).load()


if __name__ == "__main__":
    unittest.main()
