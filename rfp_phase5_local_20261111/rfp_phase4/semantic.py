"""Opt-in semantic retrieval experiments over the exact local source corpus.

Planning requires no optional SDK, key, installation, download or API request.
Live builds reuse the bounded Phase 3 embedding adapter. Comparisons only accept
dev cases and restrict both searchers to the same documents and source chunks.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from local_rag.embeddings import EmbeddingConfig, EmbeddingError, OpenAIChromaIndex, OpenAIEmbeddingClient
from local_rag.evaluation import evaluate_retrieval


class SemanticError(RuntimeError):
    """Semantic experiment configuration or source-corpus error."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _config(config=None):
    if isinstance(config, EmbeddingConfig):
        return config
    values = {'collection_name': 'rfp_phase4_openai'}
    values.update(config or {})
    return EmbeddingConfig(**values)


def _catalog(index):
    rows = index.get_catalog()
    if not isinstance(rows, list) or not rows:
        raise SemanticError('A validated local source catalog is required')
    return {row['doc_id']: row for row in rows}


def _ids(index, ids):
    if ids is None or isinstance(ids, (str, bytes)):
        raise ValueError('Provide an explicit nonempty list of selected document IDs')
    requested = list(ids)
    if not requested:
        raise ValueError('Provide an explicit nonempty list of selected document IDs')
    return sorted(set(index.resolve_doc_ids(requested)))


def plan_semantic_index(index, doc_ids, config=None):
    """Return every selected document chunk and cold-cache cost upper bounds.

    UTF-8 bytes are a conservative input token upper bound, not an exact token
    count or a currency estimate. The plan does not inspect the embedding cache.
    Source text is inherited from the already validated local search index.
    """
    config = _config(config)
    catalog = _catalog(index)
    canonical = _ids(index, doc_ids)
    if not canonical or any(doc_id not in catalog for doc_id in canonical):
        raise SemanticError('Selected documents do not resolve to this local source catalog')
    chunks, seen = [], set()
    allowed = {doc_id: catalog[doc_id]['sha256'] for doc_id in canonical}
    for doc_id in canonical:
        document_chunks = index.get_document_chunks(doc_id)
        if not document_chunks:
            raise SemanticError('A selected source has no indexed chunks')
        for chunk in document_chunks:
            if (chunk.get('doc_id') != doc_id or chunk.get('canonical_doc_id') != doc_id
                    or chunk.get('sha256') != allowed[doc_id]):
                raise SemanticError('Chunk canonical identity or original source version changed')
            cid = chunk.get('chunk_id')
            if not isinstance(cid, str) or not cid or cid in seen:
                raise SemanticError('Selected source chunks have missing or duplicate IDs')
            seen.add(cid)
            text = chunk.get('search_text', chunk.get('text'))
            if not isinstance(text, str) or not text.strip():
                raise SemanticError('A selected source chunk has no embedding input')
            if len(text.encode('utf-8')) > config.max_input_bytes:
                raise SemanticError('A selected chunk exceeds the embedding input byte bound')
            chunks.append(json.loads(_json(chunk)))
    if len(chunks) > config.max_inputs_per_operation:
        raise SemanticError('Selected corpus exceeds the configured embedding input-count bound')
    unique_inputs = list(dict.fromkeys(chunk.get('search_text', chunk['text']) for chunk in chunks))
    total_bytes = sum(len(text.encode('utf-8')) for text in unique_inputs)
    batches, count, batch_bytes = 0, 0, 0
    for text in unique_inputs:
        size = len(text.encode('utf-8'))
        if count and (count == config.batch_size or batch_bytes + size > config.max_request_token_upper_bound):
            batches += 1
            count, batch_bytes = 0, 0
        count += 1
        batch_bytes += size
    batches += bool(count)
    return {'canonical_doc_ids': canonical, 'allowed_sources': allowed, 'chunks': chunks,
            'chunk_count': len(chunks), 'unique_embedding_inputs': len(unique_inputs),
            'cold_cache_input_token_upper_bound': total_bytes,
            'cold_cache_embedding_requests_upper_bound': int(batches),
            'cold_cache_fits_total_budget': total_bytes <= config.max_total_token_upper_bound,
            'corpus_fingerprint': _digest({'chunks': chunks, 'sources': allowed}),
            'embedding_config': asdict(config), 'budget_unit': 'utf8_bytes_token_upper_bound',
            'budget_note': 'Cold-cache upper bound; cache reuse may reduce API calls. Not exact tokens or monetary cost.',
            'api_calls': 0, 'planning_only': True, 'semantic_quality_measured': False}


class _MeasuredEmbeddings:
    """Record actual reports without altering the existing embedding adapter."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.config = delegate.config
        self.reports = []
        self.failed_operations = 0

    def _guard(self):
        return self.delegate._guard()

    def embed(self, texts):
        try:
            vectors, report = self.delegate.embed(texts)
        except Exception:
            self.failed_operations += 1
            raise
        self.reports.append(dict(report))
        return vectors, report

    def usage_report(self):
        fields = ['api_calls', 'cache_hits', 'actual_prompt_tokens', 'actual_total_tokens',
                  'reserved_token_upper_bound']
        result = {field: sum(r.get(field, 0) for r in self.reports) for field in fields}
        result.update({'failed_operations': self.failed_operations,
                       'api_calls_complete': self.failed_operations == 0,
                       'budget_unit': 'utf8_bytes_token_upper_bound'})
        return result


class SemanticSearchIndex:
    """Phase 4 interface around the existing local Chroma implementation."""
    def __init__(self, backend, catalog, measured_embeddings, execution_mode):
        self.backend = backend
        self._catalog_rows = json.loads(_json(catalog))
        self.embeddings = measured_embeddings
        self.execution_mode = execution_mode

    def get_catalog(self):
        return json.loads(_json(self._catalog_rows))

    def get_metadata(self):
        return {'method': 'openai_chroma', 'corpus_doc_ids': [r['doc_id'] for r in self._catalog_rows],
                'embedding_config': asdict(self.backend.config), 'execution_mode': self.execution_mode,
                'scores_are_confidence': False,
                'stored_vector_integrity': 'Metadata/count checked; stored vector contents are not exhaustively tamper checked.'}

    def resolve_doc_ids(self, ids):
        return self.backend.resolve_doc_ids(ids)

    def get_document_chunks(self, doc_id):
        canonical = self.resolve_doc_ids([doc_id])
        if self.backend._chunks is None:
            self.backend.load()
        return [json.loads(_json(chunk)) for chunk in self.backend._chunks.values()
                if chunk['doc_id'] in canonical]

    def search(self, query, top_k=5, allowed_doc_ids=None, method='openai_chroma', fetch_k=40, mmr_lambda=.75):
        if method != 'openai_chroma':
            raise ValueError('SemanticSearchIndex supports method=openai_chroma only')
        hits = self.backend.search(query, top_k=top_k, allowed_doc_ids=allowed_doc_ids)
        for hit in hits:
            hit['score_is_confidence'] = False
        return hits

    def preflight_queries(self, queries):
        """Conservative cold-query preflight; never sends or warms query embeddings."""
        self.embeddings._guard()
        config = self.backend.config
        queries = list(dict.fromkeys(queries))
        if len(queries) > config.max_inputs_per_operation:
            raise SemanticError('Comparison query count exceeds the configured bound')
        for query in queries:
            if not isinstance(query, str) or not query.strip() or len(query.encode('utf-8')) > config.max_input_bytes:
                raise SemanticError('A comparison query is blank or exceeds the embedding input byte bound')
        byte_bound = sum(len(query.encode('utf-8')) for query in queries)
        reserved = getattr(self.embeddings.delegate, '_reserved', 0)
        if reserved + byte_bound > config.max_total_token_upper_bound:
            raise SemanticError('Comparison cold-query upper bound exceeds the remaining embedding budget')
        return {'unique_queries': len(queries), 'cold_query_token_upper_bound': byte_bound,
                'already_reserved_token_upper_bound': reserved, 'api_calls': 0,
                'note': 'Conservative query preflight, even if a prior exact-query cache entry exists.'}

    def usage_report(self):
        return self.embeddings.usage_report()


def build_semantic_index(index, doc_ids, index_dir, config=None, *, registry=None,
                         embedding_client=None, chroma_client=None):
    """Explicit opt-in build; original lexical index and its reports are untouched."""
    config = _config(config)
    plan = plan_semantic_index(index, doc_ids, config)
    if registry is not None:
        plan['chunks'] = registry.validate_chunks(plan['chunks'], allowed_doc_ids=plan['canonical_doc_ids'])
    delegate = embedding_client or OpenAIEmbeddingClient(Path(index_dir) / 'cache', config)
    if delegate.config != config:
        raise SemanticError('Embedding client and semantic experiment configurations differ')
    measured = _MeasuredEmbeddings(delegate)
    measured._guard()
    index_dir = Path(index_dir)
    mode = 'injected_client_unverified' if embedding_client is not None or chroma_client is not None else 'openai_live'
    # Phase 3's exact cache has no mock/live provenance. Bind this directory to
    # its Phase 4 execution mode before any SDK work, including failed builds,
    # so fake test vectors cannot later be presented as live semantic quality.
    provenance_path = index_dir / 'phase4_semantic_provenance.json'
    expected_provenance = {'schema_version': 1, 'execution_mode': mode,
                           'corpus_fingerprint': plan['corpus_fingerprint'],
                           'model': config.model, 'dimensions': config.dimensions}
    if provenance_path.exists():
        try:
            existing_provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise SemanticError('Semantic experiment provenance cannot be validated') from None
        if any(existing_provenance.get(key) != value for key, value in expected_provenance.items()):
            raise SemanticError('Use a fresh semantic directory for a different corpus, model, or mock/live execution mode')
    elif (index_dir / 'manifest.json').exists() or (index_dir / 'cache' / 'embeddings.sqlite3').exists():
        raise SemanticError('Existing semantic data has no Phase 4 execution provenance; use a fresh directory')
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = provenance_path.with_suffix('.json.tmp')
    temporary.write_text(_json(dict(expected_provenance, build_status='pending')), encoding='utf-8')
    temporary.replace(provenance_path)
    backend = OpenAIChromaIndex(index_dir, config=config, embedding_client=measured, chroma_client=chroma_client)
    # Existing adapter checks enabled and local environment key before importing SDKs.
    report = backend.build(plan['chunks'], allowed_sources=plan['allowed_sources'])
    temporary.write_text(_json(dict(expected_provenance, build_status='complete')), encoding='utf-8')
    temporary.replace(provenance_path)
    catalog = _catalog(index)
    wrapped = SemanticSearchIndex(backend, [catalog[doc_id] for doc_id in plan['canonical_doc_ids']], measured, mode)
    report.update({'execution_mode': mode, 'planning_fingerprint': plan['corpus_fingerprint'],
                   'canonical_doc_ids': plan['canonical_doc_ids'],
                   'quality_evaluation_performed': False,
                   'live_service_integration_measured': mode == 'openai_live',
                   'source_validation': 'raw_source_registry' if registry is not None else 'validated_local_index_provenance',
                   'original_lexical_index_modified': False})
    return wrapped, report


class _SameCorpusLexical:
    def __init__(self, index, doc_ids):
        self.index = index
        self.doc_ids = sorted(doc_ids)

    def search(self, query, top_k=5, allowed_doc_ids=None, **kwargs):
        scope = self.doc_ids if allowed_doc_ids is None else self.index.resolve_doc_ids(allowed_doc_ids)
        if not set(scope).issubset(self.doc_ids):
            raise SemanticError('A lexical comparison scope escapes the semantic source corpus')
        return self.index.search(query, top_k=top_k, allowed_doc_ids=scope, **kwargs)


def compare_semantic_on_dev(index, semantic_index, cases, alignment, *, top_ks=(5, 10, 20),
                            max_cases=10, scope_mode='context'):
    """Compare BM25 and OpenAI/Chroma with identical corpus, questions and labels.

    Eligibility depends only on user context selection, never target_doc_ids or
    gold facts. A subset experiment is not a full 99-document benchmark.
    """
    cases = list(cases)
    if any(case.get('split') != 'dev' for case in cases):
        raise ValueError('Phase 4 semantic comparisons accept dev cases only; holdout is blocked')
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError('max_cases must be a positive integer')
    if scope_mode not in {'context', 'global'}:
        raise ValueError('scope_mode must be context or global')
    top_ks = tuple(sorted(set(top_ks)))
    if not top_ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in top_ks):
        raise ValueError('top_ks must contain positive integers')
    local_catalog, semantic_catalog = _catalog(index), _catalog(semantic_index)
    corpus = set(semantic_catalog)
    for doc_id in corpus:
        if doc_id not in local_catalog or semantic_catalog[doc_id]['sha256'] != local_catalog[doc_id]['sha256']:
            raise SemanticError('The semantic and lexical corpora have different source versions')
        lexical_chunks = index.get_document_chunks(doc_id)
        semantic_chunks = semantic_index.get_document_chunks(doc_id)
        if _digest(lexical_chunks) != _digest(semantic_chunks):
            raise SemanticError('A fair comparison requires the same ordered source chunks')
    selected, excluded = [], []
    for case in cases:
        if case.get('feature') not in {'fact', 'comparison'} or not case.get('required_facts'):
            excluded.append({'case_id': case.get('case_id'), 'reason': 'not_positive_fact_or_comparison'})
            continue
        context = case.get('context', {}).get('selected_doc_ids')
        if context is not None:
            try:
                canonical = set(index.resolve_doc_ids(context))
            except (KeyError, ValueError):
                excluded.append({'case_id': case.get('case_id'), 'reason': 'unknown_user_selected_document'})
                continue
            if not canonical or not canonical.issubset(corpus):
                excluded.append({'case_id': case.get('case_id'), 'reason': 'semantic_corpus_missing_user_selected_documents'})
                continue
        if len(selected) >= max_cases:
            excluded.append({'case_id': case.get('case_id'), 'reason': 'explicit_max_cases_limit'})
            continue
        selected.append(case)
    if not selected:
        return {'comparable': False, 'case_count': 0, 'corpus_doc_ids': sorted(corpus),
                'excluded_cases': excluded, 'api_calls': 0,
                'reason': 'No eligible dev fact/comparison cases fit the selected semantic corpus'}
    query_plan = semantic_index.preflight_queries([case['question'] for case in selected])
    before = semantic_index.usage_report()
    lexical = evaluate_retrieval(_SameCorpusLexical(index, corpus), selected, alignment,
                                  top_ks=top_ks, method='bm25', scope_mode=scope_mode)
    semantic = evaluate_retrieval(semantic_index, selected, alignment,
                                  top_ks=top_ks, method='openai_chroma', scope_mode=scope_mode)
    after = semantic_index.usage_report()
    usage_delta = {key: after[key] - before.get(key, 0) for key in
                   ['api_calls', 'cache_hits', 'actual_prompt_tokens', 'actual_total_tokens', 'reserved_token_upper_bound']}
    semantic['api_calls'] = usage_delta['api_calls']
    semantic['embedding_usage'] = usage_delta
    semantic['latency_note'] = ('Per-query latency includes query embedding lookup/API on cache misses and local Chroma search; '
                                'cache state is reported, so this is not guaranteed cold-API latency.')
    semantic['execution_mode'] = semantic_index.execution_mode
    return {'comparable': True, 'split': 'dev', 'case_count': len(selected),
            'corpus_doc_ids': sorted(corpus), 'corpus_document_count': len(corpus),
            'same_ordered_source_chunks': True, 'scope_protocol': scope_mode,
            'query_policy': 'Exact case.question; scope from context.selected_doc_ids only; no gold query enrichment.',
            'excluded_cases': excluded, 'query_budget_plan': query_plan,
            'lexical_report': lexical, 'semantic_report': semantic,
            'api_calls': usage_delta['api_calls'], 'embedding_usage': usage_delta,
            'execution_mode': semantic_index.execution_mode,
            'live_semantic_quality_measured': semantic_index.execution_mode == 'openai_live',
            'answer_generation_measured': False,
            'limitations': ['Subset corpus results are not full-corpus retrieval quality.',
                            'Evidence-span recovery is not generated-answer factual correctness.',
                            'Exact model query cache reuse can reduce API calls and latency.',
                            'Chroma metadata/count validation does not detect every stored-vector alteration.']}
