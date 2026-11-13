"""Offline semantic workflow contracts; fake vectors are not quality results."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_rag.embeddings import EmbeddingConfig, EmbeddingError, OpenAIEmbeddingClient
from rfp_phase4.semantic import (SemanticError, build_semantic_index, compare_semantic_on_dev,
                                plan_semantic_index)


def chunk(doc_id, number=1):
    text = f'{doc_id} 예산 {number}억원'
    return {'chunk_id': f'{doc_id}-{number}', 'doc_id': doc_id, 'canonical_doc_id': doc_id,
            'alias_doc_ids': ['alias-' + doc_id], 'sha256': hashlib.sha256(doc_id.encode()).hexdigest(),
            'text': text, 'search_text': text, 'file_name': doc_id + '.pdf',
            'source_method': 'raw_parser', 'kind': 'prose', 'quality_flags': [],
            'source_spans': [{'collection': 'blocks', 'block_id': 'p1'}], 'locator': 'page1'}


class LocalIndex:
    def __init__(self):
        self.chunks = [chunk('A'), chunk('A', 2), chunk('B'), chunk('C')]
        self.calls = []

    def get_catalog(self):
        return [{'doc_id': doc_id, 'sha256': hashlib.sha256(doc_id.encode()).hexdigest(),
                 'file_name': doc_id + '.pdf', 'agency': '기관'} for doc_id in ['A', 'B', 'C']]

    def resolve_doc_ids(self, ids):
        if isinstance(ids, str): ids = [ids]
        result = []
        for doc_id in ids:
            canonical = doc_id.removeprefix('alias-')
            if canonical not in {'A', 'B', 'C'}: raise ValueError('unknown')
            if canonical not in result: result.append(canonical)
        return result

    def get_document_chunks(self, doc_id):
        canonical = self.resolve_doc_ids([doc_id])
        return json.loads(json.dumps([c for c in self.chunks if c['doc_id'] in canonical]))

    def search(self, query, top_k=5, allowed_doc_ids=None, **kwargs):
        self.calls.append({'query': query, 'allowed_doc_ids': allowed_doc_ids, **kwargs})
        scope = self.resolve_doc_ids(allowed_doc_ids) if allowed_doc_ids is not None else ['A', 'B', 'C']
        return [dict(c, rank=i + 1, score=1., score_is_confidence=False) for i, c in
                enumerate([c for c in self.chunks if c['doc_id'] in scope][:top_k])]


class SDK:
    def __init__(self): self.embeddings, self.calls = self, []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {'model': kwargs['model'], 'usage': {'prompt_tokens': len(kwargs['input']), 'total_tokens': len(kwargs['input'])},
                'data': [{'index': i, 'embedding': [1.] + [.5] * (kwargs['dimensions'] - 1)}
                         for i in range(len(kwargs['input']))]}


class Collection:
    def __init__(self, metadata): self.metadata, self.rows = metadata, {}
    def count(self): return len(self.rows)
    def upsert(self, ids, embeddings, metadatas):
        for cid, vector, metadata in zip(ids, embeddings, metadatas):
            self.rows[cid] = {'vector': vector, 'metadata': metadata}
    def query(self, **kwargs):
        condition = kwargs.get('where', {}).get('doc_id')
        if isinstance(condition, dict): condition = condition['$in']
        elif isinstance(condition, str): condition = [condition]
        rows = [(cid, row) for cid, row in self.rows.items()
                if condition is None or row['metadata']['doc_id'] in condition][:kwargs['n_results']]
        return {'ids': [[cid for cid, row in rows]], 'distances': [[.2] * len(rows)],
                'metadatas': [[row['metadata'] for cid, row in rows]]}


class Chroma:
    def __init__(self): self.calls, self.collection = [], None
    def get_or_create_collection(self, **kwargs):
        self.calls.append(kwargs)
        if self.collection is None: self.collection = Collection(kwargs['metadata'])
        return self.collection


def case(case_id='q1', ids=None, query='질문 원문', split='dev', feature='fact'):
    return {'case_id': case_id, 'split': split, 'feature': feature, 'question': query,
            'context': {'selected_doc_ids': ['A'] if ids is None else ids},
            'target_doc_ids': ['C'], 'required_facts': [{'fact_id': 'f1', 'doc_id': 'A',
                'expected_value': 'SECRET_GOLD_VALUE', 'evidence': [{'quote': 'GOLD_QUOTE'}]}]}


def alignment(cases):
    return {'document_aliases': {}, 'records': [{'link_id': c['case_id'] + '|fact|f1|1', 'mapped': True,
        'alternatives': [{'required_non_whitespace_characters': 1, 'chunk_coverage': {'A-1': [[0, 1]]}}]}
        for c in cases]}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root, self.local = Path(self.temp.name), LocalIndex()
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'fake-test-key'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.config = EmbeddingConfig(enabled=True, dimensions=3, collection_name='phase4_test')

    def build(self, ids=('A', 'B'), config=None, registry=None):
        config = config or self.config
        sdk, chroma = SDK(), Chroma()
        client = OpenAIEmbeddingClient(self.root / 'cache', config=config, client=sdk)
        index, report = build_semantic_index(self.local, ids, self.root / 'semantic', config=config,
                            registry=registry, embedding_client=client, chroma_client=chroma)
        return index, report, sdk, chroma

    def test_import_has_no_optional_sdk_side_effect(self):
        p = subprocess.run([sys.executable, '-c', 'import sys; import rfp_phase4.semantic; '
            'assert "openai" not in sys.modules; assert "chromadb" not in sys.modules'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_plan_is_offline_complete_deduplicated_and_byte_bounded(self):
        with patch.dict(os.environ, {}, clear=True):
            plan = plan_semantic_index(self.local, ['alias-A', 'A', 'B'])
        self.assertEqual(plan['canonical_doc_ids'], ['A', 'B'])
        self.assertEqual(plan['chunk_count'], 3)
        self.assertEqual(len(plan['allowed_sources']), 2)
        self.assertFalse(plan['embedding_config']['enabled'])
        self.assertEqual(plan['api_calls'], 0)
        self.assertTrue(plan['planning_only'])

    def test_missing_unknown_and_string_scope_are_rejected(self):
        for ids in [None, [], 'A', ['UNKNOWN']]:
            with self.subTest(ids=ids), self.assertRaises((ValueError, SemanticError)):
                plan_semantic_index(self.local, ids)

    def test_source_version_mismatch_is_rejected_before_network(self):
        self.local.chunks[0]['sha256'] = 'f' * 64
        with self.assertRaisesRegex(SemanticError, 'version'):
            plan_semantic_index(self.local, ['A'])

    def test_duplicate_chunk_and_large_embedding_input_are_rejected(self):
        self.local.chunks.append(dict(self.local.chunks[0]))
        with self.assertRaisesRegex(SemanticError, 'duplicate'): plan_semantic_index(self.local, ['A'])
        self.local.chunks.pop()
        with self.assertRaisesRegex(SemanticError, 'byte bound'):
            plan_semantic_index(self.local, ['A'], replace(self.config, max_input_bytes=4))

    def test_cold_cache_budget_fit_is_reported_not_silently_increased(self):
        plan = plan_semantic_index(self.local, ['A'], replace(self.config, max_total_token_upper_bound=1))
        self.assertFalse(plan['cold_cache_fits_total_budget'])
        self.assertEqual(plan['embedding_config']['max_total_token_upper_bound'], 1)

    def test_build_requires_enable_and_environment_key(self):
        sdk, chroma = SDK(), Chroma()
        config = replace(self.config, enabled=False)
        client = OpenAIEmbeddingClient(self.root / 'cache', config=config, client=sdk)
        with self.assertRaisesRegex(EmbeddingError, 'disabled'):
            build_semantic_index(self.local, ['A'], self.root/'semantic', config,
                                  embedding_client=client, chroma_client=chroma)
        self.assertEqual(sdk.calls, []); self.assertEqual(chroma.calls, [])
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(EmbeddingError, 'OPENAI_API_KEY'):
            self.build()

    def test_source_registry_validation_precedes_api(self):
        class Registry:
            def validate_chunks(self, chunks, allowed_doc_ids=None):
                raise ValueError('raw source mismatch')
        with self.assertRaisesRegex(ValueError, 'raw source mismatch'):
            self.build(registry=Registry())

    def test_fake_build_and_search_are_labelled_unverified_not_live_quality(self):
        index, report, sdk, chroma = self.build()
        self.assertEqual(report['chunk_count'], 3)
        self.assertFalse(report['quality_evaluation_performed'])
        self.assertFalse(report['live_service_integration_measured'])
        self.assertEqual(report['execution_mode'], 'injected_client_unverified')
        hits = index.search('질문', allowed_doc_ids=['alias-A'])
        self.assertTrue(all(h['doc_id'] == 'A' and h['score_is_confidence'] is False for h in hits))
        with self.assertRaises(ValueError): index.search('질문', method='bm25')
        self.assertEqual(index.resolve_doc_ids(['alias-A']), ['A'])
        self.assertEqual(len(index.get_document_chunks('A')), 2)

    def test_compare_uses_exact_question_and_user_scope_not_gold(self):
        index, _, sdk, _ = self.build()
        cases = [case(query='실제 사용자 질문')]
        report = compare_semantic_on_dev(self.local, index, cases, alignment(cases), top_ks=[1, 2])
        self.assertTrue(report['comparable'])
        self.assertFalse(report['live_semantic_quality_measured'])
        self.assertEqual(self.local.calls[0]['query'], cases[0]['question'])
        self.assertEqual(self.local.calls[0]['allowed_doc_ids'], ['A'])
        self.assertEqual(sdk.calls[-1]['input'], [cases[0]['question']])
        self.assertEqual(report['semantic_report']['api_calls'], 1)
        self.assertEqual(report['lexical_report']['metrics_at_k']['1']['micro_evidence_span_recall'], 1.)

    def test_global_lexical_baseline_has_identical_subset_corpus(self):
        index, _, _, _ = self.build(ids=['A'])
        cases = [case()]
        report = compare_semantic_on_dev(self.local, index, cases, alignment(cases), scope_mode='global')
        self.assertEqual(self.local.calls[0]['allowed_doc_ids'], ['A'])
        self.assertEqual(report['corpus_document_count'], 1)

    def test_cases_with_missing_context_documents_are_explicitly_not_comparable(self):
        index, _, sdk, _ = self.build(ids=['A'])
        cases = [case(ids=['A', 'C'])]
        before = len(sdk.calls)
        report = compare_semantic_on_dev(self.local, index, cases, alignment(cases))
        self.assertFalse(report['comparable'])
        self.assertIn('missing_user_selected', report['excluded_cases'][0]['reason'])
        self.assertEqual(len(sdk.calls), before)
        self.assertEqual(self.local.calls, [])

    def test_holdout_anywhere_is_blocked_before_query(self):
        index, _, sdk, _ = self.build()
        before = len(sdk.calls)
        for cases in [[case(split='holdout')], [case(), case('q2', split='holdout', feature='summary')]]:
            with self.assertRaisesRegex(ValueError, 'dev cases only'):
                compare_semantic_on_dev(self.local, index, cases, alignment(cases))
        self.assertEqual(len(sdk.calls), before)

    def test_changed_chunks_same_source_hash_cannot_be_called_fair(self):
        index, _, sdk, _ = self.build()
        index.backend._chunks['A-1']['text'] += ' changed'
        with self.assertRaisesRegex(SemanticError, 'same ordered source chunks'):
            compare_semantic_on_dev(self.local, index, [case()], alignment([case()]))

    def test_source_catalog_sha_mismatch_is_not_comparable(self):
        index, _, sdk, _ = self.build()
        index._catalog_rows[0]['sha256'] = 'f' * 64
        with self.assertRaisesRegex(SemanticError, 'different source versions'):
            compare_semantic_on_dev(self.local, index, [case()], alignment([case()]))

    def test_query_budget_preflight_precedes_all_searches(self):
        index, _, sdk, _ = self.build(config=replace(self.config, max_total_token_upper_bound=150))
        cases = [case(query='긴 질문 ' * 100)]
        before = len(sdk.calls)
        with self.assertRaisesRegex(SemanticError, 'remaining embedding budget'):
            compare_semantic_on_dev(self.local, index, cases, alignment(cases))
        self.assertEqual(len(sdk.calls), before); self.assertEqual(self.local.calls, [])

    def test_alias_context_is_accepted_and_unknown_ids_do_not_expand_scope(self):
        index, _, _, _ = self.build()
        cases = [case(ids=['alias-A']), case('q2', ids=['bad-id'])]
        report = compare_semantic_on_dev(self.local, index, cases, alignment(cases))
        self.assertEqual(report['case_count'], 1)
        self.assertEqual(report['excluded_cases'][0]['reason'], 'unknown_user_selected_document')

    def test_cache_usage_and_api_count_are_reported_separately(self):
        index, _, _, _ = self.build()
        cases = [case()]
        first = compare_semantic_on_dev(self.local, index, cases, alignment(cases))
        second = compare_semantic_on_dev(self.local, index, cases, alignment(cases))
        self.assertEqual(first['api_calls'], 1)
        self.assertEqual(second['api_calls'], 0)
        self.assertEqual(second['embedding_usage']['cache_hits'], 1)

    def test_max_cases_and_nonpositive_features_are_accounted_for(self):
        index, _, _, _ = self.build()
        cases = [case(), case('q2'), case('q3', feature='summary')]
        report = compare_semantic_on_dev(self.local, index, cases, alignment(cases), max_cases=1)
        self.assertEqual(report['case_count'], 1)
        self.assertEqual(len(report['excluded_cases']), 2)
        self.assertEqual([r['reason'] for r in report['excluded_cases']],
                         ['explicit_max_cases_limit', 'not_positive_fact_or_comparison'])

    def test_mock_index_cannot_be_reopened_as_live_quality(self):
        self.build()
        with self.assertRaisesRegex(SemanticError, 'mock/live execution mode'):
            build_semantic_index(self.local, ['A', 'B'], self.root/'semantic', config=self.config)

    def test_unlabelled_old_embedding_cache_requires_fresh_directory(self):
        cache = self.root/'semantic'/'cache'
        cache.mkdir(parents=True)
        (cache/'embeddings.sqlite3').touch()
        with self.assertRaisesRegex(SemanticError, 'no Phase 4 execution provenance'):
            build_semantic_index(self.local, ['A'], self.root/'semantic', config=self.config)


if __name__ == '__main__':
    unittest.main()
