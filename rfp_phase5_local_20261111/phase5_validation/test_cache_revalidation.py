"""Synthetic cache records exercise source replay only, never model accuracy."""
from dataclasses import asdict
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_phase5_summary as fixture
from rfp_phase4.llm import OpenAIGroundedLLM, LLMConfig
from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase5.fields import FIELD_SCHEMA, FIELD_QUESTION
from rfp_phase5.summary import FieldSummarizer, TASK

spec = importlib.util.spec_from_file_location('cache_revalidation_under_test', Path(__file__).with_name('cache_revalidation.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CacheSourceReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        docs, chunks = fixture.corpus(1)
        self.env = {'root': self.root, 'index': fixture.Index(chunks), 'registry': EvidenceRegistry(docs, chunks)}
        self.llm = OpenAIGroundedLLM(self.root/'cache', config=LLMConfig(enabled=True, max_output_tokens=6000),
                                   allowed_sources={'d1': 'a'*64})
        self.component = {'llm': self.llm}
        self.plan = FieldSummarizer(self.env['index'], self.llm, self.env['registry']).plan('d1')
        self.prepared = self.llm._prepare(TASK, FIELD_QUESTION, self.plan['batches'][0]['chunks'], schema=FIELD_SCHEMA)
        # Entirely synthetic response, deliberately marked as such by this test's name/docstring.
        payload = {'status': 'answered', 'claims': [{'doc_id': 'd1', 'category': 'budget',
            'text': '[budget_vat|fact] 원문 인용 연결을 검사하는 합성 테스트 주장',
            'citations': [{'chunk_id': chunks[0]['chunk_id'], 'quote': chunks[0]['text'][:95]}]}],
            'missing_information': []}
        self.response = {'payload': payload, 'request_fingerprint': self.prepared['fingerprint'],
            'execution_mode': 'live_openai', 'live_api': True, 'cache_hit': False, 'usage': {}}
        self.saved = {'doc_id': 'd1', 'plan_fingerprint': self.plan['plan_fingerprint'],
            'source_chunk_ids': self.plan['source_chunk_ids'], 'batch_outcomes': [{
                'batch_id': 'map-0001', 'chunk_ids': self.plan['source_chunk_ids'],
                'reason': 'source_citation_validation_failed', 'request_fingerprint': self.prepared['fingerprint']}]}
        self.path = self.root/'result.json'
        self.write_saved()

    def write_saved(self):
        self.path.write_text(json.dumps(self.saved), encoding='utf-8')

    def run_replay(self):
        with patch('socket.socket', side_effect=AssertionError('network prohibited')), patch.object(self.llm, 'generate', side_effect=AssertionError('generate prohibited')):
            return m.revalidate_cached_batches(self.env, self.path, self.component, self.plan)

    def test_matching_cache_replays_without_key_sdk_or_calls_and_preserves_bytes(self):
        self.llm._cache_write(self.prepared, self.response)
        before = self.path.read_bytes(), self.llm._cache_path().read_bytes()
        with patch.dict('os.environ', {}, clear=True): r = self.run_replay()
        self.assertTrue(r['batches'][0]['valid'])
        self.assertEqual(r['additional_api_calls'], 0)
        self.assertEqual(r['batches'][0]['claims_revalidated'], 1)
        self.assertFalse(r['full_document_complete'])
        self.assertFalse(r['answer_semantic_quality_verified'])
        self.assertEqual(before, (self.path.read_bytes(), self.llm._cache_path().read_bytes()))

    def test_missing_cache_does_not_reconstruct_claims_or_generate(self):
        r = self.run_replay()
        self.assertEqual(r['batches'][0]['status'], 'cache_not_available')
        self.assertEqual(r['revalidated_batch_count'], 0)
        self.assertNotIn('revalidated_claims', r['batches'][0])

    def test_request_fingerprint_mismatch_is_rejected(self):
        self.saved['batch_outcomes'][0]['request_fingerprint'] = 'b'*64; self.write_saved()
        with self.assertRaises(ValueError): self.run_replay()

    def test_wrong_document_or_source_ids_are_rejected(self):
        for key, value in [('doc_id', 'other'), ('source_chunk_ids', ['other'])]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.saved)
                self.saved[key] = value; self.write_saved()
                with self.assertRaises(ValueError): self.run_replay()
                self.saved = original; self.write_saved()

    def test_fabricated_quote_is_still_rejected(self):
        self.response['payload']['claims'][0]['citations'][0]['quote'] = '조작된 사업비 999조원'
        self.llm._cache_write(self.prepared, self.response)
        r = self.run_replay()
        self.assertFalse(r['batches'][0]['valid'])
        self.assertEqual(r['batches'][0]['revalidated_claims'], [])

    def test_test_client_namespace_cannot_be_accepted_as_actual_model(self):
        self.response['execution_mode'] = 'injected_test_client'
        self.llm._cache_write(self.prepared, self.response)
        with self.assertRaises(Exception): self.run_replay()


if __name__ == '__main__': unittest.main()
