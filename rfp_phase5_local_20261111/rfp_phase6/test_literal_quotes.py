"""Closed quotation choices, strict failure and aggregate budget invariants."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase4.llm import LLMConfig, LLMError, OpenAIGroundedLLM, COMPARISON_DOCUMENT_TASK
from rfp_phase6.literal_quotes import (QA_TASK, LiteralQuoteClient, accepts_citations,
                                     literal_options, literal_schema)
from rfp_phase6.runtime import make_conversation
from rfp_phase6.test_focus import GroundedClient, synthetic_environment


def citations(schema):
    return schema['properties']['claims']['items']['properties']['citations']['items']['anyOf']


class LiteralQuoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = synthetic_environment(Path(self.tmp.name))
        self.addCleanup(self.env['index'].close)
        key = patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-not-a-real-key'})
        key.start()
        self.addCleanup(key.stop)

    def llm(self, *, max_calls=8, cache=False, client=None):
        client = client or GroundedClient()
        delegate = OpenAIGroundedLLM(self.env['root'] / 'cache',
            LLMConfig(enabled=True, max_calls=max_calls, cache_enabled=cache),
            allowed_sources=self.env['registry'].source_hashes, client=client)
        return LiteralQuoteClient(delegate), delegate, client

    def test_each_choice_is_a_contiguous_original_substring_and_preserves_characters(self):
        text = '대상 업무\nFUN-공통-101 사용 승인\n' + '반출 수량은 137개, 부가세 포함. ' * 80
        options = literal_options(text)
        self.assertTrue(options)
        self.assertTrue(all(q in text for q in options))
        covered = set()
        for q in options:
            start = text.find(q)
            while start >= 0:
                covered.update(range(start, start + len(q)))
                start = text.find(q, start + 1)
        self.assertTrue(all(c.isspace() or i in covered for i, c in enumerate(text)))
        self.assertTrue(any('137개' in q for q in options))

    def test_ellipsis_joined_outline_is_not_a_permitted_quote(self):
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        schema = literal_schema(chunks)
        cid = chunks[0]['chunk_id']
        payload = {'claims': [{'doc_id': 'A', 'citations': [{'chunk_id': cid,
            'quote': '사업 범위 ... 반출 신청 ... 승인 처리 ... 반환 이력'}]}]}
        self.assertFalse(accepts_citations(payload, schema))

    def test_source_ellipsis_is_allowed_only_when_it_is_actually_in_the_source(self):
        chunks = copy.deepcopy(self.env['index'].get_document_chunks('A')[-1:])
        chunks[0]['text'] = '원문 자체의 표기 ... 를 그대로 보존한다.'
        schema = literal_schema(chunks)
        self.assertEqual(citations(schema)[0]['properties']['quote']['enum'], [chunks[0]['text']])

    def test_schema_pairs_each_quote_with_its_actual_chunk(self):
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        schema = literal_schema(chunks)
        for branch, chunk in zip(citations(schema), chunks):
            self.assertEqual(branch['properties']['chunk_id']['enum'], [chunk['chunk_id']])
            self.assertFalse(branch['additionalProperties'])
            for quote in branch['properties']['quote']['enum']:
                result = self.env['registry'].validate_answer({'status': 'answered', 'claims': [{
                    'doc_id': 'A', 'category': 'requirements', 'text': quote,
                    'citations': [{'chunk_id': chunk['chunk_id'], 'quote': quote}]}],
                    'missing_information': []}, chunks, allowed_doc_ids=['A'])
                self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(ANSWER_BASELINE, self._schema_hash())

    @staticmethod
    def _schema_hash():
        from rfp_phase4.evidence import ANSWER_SCHEMA
        return hashlib.sha256(json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()).hexdigest()

    def test_empty_duplicate_and_excessive_choices_fail_before_dispatch(self):
        for evidence in [[], self.env['index'].get_document_chunks('A')[-1:] * 2]:
            with self.assertRaises(ValueError):
                literal_schema(evidence)
        with self.assertRaises(ValueError):
            literal_options(' \n ')
        huge = copy.deepcopy(self.env['index'].get_document_chunks('A')[-1:])
        huge[0]['text'] = '\n'.join(f'{i:04d} 별도 원문 구간 ' + '가' * 150 for i in range(300))
        with self.assertRaises(ValueError):
            literal_schema(huge)

    def test_preflight_and_generation_use_identical_transformed_requests_and_one_client(self):
        wrapper, delegate, client = self.llm()
        evidence = self.env['index'].get_document_chunks('A')[-4:]
        request = {'task': 'qa', 'question': '대상 업무는?', 'evidence': evidence}
        with patch.object(delegate, 'preflight_requests', wraps=delegate.preflight_requests) as planned:
            wrapper.preflight_requests([request])
            result = wrapper.generate(**request)
        expected = planned.call_args.args[0][0]
        self.assertEqual(expected['task'], QA_TASK)
        self.assertEqual(client.bodies[0]['text']['format']['schema'], expected['schema'])
        body = json.loads(client.bodies[0]['input'][1]['content'])
        self.assertEqual(body['question'], expected['question'])
        self.assertTrue(result['literal_citation_contract']['selection_valid'])
        self.assertEqual(delegate.usage_report()['mock_calls'], 1)

    def test_invalid_model_selection_stays_blocked_without_retry_or_claim_dropping(self):
        wrapper, delegate, client = self.llm(client=GroundedClient('wrong_quote'))
        evidence = self.env['index'].get_document_chunks('A')[-4:]
        with self.assertRaises(LLMError):
            wrapper.generate('qa', '대상 업무는?', evidence)
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(delegate.usage_report()['mock_calls'], 1)

    def test_existing_free_quote_cache_and_literal_quote_cache_are_separate(self):
        wrapper, delegate, client = self.llm(cache=True)
        evidence = self.env['index'].get_document_chunks('A')[-4:]
        delegate.generate('qa', '대상 업무는?', evidence)
        wrapper.generate('qa', '대상 업무는?', evidence)
        cached = wrapper.generate('qa', '대상 업무는?', evidence)
        self.assertEqual(len(client.bodies), 2)
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(delegate.usage_report()['cache_hits'], 1)

    def test_comparison_checks_two_requests_against_remaining_budget_before_first_call(self):
        wrapper, delegate, client = self.llm(max_calls=1)
        requests = [{'task': COMPARISON_DOCUMENT_TASK, 'question': '구축 범위 비교',
            'evidence': self.env['index'].get_document_chunks(did)[-4:]} for did in ['A', 'B']]
        with self.assertRaises(LLMError):
            wrapper.preflight_requests(requests)
        self.assertEqual(client.bodies, [])

    def test_context_and_schema_stay_in_the_existing_request_byte_budget(self):
        client = GroundedClient()
        chat = make_conversation(self.env, enabled=True, approved_doc_ids=['A'], client=client,
            retrieval={'method': 'rrf', 'top_k': 8, 'fetch_k': 40, 'max_context_bytes': 42000})
        chat.select_documents(['A'])
        result = chat.ask('구체적인 공사의 내용은?')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertTrue(result['validation']['valid'])
        self.assertLessEqual(result['generation']['request_input_bytes'], chat.llm.config.max_request_bytes)
        self.assertEqual(chat.llm.usage_report()['max_calls'], 30)

    def test_scope_comparison_keeps_per_document_validation_and_shared_usage(self):
        client = GroundedClient()
        chat = make_conversation(self.env, enabled=True, approved_doc_ids=['A', 'B'], client=client,
            retrieval={'method': 'rrf', 'top_k': 8, 'per_doc_top_k': 6, 'fetch_k': 40})
        chat.select_documents(['A', 'B'])
        result = chat.ask('두 사업의 구체적인 구축 대상 업무 범위를 비교해줘.')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertTrue(result['validation']['valid'])
        self.assertEqual(len(client.bodies), 2)
        self.assertEqual(chat.llm.usage_report()['mock_calls'], 2)
        self.assertEqual({c['doc_id'] for c in result['claims']}, {'A', 'B'})

    def test_budget_request_keeps_existing_schema_and_task(self):
        from rfp_phase4.evidence import ANSWER_SCHEMA
        client = GroundedClient()
        chat = make_conversation(self.env, enabled=True, approved_doc_ids=['A'], client=client)
        chat.select_documents(['A'])
        chat.ask('사업예산은 얼마야?')
        self.assertEqual(client.bodies[0]['text']['format']['schema'], ANSWER_SCHEMA)
        self.assertEqual(json.loads(client.bodies[0]['input'][1]['content'])['task'], 'qa')


ANSWER_BASELINE = LiteralQuoteTests._schema_hash()


if __name__ == '__main__':
    unittest.main()
