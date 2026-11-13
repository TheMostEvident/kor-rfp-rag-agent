"""Offline source-backed tests for full-input planning and partial summary failure."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase4.summarization import FullDocumentSummarizer, SummaryError, DEFAULT_CATEGORIES


def corpus(count=6, *, body_size=45):
    blocks, chunks = [], []
    for i in range(count):
        text = f'예산 {1000+i:,}원. ' + ('요구사항 ' * body_size)
        block = {'block_id': f'b{i}', 'text': text, 'locator': f'PDF p.{i+1}', 'page': i+1}
        blocks.append(block)
        chunks.append({'chunk_id': f'c{i}', 'doc_id': 'd1', 'canonical_doc_id': 'd1',
            'alias_doc_ids': ['d1', 'alias'], 'sha256': 'a'*64, 'file_name': '제안요청서.pdf',
            'text': text, 'text_bytes': len(text.encode()), 'length_unit': 'utf8_bytes',
            'source_method': 'raw_parser', 'kind': 'prose', 'page': i+1,
            'locator': block['locator'], 'source_spans': [{'collection': 'blocks', 'block_id': block['block_id'],
                'locator': block['locator'], 'page': i+1, 'char_start': 0, 'char_end': len(text),
                'chunk_start': 0, 'chunk_end': len(text)}]})
    doc = {'doc_id': 'd1', 'canonical_doc_id': 'd1', 'sha256': 'a'*64,
           'file_name': '제안요청서.pdf', 'blocks': blocks, 'supplemental_blocks': []}
    alias = {'doc_id': 'alias', 'canonical_doc_id': 'd1', 'sha256': 'a'*64,
             'is_duplicate_alias': True, 'file_name': '사본.pdf', 'blocks': []}
    return [doc, alias], chunks


class Index:
    def __init__(self, chunks):
        self.chunks = chunks

    def resolve_doc_ids(self, ids):
        if any(i not in {'d1', 'alias'} for i in ids):
            raise ValueError('unknown scope')
        return list(dict.fromkeys('d1' for _ in ids))

    def get_document_chunks(self, doc_id):
        return list(self.chunks) if doc_id == 'd1' else []


class OfflineLLM:
    is_mock = True

    def __init__(self, *, max_request_bytes=2800, failure_at=None, invalid_at=None,
                 preflight_error=False, categories=('budget',), repeated_text=False):
        self.config = SimpleNamespace(enabled=False, max_request_bytes=max_request_bytes,
            max_output_tokens=3000)
        self.failure_at, self.invalid_at, self.preflight_error = failure_at, invalid_at, preflight_error
        self.categories, self.repeated_text = categories, repeated_text
        self.requests, self.events, self.preflight_seen = [], [], []

    def request_input_bytes(self, task, question, evidence, *, schema=None):
        return 200 + len(question.encode()) + sum(len(c['text'].encode()) for c in evidence)

    def preflight_requests(self, requests):
        self.events.append('preflight')
        self.preflight_seen = copy.deepcopy(requests)
        if self.preflight_error:
            raise RuntimeError('secret-key-must-not-be-rendered')
        return {'valid': True, 'request_count': len(requests)}

    def generate(self, task, question, evidence, *, schema=None):
        self.events.append('generate')
        self.requests.append({'task': task, 'question': question, 'evidence': evidence, 'schema': schema})
        n = len(self.requests)
        if n == self.failure_at:
            raise RuntimeError('secret-key-must-not-be-rendered')
        quote = '근거에 없는 1억원' if n == self.invalid_at else evidence[0]['text'][:12]
        payload = {'status': 'answered', 'claims': [
            {'doc_id': 'd1', 'category': category,
             'text': '같은 예산 사실' if self.repeated_text else f'입력 묶음 {n}의 {category} 사실',
             'citations': [{'chunk_id': evidence[0]['chunk_id'], 'quote': quote}]}
            for category in self.categories], 'missing_information': []}
        return {'payload': payload, 'usage': {'input_tokens': 50, 'output_tokens': 30},
                'live_api': False, 'cache_hit': False, 'request_fingerprint': str(n)}

    def usage_report(self):
        return {'mock_calls': len(self.requests), 'live_api_calls': 0}


class FullDocumentSummaryTests(unittest.TestCase):
    def make(self, count=6, body_size=45, llm=None, config=None):
        documents, chunks = corpus(count, body_size=body_size)
        llm = llm or OfflineLLM()
        registry = EvidenceRegistry(documents, chunks)
        return FullDocumentSummarizer(Index(chunks), llm, registry, config), llm, chunks

    def test_plan_is_offline_deterministic_and_keeps_every_chunk_in_order(self):
        summary, llm, chunks = self.make()
        first = summary.plan('d1')
        second = summary.plan('alias')
        self.assertEqual(first['plan_fingerprint'], second['plan_fingerprint'])
        self.assertEqual(first['source_chunk_ids'], [c['chunk_id'] for c in chunks])
        self.assertEqual([c['chunk_id'] for b in first['batches'] for c in b['chunks']], first['source_chunk_ids'])
        self.assertTrue(first['can_execute'])
        self.assertTrue(all(b['request_bytes'] <= summary.batch_max_bytes for b in first['batches']))
        self.assertEqual(llm.events, [])
        self.assertEqual(first['api_calls'], 0)
        self.assertFalse(first['byte_budget_is_token_count'])
        self.assertIsNone(first['estimated_price'])

    def test_max_map_calls_blocks_without_truncating_source_plan(self):
        summary, llm, chunks = self.make(config={'max_map_calls': 1, 'max_batch_bytes': 2200})
        plan = summary.plan('d1')
        self.assertFalse(plan['can_execute'])
        self.assertIn('max_map_calls_exceeded', plan['block_reasons'])
        self.assertEqual(sum(len(b['chunks']) for b in plan['batches']), len(chunks))
        result = summary.summarize('d1')
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['answer_generated'])
        self.assertEqual(llm.events, [])

    def test_individual_oversized_chunk_is_preserved_but_never_sent(self):
        summary, llm, chunks = self.make(count=2, config={'max_batch_bytes': 100})
        plan = summary.plan('d1')
        self.assertEqual(plan['source_chunk_ids'], [c['chunk_id'] for c in chunks])
        self.assertEqual(plan['chunk_count'], 2)
        self.assertFalse(plan['can_execute'])
        self.assertIn('one_or_more_source_chunks_exceed_request_byte_budget', plan['block_reasons'])
        self.assertEqual(summary.summarize('d1')['status'], 'blocked')
        self.assertEqual(llm.events, [])

    def test_preflight_has_every_exact_request_before_first_generate(self):
        summary, llm, _ = self.make()
        plan = summary.plan('d1')
        summary.summarize('d1')
        self.assertEqual(llm.events[0], 'preflight')
        self.assertEqual(len(llm.preflight_seen), plan['map_calls'])
        self.assertEqual(llm.preflight_seen, llm.requests)

    def test_whole_plan_budget_failure_spends_no_calls_and_hides_exception(self):
        summary, llm, _ = self.make(llm=OfflineLLM(preflight_error=True))
        result = summary.summarize('d1')
        self.assertEqual(result['execution_status'], 'preflight_blocked')
        self.assertEqual(result['claims'], [])
        self.assertEqual(llm.events, ['preflight'])
        self.assertNotIn('secret-key', str(result))

    def test_false_preflight_result_is_not_treated_as_approval(self):
        llm = OfflineLLM()
        llm.preflight_requests = lambda requests: {'valid': False}
        summary, _, _ = self.make(llm=llm)
        result = summary.summarize('d1')
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(llm.requests, [])

    def test_all_inputs_processed_does_not_mean_all_summary_topics_supported(self):
        summary, llm, chunks = self.make()
        result = summary.summarize('d1')
        self.assertEqual(result['processed_chunk_ids'], [c['chunk_id'] for c in chunks])
        self.assertEqual(result['input_chunk_coverage'], 1.0)
        self.assertEqual(result['validated_chunk_coverage'], 1.0)
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['semantic_coverage_verified'])
        self.assertFalse(result['semantic_entailment_verified'])
        self.assertTrue(result['missing_information'])
        self.assertTrue(result['answer_generated'])
        self.assertEqual(result['llm_session_usage']['live_api_calls'], 0)

    def test_complete_category_input_still_does_not_claim_semantic_verification(self):
        summary, _, _ = self.make(count=1, llm=OfflineLLM(categories=DEFAULT_CATEGORIES))
        result = summary.summarize('d1')
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['missing_information'], [])
        self.assertFalse(result['semantic_coverage_verified'])
        self.assertIn('의미적', result['markdown'])

    def test_map_uncertainty_survives_even_when_all_categories_have_claims(self):
        llm = OfflineLLM(categories=DEFAULT_CATEGORIES)
        original = llm.generate
        def uncertain(*args, **kwargs):
            response = original(*args, **kwargs)
            response['payload']['status'] = 'partial'
            response['payload']['missing_information'] = ['부가세 포함 여부']
            return response
        llm.generate = uncertain
        summary, _, _ = self.make(count=1, llm=llm)
        result = summary.summarize('d1')
        self.assertEqual(result['input_chunk_coverage'], 1.0)
        self.assertEqual(len({c['category'] for c in result['claims']}), len(DEFAULT_CATEGORIES))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['partial_batch_count'], 1)
        self.assertTrue(any('부가세 포함 여부' in label for label in result['missing_information']))
        self.assertIn('부분 입력', result['markdown'])

    def test_one_batch_cannot_implicitly_resolve_another_batch_uncertainty(self):
        llm = OfflineLLM(categories=DEFAULT_CATEGORIES)
        original = llm.generate
        def uncertain_first(*args, **kwargs):
            response = original(*args, **kwargs)
            if len(llm.requests) == 1:
                response['payload']['status'] = 'partial'
                response['payload']['missing_information'] = ['예산 부가세 기준']
            return response
        llm.generate = uncertain_first
        summary, _, _ = self.make(llm=llm)
        result = summary.summarize('d1')
        self.assertGreater(len(result['batch_outcomes']), 1)
        self.assertEqual(result['status'], 'partial')
        self.assertTrue(any('예산 부가세 기준' in label for label in result['missing_information']))

    def test_fabricated_quote_is_not_rendered_and_remaining_input_is_marked_skipped(self):
        summary, llm, _ = self.make(llm=OfflineLLM(invalid_at=1))
        result = summary.summarize('d1')
        self.assertFalse(result['answer_generated'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['claims'], [])
        self.assertEqual(result['processed_chunk_ids'], [])
        self.assertTrue(result['response_received_chunk_ids'])
        self.assertNotIn('1억원', result['markdown'])
        self.assertEqual(len(llm.requests), 1)
        self.assertTrue(all(b['status'] == 'skipped' for b in result['batch_outcomes'][1:]))

    def test_later_failure_retains_valid_source_claims_and_marks_summary_partial(self):
        summary, llm, _ = self.make(llm=OfflineLLM(failure_at=2))
        result = summary.summarize('d1')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['execution_status'], 'incomplete')
        self.assertTrue(result['valid'])
        self.assertTrue(result['claims'])
        self.assertLess(result['validated_chunk_coverage'], 1)
        self.assertTrue(any('일부 원문 청크' in label for label in result['missing_information']))
        self.assertNotIn('secret-key', str(result))

    def test_explicit_continue_after_failure_still_never_marks_full_success(self):
        summary, llm, _ = self.make(count=8, llm=OfflineLLM(failure_at=2), config={'stop_on_error': False})
        result = summary.summarize('d1')
        self.assertEqual(len(llm.requests), result['map_calls'])
        self.assertEqual(result['status'], 'partial')
        self.assertLess(result['validated_chunk_coverage'], 1)
        self.assertTrue(any(b['valid'] for b in result['batch_outcomes'][2:]))

    def test_exact_duplicate_claims_merge_citations_without_synthesis(self):
        summary, _, _ = self.make(llm=OfflineLLM(repeated_text=True))
        result = summary.summarize('d1')
        self.assertEqual(len(result['claims']), 1)
        self.assertEqual(result['claims'][0]['text'], '같은 예산 사실')
        self.assertEqual(len(result['claims'][0]['citations']), result['map_calls'])
        self.assertTrue(all(c.get('source_ranges') for c in result['claims'][0]['citations']))

    def test_plan_changes_if_summary_question_or_batch_budget_changes(self):
        summary, _, _ = self.make()
        first = summary.plan('d1')
        other = summary.plan('d1', question='예산을 정리해 주세요.')
        self.assertNotEqual(first['plan_fingerprint'], other['plan_fingerprint'])

    def test_empty_document_returns_blocked_without_call(self):
        summary, llm, _ = self.make(count=0)
        result = summary.summarize('d1')
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['source_chunk_ids'], [])
        self.assertIn('document_has_no_indexed_chunks', result['errors'])
        self.assertEqual(llm.events, [])

    def test_mutated_registered_chunk_cannot_be_used_for_plan(self):
        summary, llm, chunks = self.make()
        chunks[0]['text'] = '원문 바꾸기'
        with self.assertRaises(ValueError):
            summary.plan('d1')
        self.assertEqual(llm.events, [])

    def test_configuration_and_plan_fail_closed(self):
        for config in [{'max_map_calls': 0}, {'max_map_calls': True}, {'max_batch_bytes': False},
                       {'max_batch_bytes': 1.5}, {'requested_categories': ['invented']},
                       {'requested_categories': ['budget', 'budget']}, {'stop_on_error': 1},
                       {'max_batch_bytes': 1000, 'batch_max_bytes': 1000}, {'ignored': 1}]:
            with self.subTest(config=config), self.assertRaises(SummaryError):
                self.make(config=config)
        summary, _, _ = self.make()
        with self.assertRaises(SummaryError):
            summary.plan('d1', question=' ')
        with self.assertRaises(ValueError):
            summary.plan('missing')

    def test_real_client_offline_planner_does_not_require_sdk_key_or_enable(self):
        from rfp_phase4.llm import OpenAIGroundedLLM, LLMConfig
        with tempfile.TemporaryDirectory() as directory:
            llm = OpenAIGroundedLLM(Path(directory), config=LLMConfig(enabled=False),
                                   allowed_sources={'d1': 'a'*64})
            summary, _, chunks = self.make(llm=llm)
            plan = summary.plan('d1')
            self.assertTrue(plan['can_execute'])
            self.assertEqual(plan['chunk_count'], len(chunks))
            self.assertTrue(all(b['request_bytes'] <= 35000 for b in plan['batches']))
            result = summary.summarize('d1')
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['processed_chunk_ids'], [])

    def test_real_client_oversize_measurement_returns_a_blocked_full_plan(self):
        from rfp_phase4.llm import OpenAIGroundedLLM, LLMConfig
        with tempfile.TemporaryDirectory() as directory:
            llm = OpenAIGroundedLLM(Path(directory), config=LLMConfig(enabled=False, max_request_bytes=1000),
                                   allowed_sources={'d1': 'a'*64})
            summary, _, chunks = self.make(llm=llm)
            plan = summary.plan('d1')
            self.assertFalse(plan['can_execute'])
            self.assertEqual(plan['source_chunk_ids'], [chunk['chunk_id'] for chunk in chunks])
            self.assertTrue(all(batch['request_bytes'] > 1000 for batch in plan['batches']))
            self.assertEqual(summary.summarize('d1')['execution_status'], 'plan_blocked')
            self.assertEqual(llm.usage_report()['api_calls'], 0)

    def test_usage_is_for_this_summary_not_previous_client_calls(self):
        llm = OfflineLLM()
        llm.requests.extend([{'earlier': True}, {'earlier': True}])
        summary, _, _ = self.make(count=1, llm=llm)
        result = summary.summarize('d1')
        self.assertEqual(result['usage']['mock_calls'], 1)
        self.assertEqual(result['llm_session_usage']['mock_calls'], 3)

    def test_cached_real_origin_is_not_counted_as_a_new_live_call(self):
        llm = OfflineLLM()
        original = llm.generate
        def cached(*args, **kwargs):
            response = original(*args, **kwargs)
            response.update({'live_api': True, 'cache_hit': True})
            return response
        llm.generate = cached
        summary, _, _ = self.make(count=1, llm=llm)
        result = summary.summarize('d1')
        self.assertTrue(result['batch_outcomes'][0]['live_api'])
        self.assertTrue(result['batch_outcomes'][0]['cache_hit'])
        self.assertFalse(result['batch_outcomes'][0]['api_called'])


if __name__ == '__main__':
    unittest.main()
