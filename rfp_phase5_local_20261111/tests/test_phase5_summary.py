"""Bounded full-input wiring with source-backed offline doubles, not quality scores."""
import copy
from types import SimpleNamespace
import unittest

from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase5.fields import FIELDS, FIELD_SCHEMA, FIELD_QUESTION
from rfp_phase5.summary import FieldSummarizer, TASK


def corpus(count=4):
    blocks, chunks = [], []
    for number in range(count):
        text = (f'[{number}] 용역비 650,000,000원(부가세 포함). 계약일로부터 5개월이며 '
                '안정화 1개월 포함. 제출기한은 입찰공고문 참조. ' + '세부 요건을 준수한다. ' * 24)
        block = {'block_id': f'b{number}', 'text': text, 'locator': f'PDF p.{number+1}', 'page': number+1}
        blocks.append(block)
        chunks.append({'chunk_id': f'c{number}', 'doc_id': 'd1', 'canonical_doc_id': 'd1',
            'alias_doc_ids': ['d1'], 'sha256': 'a'*64, 'file_name': '공개RFP.pdf',
            'text': text, 'text_bytes': len(text.encode()), 'length_unit': 'utf8_bytes',
            'source_method': 'raw_parser', 'kind': 'prose', 'page': number+1,
            'locator': block['locator'], 'source_spans': [{'collection': 'blocks',
                'block_id': block['block_id'], 'locator': block['locator'], 'page': number+1,
                'char_start': 0, 'char_end': len(text), 'chunk_start': 0, 'chunk_end': len(text)}]})
    documents = [{'doc_id': 'd1', 'canonical_doc_id': 'd1', 'sha256': 'a'*64,
                  'file_name': '공개RFP.pdf', 'blocks': blocks, 'supplemental_blocks': []}]
    return documents, chunks


class Index:
    def __init__(self, chunks): self.chunks = chunks
    def resolve_doc_ids(self, ids):
        if set(ids) - {'d1'}: raise ValueError('unknown document')
        return list(dict.fromkeys(ids))
    def get_document_chunks(self, doc_id): return self.chunks if doc_id == 'd1' else []


class OfflineLLM:
    """Extractive doubles exercise provenance, not semantic field assignment."""
    def __init__(self, *, max_request_bytes=11000, invalid_quote_at=None,
                 malformed_at=None, failed_at=None, all_fields=False, preflight_fail=False,
                 mode='fact', uncertainty=False):
        self.config = SimpleNamespace(enabled=False, max_request_bytes=max_request_bytes,
             max_output_tokens=3000, max_calls=30, max_total_input_bytes=600000)
        self.calls, self.preflight, self.measures, self.events = [], [], [], []
        self.invalid_quote_at, self.malformed_at, self.failed_at = invalid_quote_at, malformed_at, failed_at
        self.all_fields, self.preflight_fail, self.mode, self.uncertainty = all_fields, preflight_fail, mode, uncertainty

    def request_input_bytes(self, task, question, evidence, *, schema=None):
        self.measures.append({'task': task, 'question': question, 'schema': copy.deepcopy(schema)})
        return 6500 + sum(len(chunk['text'].encode()) for chunk in evidence)

    def preflight_requests(self, requests):
        self.events.append('preflight')
        self.preflight = copy.deepcopy(requests)
        if self.preflight_fail: raise ValueError('key-must-never-leak')
        return {'valid': True}

    def usage_report(self):
        return {'mock_calls': len(self.calls), 'live_api_calls': 0, 'api_calls': len(self.calls),
                'usage_unknown_attempts': 0}

    def generate(self, task, question, evidence, *, schema=None):
        self.events.append('generate')
        self.calls.append({'task': task, 'question': question, 'evidence': copy.deepcopy(evidence), 'schema': copy.deepcopy(schema)})
        number = len(self.calls)
        if number == self.failed_at: raise RuntimeError('key-must-never-leak')
        quote = '꾸며낸 금액 999억원' if number == self.invalid_quote_at else evidence[0]['text'][:95]
        chosen = FIELDS if self.all_fields else [next(field for field in FIELDS if field['id']=='budget_vat')]
        claims = [{'doc_id': 'd1', 'category': field['category'],
                   'text': f'[{field["id"]}|{self.mode}] 입력 묶음 {number}에 따른 추출 사실',
                   'citations': [{'chunk_id': evidence[0]['chunk_id'], 'quote': quote}]} for field in chosen]
        if number == self.malformed_at: claims[0]['text'] = '표지가 없는 추출 사실'
        missing = ['[budget_vat] VAT 세부 적용 범위'] if self.uncertainty and number == 1 else []
        return {'payload': {'status': 'partial' if missing else 'answered', 'claims': claims,
                            'missing_information': missing},
                'usage': {}, 'execution_mode': 'injected_test_client', 'live_api': False, 'cache_hit': False,
                'request_fingerprint': str(number)}


class FieldSummaryTests(unittest.TestCase):
    def make(self, count=4, llm=None, config=None):
        docs, chunks = corpus(count)
        llm = llm or OfflineLLM()
        return FieldSummarizer(Index(chunks), llm, EvidenceRegistry(docs,chunks), config), llm, chunks

    def test_plan_and_preflight_and_generate_use_identical_phase5_contract(self):
        summary, llm, chunks = self.make(config={'max_batch_bytes': 9000})
        plan = summary.plan('d1')
        self.assertEqual(llm.calls, [])
        self.assertEqual([chunk['chunk_id'] for batch in plan['batches'] for chunk in batch['chunks']],
                         [chunk['chunk_id'] for chunk in chunks])
        self.assertEqual(plan['task'], TASK)
        self.assertEqual(len(plan['field_schema_fingerprint']), 64)
        self.assertNotEqual(plan['plan_fingerprint'], plan['base_plan_fingerprint'])
        result = summary.summarize('d1')
        self.assertEqual(result['plan_fingerprint'], plan['plan_fingerprint'])
        self.assertEqual(llm.preflight, llm.calls)
        self.assertEqual(llm.events[0], 'preflight')
        self.assertTrue(all(call['schema']==FIELD_SCHEMA and call['task']==TASK and call['question']==FIELD_QUESTION
                            for call in llm.calls + llm.measures))
        self.assertEqual(result['processed_chunk_ids'], plan['source_chunk_ids'])

    def test_eleven_rows_distinguish_missing_from_source_absence(self):
        summary, _, _ = self.make(count=1)
        result = summary.summarize('d1')
        self.assertEqual(len(result['fields']), 11)
        by_id = {row['field_id']: row for row in result['fields']}
        self.assertEqual(by_id['budget_vat']['state'], 'evidence_found')
        self.assertEqual(by_id['submission_deadline']['state'], 'not_found_in_processed_input')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['validated_chunk_coverage'], 1)
        self.assertFalse(result['semantic_quality_verified'])
        self.assertFalse(result['semantic_entailment_verified'])
        self.assertNotIn('[budget_vat|fact]', result['markdown'])
        self.assertTrue(result['claims'][0]['text'].startswith('[budget_vat|fact]'))

    def test_all_field_labels_is_still_not_semantic_verification(self):
        summary, _, _ = self.make(count=1, llm=OfflineLLM(all_fields=True))
        result = summary.summarize('d1')
        self.assertEqual(result['status'], 'answered')
        self.assertFalse(result['semantic_quality_verified'])
        self.assertFalse(result['answer_correctness_verified'])

    def test_whole_plan_block_spends_no_calls_and_retains_all_ids(self):
        summary, llm, chunks = self.make(config={'max_batch_bytes': 9000, 'max_map_calls': 1})
        result = summary.summarize('d1')
        self.assertEqual(result['execution_status'], 'plan_blocked')
        self.assertEqual(result['source_chunk_ids'], [chunk['chunk_id'] for chunk in chunks])
        self.assertEqual(llm.calls, [])
        self.assertEqual({row['state'] for row in result['fields']}, {'not_processed'})

    def test_fresh_uncached_budget_warning_is_distinct_from_batch_plan_limits(self):
        llm = OfflineLLM()
        llm.config.max_total_input_bytes = 1
        llm.config.max_calls = 1
        summary, llm, _ = self.make(llm=llm, config={'max_batch_bytes': 9000})
        plan = summary.plan('d1')
        self.assertTrue(plan['can_execute'])
        self.assertFalse(plan['fresh_session_uncached_within_caps'])
        self.assertEqual(set(plan['fresh_session_uncached_block_reasons']),
                         {'max_calls_exceeded', 'max_total_input_bytes_exceeded'})
        self.assertEqual(llm.calls, [])
        self.assertIn('cache', plan['fresh_session_uncached_scope'])

    def test_preflight_failure_hides_secrets_and_does_not_invoke_generator(self):
        summary, llm, _ = self.make(llm=OfflineLLM(preflight_fail=True))
        result = summary.summarize('d1')
        self.assertEqual(result['execution_status'], 'preflight_blocked')
        self.assertEqual(llm.calls, [])
        self.assertNotIn('key-must-never-leak', str(result))

    def test_bad_field_marker_cannot_count_as_validated_input(self):
        summary, llm, _ = self.make(llm=OfflineLLM(malformed_at=1), config={'max_batch_bytes': 9000})
        result = summary.summarize('d1')
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result['validated_chunk_coverage'], 0)
        self.assertEqual(result['processed_chunk_ids'], [])
        self.assertEqual(result['claims'], [])
        self.assertNotIn('표지가 없는', result['markdown'])
        self.assertEqual(result['usage']['mock_calls'], 1)
        self.assertGreater(result['input_chunk_coverage'], 0)
        self.assertEqual(result['field_contract_failures'][0]['reason'], 'field_marker_or_category_invalid')

    def test_fabricated_quote_not_shown_and_cannot_count_as_valid_input(self):
        summary, llm, _ = self.make(llm=OfflineLLM(invalid_quote_at=1), config={'max_batch_bytes': 9000})
        result = summary.summarize('d1')
        self.assertEqual(result['validated_chunk_coverage'], 0)
        self.assertEqual(result['claims'], [])
        self.assertNotIn('999억', result['markdown'])
        self.assertEqual(result['batch_outcomes'][0]['status'], 'invalid')
        self.assertEqual(len(llm.calls), 1)

    def test_later_failure_retains_verified_prefix_without_full_summary_claim(self):
        summary, _, _ = self.make(llm=OfflineLLM(failed_at=2), config={'max_batch_bytes': 9000})
        result = summary.summarize('d1')
        self.assertTrue(result['claims'])
        self.assertLess(result['validated_chunk_coverage'], 1)
        self.assertEqual(result['execution_status'], 'incomplete')
        self.assertEqual(result['status'], 'partial')
        self.assertNotIn('key-must-never-leak', str(result))

    def test_uncertainty_from_first_batch_not_cleared_by_later_fact(self):
        summary, _, _ = self.make(llm=OfflineLLM(uncertainty=True), config={'max_batch_bytes': 9000})
        result = summary.summarize('d1')
        row = next(row for row in result['fields'] if row['field_id']=='budget_vat')
        self.assertEqual(row['state'], 'evidence_found')
        self.assertTrue(any('부분 입력 map-0001' in note and 'VAT 세부' in note for note in row['notes']))
        self.assertEqual(result['status'], 'partial')

    def test_explicit_conflict_candidates_preserved_without_resolution(self):
        summary, _, _ = self.make(llm=OfflineLLM(mode='conflict_candidate'), config={'max_batch_bytes': 9000})
        result = summary.summarize('d1')
        row = next(row for row in result['fields'] if row['field_id']=='budget_vat')
        self.assertGreater(len(row['claims']), 1)
        self.assertEqual(row['state'], 'conflict_candidates')
        self.assertFalse(result['automatic_conflict_resolution'])

    def test_notice_reference_stays_separate_from_a_known_value(self):
        summary, _, _ = self.make(count=1, llm=OfflineLLM(mode='notice_reference'))
        result = summary.summarize('d1')
        row = next(row for row in result['fields'] if row['field_id']=='budget_vat')
        self.assertEqual(row['state'], 'notice_reference')
        self.assertTrue(any('공고문' in note for note in row['notes']))
        self.assertEqual(result['status'], 'partial')

    def test_untrusted_markdown_does_not_render_external_image_or_link(self):
        summary, llm, _ = self.make(count=1)
        original = llm.generate
        def malicious(*args, **kwargs):
            response = original(*args, **kwargs)
            response['payload']['claims'][0]['text'] = '[budget_vat|fact] ![image](https://host/track) [link](https://host)'
            return response
        llm.generate = malicious
        result = summary.summarize('d1')
        self.assertTrue(result['claims'])
        self.assertIn(r'\!\[image\]\(https://host/track\)', result['markdown'])
        self.assertNotIn('![image](', result['markdown'])
        self.assertNotIn('[link](', result['markdown'])

    def test_received_metadata_is_reset_between_invocations(self):
        summary, llm, _ = self.make(count=1)
        first = summary.summarize('d1')
        self.assertEqual(first['input_chunk_coverage'], 1)
        llm.preflight_fail = True
        second = summary.summarize('d1')
        self.assertEqual(second['response_received_chunk_ids'], [])
        self.assertEqual(second['input_chunk_coverage'], 0)


if __name__ == '__main__':
    unittest.main()
