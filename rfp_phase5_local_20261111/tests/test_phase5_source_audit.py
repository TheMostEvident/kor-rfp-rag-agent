"""Limited omission hints must remain source-bound and separate from answers."""
import copy
import unittest

from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase5.source_audit import audit_summary_sources, render_source_audit
from rfp_phase5.summary import FieldSummarizer
import test_phase5_summary as fixture


def source_case():
    docs, chunks = fixture.corpus(count=1)
    text = ('사업예산 : 12,345,000원(부가세 제외)\n'
            '본 사업은 단가계약으로, 대금은 실제 대상인원에 대하여 정산하여 지급\n'
            '기간 : 계약체결일 ~ 2030. 4. 1\n'
            '후속절차 지원은 과업기간 이후에도 포함\n'
            '낙찰자 선정방식 : 적격심사(제한적최저가) 낙찰제')
    docs[0]['blocks'][0]['text'] = text
    chunks[0].update(text=text, text_bytes=len(text.encode()))
    span = chunks[0]['source_spans'][0]
    span.update(char_end=len(text), chunk_end=len(text))
    claims = [{'doc_id': 'd1', 'category': 'budget', 'text': '[budget_vat|fact] 12,345,000원(부가세 제외)',
               'citations': [{'chunk_id': 'c0', 'quote': text.splitlines()[0]}]}]
    result = {'doc_id': 'd1', 'status': 'partial', 'execution_status': 'completed',
              'claims': claims, 'missing_information': ['기타 미확인'],
              'source_chunk_ids': ['c0'], 'processed_chunk_ids': ['c0']}
    return fixture.Index(chunks), EvidenceRegistry(docs, chunks), result


class SourceAuditTests(unittest.TestCase):
    def test_three_hints_use_processed_original_source_without_mutating_answer(self):
        index, registry, result = source_case()
        before = copy.deepcopy(result)
        audit = audit_summary_sources(index, registry, result)
        self.assertEqual(audit['candidate_count'], 3)
        self.assertEqual({c['field_id'] for c in audit['candidates']}, {'budget_vat', 'duration', 'evaluation'})
        self.assertTrue(audit['original_quotes_revalidated'])
        self.assertEqual(audit['api_calls'], 0)
        self.assertFalse(audit['semantic_completeness_verified'])
        self.assertFalse(audit['human_review_completed'])
        self.assertEqual(result, before)
        self.assertTrue(all(c['citations'][0]['provenance_verified'] for c in audit['candidates']))
        self.assertTrue(all(not c['automatically_added_to_answer'] for c in audit['candidates']))

    def test_unprocessed_chunks_are_never_scanned(self):
        index, registry, result = source_case()
        result.update(claims=[], processed_chunk_ids=[], status='error')
        audit = audit_summary_sources(index, registry, result)
        self.assertEqual(audit['candidate_count'], 0)
        self.assertFalse(audit['all_input_chunks_processed'])

    def test_unknown_scope_and_duplicate_source_ids_are_rejected(self):
        for mode in ('doc', 'processed', 'source'):
            index, registry, result = source_case()
            if mode == 'doc': result['doc_id'] = 'other'
            if mode == 'processed': result['processed_chunk_ids'] *= 2
            if mode == 'source': result['source_chunk_ids'] = []
            with self.assertRaises(ValueError): audit_summary_sources(index, registry, result)

    def test_fabricated_existing_quote_is_not_accepted_by_source_audit(self):
        index, registry, result = source_case()
        result['claims'][0]['citations'][0]['quote'] = '원문에 없는 금액 999원'
        with self.assertRaises(ValueError): audit_summary_sources(index, registry, result)

    def test_same_condition_in_another_field_remains_a_field_review_hint(self):
        index, registry, result = source_case()
        result['claims'][0]['text'] = '[budget_vat|fact] 후속지원은 과업기간 이후에도 포함'
        audit = audit_summary_sources(index, registry, result)
        self.assertIn('duration', [c['field_id'] for c in audit['candidates']])

    def test_presence_check_is_limited_and_does_not_infer_scores(self):
        index, registry, result = source_case()
        result['claims'].append({'doc_id': 'd1', 'category': 'evaluation',
            'text': '[evaluation|fact] 적격심사 방식이다. 배점은 미확인이다.',
            'citations': [{'chunk_id': 'c0', 'quote': index.chunks[0]['text'].splitlines()[-1]}]})
        audit = audit_summary_sources(index, registry, result)
        self.assertNotIn('evaluation', [c['field_id'] for c in audit['candidates']])
        self.assertIsNone(audit['semantic_accuracy'])

    def test_selection_heading_is_not_an_operative_method_value(self):
        import re
        from rfp_phase5.source_audit import RULES
        pattern = next(rule[2] for rule in RULES if rule[0] == 'selection_method')
        self.assertIsNone(re.search(pattern, '3. 사업예산 및 낙찰자 선정방식\t1'))
        self.assertIsNone(re.search(pattern, '낙찰자 선정방식'))
        self.assertIsNotNone(re.search(pattern, '낙찰자 선정방식 : 적격심사'))

    def test_renderer_escapes_links_images_and_table_delimiters(self):
        index, registry, result = source_case()
        audit = audit_summary_sources(index, registry, result)
        audit['candidates'][0]['source_quote'] += ' ![unsafe](https://example.test) | <script>'
        rendered = render_source_audit(audit)
        self.assertNotIn('![unsafe]', rendered)
        self.assertIn('&#124;', rendered)
        self.assertNotIn('<script>', rendered)

    def test_regular_summarizer_attaches_hints_without_extra_model_calls(self):
        index, registry, _ = source_case()
        llm = fixture.OfflineLLM(all_fields=True)
        result = FieldSummarizer(index, llm, registry).summarize('d1')
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(result['claims']), 11)
        self.assertEqual(result['status'], 'partial')
        # This fixture's budget citation already includes the unit-price line;
        # only the duration and selection-method candidates remain new hints.
        self.assertEqual({c['field_id'] for c in result['source_audit']['candidates']}, {'duration', 'evaluation'})
        self.assertIn('원문 추가 확인 필요:', '\n'.join(result['missing_information']))


if __name__ == '__main__':
    unittest.main()
