import copy
import unittest

from rfp_phase5.quality import audit_result


def claim(text='예산은 688백만원(부가세 포함)입니다.', quote='사업예산: 688백만원(부가세 포함)', **kw):
    value = {'doc_id': 'd1', 'category': 'budget', 'text': text,
             'citations': [{'chunk_id': 'c1', 'quote': quote, 'provenance_verified': True,
                            'sha256': 'a'*64, 'doc_id': 'd1'}]}
    value.update(kw)
    return value


def result(*claims):
    return {'valid': True, 'claims': list(claims), 'execution_status': 'completed'}


def codes(value):
    return {row['code'] for row in audit_result(value)['flags']}


class SurfaceAuditTests(unittest.TestCase):
    def test_literal_budget_vat_matches_but_never_truth_score(self):
        report = audit_result(result(claim()))
        self.assertEqual(report['flag_count'], 0)
        self.assertIsNone(report['semantic_accuracy'])
        self.assertFalse(report['zero_flags_means_semantically_correct'])
        self.assertEqual(report['api_calls'], 0)

    def test_number_in_other_claim_quote_does_not_support_this_claim(self):
        value = result(claim('예산 280백만원', '예산 688백만원'),
                       claim('다른 문서 280백만원', '다른 문서 280백만원', doc_id='d2'))
        self.assertIn('numeric_surface_not_in_claim_quotes', codes(value))
        flagged = [row for row in audit_result(value)['flags'] if row['code'] == 'numeric_surface_not_in_claim_quotes']
        self.assertEqual(flagged[0]['claim_index'], 1)
        self.assertEqual(flagged[0]['tokens'], ['280'])

    def test_number_unit_pair_matters(self):
        value = result(claim('예산 688원', '예산 688백만원'))
        self.assertNotIn('numeric_surface_not_in_claim_quotes', codes(value))
        self.assertIn('number_unit_pair_not_in_claim_quotes', codes(value))

    def test_whitespace_in_unit_is_not_amount_conversion(self):
        value = result(claim('예산 280백만 원 이내', '예산 280백만원 이내'))
        self.assertEqual(audit_result(value)['flag_count'], 0)

    def test_reformatted_or_converted_number_is_reviewed(self):
        for text in ('예산 688000000원', '예산 688,000,000원'):
            with self.subTest(text=text):
                self.assertIn('numeric_surface_not_in_claim_quotes', codes(result(claim(text, '예산 688백만원'))))

    def test_vat_omission_and_opposite_state(self):
        self.assertIn('vat_detail_omitted_from_claim', codes(result(claim('예산 688백만원'))))
        self.assertIn('vat_state_not_matched_in_claim_quotes',
                      codes(result(claim('예산 688백만원(부가세 별도)'))))
        self.assertIn('vat_state_not_matched_in_claim_quotes',
                      codes(result(claim('예산 688백만원(부가세 포함)', '예산 688백만원(부가세 불포함)'))))

    def test_anchor_and_stabilization_conditions_are_preserved(self):
        quote = '사업기간: 계약일로부터 5개월(안정화·시스템모니터링 1개월 포함)'
        report = audit_result(result(claim('기간은 5개월입니다.', quote, category='duration')))
        conditions = {row.get('condition') for row in report['flags']}
        self.assertTrue({'contract_start', 'stabilization', 'monitoring'} <= conditions)
        self.assertEqual(audit_result(result(claim(quote, quote, category='duration')))['flag_count'], 0)

    def test_notice_reference_is_not_claimed_missing_document_fact(self):
        text = '[submission_deadline|notice_reference] 제출기한은 입찰공고문 참조'
        quote = '제출기한은 입찰공고문 참조'
        self.assertEqual(audit_result(result(claim(text, quote, category='submission')))['flag_count'], 0)

    def test_form_warning_even_when_another_quote_has_budget(self):
        value = claim()
        value['citations'].append({'chunk_id': 'form', 'quote': '구 분\n금 액\n비 고\n용 역 비\n부가가치세 포함',
                                   'provenance_verified': True})
        report = audit_result(result(value))
        form = [row for row in report['flags'] if row['code'] == 'possible_blank_form_or_weak_quote']
        self.assertEqual(form[0]['citation_index'], 2)
        self.assertNotIn('numeric_surface_not_in_claim_quotes', codes(result(value)))
        value['citations'][1]['quote'] = '<부가세 포함, 금액단위 : 원>'
        self.assertIn('possible_blank_form_or_weak_quote', codes(result(value)))

    def test_missing_provenance_and_empty_citations_are_not_confirmed(self):
        for value in (result(claim(citations=[])), result(claim(citations=[{'quote': '688'}])),
                      {'validation': {'valid': False, 'claims': []}}):
            with self.subTest(value=value):
                self.assertFalse(audit_result(value)['source_validation_reported'])
                self.assertIn('citation_provenance_not_confirmed', codes(value))

    def test_inputs_unchanged_and_counts_consistent(self):
        value = result(claim('688원', '688백만원 이내(부가세 포함)'))
        original = copy.deepcopy(value)
        report = audit_result(value)
        self.assertEqual(value, original)
        self.assertEqual(report['flag_count'], sum(report['flag_counts'].values()))

    def test_no_semantic_entailment_from_exact_surfaces(self):
        value = result(claim('688백만원은 기간을 뜻한다.', '사업예산 688백만원'))
        report = audit_result(value)
        self.assertEqual(report['flag_count'], 0)
        self.assertIsNone(report['automatic_answer_correctness'])
        self.assertFalse(report['semantic_entailment_verified'])


if __name__ == '__main__':
    unittest.main()
