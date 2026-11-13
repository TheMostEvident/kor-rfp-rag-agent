import copy
import unittest

from rfp_phase5.review import answer_fingerprint, assess_review, build_review


def qa(live=True):
    claim = {'doc_id': 'd1', 'category': 'budget', 'text': '예산 688백만원(부가세 포함)',
             'citations': [{'chunk_id': 'c1', 'doc_id': 'd1', 'sha256': 'a'*64,
                            'quote': '예산 688백만원(부가세 포함)', 'source_ranges': [{'char_start': 0, 'char_end': 18}],
                            'provenance_verified': True}]}
    return {'status': 'answered', 'execution_status': 'completed', 'answer_generated': True,
            'claims': [claim], 'validation': {'valid': True, 'claims': [claim]},
            'generation': {'execution_mode': 'live_openai' if live else 'injected_test_client',
                           'live_api': live, 'cache_hit': False},
            'source_chunks': [{'chunk_id': 'c1', 'doc_id': 'd1', 'sha256': 'a'*64, 'text': claim['text']}],
            'usage': {'input_tokens': 123}}


def positive(value):
    review = build_review(value)
    review['reviewer'] = '검토자'
    for row in review['rows']:
        for key in ('supported', 'complete', 'conditions_preserved'):
            row[key] = True
    return review


class HumanReviewTests(unittest.TestCase):
    def test_template_is_pending(self):
        value = qa()
        review = build_review(value, case_id='F001')
        self.assertEqual(review['case_id'], 'F001')
        self.assertEqual(review['document_ids'], ['d1'])
        self.assertIsNone(review['rows'][0]['supported'])
        result = assess_review(value, review)
        self.assertEqual(result['status'], 'review_pending')
        self.assertFalse(result['live_readiness'])

    def test_positive_review_for_one_live_case_never_systemwide_accuracy(self):
        value = qa()
        report = assess_review(value, positive(value))
        self.assertEqual(report['status'], 'passed_human_review')
        self.assertTrue(report['live_readiness'])
        self.assertTrue(report['case_only'])
        self.assertFalse(report['system_quality_verified'])
        self.assertIsNone(report['semantic_accuracy'])

    def test_cached_live_response_does_not_need_new_paid_request(self):
        value = qa()
        value['generation']['cache_hit'] = True
        self.assertTrue(assess_review(value, positive(value))['live_readiness'])

    def test_positive_mock_is_not_live_ready(self):
        value = qa(live=False)
        report = assess_review(value, positive(value))
        self.assertEqual(report['status'], 'passed_human_review')
        self.assertFalse(report['live_readiness'])
        self.assertIn('live_model_origin_not_confirmed', report['readiness_blockers'])

    def test_changed_text_raw_quote_sha_and_ranges_all_invalidate(self):
        for mutation in (
            lambda v: v['claims'][0].update(text='예산 1원'),
            lambda v: v['claims'][0]['citations'][0].update(quote='예산 1원'),
            lambda v: v['claims'][0]['citations'][0].update(sha256='b'*64),
            lambda v: v['claims'][0]['citations'][0]['source_ranges'][0].update(char_start=1),
            lambda v: v['source_chunks'][0].update(text='변경된 원문'),
        ):
            value = qa(); review = positive(value); mutation(value)
            with self.subTest(mutation=mutation):
                self.assertEqual(assess_review(value, review)['status'], 'invalid_review')

    def test_token_counters_and_timestamps_do_not_invalidate(self):
        value = qa(); review = positive(value)
        value['usage']['input_tokens'] = 555
        value['timestamp'] = '2026-11-11'
        self.assertTrue(assess_review(value, review)['live_readiness'])

    def test_false_review_blocks_even_if_other_checks_pending(self):
        value = qa(); review = build_review(value)
        review['rows'][0]['supported'] = False
        report = assess_review(value, review)
        self.assertEqual(report['status'], 'review_failed')
        self.assertFalse(report['live_readiness'])

    def test_missing_row_swapped_content_and_nonboolean_judgement_invalid(self):
        value = qa()
        for mutation in (
            lambda r: r.update(rows=[]),
            lambda r: r['rows'][0].update(claims=[]),
            lambda r: r['rows'][0].update(supported=1),
            lambda r: r['rows'][0].update(conditions_preserved='true'),
        ):
            review = positive(value); mutation(review)
            self.assertEqual(assess_review(value, review)['status'], 'invalid_review')

    def test_reviewer_identity_required_before_pass(self):
        value = qa(); review = positive(value); review['reviewer'] = ''
        report = assess_review(value, review)
        self.assertEqual(report['status'], 'review_pending')
        self.assertIn('reviewer_not_recorded', report['readiness_blockers'])

    def test_source_validation_failure_and_partial_execution_not_ready(self):
        for mutation in (
            lambda v: v.update(status='partial'),
            lambda v: v.update(execution_status='incomplete'),
            lambda v: v['validation'].update(valid=False),
            lambda v: v['claims'][0]['citations'][0].update(provenance_verified=False),
        ):
            value = qa(); mutation(value)
            self.assertFalse(assess_review(value, positive(value))['live_readiness'])

    def test_summary_all_fields_reviewed_and_entire_input_required(self):
        value = qa(); value.pop('generation'); value.pop('validation'); value['valid'] = True
        value.update(source_chunk_ids=['c1', 'c2'], processed_chunk_ids=['c1', 'c2'],
                     validated_chunk_coverage=1.0,
                     batch_outcomes=[{'batch_id': 'map-0001', 'chunk_ids': ['c1', 'c2'], 'valid': True,
                                      'status': 'answered', 'execution_mode': 'live_openai', 'live_api': True}],
                     fields=[{'field_id': 'budget_vat', 'label': '예산·VAT', 'state': 'evidence_found',
                              'claims': value['claims'], 'notes': []},
                             {'field_id': 'submission_deadline', 'label': '제출기한', 'state': 'notice_reference',
                              'claims': [], 'notes': ['공고문 참조']}])
        review = positive(value)
        self.assertEqual(len(review['rows']), 2)
        self.assertTrue(assess_review(value, review)['live_readiness'])
        value['processed_chunk_ids'] = ['c1']
        report = assess_review(value, positive(value))
        self.assertFalse(report['live_readiness'])
        self.assertIn('full_input_not_validated', report['readiness_blockers'])

    def test_status_only_result_never_ready_or_silently_passed(self):
        value = {'status': 'insufficient_evidence', 'execution_status': 'completed', 'answer_generated': False}
        review = build_review(value)
        self.assertEqual(review['rows'][0]['row_id'], 'answer-status')
        self.assertFalse(assess_review(value, positive(value))['live_readiness'])

    def test_fingerprint_binding_display_field_changes(self):
        value = qa(); value['fields'] = [{'field_id': 'budget_vat', 'label': '예산',
                                       'state': 'evidence_found', 'claims': value['claims']}]
        fingerprint = answer_fingerprint(value)
        value['fields'][0]['state'] = 'conflict_candidates'
        self.assertNotEqual(fingerprint, answer_fingerprint(value))

    def test_input_unchanged(self):
        value = qa(); original = copy.deepcopy(value)
        review = positive(value); saved = copy.deepcopy(review)
        assess_review(value, review)
        self.assertEqual(value, original)
        self.assertEqual(review, saved)


if __name__ == '__main__':
    unittest.main()
