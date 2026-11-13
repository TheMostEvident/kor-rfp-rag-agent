import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

from rfp_phase4.evaluation import evaluate_answer_case, run_dev_evaluation


SHA = 'a' * 64
TEXT = '사업 예산: 100원 (부가세 포함)'


def source_documents():
    return [{'doc_id': 'A', 'sha256': SHA, 'blocks': [
        {'block_id': 'b1', 'text': TEXT, 'locator': 'page 1', 'page': 1}]}]


def case(feature='fact', case_id='F1'):
    return {'case_id': case_id, 'feature': feature, 'split': 'dev',
        'question': '이 사업의 예산을 알려줘.', 'context': {'selected_doc_ids': ['A']},
        'target_doc_ids': ['GOLD_DO_NOT_PASS'], 'expected_status': 'answered',
        'required_facts': [{'fact_id': 'f1', 'doc_id': 'A', 'field': 'budget',
            'expected_value': '100원 (부가세 포함)',
            'evidence': [{'doc_id': 'A', 'sha256': SHA, 'block_id': 'old_b1', 'quote': TEXT}]}],
        'required_behaviors': ['GOLD_BEHAVIOR_DO_NOT_PASS']}


def alignment(case_id='F1'):
    return {'document_aliases': {'aliasA': 'A'}, 'records': [{
        'case_id': case_id, 'fact_id': 'f1', 'evidence_kind': 'fact', 'evidence_index': 1,
        'link_id': f'{case_id}|fact|f1|1', 'mapped': True,
        'doc_id': 'A', 'canonical_doc_id': 'A', 'alternatives': [{
            'source_fragments': [{'collection': 'blocks', 'block_id': 'b1',
                                 'char_start': 0, 'char_end': len(TEXT)}]}]}]}


def result(start=0, end=None, *, status='answered', valid=True, text='예산은 100원이며 부가세가 포함됩니다.'):
    end = len(TEXT) if end is None else end
    citation = {'chunk_id': 'c1', 'quote': TEXT[start:end], 'doc_id': 'A', 'sha256': SHA,
        'source_ranges': [{'collection': 'blocks', 'block_id': 'b1',
                          'char_start': start, 'char_end': end}]}
    claim = {'doc_id': 'A', 'category': 'budget', 'text': text, 'citations': [citation]}
    return {'status': status, 'execution_status': 'completed', 'claims': [claim],
        'validation': {'valid': valid, 'claims': [claim] if valid else []},
        'generation': {'live_api': False, 'cache_hit': False,
                       'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}}


def actual_registry(two_documents=False):
    from rfp_phase4.evidence import EvidenceRegistry
    docs, chunks = [], []
    for did, sha in [('A', SHA)] + ([('B', 'b' * 64)] if two_documents else []):
        doc = {'doc_id': did, 'canonical_doc_id': did, 'sha256': sha,
               'file_name': did + '.pdf', 'supplemental_blocks': [],
               'blocks': [{'block_id': 'b1', 'text': TEXT, 'locator': 'page 1', 'page': 1}]}
        c = {'chunk_id': 'c1' if did == 'A' else 'c2', 'doc_id': did,
            'canonical_doc_id': did, 'sha256': sha, 'file_name': did + '.pdf',
            'kind': 'prose', 'source_method': 'raw_parser', 'text': TEXT,
            'locator': 'page 1', 'page': 1, 'length_unit': 'utf8_bytes',
            'text_bytes': len(TEXT.encode()), 'source_spans': [{
                'collection': 'blocks', 'block_id': 'b1', 'locator': 'page 1', 'page': 1,
                'char_start': 0, 'char_end': len(TEXT), 'chunk_start': 0, 'chunk_end': len(TEXT)}]}
        docs.append(doc); chunks.append(c)
    return docs, chunks, EvidenceRegistry(docs, chunks)


class MockSession:
    is_mock = True

    def __init__(self, log, active_doc_ids=None, responses=None):
        self.log = log
        self.responses = responses or [result()]
        self.log.append(('init', list(active_doc_ids or [])))

    def ask(self, query, *, feature='qa', candidate_doc_ids=None):
        self.log.append(('ask', query, feature, candidate_doc_ids))
        value = self.responses[min(sum(row[0] == 'ask' for row in self.log) - 1, len(self.responses) - 1)]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


class AnswerDiagnosticTests(unittest.TestCase):
    def score(self, value, **kwargs):
        return evaluate_answer_case(case(), value, alignment(), source_documents=source_documents(), **kwargs)

    def test_complete_valid_quote_recovers_gold_source(self):
        score = self.score(result())
        self.assertEqual(score['gold_source_span_recovery']['recall'], 1)
        self.assertEqual(score['gold_source_span_recovery']['mean_source_character_coverage'], 1)
        self.assertTrue(score['citation_source_validity']['validation_passed'])

    def test_partial_quote_is_not_full_recovery_even_when_chunk_id_is_correct(self):
        score = self.score(result(0, 5))
        self.assertEqual(score['gold_source_span_recovery']['recall'], 0)
        self.assertLess(score['gold_source_span_recovery']['mean_source_character_coverage'], 1)

    def test_union_of_two_quoted_source_ranges_can_recover_the_full_link(self):
        value = result(0, 9)
        value['claims'][0]['citations'] += result(9)['claims'][0]['citations']
        self.assertEqual(self.score(value)['gold_source_span_recovery']['recall'], 1)

    def test_same_quote_from_wrong_document_cannot_earn_source_credit(self):
        value = result()
        value['claims'][0]['citations'][0]['doc_id'] = 'B'
        self.assertEqual(self.score(value)['gold_source_span_recovery']['recall'], 0)

    def test_wrong_raw_sha_cannot_earn_source_credit(self):
        value = result()
        value['claims'][0]['citations'][0]['sha256'] = 'b' * 64
        self.assertEqual(self.score(value)['gold_source_span_recovery']['recall'], 0)

    def test_source_alias_canonicalization_preserves_legitimate_credit(self):
        value = result()
        value['claims'][0]['citations'][0]['doc_id'] = 'aliasA'
        self.assertEqual(self.score(value)['gold_source_span_recovery']['recall'], 1)

    def test_invalid_pipeline_validation_discards_all_source_credit(self):
        score = self.score(result(valid=False))
        self.assertEqual(score['gold_source_span_recovery']['recall'], 0)
        self.assertEqual(score['citation_source_validity']['validated_claim_count'], 0)

    def test_missing_source_documents_does_not_fallback_to_literal_overlap(self):
        score = evaluate_answer_case(case(), result(), alignment())
        self.assertIsNone(score['gold_source_span_recovery']['recall'])
        self.assertFalse(score['gold_source_span_recovery']['measurement_available'])

    def test_missing_alignment_label_is_explicitly_unmeasured(self):
        score = evaluate_answer_case(case(), result(), {'records': []}, source_documents=source_documents())
        self.assertIsNone(score['gold_source_span_recovery']['recall'])
        self.assertFalse(score['gold_source_span_recovery']['evidence_results'][0]['mapped'])

    def test_true_quote_with_false_claim_is_never_called_answer_accuracy(self):
        score = self.score(result(text='예산은 100원이며 부가세는 별도입니다.'))
        self.assertEqual(score['gold_source_span_recovery']['recall'], 1)
        self.assertIsNone(score['answer_correctness'])
        self.assertIsNone(score['semantic_entailment_score'])
        self.assertFalse(score['citation_source_validity']['semantic_entailment_verified'])
        self.assertEqual(score['human_review']['status'], 'not_reviewed')

    def test_numeric_presence_probe_is_only_a_literal_diagnostic(self):
        score = self.score(result(text='예산은 100억원입니다.'))
        diagnostic = score['literal_diagnostics'][0]
        self.assertEqual(diagnostic['present_numeric_tokens'], ['100'])
        self.assertIn('unit conversion', diagnostic['purpose'])
        self.assertIsNone(score['answer_correctness'])

    def test_holdout_case_diagnostic_is_always_rejected(self):
        c = case(); c['split'] = 'holdout'
        with self.assertRaises(ValueError):
            evaluate_answer_case(c, result())

    def test_registry_revalidation_has_priority_over_a_claimed_valid_flag(self):
        class Registry:
            def validate_answer(self, payload, context, allowed_doc_ids=None):
                self.payload = payload
                return {'valid': False, 'claims': [], 'status': 'insufficient_evidence'}
        registry = Registry()
        value = result(); value['source_chunks'] = []; value['active_doc_ids'] = ['A']
        score = self.score(value, registry=registry)
        self.assertTrue(score['citation_source_validity']['independently_checked'])
        self.assertFalse(score['citation_source_validity']['validation_passed'])
        self.assertEqual(score['gold_source_span_recovery']['recall'], 0)
        self.assertEqual(set(registry.payload['claims'][0]['citations'][0]), {'chunk_id', 'quote'})

    def test_actual_evidence_registry_integration_revalidates_quote_and_source_offsets(self):
        from rfp_phase4.evidence import EvidenceRegistry
        docs = source_documents()
        docs[0].update(canonical_doc_id='A', file_name='A.pdf', supplemental_blocks=[])
        c = {'chunk_id': 'c1', 'doc_id': 'A', 'canonical_doc_id': 'A', 'sha256': SHA,
            'file_name': 'A.pdf', 'kind': 'prose', 'source_method': 'raw_parser',
            'text': TEXT, 'locator': 'page 1', 'page': 1, 'length_unit': 'utf8_bytes',
            'text_bytes': len(TEXT.encode()), 'source_spans': [{
                'collection': 'blocks', 'block_id': 'b1', 'locator': 'page 1', 'page': 1,
                'char_start': 0, 'char_end': len(TEXT), 'chunk_start': 0, 'chunk_end': len(TEXT)}]}
        registry = EvidenceRegistry(docs, [c])
        value = result(); value['source_chunks'] = [c]; value['active_doc_ids'] = ['A']
        score = evaluate_answer_case(case(), value, alignment(), source_documents=docs, registry=registry)
        self.assertTrue(score['citation_source_validity']['independently_checked'])
        self.assertTrue(score['citation_source_validity']['validation_passed'])
        self.assertEqual(score['gold_source_span_recovery']['recall'], 1)
        value['claims'][0]['citations'][0]['quote'] = '사업 예산: 999원'
        score = evaluate_answer_case(case(), value, alignment(), source_documents=docs, registry=registry)
        self.assertFalse(score['citation_source_validity']['validation_passed'])
        self.assertEqual(score['gold_source_span_recovery']['recall'], 0)

    def test_legitimate_ambiguous_alternative_can_complete_a_source_link(self):
        a = alignment()
        missing = copy.deepcopy(a['records'][0]['alternatives'][0])
        missing['source_fragments'][0]['block_id'] = 'unavailable_alternative'
        a['records'][0]['alternatives'].insert(0, missing)
        score = evaluate_answer_case(case(), result(), a, source_documents=source_documents())
        self.assertEqual(score['gold_source_span_recovery']['recall'], 1)

    def test_failed_or_blocked_executions_are_not_status_label_evaluable(self):
        for execution in ['validation_failed', 'plan_blocked', 'preflight_blocked',
                          'disabled', 'blocked', 'error', 'not_started', 'unverified']:
            with self.subTest(execution=execution):
                value = result(); value['execution_status'] = execution
                self.assertFalse(self.score(value)['status_label_evaluable'])
        value = result(); value['execution_status'] = 'not_requested'
        self.assertTrue(self.score(value)['status_label_evaluable'])

    def test_summary_context_is_reconstructed_from_processed_chunk_ids(self):
        docs, chunks, registry = actual_registry()
        value = result(); value['processed_chunk_ids'] = ['c1']; value['source_chunk_ids'] = ['c1']
        score = evaluate_answer_case(case('summary'), value, alignment(), source_documents=docs, registry=registry)
        self.assertTrue(score['citation_source_validity']['independently_checked'])
        self.assertTrue(score['citation_source_validity']['validation_passed'])
        self.assertEqual(score['gold_source_span_recovery']['recall'], 1)


class DevRunnerTests(unittest.TestCase):
    def factory(self, log, responses=None):
        return lambda active_doc_ids=None: MockSession(log, active_doc_ids, responses)

    def test_only_question_and_user_context_reach_the_answer_session(self):
        log = []
        report = run_dev_evaluation([case()], self.factory(log), max_cases=1)
        self.assertEqual(log, [('init', ['A']), ('ask', case()['question'], 'qa', None)])
        self.assertNotIn('GOLD_DO_NOT_PASS', repr(log))
        self.assertNotIn('GOLD_BEHAVIOR_DO_NOT_PASS', repr(log))
        self.assertFalse(report['automatic_answer_accuracy'])

    def test_comparison_and_uncertainty_use_user_context_only(self):
        log = []
        compare = case('comparison'); compare['context']['selected_doc_ids'] = ['A', 'B']
        uncertain = case('uncertainty', 'U1')
        uncertain['context'] = {'selected_doc_ids': [], 'candidate_doc_ids': ['A', 'B']}
        run_dev_evaluation([compare, uncertain], self.factory(log), max_cases=2)
        self.assertIn(('ask', compare['question'], 'comparison', None), log)
        self.assertIn(('ask', uncertain['question'], 'qa', ['A', 'B']), log)

    def test_followup_preserves_one_session_without_expected_future_scope_leak(self):
        log = []
        c = case('followup', 'D1')
        c['turns'] = [
            {'turn_id': 't1', 'user': '첫 질문', 'expected_status': 'answered', 'required_fact_ids': ['f1'], 'expected_doc_ids': ['A']},
            {'turn_id': 't2', 'user': '그것은?', 'expected_status': 'answered', 'required_fact_ids': ['f1'], 'expected_doc_ids': ['GOLD_FUTURE']}]
        report = run_dev_evaluation([c], self.factory(log), alignment=alignment('D1'), source_documents=source_documents())
        self.assertEqual(sum(row[0] == 'init' for row in log), 1)
        self.assertEqual([row[1] for row in log if row[0] == 'ask'], ['첫 질문', '그것은?'])
        self.assertNotIn('GOLD_FUTURE', repr(log))
        self.assertEqual(len(report['case_results'][0]['turn_results']), 2)
        self.assertTrue(report['case_results'][0]['all_turn_statuses_match'])

    def test_each_followup_turn_is_scored_instead_of_only_the_last(self):
        log = []
        c = case('followup', 'D1')
        c['turns'] = [{'turn_id': 't1', 'user': '첫 질문', 'expected_status': 'answered'},
                      {'turn_id': 't2', 'user': '둘째 질문', 'expected_status': 'answered'}]
        values = [result(status='insufficient_evidence'), result()]
        report = run_dev_evaluation([c], self.factory(log, values))
        self.assertFalse(report['case_results'][0]['all_turn_statuses_match'])
        self.assertTrue(report['case_results'][0]['turn_results'][1]['status_match'])

    def test_followup_local_clarification_does_not_erase_earlier_verified_quotes(self):
        log = []
        c = case('followup', 'D1'); c['expected_status'] = 'needs_clarification'
        c['turns'] = [
            {'turn_id': 't1', 'user': '첫 질문', 'expected_status': 'answered', 'required_fact_ids': ['f1']},
            {'turn_id': 't2', 'user': '다른 사업은?', 'expected_status': 'needs_clarification', 'required_fact_ids': []}]
        clarify = {'status': 'needs_clarification', 'execution_status': 'not_requested',
                   'claims': [], 'validation': None, 'answer_generated': False}
        report = run_dev_evaluation([c], self.factory(log, [result(), clarify]),
                                   alignment=alignment('D1'), source_documents=source_documents())
        row = report['case_results'][0]
        self.assertTrue(row['all_turn_statuses_match'])
        self.assertEqual(row['gold_source_span_recovery']['recall'], 1)
        self.assertEqual(row['citation_source_validity']['validated_claim_count'], 1)

    def test_summary_uses_all_document_route_not_qa_search(self):
        log = []
        class Summary:
            is_mock = True
            def summarize(self, doc_id, question):
                log.append(('summary', doc_id, question)); return result()
        c = case('summary')
        run_dev_evaluation([c], self.factory(log), summary_factory=Summary)
        self.assertEqual(log, [('summary', 'A', c['question'])])

    def test_multiple_summaries_revalidate_with_status_missing_fields_and_actual_source_ids(self):
        log = []
        docs, chunks, registry = actual_registry(two_documents=True)
        class Summary:
            is_mock = True
            def summarize(self, doc_id, question):
                chunk = next(value for value in chunks if value['doc_id'] == doc_id)
                payload = {'status': 'answered', 'claims': [{'doc_id': doc_id, 'category': 'budget',
                    'text': '예산은 100원입니다.', 'citations': [{'chunk_id': chunk['chunk_id'], 'quote': TEXT}]}],
                    'missing_information': []}
                checked = registry.validate_answer(payload, [chunk], allowed_doc_ids=[doc_id])
                return {**checked, 'execution_status': 'completed',
                        'source_chunk_ids': [chunk['chunk_id']], 'processed_chunk_ids': [chunk['chunk_id']]}
        c = case('summary'); c['context']['selected_doc_ids'] = ['A', 'B']
        report = run_dev_evaluation([c], self.factory(log), summary_factory=Summary,
            source_documents=docs, alignment=alignment(), registry=registry)
        row = report['case_results'][0]
        self.assertTrue(row['citation_source_validity']['independently_checked'])
        self.assertTrue(row['citation_source_validity']['validation_passed'])
        self.assertEqual(row['citation_source_validity']['validated_claim_count'], 2)
        self.assertEqual(row['gold_source_span_recovery']['recall'], 1)

    def test_multiple_summaries_preserve_valid_document_when_another_plan_is_blocked(self):
        log = []
        docs, chunks, registry = actual_registry(two_documents=True)
        class Summary:
            is_mock = True
            def summarize(self, doc_id, question):
                if doc_id == 'B':
                    return {'status': 'blocked', 'execution_status': 'plan_blocked', 'valid': False,
                            'claims': [], 'source_chunk_ids': ['c2'], 'processed_chunk_ids': []}
                checked = registry.validate_answer({'status': 'answered', 'claims': [{
                    'doc_id': 'A', 'category': 'budget', 'text': '예산은 100원입니다.',
                    'citations': [{'chunk_id': 'c1', 'quote': TEXT}]}], 'missing_information': []}, [chunks[0]])
                return {**checked, 'execution_status': 'completed',
                        'source_chunk_ids': ['c1'], 'processed_chunk_ids': ['c1']}
        c = case('summary'); c['context']['selected_doc_ids'] = ['A', 'B']
        report = run_dev_evaluation([c], self.factory(log), summary_factory=Summary,
            source_documents=docs, alignment=alignment(), registry=registry)
        row = report['case_results'][0]
        self.assertEqual(row['actual_status'], 'partial')
        self.assertEqual(row['execution_status'], 'incomplete')
        self.assertTrue(row['citation_source_validity']['validation_passed'])
        self.assertEqual(row['citation_source_validity']['validated_claim_count'], 1)
        self.assertEqual(row['gold_source_span_recovery']['recall'], 1)

    def test_missing_summary_factory_is_blocked_without_topk_fallback(self):
        log = []
        report = run_dev_evaluation([case('summary')], self.factory(log))
        self.assertEqual(log, [])
        self.assertEqual(report['case_results'][0]['execution_status'], 'blocked')

    def test_holdout_anywhere_is_rejected_before_factory_even_beyond_max_cases(self):
        log = []
        c = case(); c['split'] = 'holdout'
        with self.assertRaises(ValueError):
            run_dev_evaluation([case(), c], self.factory(log), max_cases=1)
        self.assertEqual(log, [])

    def test_unapproved_user_scope_is_rejected_before_any_generation(self):
        log = []
        with self.assertRaises(ValueError):
            run_dev_evaluation([case()], self.factory(log), allowed_doc_ids=['B'])
        self.assertEqual(log, [])

    def test_enabled_llm_is_blocked_before_ask_without_live_opt_in(self):
        log = []
        class Session:
            llm = types.SimpleNamespace(config=types.SimpleNamespace(enabled=True))
            def ask(self, *args, **kwargs):
                log.append('CALLED'); raise AssertionError('should be blocked')
        report = run_dev_evaluation([case()], lambda active_doc_ids=None: Session())
        self.assertEqual(log, [])
        self.assertEqual(report['execution_modes'], {'blocked': 1})
        self.assertEqual(report['status_label_evaluable_cases'], 0)

    def test_unknown_client_is_also_blocked_before_ask(self):
        log = []
        class Session:
            def ask(self, *args, **kwargs): log.append('CALLED')
        run_dev_evaluation([case()], lambda active_doc_ids=None: Session())
        self.assertEqual(log, [])

    def test_live_opt_in_and_live_result_are_reported_separately_from_truth(self):
        log = []
        value = result(); value['generation']['live_api'] = True
        report = run_dev_evaluation([case()], self.factory(log, [value]), allow_live=True)
        self.assertEqual(report['execution_modes'], {'live_api': 1})
        self.assertIsNone(report['automatic_answer_accuracy'])
        self.assertTrue(report['human_semantic_review_required'])

    def test_cached_live_origin_is_not_counted_as_a_new_live_api_call(self):
        log = []
        value = result()
        value['generation'].update(live_api=True, cache_hit=True,
                                   usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
        report = run_dev_evaluation([case()], self.factory(log, [value]), allow_live=True)
        self.assertEqual(report['execution_modes'], {'cached_from_live_api': 1})
        self.assertFalse(report['live_api_executed'])
        self.assertFalse(report['live_api_quality_measured'])
        self.assertEqual(report['usage_totals']['total_tokens'], 0)

    def test_errors_do_not_disappear_from_source_recall_denominator_or_expose_secrets(self):
        log = []
        report = run_dev_evaluation([case()], self.factory(log, [RuntimeError('SECRET_DO_NOT_SAVE')]),
            alignment=alignment(), source_documents=source_documents())
        self.assertNotIn('SECRET_DO_NOT_SAVE', json.dumps(report))
        self.assertEqual(report['quoted_gold_source_evidence_links_measured'], 1)
        self.assertEqual(report['quoted_gold_source_span_recall'], 0)
        self.assertEqual(report['case_results'][0]['execution_status'], 'error')

    def test_max_cases_is_a_bounded_prefix_not_gold_driven_selection(self):
        log = []
        report = run_dev_evaluation([case(case_id='F1'), case(case_id='F2')], self.factory(log), max_cases=1)
        self.assertEqual(report['evaluated_cases'], 1)
        self.assertEqual(report['supplied_cases'], 2)
        self.assertEqual(len([row for row in log if row[0] == 'ask']), 1)

    def test_usage_and_human_review_artifacts_are_saved_with_unfilled_semantic_scores(self):
        log = []
        with tempfile.TemporaryDirectory() as directory:
            report = run_dev_evaluation([case()], self.factory(log), output_dir=directory)
            self.assertEqual(report['usage_totals']['total_tokens'], 15)
            saved = json.loads((Path(directory) / 'dev_answer_evaluation.json').read_text())
            review = json.loads((Path(directory) / 'human_review.jsonl').read_text())
            self.assertIsNone(saved['automatic_answer_accuracy'])
            self.assertIsNone(review['answer_correct'])
            self.assertEqual(review['status'], 'not_reviewed')

    def test_summary_usage_and_live_flags_are_collected_from_batch_outcomes(self):
        log = []
        class Summary:
            is_mock = True
            def summarize(self, doc_id, question):
                value = result(); value.pop('generation')
                value['batch_outcomes'] = [
                    {'live_api': True, 'usage': {'input_tokens': 5, 'output_tokens': 2, 'total_tokens': 7}},
                    {'live_api': False, 'usage': {'input_tokens': 9, 'output_tokens': 4, 'total_tokens': 13}}]
                return value
        report = run_dev_evaluation([case('summary')], self.factory(log), summary_factory=Summary, allow_live=True)
        self.assertEqual(report['usage_totals']['total_tokens'], 20)
        self.assertEqual(report['execution_modes'], {'live_api': 1})


if __name__ == '__main__':
    unittest.main()
