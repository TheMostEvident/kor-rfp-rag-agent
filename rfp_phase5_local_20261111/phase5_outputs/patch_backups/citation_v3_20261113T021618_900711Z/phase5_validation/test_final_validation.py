"""Acceptance regressions with synthetic source-backed doubles. API calls: zero."""
import copy
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('final_validation_under_test', Path(__file__).with_name('final_validation.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

import test_phase5_summary as fixture


def case():
    summary, llm, chunks = fixture.FieldSummaryTests().make(count=3,
        llm=fixture.OfflineLLM(all_fields=True), config={'max_batch_bytes': 9000})
    result = summary.summarize('d1')
    plan = summary.plan('d1')
    for n, batch in enumerate(result['batch_outcomes']):
        batch.update(execution_mode='live_openai', live_api=True, request_fingerprint=str(n+1))
    summary.index.get_catalog = lambda: [{'doc_id': 'd1', 'sha256': 'a'*64}]
    env = {'index': summary.index, 'registry': summary.registry}
    sample = {'sample_id': 'D01', 'doc_id': 'd1', 'source_sha256': 'a'*64}
    fps = [str(n+1) for n in range(len(result['batch_outcomes']))]
    refs = {'rows': [{'sample_id': 'D01', 'field_id': f['id'], 'label': f['label'],
        'expected': '원문 대조용 테스트 기준', 'conditions': '조건 보존 검토', 'source_reference': 'PDF p.1'}
        for f in fixture.FIELDS]}
    return result, sample, env, plan, fps, refs


class FinalValidationTests(unittest.TestCase):
    def check(self, result, sample, env, plan, fps):
        return v.technical_check(result, sample, env, plan, fps)

    def test_complete_multibatch_sources_pass_technical_only(self):
        r, s, e, p, fps, _ = case()
        self.assertGreater(p['map_calls'], 1)
        check = self.check(r, s, e, p, fps)
        self.assertTrue(check['passed'], check['failed_checks'])
        self.assertFalse(check['semantic_quality_verified'])
        self.assertFalse(v.case_verdict(check, {'passed': False, 'status': 'review_pending'})['accepted'])

    def test_partial_status_requires_complete_processing_and_human_review(self):
        r, s, e, p, fps, _ = case()
        r['status'] = 'partial'
        r['missing_information'] = ['제출기한은 공고문 별도 확인']
        check = self.check(r, s, e, p, fps)
        self.assertTrue(check['passed'], check['failed_checks'])
        self.assertTrue(v.case_verdict(check, {'passed': True, 'status': 'passed_human_review'})['accepted'])
        r['processed_chunk_ids'] = r['processed_chunk_ids'][:-1]
        self.assertFalse(self.check(r, s, e, p, fps)['passed'])

    def test_fabricated_quote_rejected_despite_saved_valid_flag(self):
        r, s, e, p, fps, _ = case()
        r['claims'][0]['citations'][0]['quote'] = '존재하지 않는 예산 999조원'
        r['valid'] = True
        self.assertFalse(self.check(r, s, e, p, fps)['passed'])

    def test_display_can_not_diverge_from_validated_claims(self):
        r, s, e, p, fps, _ = case()
        r['fields'][0]['claims'][0]['text'] = '모델 출력에 없는 사업명'
        self.assertIn('field_display_matches_validated_claims', self.check(r, s, e, p, fps)['failed_checks'])

    def test_field_state_can_not_be_changed_to_hide_uncertainty(self):
        r, s, e, p, fps, _ = case()
        r['fields'][0]['state'] = 'not_found_in_processed_input'
        self.assertIn('field_states_match_claims', self.check(r, s, e, p, fps)['failed_checks'])

    def test_wrong_scope_source_and_request_all_rejected(self):
        for mode in ['scope', 'source', 'request', 'mock', 'missing_field', 'reorder']:
            r, s, e, p, fps, _ = case()
            if mode == 'scope': r['doc_id'] = 'other'
            if mode == 'source': s['source_sha256'] = 'b'*64
            if mode == 'request': fps[0] = 'different-config-or-prompt'
            if mode == 'mock': r['batch_outcomes'][0]['live_api'] = False
            if mode == 'missing_field': r['fields'].pop()
            if mode == 'reorder': r['batch_outcomes'].reverse()
            with self.subTest(mode=mode): self.assertFalse(self.check(r, s, e, p, fps)['passed'])

    def test_duplicate_processing_ids_do_not_count_as_full_coverage(self):
        r, s, e, p, fps, _ = case()
        r['processed_chunk_ids'] += r['processed_chunk_ids']
        self.assertFalse(self.check(r, s, e, p, fps)['passed'])

    def test_cached_live_origin_allowed_without_new_paid_call(self):
        r, s, e, p, fps, _ = case()
        for batch in r['batch_outcomes']:
            batch.update(cache_hit=True, api_called=False)
        self.assertTrue(self.check(r, s, e, p, fps)['passed'])

    def review_fixture(self, folder):
        r, s, e, p, fps, refs = case()
        rows = v.make_review_rows(r, s, refs)
        path = Path(folder) / 'comparison.csv'
        v.write_review(path, rows)
        return path, rows

    def fill(self, path, rows, value='통과'):
        output = copy.deepcopy(rows)
        for row in output:
            for key in v.JUDGEMENTS: row[key] = value
        path.unlink()
        v.write_review(path, output)
        return output

    def test_review_is_blank_and_never_implicitly_positive(self):
        with tempfile.TemporaryDirectory() as d:
            path, rows = self.review_fixture(d)
            report = v.assess_csv(path, rows, '검토자')
            self.assertEqual(report['status'], 'review_pending')
            self.assertEqual(report['reviewed'], 0)

    def test_all_33_judgements_and_reviewer_required(self):
        with tempfile.TemporaryDirectory() as d:
            path, rows = self.review_fixture(d)
            self.fill(path, rows)
            self.assertFalse(v.assess_csv(path, rows, '')['passed'])
            report = v.assess_csv(path, rows, '원문 검토자')
            self.assertTrue(report['passed'])
            self.assertEqual(report['positive'], 33)

    def test_negative_and_invalid_judgements_block_acceptance(self):
        for value in ['실패', '1', '=TRUE()']:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as d:
                path, rows = self.review_fixture(d)
                self.fill(path, rows, value)
                self.assertFalse(v.assess_csv(path, rows, '검토자')['passed'])

    def test_review_bound_to_unchanged_answer_and_reference(self):
        with tempfile.TemporaryDirectory() as d:
            path, rows = self.review_fixture(d)
            changed = self.fill(path, rows)
            changed[0]['실제요약'] = '사용자가 표에서 바꾼 답변'
            path.unlink(); v.write_review(path, changed)
            self.assertEqual(v.assess_csv(path, rows, '검토자')['status'], 'invalid_review')

    def test_csv_input_is_preserved_on_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            path, rows = self.review_fixture(d)
            self.fill(path, rows)
            before = path.read_bytes()
            v.write_review(path, rows)
            self.assertEqual(before, path.read_bytes())

    def test_csv_formula_text_is_escaped(self):
        self.assertEqual(v._csv_text('=HYPERLINK("bad")'), '\'=HYPERLINK("bad")')
        self.assertEqual(v._csv_text('650,000,000원'), '650,000,000원')

    def test_invalid_suite_id_rejected_before_any_components_or_api(self):
        with self.assertRaises(ValueError): v.ValidationSuite({}, {}, '../other')

    def stub_suite(self, folder):
        obj = v.ValidationSuite.__new__(v.ValidationSuite)
        obj.folder = Path(folder)
        obj.samples = [{'sample_id': 'D01', 'split': 'dev', 'doc_id': 'd1', 'file_name': '공개 문서'}]
        obj.assert_frozen = lambda: None
        obj.report = lambda reviewer: {'dev_passed': False}
        obj.import_existing = lambda split: []
        obj.used_budget = lambda: {'api_calls': 0, 'attempted_input_bytes': 0}
        obj.plans = {'D01': {'question': '형식 검사', 'batches': []}}
        obj.suite_calls, obj.suite_bytes = 1, 100
        return obj

    def test_final_stage_does_not_generate_before_dev_human_review(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self.stub_suite(d)
            obj._component = lambda *a, **kw: self.fail('검토 전 API 준비도 하지 않아야 함')
            self.assertEqual(obj.run_stage('final', enabled=True)['status'], 'blocked_until_dev_review_passes')

    def test_whole_suite_cap_blocks_before_generate(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self.stub_suite(d)
            summary = SimpleNamespace(_adapter=SimpleNamespace(preflight_requests=lambda _: {
                'uncached_requests': 2, 'input_bytes_to_reserve': 50}))
            summary.summarize = lambda *a: self.fail('상한을 넘으면 생성하면 안 됨')
            obj._component = lambda *a, **kw: {'summarizer': summary}
            with self.assertRaises(ValueError): obj.run_stage('dev', enabled=True)

    def test_interrupted_attempt_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self.stub_suite(d)
            path = Path(d)/'D01';path.mkdir();(path/'started.json').write_text('{}')
            obj._component = lambda *a, **kw: self.fail('중단된 요청의 자동 재시도 금지')
            self.assertEqual(obj.run_stage('dev', enabled=True)['status'], 'interrupted_attempt_requires_review')

    def test_saved_result_is_not_generated_again_on_run_all(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self.stub_suite(d)
            path = Path(d)/'D01';path.mkdir();(path/'result.json').write_text('{"status":"error"}')
            obj._component = lambda *a, **kw: self.fail('실패도 명시적 새 실행 전까지 보존')
            self.assertEqual(obj.run_stage('dev', enabled=True)['samples_generated'], [])

    def test_saved_technical_failure_stops_remaining_document_calls(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self.stub_suite(d)
            obj.report = lambda reviewer: {'dev_passed': False, 'cases': [
                {'sample_id': 'D01', 'split': 'dev', 'status': 'technical_failed'}]}
            obj._component = lambda *a, **kw: self.fail('기존 실패가 있으면 추가 요청 중단')
            self.assertEqual(obj.run_stage('dev', enabled=True)['status'], 'blocked_by_existing_technical_failure')


if __name__ == '__main__':
    unittest.main()
