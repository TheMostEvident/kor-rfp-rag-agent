"""Orchestration policy with one invented document and mocked retrieval metrics.

No project evaluation questions, real holdout queries, API keys or network are
used. Each test builds and discards a separate temporary experiment directory.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_phase3 as pipeline


class FixtureIndex:
    """A local artifact stub; it has no search or remote-client implementation."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.build_calls = 0
        self.metadata_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def build(self, chunks, config):
        self.build_calls += 1
        pipeline.write_json(self.directory / 'index_meta.json',
                            {'fixture': True, 'chunk_ids': [c['chunk_id'] for c in chunks]})
        return {'chunks': len(chunks), 'cache_reused': False}

    def get_metadata(self):
        self.metadata_calls += 1
        return {'fixture': True}


def fixture_retrieval(index, cases, alignment, **arguments):
    method, scope = arguments['method'], arguments['scope_mode']
    # Global results favour rrf; the declared context-only criterion favours bm25.
    recall = .8 if method == 'bm25' else .7
    if scope == 'global':
        recall = .95 if method == 'rrf' else .6
    return {'method': method, 'scope_protocol': scope, 'case_count': len(cases),
            'excluded_case_count': 0,
            'metrics_at_k': {str(k): {'micro_evidence_span_recall': recall} for k in [5, 10, 20]},
            'mrr_first_source_overlap': recall, 'latency_ms': {'median': 0},
            'scope': {'fixture': True}}


class PipelinePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {'version': 'fixture-v1',
                       'chunking': {'chunk_size_bytes': 1800, 'overlap_bytes': 240},
                       'retrieval': {},
                       'search': {'method': 'rrf', 'top_k': 5, 'fetch_k': 40, 'mmr_lambda': .75},
                       'evaluation': {'top_ks': [5, 10, 20], 'scope_modes': ['context', 'global'],
                                      'methods': ['bm25', 'rrf']},
                       'openai': {'enabled': False}}
        doc = {'doc_id': 'fixture_document', 'canonical_doc_id': 'fixture_document',
               'sha256': 'a' * 64, 'file_name': 'invented_fixture.hwp',
               'detected_format': 'hwp', 'status': 'parsed',
               'blocks': [{'block_id': 'fixture_block', 'locator': 'BodyText/Section0 / fixture',
                           'section': 'BodyText/Section0', 'text': '가상의 문서와 가상의 수치입니다.'}],
               'supplemental_blocks': [], 'reference_blocks': [], 'tables': [], 'warnings': []}
        pipeline.write_json(self.root / 'configs/default.json', self.config)
        pipeline.write_jsonl(self.root / 'data/source_documents.jsonl', [doc])
        pipeline.write_json(self.root / 'data/documents_manifest.json', {'fixture': True})
        pipeline.write_json(self.root / 'data/source_evidence_validation.json', {'fixture': True})
        for split in ['dev', 'holdout']:
            pipeline.write_jsonl(self.root / f'eval/questions.{split}.jsonl',
                                 [{'case_id': f'invented_{split}', 'split': split,
                                   'question': 'Invented fixture question.'}])
        (self.root / 'run_phase3.py').write_text('# invented fixture code version\n')
        (self.root / 'local_rag').mkdir()
        (self.root / 'local_rag/fixture.py').write_text('# invented fixture module version\n')
        self.alignment = {'total_links': 1, 'covered_links': 1, 'uncovered_links': 0,
                          'source_ambiguous_links': 0, 'chunk_source_issue_count': 0, 'valid': True}
        self.indices = []

        def make_index(path):
            instance = FixtureIndex(path)
            self.indices.append(instance)
            return instance

        stack = ExitStack()
        self.addCleanup(stack.close)
        self.index_mock = stack.enter_context(patch.object(pipeline, 'LocalSearchIndex', side_effect=make_index))
        self.alignment_mock = stack.enter_context(patch.object(pipeline, 'audit_chunk_coverage', return_value=self.alignment))
        self.retrieval_mock = stack.enter_context(patch.object(pipeline, 'evaluate_retrieval', side_effect=fixture_retrieval))
        self.summary_mock = stack.enter_context(patch.object(pipeline, 'evaluate_summary_coverage', return_value={'fixture': True}))

    def build(self):
        return pipeline.build_pipeline(self.root)

    def prepare_selected(self):
        self.build()
        pipeline.evaluate_split(self.root, 'dev')
        return pipeline.select_on_dev(self.root)

    def prepare_frozen_without_reading_questions(self):
        self.prepare_selected()
        with patch.object(pipeline, 'evaluate_split', return_value={'fixture': 'not evaluated'}):
            pipeline.freeze_and_evaluate_holdout(self.root)
        return json.loads((self.root / 'reports/frozen_configuration.json').read_text())

    def assert_denied_before_questions(self, function, *args, **kwargs):
        with patch.object(pipeline, 'read_jsonl', side_effect=AssertionError('Questions must not be parsed')):
            with self.assertRaises(ValueError):
                function(*args, **kwargs)

    def test_build_does_not_parse_questions_or_evaluate_or_call_api(self):
        with patch.object(pipeline, 'read_jsonl', side_effect=AssertionError('Build must not parse questions')):
            result = self.build()
        self.assertEqual(result['chunks'], 1)
        self.assertEqual(result['openai_api_calls'], 0)
        self.assertFalse(result['openai_live_measured'])
        self.retrieval_mock.assert_not_called()
        self.summary_mock.assert_not_called()
        manifest = pipeline.verify_saved_run(self.root)
        self.assertEqual(manifest['network_calls'], 0)
        self.assertIn('eval/questions.dev.jsonl', manifest['files_sha256'])
        self.assertIn('eval/questions.holdout.jsonl', manifest['files_sha256'])

    def test_successful_force_rebuild_invalidates_previous_measurements(self):
        self.build()
        for name in ['dev_evaluation.json', 'holdout_evaluation.json', 'frozen_configuration.json', 'selection.json']:
            pipeline.write_json(self.root / 'reports' / name, {'stale_fixture': True})
        result = pipeline.build_pipeline(self.root, force=True)
        for name in ['dev_evaluation.json', 'holdout_evaluation.json', 'frozen_configuration.json', 'selection.json']:
            self.assertFalse((self.root / 'reports' / name).exists(), name)
        self.assertIsNone(result['dev_evaluation_summary'])
        self.assertIsNone(result['holdout_evaluation_summary'])
        pipeline.verify_saved_run(self.root)

    def test_cached_build_rejects_changed_source_chunks_config_code_index_and_questions(self):
        self.build()
        for relative in ['data/source_documents.jsonl', 'data/chunks.jsonl', 'configs/default.json',
                         'run_phase3.py', 'local_rag/fixture.py', 'index/index_meta.json',
                         'eval/questions.dev.jsonl', 'eval/questions.holdout.jsonl']:
            with self.subTest(relative=relative):
                path = self.root / relative
                previous = path.read_bytes()
                path.write_bytes(previous + b'\n ')
                try:
                    with self.assertRaises(ValueError):
                        pipeline.build_pipeline(self.root)
                finally:
                    path.write_bytes(previous)

    def test_alignment_artifact_tampering_is_detected_before_evaluation(self):
        self.build()
        pipeline.write_json(self.root / 'reports/evidence_chunk_alignment.json', {'valid': True, 'tampered': True})
        self.assert_denied_before_questions(pipeline.evaluate_split, self.root, 'dev')
        self.retrieval_mock.assert_not_called()

    def test_failed_source_integrity_does_not_build_index_or_parse_questions(self):
        quality = {'valid': False, 'errors': ['invented corruption']}
        with patch.object(pipeline, 'validate_chunks', return_value=quality):
            self.assert_denied_before_questions(pipeline.build_pipeline, self.root)
        self.index_mock.assert_not_called()
        self.alignment_mock.assert_not_called()

    def test_dev_selection_uses_context_metrics_and_does_not_parse_holdout(self):
        self.build()
        with patch.object(pipeline, 'read_jsonl', wraps=pipeline.read_jsonl) as reader:
            report = pipeline.evaluate_split(self.root, 'dev')
            selection = pipeline.select_on_dev(self.root)
        self.assertEqual(report['split'], 'dev')
        self.assertEqual(selection['selected_default_method'], 'bm25')
        self.assertFalse(selection['holdout_outcomes_used'])
        self.assertEqual([call.args[0].name for call in reader.call_args_list], ['questions.dev.jsonl'])
        self.assertFalse((self.root / 'reports/frozen_configuration.json').exists())
        pipeline.verify_saved_run(self.root)

    def test_dev_selection_rejects_stale_config_and_a_mislabelled_report(self):
        self.build()
        pipeline.evaluate_split(self.root, 'dev')
        old_report = json.loads((self.root / 'reports/dev_evaluation.json').read_text())
        changed = copy.deepcopy(self.config)
        changed['search']['fetch_k'] = 90
        pipeline.write_json(self.root / 'configs/default.json', changed)
        pipeline.refresh_manifest(self.root)
        with self.assertRaises(ValueError):
            pipeline.select_on_dev(self.root)
        # A file named dev_evaluation must also identify itself as the dev split.
        pipeline.write_json(self.root / 'configs/default.json', self.config)
        old_report['split'] = 'holdout'
        pipeline.write_json(self.root / 'reports/dev_evaluation.json', old_report)
        pipeline.refresh_manifest(self.root)
        with self.assertRaises(ValueError):
            pipeline.select_on_dev(self.root)

    def test_holdout_requires_both_explicit_opt_in_and_a_frozen_record(self):
        self.build()
        self.assert_denied_before_questions(pipeline.evaluate_split, self.root, 'holdout', allow_holdout=True)
        pipeline.write_json(self.root / 'reports/frozen_configuration.json', {})
        self.assert_denied_before_questions(pipeline.evaluate_split, self.root, 'holdout')
        self.retrieval_mock.assert_not_called()

    def test_freeze_exists_before_first_holdout_parse_and_repeated_run_is_blocked(self):
        self.prepare_selected()
        original_reader = pipeline.read_jsonl

        def checked_reader(path):
            if Path(path).name == 'questions.holdout.jsonl':
                frozen = json.loads((self.root / 'reports/frozen_configuration.json').read_text())
                self.assertEqual(frozen['configuration_sha256'], pipeline.configuration_hash(pipeline._config(self.root)))
                self.assertEqual(frozen['planned_comparisons']['methods'], ['bm25', 'rrf'])
            return original_reader(path)

        with patch.object(pipeline, 'read_jsonl', side_effect=checked_reader):
            report = pipeline.freeze_and_evaluate_holdout(self.root)
        self.assertEqual(report['split'], 'holdout')
        self.assertEqual(report['api_calls'], 0)
        self.assertTrue((self.root / 'reports/holdout_evaluation.json').exists())
        self.assert_denied_before_questions(pipeline.freeze_and_evaluate_holdout, self.root)
        self.assert_denied_before_questions(pipeline.evaluate_split, self.root, 'holdout', allow_holdout=True)
        self.assert_denied_before_questions(pipeline.evaluate_split, self.root, 'dev')
        with self.assertRaises(ValueError):
            pipeline.select_on_dev(self.root)

    def test_all_frozen_fingerprints_are_checked_before_holdout_questions(self):
        frozen = self.prepare_frozen_without_reading_questions()
        for key in ['configuration_sha256', 'run_manifest_sha256', 'dev_report_sha256',
                    'selection_sha256', 'holdout_questions_sha256']:
            with self.subTest(fingerprint=key):
                altered = copy.deepcopy(frozen)
                altered[key] = '0' * 64
                pipeline.write_json(self.root / 'reports/frozen_configuration.json', altered)
                self.assert_denied_before_questions(pipeline.evaluate_split, self.root, 'holdout', allow_holdout=True)
        pipeline.write_json(self.root / 'reports/frozen_configuration.json', frozen)
        self.assert_denied_before_questions(pipeline.freeze_and_evaluate_holdout, self.root)

    def test_wrong_split_cases_cannot_bypass_dev_guard_with_holdout_opt_in(self):
        self.build()
        with patch.object(pipeline, 'read_jsonl', return_value=[{'case_id': 'invented', 'split': 'holdout'}]):
            with self.assertRaises(ValueError):
                pipeline.evaluate_split(self.root, 'dev', allow_holdout=True)
        self.retrieval_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
