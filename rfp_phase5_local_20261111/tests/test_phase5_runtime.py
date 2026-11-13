"""Offline run isolation, effective flags, and invocation cost boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_phase5.runtime import make_components, save_run


class Index:
    def resolve_doc_ids(self, ids):
        if any(i not in {'d1', 'alias'} for i in ids):
            raise ValueError('unknown document')
        return ['d1']

    def get_catalog(self):
        return [{'doc_id': 'd1', 'sha256': 'a' * 64}]


class RuntimeTests(unittest.TestCase):
    def env(self, root):
        return {'root': root, 'index': Index(), 'registry': object(),
                'phase5_config': {'generation': {'enabled': False},
                                  'retrieval': {}, 'summary': {}}}

    def test_effective_enable_overrides_file_default_without_sdk_or_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('rfp_phase5.summary.FieldSummarizer'), patch('rfp_phase4.rag.AnswerSession'):
                value = make_components(self.env(root), doc_ids=['alias'], enabled=True)
            self.assertTrue(value['llm'].config.enabled)
            self.assertEqual(value['approved_sources'], {'d1': 'a' * 64})
            self.assertEqual(value['llm'].usage_report()['api_calls'], 0)
            self.assertEqual(list(root.iterdir()), [])

    def test_empty_approval_and_truthy_string_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                make_components(self.env(Path(tmp)), doc_ids=['d1'], enabled='False')
            with self.assertRaises(ValueError):
                make_components(self.env(Path(tmp)), doc_ids=[], enabled=False)

    def test_runs_unique_review_pending_and_cost_not_cumulative(self):
        result = {'status': 'not_generated', 'execution_status': 'disabled',
                  'answer_generated': False, 'claims': [], 'markdown': '미실행',
                  'usage': {'input_tokens': 10, 'output_tokens': 5,
                            'cached_input_tokens': 0, 'api_calls': 0}}
        original = copy.deepcopy(result)
        pricing = {'model': 'test', 'input_per_million': 1,
                   'cached_input_per_million': 0.1, 'output_per_million': 2}
        with tempfile.TemporaryDirectory() as tmp:
            first = save_run(tmp, result, operation='qa',
                             usage={'reported_input_tokens': 1000000}, pricing=pricing)
            second = save_run(tmp, result, operation='qa', pricing=pricing)
            self.assertNotEqual(first, second)
            self.assertEqual({p.name for p in first.iterdir()},
                             {'run.json', 'result.json', 'quality.json', 'human_review.json', 'answer.md'})
            record = json.loads((first / 'run.json').read_text())
            self.assertEqual(record['cost_estimate']['input_tokens'], 10)
            self.assertEqual(record['cost_estimate']['estimated_usd_for_reported_tokens'], 0.00002)
            self.assertEqual(record['human_review_status'], 'pending')
            self.assertIsNone(record['semantic_accuracy'])
        self.assertEqual(result, original)

    def test_bad_operation_and_nonfinite_data_leave_no_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_run(tmp, {}, operation='../danger')
            with self.assertRaises(ValueError):
                save_run(tmp, {'value': float('nan')}, operation='qa')
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
