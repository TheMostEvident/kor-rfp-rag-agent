"""No-call inspection and safe propagation of failed-generation diagnostics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_phase4.llm import LLMError
from rfp_phase5.diagnostics import inspect_saved_failure

# Reuse source-backed fixtures; these doubles never import the live SDK.
import test_phase5_summary as fixtures


class SavedDiagnosticTests(unittest.TestCase):
    def write_result(self, root, result):
        path = root / 'phase5_outputs/runs/old/result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
        return path

    def test_old_result_cannot_reconstruct_lost_cause_or_turn_into_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = {'status': 'error', 'execution_status': 'incomplete',
                      'usage': {'api_calls': 1, 'input_tokens': 1200, 'output_tokens': 6000},
                      'errors': ['map-0001:batch_failed:LLMError'],
                      'claims': [{'text': 'do-not-print-source-text'}]}
            path = self.write_result(root, result)
            before = path.read_bytes()
            with patch('socket.socket', side_effect=AssertionError('network prohibited')):
                report = inspect_saved_failure(root)
            self.assertEqual(report['diagnostic_classification'], 'response_usage_received_before_failure')
            self.assertFalse(report['root_cause_confirmed'])
            self.assertFalse(report['past_exception_reconstructed'])
            self.assertEqual(report['api_calls_by_diagnostic'], 0)
            self.assertNotIn('do-not-print-source-text', json.dumps(report))
            self.assertEqual(before, path.read_bytes())

    def test_unknown_usage_is_not_labelled_definite_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_result(root, {'usage': {'usage_unknown_attempts': 1}})
            report = inspect_saved_failure(root)
            self.assertEqual(report['diagnostic_classification'], 'failed_request_or_unreported_usage')
            self.assertFalse(report['root_cause_confirmed'])

    def test_session_usage_alone_does_not_attribute_previous_tokens_to_this_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_result(root, {'llm_session_usage': {'reported_output_tokens': 9999}})
            report = inspect_saved_failure(root)
            self.assertEqual(report['diagnostic_classification'], 'original_error_detail_unavailable')

    def test_no_saved_result_is_zero_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(inspect_saved_failure(tmp)['status'], 'no_saved_result')

    def test_summary_retains_controlled_error_and_hides_arbitrary_exception_text(self):
        fixture = fixtures.FieldSummaryTests()
        llm = fixtures.OfflineLLM()
        summary, _, _ = fixture.make(count=1, llm=llm)
        with patch.object(llm, 'generate', side_effect=LLMError(
                'API response is incomplete or failed; no factual answer was accepted')):
            result = summary.summarize('d1')
        self.assertEqual(result['execution_status'], 'incomplete')
        self.assertEqual(result['validated_chunk_coverage'], 0)
        self.assertIn('error_detail', result)
        self.assertTrue(result['error_diagnostics'])
        self.assertEqual(result['batch_outcomes'][0]['error_detail'], result['error_detail'])
        with patch.object(llm, 'generate', side_effect=RuntimeError('sk-private-secret-do-not-print')):
            hidden = summary.summarize('d1')
        self.assertNotIn('sk-private', json.dumps(hidden))

    def test_preflight_retains_controlled_error_without_calling_generate(self):
        fixture = fixtures.FieldSummaryTests()
        summary, llm, _ = fixture.make(count=1)
        with patch.object(llm, 'preflight_requests', side_effect=LLMError(
                "Planned inputs exceed this session's max_total_input_bytes")):
            result = summary.summarize('d1')
        self.assertEqual(result['execution_status'], 'preflight_blocked')
        self.assertEqual(llm.calls, [])
        self.assertIn('error_detail', result)


if __name__ == '__main__':
    unittest.main()
