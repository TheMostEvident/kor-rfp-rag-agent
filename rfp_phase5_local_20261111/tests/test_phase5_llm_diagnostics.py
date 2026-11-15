"""Failure diagnostics use fixed labels; API/source text never becomes a message."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_phase4.llm import LLMConfig, LLMError, OpenAIGroundedLLM, safe_llm_diagnostic
from test_phase4_llm import FakeClient, PAYLOAD, SCHEMA, SHA, evidence


class HTTPFailure(RuntimeError):
    def __init__(self, status, code):
        super().__init__('SECRET_KEY and private source must not appear')
        self.status_code, self.code = status, code
        self.body = {'error': {'message': 'SECRET_KEY and private RFP'}}


class IncompleteClient(FakeClient):
    def __init__(self, reason):
        super().__init__(status='incomplete')
        self.reason = reason

    def create(self, **kwargs):
        result = super().create(**kwargs)
        result.incomplete_details = {'reason': self.reason}
        return result


class LLMDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-dummy-only'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temporary.cleanup()

    def make(self, fake=None):
        fake = fake or FakeClient()
        return OpenAIGroundedLLM(self.cache, LLMConfig(enabled=True), {'A': SHA}, fake), fake

    def generate(self, llm, chunks=None):
        return llm.generate('qa', '예산은?', evidence() if chunks is None else chunks, schema=SCHEMA)

    def fail(self, fake):
        llm, fake = self.make(fake)
        with self.assertRaises(LLMError) as caught:
            self.generate(llm)
        return llm, fake, safe_llm_diagnostic(caught.exception)

    def test_exact_request_fingerprint_remains_unchanged(self):
        llm, _ = self.make()
        # Independently computed using the original untouched Phase 4 client.
        expected = '698e81792199cb4f15bc9d79acbb2f2361f6b7009dd2bcdcabc7d88c0a218b34'
        self.assertEqual(llm._prepare('qa', '예산은?', evidence(), SCHEMA)['fingerprint'], expected)

    def test_token_limit_is_reported_without_accepting_or_caching_answer(self):
        llm, fake, diagnostic = self.fail(IncompleteClient('max_output_tokens'))
        self.assertEqual(diagnostic['code'], 'response_incomplete')
        self.assertEqual(diagnostic['incomplete_reason'], 'max_output_tokens')
        self.assertEqual(diagnostic['response_status'], 'incomplete')
        self.assertIs(diagnostic['response_received'], True)
        self.assertEqual(diagnostic['request_fingerprint'], llm._prepare('qa', '예산은?', evidence(), SCHEMA)['fingerprint'])
        stats = llm.usage_report()
        self.assertEqual(stats['api_calls'], 1)
        self.assertEqual(stats['mock_calls'], 1)
        self.assertEqual(stats['failed_attempts'], 1)
        self.assertEqual(stats['usage_unknown_attempts'], 0)
        self.assertEqual(stats['reported_total_tokens'], 132)
        self.assertEqual(len(fake.calls), 1)
        self.assertFalse((self.cache / 'grounded_responses.sqlite3').exists())

    def test_content_filter_and_unknown_incomplete_reasons_are_bounded(self):
        for reason, expected in [('content_filter', 'content_filter'), ('SECRET_KEY', 'unknown'), (['SECRET_KEY'], 'unknown')]:
            with self.subTest(reason=reason):
                _, _, diagnostic = self.fail(IncompleteClient(reason))
                self.assertEqual(diagnostic['incomplete_reason'], expected)
                self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_refusal_does_not_include_refusal_text(self):
        _, _, diagnostic = self.fail(FakeClient(output=[{'content': [
            {'type': 'refusal', 'refusal': 'SECRET_KEY and private source'}]}]))
        self.assertEqual(diagnostic['code'], 'response_refused')
        self.assertIs(diagnostic['response_received'], True)
        self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_http400_schema_code_and_unknown_usage_are_preserved(self):
        llm, fake, diagnostic = self.fail(FakeClient(fail=HTTPFailure(400, 'invalid_json_schema')))
        self.assertEqual(diagnostic['code'], 'api_http_error')
        self.assertEqual(diagnostic['http_status'], 400)
        self.assertEqual(diagnostic['api_error_code'], 'invalid_json_schema')
        self.assertIs(diagnostic['response_received'], False)
        self.assertEqual(llm.usage_report()['usage_unknown_attempts'], 1)
        self.assertEqual(llm.usage_report()['reported_total_tokens'], 0)
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_http429_quota_is_distinguished_without_claiming_a_specific_balance(self):
        _, _, diagnostic = self.fail(FakeClient(fail=HTTPFailure(429, 'insufficient_quota')))
        self.assertEqual(diagnostic['code'], 'api_rate_or_quota_limit')
        self.assertEqual(diagnostic['http_status'], 429)
        self.assertEqual(diagnostic['api_error_code'], 'insufficient_quota')

    def test_schema_rule_and_context_survive_without_reflected_values_or_extra_attempts(self):
        error = HTTPFailure(400, 'invalid_json_schema')
        error.body = {'message': "Invalid schema for response_format 'rfp_grounded_answer': "
            "In context=('properties', 'claims', 'items', 'properties', 'citations', 'items', 'properties', 'quote'), "
            "enum values must not contain trailing whitespace: 'SECRET_KEY private source'.",
            'param': 'text.format.schema'}
        error.request_id = 'req_' + 'a' * 32
        llm, fake, diagnostic = self.fail(FakeClient(fail=error))
        detail = diagnostic['schema_error_message']
        self.assertIn("'quote'", detail)
        self.assertIn('enum values must not contain trailing whitespace', detail)
        self.assertNotIn('SECRET', json.dumps(diagnostic))
        self.assertNotIn('private source', json.dumps(diagnostic))
        self.assertTrue(diagnostic['schema_error_detail_available'])
        self.assertEqual(diagnostic['api_error_param'], 'text.format.schema')
        self.assertEqual(diagnostic['api_request_id'], error.request_id)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(llm.usage_report()['failed_attempts'], 1)
        self.assertFalse((self.cache / 'grounded_responses.sqlite3').exists())
        # Runtime boundaries sanitize the same diagnostic more than once.
        again = safe_llm_diagnostic(LLMError('ignored', diagnostic=diagnostic))
        self.assertEqual(again, diagnostic)

    def test_wrapped_sdk_body_can_supply_schema_code_and_detail(self):
        error = HTTPFailure(400, None)
        error.body = {'error': {'code': 'invalid_json_schema', 'message':
            "Invalid schema for response_format 'rfp_grounded_answer': "
            "In context=(), 'required' must include every key in properties.",
            'param': 'text.format'}}
        _, _, diagnostic = self.fail(FakeClient(fail=error))
        self.assertEqual(diagnostic['api_error_code'], 'invalid_json_schema')
        self.assertIn("'required' must include every key", diagnostic['schema_error_message'])
        self.assertEqual(diagnostic['api_error_param'], 'text.format')

    def test_old_or_unrecognized_provider_details_remain_explicitly_unavailable(self):
        for body in ({'message': 'SECRET_KEY private source'}, {'message': ['SECRET_KEY']}, None):
            with self.subTest(body=body):
                error = HTTPFailure(400, 'invalid_json_schema')
                error.body = body
                error.request_id = 'req_SECRET_KEY'
                error.param = 'SECRET_KEY'
                _, _, diagnostic = self.fail(FakeClient(fail=error))
                self.assertIs(diagnostic['schema_error_detail_available'], False)
                self.assertNotIn('schema_error_message', diagnostic)
                self.assertNotIn('api_error_param', diagnostic)
                self.assertNotIn('api_request_id', diagnostic)
                self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_schema_message_is_bounded_and_unknown_words_are_redacted(self):
        error = HTTPFailure(400, 'invalid_json_schema')
        error.body = {'message': 'Invalid schema: SECRET_KEY 비밀원문 ' + 'enum values ' * 2000}
        _, _, diagnostic = self.fail(FakeClient(fail=error))
        self.assertLessEqual(len(diagnostic['schema_error_message']), 2400)
        self.assertNotIn('SECRET', json.dumps(diagnostic))
        self.assertNotIn('비밀원문', json.dumps(diagnostic, ensure_ascii=False))

    def test_schema_details_cannot_be_injected_through_other_error_codes(self):
        diagnostic = safe_llm_diagnostic(LLMError('ignored', diagnostic={
            'code': 'api_http_error', 'http_status': 401, 'api_error_code': 'invalid_api_key',
            'schema_error_message': 'Invalid schema: enum values are required',
            'api_error_param': 'text.format.schema', 'schema_error_detail_available': True,
            'api_request_id': 'req_' + 'a' * 32}))
        self.assertNotIn('schema_error_message', diagnostic)
        self.assertNotIn('schema_error_detail_available', diagnostic)
        self.assertNotIn('api_error_param', diagnostic)

    def test_authentication_error_is_controlled(self):
        _, _, diagnostic = self.fail(FakeClient(fail=HTTPFailure(401, 'invalid_api_key')))
        self.assertEqual(diagnostic['code'], 'api_authentication_failed')
        self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_unknown_sdk_messages_codes_and_exception_bodies_are_not_exposed(self):
        for error in (HTTPFailure(400, 'SECRET_KEY'), RuntimeError('SECRET_KEY private source'), LLMError('SECRET_KEY')):
            with self.subTest(error_type=type(error).__name__):
                _, _, diagnostic = self.fail(FakeClient(fail=error))
                self.assertNotIn('SECRET', json.dumps(diagnostic))
                self.assertNotIn('api_error_code', diagnostic)

    def test_unknown_arbitrary_exceptions_and_diagnostic_fields_are_redacted(self):
        for error in (RuntimeError('SECRET_KEY'), LLMError('SECRET_KEY'), LLMError('SECRET_KEY', diagnostic={
                'code': 'SECRET_KEY', 'message': 'SECRET_KEY', 'http_status': True,
                'request_fingerprint': 'SECRET_KEY', 'response_status': 'SECRET_KEY',
                'incomplete_reason': 'SECRET_KEY', 'body': 'SECRET_KEY'})):
            diagnostic = safe_llm_diagnostic(error)
            self.assertEqual(diagnostic['code'], 'unknown_error')
            self.assertEqual(set(diagnostic), {'code', 'message'})
            self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_legacy_sdk_install_and_field_contract_errors_have_specific_labels(self):
        errors = [('OpenAI SDK is not installed in this notebook kernel; install requirements-phase4.txt', 'sdk_missing'),
                  ('Phase 5 field marker/category validation failed', 'field_contract_invalid')]
        for message, code in errors:
            self.assertEqual(safe_llm_diagnostic(LLMError(message))['code'], code)

    def test_inconsistent_statuses_are_rejected_without_repair_or_cache(self):
        cases = []
        payload = deepcopy(PAYLOAD); payload['missing_information'] = ['VAT']; cases.append((payload, 'answered_with_missing'))
        payload = deepcopy(PAYLOAD); payload['status'] = 'partial'; cases.append((payload, 'partial_without_missing'))
        payload = deepcopy(PAYLOAD); payload['status'] = 'insufficient_evidence'; cases.append((payload, 'answer_status_claim_mismatch'))
        for payload, expected in cases:
            with self.subTest(expected=expected):
                _, _, diagnostic = self.fail(FakeClient(payload=payload))
                self.assertEqual(diagnostic['code'], expected)
                self.assertIs(diagnostic['response_received'], True)
                self.assertFalse((self.cache / 'grounded_responses.sqlite3').exists())

    def test_malformed_json_and_uncited_claim_still_fail_closed(self):
        _, _, diagnostic = self.fail(FakeClient(raw='SECRET_KEY is not JSON'))
        self.assertEqual(diagnostic['code'], 'invalid_structured_json')
        payload = deepcopy(PAYLOAD); payload['claims'][0]['citations'] = []
        _, _, diagnostic = self.fail(FakeClient(payload=payload))
        self.assertEqual(diagnostic['code'], 'invalid_answer_structure')
        self.assertNotIn('SECRET', json.dumps(diagnostic))

    def test_source_sha_guard_still_rejects_before_any_call(self):
        llm, fake = self.make()
        chunks = evidence(); chunks[0]['sha256'] = 'b' * 64
        with self.assertRaises(LLMError) as caught:
            self.generate(llm, chunks)
        self.assertEqual(safe_llm_diagnostic(caught.exception)['code'], 'source_not_approved')
        self.assertEqual(fake.calls, [])
        self.assertEqual(llm.usage_report()['api_calls'], 0)


if __name__ == '__main__':
    unittest.main()
