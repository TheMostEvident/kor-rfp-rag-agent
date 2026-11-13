"""Exercise real-source Phase 5 wiring with an explicitly injected SDK double.

No model quality is measured; all fake outputs live in temporary directories.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rfp_phase5 import runtime
from rfp_phase5.review import assess_review, build_review
from rfp_phase5.quality import audit_result


class OfflineResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **body):
        self.calls += 1
        request = json.loads(body['input'][1]['content'])
        chunk = request['evidence'][0]
        quote = chunk['text'][:120]
        field = request['task'].startswith('phase5_')
        text = '[scope_requirements|fact] ' + quote if field else quote
        payload = {'status': 'answered', 'claims': [{
            'doc_id': chunk['doc_id'], 'category': 'requirements' if field else 'other',
            'text': text, 'citations': [{'chunk_id': chunk['chunk_id'], 'quote': quote}]}],
            'missing_information': []}
        return SimpleNamespace(status='completed', output=[],
            output_text=json.dumps(payload, ensure_ascii=False),
            usage=SimpleNamespace(input_tokens=100, output_tokens=70, total_tokens=170,
                                 input_tokens_details=SimpleNamespace(cached_tokens=0)))


def main():
    notebook_path = ROOT / 'notebooks/08_younggwan_phase5_grounded_summary.ipynb'
    notebook = json.loads(notebook_path.read_text())
    responses = OfflineResponses()
    fake = SimpleNamespace(responses=responses)
    original_make, original_save = runtime.make_components, runtime.save_run
    from importlib import metadata
    original_version = metadata.version
    saved, results = [], []
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        def make(env, **kwargs):
            return original_make(env, **kwargs, client=fake, cache_dir=temporary / 'cache')
        def save(root, result, **kwargs):
            folder = original_save(temporary, result, **kwargs)
            saved.append(folder)
            results.append(result)
            return folder
        def sdk_version(name):
            return '3.49.0-offline-double' if name == 'openai' else original_version(name)
        namespace = {'__name__': '__phase5_offline_live_branch_test__'}
        out = io.StringIO()
        with (patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-sdk-double-only'}),
              patch.object(runtime, 'make_components', make),
              patch.object(runtime, 'save_run', save),
              patch.object(metadata, 'version', sdk_version),
              contextlib.redirect_stdout(out)):
            previous = Path.cwd()
            os.chdir(ROOT)
            try:
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(cell['source'])
                    source = source.replace('ENABLE_OPENAI = False', 'ENABLE_OPENAI = True')
                    source = source.replace('RUN_LIVE_QA = False', 'RUN_LIVE_QA = True')
                    source = source.replace('RUN_LIVE_SUMMARY = False', 'RUN_LIVE_SUMMARY = True')
                    exec(compile(source, '<phase5-notebook-mock-live-branch>', 'exec'), namespace)
            finally:
                os.chdir(previous)
                if namespace.get('app', {}).get('index'):
                    namespace['app']['index'].close()
        assert len(saved) == 2, len(saved)
        qa, summary = results
        assert qa['execution_status'] == 'completed'
        assert summary['execution_status'] == 'completed', summary
        assert len(summary['fields']) == 11
        assert summary['validated_chunk_coverage'] == 1.0
        assert summary['semantic_quality_verified'] is False
        assert summary['status'] == 'partial'  # fake never supplies all 11 fields
        checks = []
        for result, folder in zip(results, saved):
            assert (folder / 'human_review.json').is_file()
            review = build_review(result)
            assessed = assess_review(result, review)
            assert assessed['live_readiness'] is False, assessed
            checks.append({'execution_status': result['execution_status'],
                           'answer_status': result['status'],
                           'source_validation': result.get('valid', result.get('validation', {}).get('valid')),
                           'human_review': assessed,
                           'quality': audit_result(result)})
        usage = namespace['components']['llm'].usage_report()
        assert usage['live_api_calls'] == 0, usage
        assert usage['mock_calls'] >= 2, usage
        report = {'scope': 'Actual original sources; notebook opt-in branches executed with an injected SDK double only.',
                  'live_api_calls': 0, 'mock_calls': usage['mock_calls'],
                  'notebook_live_branches_saved': len(saved), 'summary_fields': len(summary['fields']),
                  'summary_chunks': len(summary['source_chunk_ids']),
                  'summary_validated_input_coverage': summary['validated_chunk_coverage'],
                  'semantic_quality_measured': False, 'human_approval_recorded': False,
                  'quality_checks': checks}
        (ROOT / 'phase5_reports/mock_integration.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'quality_checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
