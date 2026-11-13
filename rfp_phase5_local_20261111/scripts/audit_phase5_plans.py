"""Audit all canonical summary *input plans*, with no SDK or API execution.

This deliberately does not load evaluation questions or gold labels. All
budgets below are uncached upper bounds for a fresh client, not spending caps.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfp_phase4.llm import LLMConfig, OpenAIGroundedLLM
from rfp_phase5.runtime import bootstrap_phase5
from rfp_phase5.summary import FieldSummarizer
from rfp_phase5.fields import FIELD_QUESTION, FIELD_SCHEMA


S002_DOC = '1Eq8gIdUnnLRbbEBiaU29wYji2YglNNuu'


def sha_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def main():
    started = time.perf_counter()
    prompt_file = ROOT / 'rfp_phase5/fields.py'
    before = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    old_key = os.environ.pop('OPENAI_API_KEY', None)
    env = bootstrap_phase5(ROOT)
    index = env['index']
    try:
        config = env['phase5_config']
        llm = OpenAIGroundedLLM(ROOT / 'phase5_outputs/unused_offline_plan_cache',
            config=LLMConfig(**config['generation']),
            allowed_sources=env['registry'].source_hashes)
        assert llm.config.enabled is False
        summarizer = FieldSummarizer(index, llm, env['registry'], config['summary'])
        catalog = index.get_catalog()
        rows = []
        for position, document in enumerate(catalog, 1):
            doc_id = document['doc_id']
            plan = summarizer.plan(doc_id)
            expected = [c['chunk_id'] for c in index.get_document_chunks(doc_id)]
            actual = [c['chunk_id'] for batch in plan['batches'] for c in batch['chunks']]
            assert plan['source_chunk_ids'] == expected == actual, doc_id
            assert len(actual) == len(set(actual)), doc_id
            sizes = [batch['request_bytes'] for batch in plan['batches']]
            oversized = [size for size in sizes if size > plan['max_batch_bytes']]
            assert not oversized or 'one_or_more_source_chunks_exceed_request_byte_budget' in plan['block_reasons']
            budget_reasons = []
            if plan['map_calls'] > llm.config.max_calls:
                budget_reasons.append('fresh_session_max_calls_exceeded')
            if plan['total_request_bytes'] > llm.config.max_total_input_bytes:
                budget_reasons.append('fresh_session_max_total_input_bytes_exceeded')
            rows.append({
                'doc_id': doc_id, 'file_name': document['file_name'],
                'sha256': document['sha256'],
                'chunk_count': plan['chunk_count'], 'map_calls': plan['map_calls'],
                'total_request_bytes': plan['total_request_bytes'],
                'max_batch_request_bytes': max(sizes, default=0),
                'oversized_batch_count': len(oversized),
                'all_source_chunks_once_in_source_order': True,
                'plan_can_execute': plan['can_execute'],
                'plan_block_reasons': plan['block_reasons'],
                'fresh_uncached_session_budget_reasons': budget_reasons,
                'fresh_uncached_session_within_caps': not budget_reasons and plan['can_execute'],
                'plan_fingerprint': plan['plan_fingerprint'],
            })
            if position % 10 == 0:
                print(f'Input plans checked: {position}/{len(catalog)}', flush=True)
        usage = llm.usage_report()
        assert usage['api_calls'] == usage['live_api_calls'] == usage['mock_calls'] == 0
        after = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError('fields.py changed during audit; rerun against the final prompt/schema.')
        report = {
            'scope': 'Offline input-plan completeness and fresh-session uncached byte/call limits only; not semantic summary quality.',
            'canonical_document_count': len(rows),
            'total_indexed_chunks': sum(row['chunk_count'] for row in rows),
            'all_source_chunks_once_in_source_order': True,
            'all_chunk_sizes_under_batch_limit': all(row['oversized_batch_count'] == 0 for row in rows),
            'maximum_map_calls': max(row['map_calls'] for row in rows),
            'maximum_total_request_bytes': max(row['total_request_bytes'] for row in rows),
            'plan_blocked_document_count': sum(not row['plan_can_execute'] for row in rows),
            'fresh_uncached_session_blocked_document_count': sum(not row['fresh_uncached_session_within_caps'] for row in rows),
            'plan_block_reason_counts': dict(Counter(reason for row in rows for reason in row['plan_block_reasons'])),
            'fresh_uncached_session_budget_reason_counts': dict(Counter(reason for row in rows for reason in row['fresh_uncached_session_budget_reasons'])),
            'max_map_calls': config['summary']['max_map_calls'],
            'max_batch_bytes': config['summary']['max_batch_bytes'],
            'fresh_session_max_calls': llm.config.max_calls,
            'fresh_session_max_total_input_bytes': llm.config.max_total_input_bytes,
            'selected_short_summary_document': next(row for row in rows if row['doc_id'] == S002_DOC),
            'field_prompt_fingerprint': sha_json(FIELD_QUESTION),
            'field_schema_fingerprint': sha_json(FIELD_SCHEMA),
            'field_module_sha256': before,
            'usage': usage,
            'api_calls': 0, 'sdk_imported': 'openai' in sys.modules,
            'holdout_questions_loaded': 0, 'gold_labels_loaded': 0,
            'semantic_quality_verified': False,
            'elapsed_seconds': round(time.perf_counter() - started, 3),
            'documents': rows,
        }
        assert report['sdk_imported'] is False
        destination = ROOT / 'phase5_reports/summary_plan_audit.json'
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in report.items()
                          if key not in {'documents', 'usage'}}, ensure_ascii=False, indent=2))
    finally:
        index.close()
        if old_key is not None:
            os.environ['OPENAI_API_KEY'] = old_key


if __name__ == '__main__':
    main()
