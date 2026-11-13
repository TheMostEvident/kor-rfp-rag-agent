"""Inspect saved Phase 5 failures without an SDK, key, network or model call.

Old exceptions were not persisted and cannot be reconstructed from usage alone.
This module describes available evidence and labels inferences as inferences.
"""
from __future__ import annotations

import json
from pathlib import Path


def inspect_saved_failure(root, result_path=None):
    root = Path(root).expanduser().resolve()
    if result_path is None:
        paths = sorted((root / 'phase5_outputs/runs').glob('*/result.json'),
                       key=lambda p: (p.stat().st_mtime_ns, str(p)))
        if not paths:
            return {'status': 'no_saved_result', 'api_calls_by_diagnostic': 0,
                    'message': 'phase5_outputs/runs 아래에 저장된 result.json이 없습니다.'}
        path = paths[-1]
    else:
        path = Path(result_path).expanduser()
        path = path if path.is_absolute() else root / path
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('result.json은 실행 결과 객체여야 합니다.')
    usage = result.get('usage') or {}
    session = result.get('llm_session_usage') or {}
    invocation_keys = ('api_calls', 'live_api_calls', 'mock_calls', 'input_tokens',
                       'output_tokens', 'total_tokens', 'cached_input_tokens', 'usage_unknown_attempts')
    session_keys = ('api_calls', 'live_api_calls', 'mock_calls', 'cache_hits',
                    'failed_attempts', 'reported_input_tokens', 'reported_output_tokens',
                    'reported_total_tokens', 'usage_unknown_attempts', 'reserved_max_output_tokens')
    batch_keys = ('batch_id', 'status', 'reason', 'valid', 'api_called', 'live_api',
                  'cache_hit', 'execution_mode', 'error_detail', 'error_diagnostic')
    # Do not print raw claims, prompts, API bodies, source text, or environment.
    batches = [{k: batch[k] for k in batch_keys if k in batch}
               for batch in result.get('batch_outcomes') or []]
    contracts = [{'reason': item.get('reason'), 'chunk_count': len(item.get('chunk_ids') or [])}
                 for item in result.get('field_contract_failures') or []]
    recorded = result.get('error_diagnostics') or [b['error_diagnostic'] for b in batches if b.get('error_diagnostic')]
    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    if result.get('execution_status') == 'completed':
        inference = 'execution_completed'
        explanation = '선택한 결과는 실행이 완료된 기록입니다. 의미적 정답 판정은 별도입니다.'
    elif recorded:
        inference = 'controlled_error_recorded'
        explanation = '오류 코드와 안전한 설명이 기록되어 있습니다. 아래 진단을 확인하세요.'
    elif contracts:
        inference = 'field_contract_rejected'
        explanation = '항목 표지 또는 category 검사에서 거절된 응답 기록이 있습니다.'
    elif any(positive(usage.get(k)) for k in ('input_tokens', 'output_tokens', 'total_tokens')):
        inference = 'response_usage_received_before_failure'
        explanation = ('이 실행의 사용량을 받은 뒤 응답 수락 과정에서 실패한 정황입니다. '
                       '출력 중단·거절·JSON·상태 검증 중 무엇인지는 과거 기록만으로 확정할 수 없습니다.')
    elif positive(usage.get('usage_unknown_attempts')):
        inference = 'failed_request_or_unreported_usage'
        explanation = ('요청 실패 또는 사용량이 제공되지 않은 응답이 있을 수 있습니다. '
                       '잔액·네트워크·HTTP 오류 중 하나로 단정할 수 없습니다.')
    else:
        inference = 'original_error_detail_unavailable'
        explanation = '기존 기록에 원래 오류 설명이 없어 정확한 원인을 복원할 수 없습니다.'
    return {'result_path': str(path.resolve()), 'status': result.get('status'),
            'execution_status': result.get('execution_status'),
            'answer_generated': result.get('answer_generated'),
            'input_chunk_coverage': result.get('input_chunk_coverage'),
            'validated_chunk_coverage': result.get('validated_chunk_coverage'),
            'errors': result.get('errors') or [],
            'invocation_usage': {k: usage[k] for k in invocation_keys if k in usage},
            'session_usage': {k: session[k] for k in session_keys if k in session},
            'batch_outcomes': batches, 'field_contract_failures': contracts,
            'recorded_error_diagnostics': recorded,
            'diagnostic_classification': inference, 'explanation': explanation,
            'original_error_detail_available': bool(recorded),
            'root_cause_confirmed': False,
            'past_exception_reconstructed': False,
            'api_calls_by_diagnostic': 0, 'result_modified': False,
            'retry_executed': False,
            'next_step': '이 진단 출력을 공유해 주세요. 추가 API 요청은 실행하지 않습니다.'}
