"""Read-only replay of failed summary batches. No generate, SDK, or API calls."""
from __future__ import annotations

import json
from pathlib import Path


def revalidate_cached_batches(env, result_path, component, plan):
    from rfp_phase4.citation_recovery import validate_with_quote_recovery
    from rfp_phase5.fields import FIELD_QUESTION, FIELD_SCHEMA, parse_field_claim
    from rfp_phase5.summary import TASK

    path = Path(result_path).expanduser().resolve()
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('doc_id') != plan['doc_id'] or saved.get('plan_fingerprint') != plan['plan_fingerprint']:
        raise ValueError('기존 결과의 문서·질문·요약 계획이 현재 요청과 다릅니다.')
    if saved.get('source_chunk_ids') != plan['source_chunk_ids']:
        raise ValueError('기존 결과와 현재 원문의 전체 청크 범위가 다릅니다.')
    by_id = {b['batch_id']: b for b in plan['batches']}
    outcomes = []
    llm = component['llm']
    before = llm.usage_report()
    for old in saved.get('batch_outcomes', []):
        if old.get('reason') != 'source_citation_validation_failed':
            continue
        batch = by_id.get(old.get('batch_id'))
        if batch is None or old.get('chunk_ids') != [c['chunk_id'] for c in batch['chunks']]:
            raise ValueError('기존 실패 배치의 청크·순서가 현재 계획과 다릅니다.')
        prepared = llm._prepare(TASK, FIELD_QUESTION, batch['chunks'], schema=FIELD_SCHEMA)
        if old.get('request_fingerprint') != prepared['fingerprint']:
            raise ValueError('모델·설정·질문·원문이 기존 API 요청과 다릅니다. 재사용하지 않았습니다.')
        response = llm._cache_read(prepared)  # Read-only checksum-bound lookup; no _guard or SDK.
        item = {'batch_id': batch['batch_id'], 'chunk_count': len(batch['chunks']),
            'original_request_fingerprint': prepared['fingerprint'],
            'original_api_usage': old.get('usage', {}), 'additional_api_calls': 0}
        if response is None:
            item.update(status='cache_not_available', valid=False,
                        explanation='원래 응답 캐시를 찾지 못했습니다. 진단용 인용만으로 답변을 재구성하지 않았습니다.')
        else:
            if (response.get('request_fingerprint') != prepared['fingerprint']
                or response.get('execution_mode') != 'live_openai' or response.get('live_api') is not True):
                raise ValueError('기존 실제 모델 응답의 실행 구분 또는 요청 지문을 확인하지 못했습니다.')
            payload = response['payload']
            for claim in payload.get('claims', []):
                parse_field_claim(claim)
            checked = validate_with_quote_recovery(env['registry'], payload, batch['chunks'],
                                                   allowed_doc_ids=[plan['doc_id']])
            item.update(status='cached_batch_revalidated' if checked['valid'] else 'validation_failed',
                valid=checked['valid'], claims_revalidated=len(checked['claims']),
                citation_recovery=checked['citation_recovery'],
                citation_diagnostics=checked['citation_diagnostics'],
                validation_errors=checked.get('errors', []),
                revalidated_claims=checked['claims'] if checked['valid'] else [])
        outcomes.append(item)
    after = llm.usage_report()
    if before != after:
        raise RuntimeError('읽기 전용 재검증에서 사용량 카운터가 변경됐습니다.')
    return {'version': 'phase5-cache-source-revalidation-v1', 'original_result_path': str(path),
        'doc_id': plan['doc_id'], 'document_chunk_count': plan['chunk_count'],
        'batches': outcomes, 'revalidated_batch_count': sum(b['valid'] for b in outcomes),
        'full_document_complete': False, 'answer_semantic_quality_verified': False,
        'additional_api_calls': 0, 'cache_or_original_result_modified': False,
        'scope': '원래 API 응답이 캐시에 있을 때 실패한 배치의 인용만 재검증합니다. 미실행 배치의 결과나 정답 판정을 만들지 않습니다.'}
