"""Fingerprint-bound human review; no automatic semantic pass or model judge."""
from __future__ import annotations

import copy
import hashlib
import json

_VERSION = 'phase5-human-review-v1'
_JUDGEMENTS = ('supported', 'complete', 'conditions_preserved')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _snapshot(result):
    """Bind answer, quoted source versions, coverage and execution origin.

    Timestamps, paths and token/cost counters do not change a review's identity.
    Source text and source ranges do. Both field view and raw validated claims
    are included, so editing the displayed structured answer invalidates review.
    """
    keys = ('status', 'execution_status', 'answer_generated', 'valid', 'claims',
            'fields', 'missing_information', 'source_chunk_ids', 'processed_chunk_ids',
            'input_chunk_coverage', 'validated_chunk_coverage', 'doc_id', 'active_doc_ids')
    value = {key: result.get(key) for key in keys}
    validation = result.get('validation')
    if isinstance(validation, dict):
        value['validation'] = {key: validation.get(key) for key in
                               ('valid', 'status', 'claims', 'missing_information')}
    value['source_chunks'] = [{key: chunk.get(key) for key in
                              ('chunk_id', 'doc_id', 'sha256', 'text', 'source_spans')}
                             for chunk in result.get('source_chunks', [])]
    generation = result.get('generation') or {}
    value['generation_origin'] = {key: generation.get(key) for key in ('execution_mode', 'live_api')}
    value['batch_origin'] = [{key: batch.get(key) for key in
                             ('batch_id', 'chunk_ids', 'valid', 'status', 'execution_mode', 'live_api')}
                            for batch in result.get('batch_outcomes', [])]
    return value


def answer_fingerprint(result):
    if not isinstance(result, dict):
        raise TypeError('result는 실행 결과 dict여야 합니다.')
    return hashlib.sha256(_json(_snapshot(result)).encode('utf-8')).hexdigest()


def _claims(result):
    validation = result.get('validation')
    return list((validation.get('claims') if isinstance(validation, dict)
                 else result.get('claims')) or [])


def _rows(result):
    fields = result.get('fields')
    if isinstance(fields, list) and fields:
        rows = [{'row_id': str(field.get('field_id', '')), 'label': field.get('label'),
                 'state': field.get('state'), 'claims': copy.deepcopy(field.get('claims') or [])}
                for field in fields]
    else:
        rows = [{'row_id': f'claim-{index:04d}', 'label': claim.get('category'),
                 'state': 'claim', 'claims': [copy.deepcopy(claim)]}
                for index, claim in enumerate(_claims(result), 1)]
    if not rows:
        rows = [{'row_id': 'answer-status', 'label': '답변 상태·근거 부족 설명',
                 'state': result.get('status'), 'claims': []}]
    identifiers = [row['row_id'] for row in rows]
    if any(not row for row in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError('검토 항목 ID가 비어 있거나 중복됐습니다.')
    return rows


def build_review(result, *, case_id='manual'):
    """Create an unfilled review for this exact result and its source versions."""
    if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 200:
        raise ValueError('case_id는 1~200자 문자열이어야 합니다.')
    rows = _rows(result)
    for row in rows:
        row.update({key: None for key in _JUDGEMENTS})
        row['notes'] = ''
    return {'version': _VERSION, 'case_id': case_id,
            'answer_fingerprint': answer_fingerprint(result),
            'document_ids': sorted({claim.get('doc_id') for claim in _claims(result)
                                    if isinstance(claim.get('doc_id'), str)}
                                   | ({result['doc_id']} if isinstance(result.get('doc_id'), str) else set())),
            'reviewer': '', 'rows': rows,
            'instructions': '각 항목의 주장과 인용을 대조한 뒤 supported, complete, conditions_preserved를 true/false로 기록하세요. 미확인·공고문 참조 항목도 상태가 적절한지 검토하세요. null은 미검토입니다.',
            'automatic_semantic_pass': False, 'api_calls': 0}


def _live_origin(result):
    generation = result.get('generation') or {}
    if generation:
        return generation.get('execution_mode') == 'live_openai' and generation.get('live_api') is True
    batches = result.get('batch_outcomes') or []
    return bool(batches) and all(batch.get('valid') is True
        and batch.get('execution_mode') == 'live_openai' and batch.get('live_api') is True
        for batch in batches)


def _readiness_blockers(result):
    blockers = []
    if result.get('execution_status') != 'completed':
        blockers.append('execution_not_completed')
    if result.get('status') != 'answered':
        blockers.append('answer_not_fully_answered')
    if result.get('answer_generated') is not True:
        blockers.append('answer_not_generated')
    if not _live_origin(result):
        blockers.append('live_model_origin_not_confirmed')
    validation = result.get('validation')
    valid = validation.get('valid') if isinstance(validation, dict) else result.get('valid')
    claims = _claims(result)
    if (valid is not True or not claims or any(not claim.get('citations') for claim in claims)
        or any(citation.get('provenance_verified') is not True
               for claim in claims for citation in claim.get('citations', []))):
        blockers.append('source_provenance_not_confirmed')
    if 'source_chunk_ids' in result:
        expected, processed = result.get('source_chunk_ids') or [], result.get('processed_chunk_ids') or []
        if (not expected or len(expected) != len(set(expected)) or expected != processed
            or result.get('validated_chunk_coverage') != 1.0):
            blockers.append('full_input_not_validated')
    if any(field.get('state') in {'not_processed', 'conflict_candidates'}
           for field in result.get('fields') or []):
        blockers.append('unprocessed_or_conflicting_fields')
    return blockers


def assess_review(result, review):
    """Assess this review only; never promotes one case to system-wide quality.

    A complete positive mock review can be recorded but is never live-ready.
    Exact matching of row contents prevents retaining `true` while swapping the
    evidence printed in the human-review file.
    """
    if not isinstance(result, dict) or not isinstance(review, dict):
        raise TypeError('실행 결과와 검토 자료는 dict여야 합니다.')
    errors = []
    try:
        template = build_review(result, case_id=review.get('case_id', 'manual'))
    except (ValueError, TypeError, KeyError):
        return {'status': 'invalid_review', 'live_readiness': False,
                'errors': ['invalid_result_or_case_id'], 'case_only': True,
                'system_quality_verified': False, 'semantic_accuracy': None}
    if review.get('version') != _VERSION:
        errors.append('review_version_mismatch')
    if review.get('answer_fingerprint') != template['answer_fingerprint']:
        errors.append('stale_answer_or_source_fingerprint')
    rows = review.get('rows')
    if not isinstance(rows, list) or len(rows) != len(template['rows']):
        errors.append('review_rows_mismatch')
    else:
        for actual, expected in zip(rows, template['rows']):
            if not isinstance(actual, dict) or any(actual.get(key) != expected.get(key)
                        for key in ('row_id', 'label', 'state', 'claims')):
                errors.append('review_row_content_mismatch'); break
            if any(actual.get(key) is not None and type(actual.get(key)) is not bool for key in _JUDGEMENTS):
                errors.append('judgement_must_be_boolean_or_null'); break
            if not isinstance(actual.get('notes', ''), str):
                errors.append('review_notes_must_be_text'); break
    blockers = _readiness_blockers(result)
    if errors:
        return {'status': 'invalid_review', 'live_readiness': False,
                'errors': list(dict.fromkeys(errors)), 'readiness_blockers': blockers,
                'case_only': True, 'system_quality_verified': False, 'semantic_accuracy': None}
    pending = [(row['row_id'], key) for row in rows for key in _JUDGEMENTS if row.get(key) is None]
    rejected = [(row['row_id'], key) for row in rows for key in _JUDGEMENTS if row.get(key) is False]
    if rejected:
        status = 'review_failed'
    elif pending:
        status = 'review_pending'
    elif not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        status = 'review_pending'; blockers.append('reviewer_not_recorded')
    else:
        status = 'passed_human_review'
    return {'status': status, 'live_readiness': status == 'passed_human_review' and not blockers,
            'errors': [], 'pending_judgements': [{'row_id': row, 'judgement': key} for row, key in pending],
            'failed_judgements': [{'row_id': row, 'judgement': key} for row, key in rejected],
            'readiness_blockers': blockers, 'case_only': True,
            'system_quality_verified': False, 'semantic_accuracy': None,
            'scope': '이 실행 결과에 대한 사람 검토 기록입니다. 전체 RAG 성능이나 모든 문서의 정확성을 확정하지 않습니다.'}
