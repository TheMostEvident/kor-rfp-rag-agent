"""Source-backed omission hints over processed input, without generation.

Quote provenance does not establish completeness. These limited surface rules
find source lines absent from the corresponding field's text/quotes. They do
not establish semantic error, resolve conflicts, or fill missing answer fields.
"""
from __future__ import annotations

import hashlib
import html
import json
import re

from .fields import FIELD_BY_ID, parse_field_claim


RULES = (
    ('unit_price_settlement', 'budget_vat', r'단가\s*계약',
     ('단가계약',), '단가계약과 실제 대상·물량에 따른 정산 조건을 확인하세요.'),
    ('post_period_support', 'duration', r'(?:과업|용역|사업|계약)\s*기간\s*(?:이후|종료\s*후)',
     ('기간이후', '기간종료후'), '계약상 기간이 끝난 뒤에도 남는 지원 의무가 있는지 확인하세요.'),
    ('selection_method', 'evaluation', r'(?:낙찰자\s*선정\s*방식|낙찰자\s*결정\s*방법)\s*[:：]\s*\S',
     ('선정방식', '결정방법', '적격심사', '협상에의한계약'),
     '선정방식과 세부 평가항목·배점을 구분하세요. 선정방식만으로 배점표를 추정하지 않습니다.'),
)


def _compact(value):
    return ''.join(str(value).split())


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def audit_summary_sources(index, registry, result):
    """Revalidate original claims, then inspect only truly processed source chunks.

    ``fields`` is a display projection and is never trusted as the source of
    claims or facts. Original generated claims and quotes remain unchanged.
    """
    did = result.get('doc_id')
    canonical = list(index.resolve_doc_ids([did]))
    if len(canonical) != 1 or canonical[0] != did:
        raise ValueError('대표 문서 ID 한 개의 요약 결과가 필요합니다.')
    full = list(registry.validate_chunks(index.get_document_chunks(did), allowed_doc_ids=[did]))
    by_id = {chunk['chunk_id']: chunk for chunk in full}
    full_ids = [chunk['chunk_id'] for chunk in full]
    declared = result.get('source_chunk_ids') or []
    processed = result.get('processed_chunk_ids') or []
    if (declared != full_ids or len(processed) != len(set(processed))
            or any(cid not in by_id for cid in processed)
            or [cid for cid in full_ids if cid in set(processed)] != processed):
        raise ValueError('요약의 원문 청크 범위·순서 또는 처리 목록이 현재 문서와 다릅니다.')
    chunks = [by_id[cid] for cid in processed]
    claims = result.get('claims') or []
    by_field = {key: [] for key in FIELD_BY_ID}
    payload_claims = []
    for claim in claims:
        parsed = parse_field_claim(claim)
        by_field[parsed['field_id']].append(claim)
        payload_claims.append({key: claim[key] for key in ('doc_id', 'category', 'text')} | {
            'citations': [{key: cite[key] for key in ('chunk_id', 'quote')}
                          for cite in claim.get('citations') or []]})
    validation = None
    if payload_claims:
        payload = {'status': result.get('status'), 'claims': payload_claims,
                   'missing_information': result.get('missing_information') or []}
        validation = registry.validate_answer(payload, chunks, allowed_doc_ids=[did])
        if not validation['valid']:
            raise ValueError('저장된 요약의 원문 인용을 다시 검증하지 못했습니다.')
    elif processed:
        # A completed empty-evidence map can be audited, but it cannot receive
        # a provenance-valid answer verdict without any claims.
        validation = {'valid': False, 'claims': []}
    candidates, seen = [], set()
    for rule_id, field_id, expression, markers, explanation in RULES:
        related = by_field[field_id]
        field_text = _compact('\n'.join(c['text'] + '\n' + '\n'.join(
            cite['quote'] for cite in c.get('citations') or []) for c in related))
        # Surface presence only. A paraphrase may still be flagged; never turn
        # this decision into a semantic-error label or an automatic correction.
        if any(marker in field_text for marker in markers):
            continue
        for chunk in chunks:
            for quote in chunk['text'].splitlines():
                if not quote.strip() or not re.search(expression, quote):
                    continue
                key = rule_id, did, quote
                if key in seen:
                    continue
                seen.add(key)
                candidate_claim = {'doc_id': did, 'category': FIELD_BY_ID[field_id]['category'],
                    'text': quote, 'citations': [{'chunk_id': chunk['chunk_id'], 'quote': quote}]}
                verified = registry.validate_answer({'status': 'answered', 'claims': [candidate_claim],
                    'missing_information': []}, [chunk], allowed_doc_ids=[did])
                if not verified['valid']:
                    raise ValueError('원문 추가 확인 후보의 인용 검증에 실패했습니다.')
                candidates.append({'rule_id': rule_id, 'field_id': field_id,
                    'label': FIELD_BY_ID[field_id]['label'], 'source_quote': quote,
                    'citations': verified['claims'][0]['citations'],
                    'explanation': explanation, 'requires_review': True,
                    'automatic_semantic_error': False, 'automatically_added_to_answer': False})
    return {'version': 'phase5-source-audit-v1', 'doc_id': did,
            'source_sha256': registry.source_hashes[did],
            'input_result_fingerprint': _digest({'doc_id': did, 'claims': claims,
                'source_chunk_ids': declared, 'processed_chunk_ids': processed}),
            'processed_chunk_count': len(chunks), 'document_chunk_count': len(full),
            'all_input_chunks_processed': processed == full_ids,
            'original_claim_count': len(claims),
            'original_citation_count': sum(len(c.get('citations') or []) for c in claims),
            'original_quotes_revalidated': bool(validation and validation['valid']),
            'candidate_count': len(candidates), 'candidates': candidates,
            'semantic_accuracy': None, 'semantic_completeness_verified': False,
            'human_review_completed': False, 'api_calls': 0,
            'scope': '이미 처리된 원문의 일부 표면 규칙을 대조한 추가 확인 후보. 모든 누락 탐지나 의미적 정답 판정은 아닙니다.'}


def render_source_audit(audit):
    def escape(value):
        text = html.escape(str(value), quote=False)
        text = re.sub(r'([\\`*_{}\[\]()#!+\-.])', r'\\\1', text)
        return text.replace('|', '&#124;').replace('\n', ' / ')
    lines = ['원문 추가 확인', '',
        '이미 전달된 원문에서 찾은 내용입니다. 기존 생성 답변에 자동으로 추가하지 않았습니다.', '',
        '| 항목 | 원문에서 찾은 내용 | 확인할 점 |', '| --- | --- | --- |']
    for candidate in audit.get('candidates') or []:
        lines.append('| ' + ' | '.join(escape(candidate[key]) for key in
                     ('label', 'source_quote', 'explanation')) + ' |')
    if not audit.get('candidates'):
        lines.append('| 추가 후보 없음 | 지정된 규칙에서 발견하지 못함 | 요약 완전성을 뜻하지 않음 |')
    if audit.get('candidates'):
        lines.extend(['', '원문 위치:', ''])
        for item in audit['candidates']:
            c = item['citations'][0]
            locations = ' / '.join(dict.fromkeys(span.get('locator', '') for span in c.get('source_ranges') or []))
            lines.append('- ' + escape(item['label']) + ': ' + escape(c.get('file_name', '')) + ' · ' + escape(locations or c.get('locator', '')))
    lines.extend(['', '추가 API 호출: 0회. 자동 의미 검증·사람 검토 완료 판정은 하지 않았습니다.'])
    return '\n'.join(lines)
