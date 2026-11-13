"""Local review hints for claim/quotation surfaces, never a truth grader.

Only a claim's own quotations are compared. Literal matches can still be
irrelevant, and an arithmetic conversion can be correct while being flagged.
These hints deliberately do not infer semantic entailment or legal eligibility.
"""
from __future__ import annotations

import re

_NUMBERS = re.compile(r'(?<![\d,])\d[\d,]*(?:\.\d+)?')
_MEASURES = re.compile(
    r'(?<![\d,])(\d[\d,]*(?:\.\d+)?)\s*'
    r'(억\s*원|천만\s*원|백만\s*원|만\s*원|천\s*원|원|개\s*월|년|일|월|시간|%)')
_PREFIX = re.compile(r'^\[[a-z_]+\|(?:fact|notice_reference|conflict_candidate)\]\s*')
_VAT = re.compile(r'부가\s*(?:가치\s*)?세|\bVAT\b', re.IGNORECASE)
# Each group names a surface detail worth preserving, rather than its meaning.
_CONDITIONS = (
    ('contract_start', re.compile(r'계약\s*(?:체결\s*)?일(?:로부터|부터)?')),
    ('stabilization', re.compile(r'안정화')),
    ('monitoring', re.compile(r'(?:시스템\s*)?모니터링')),
    ('delay_condition', re.compile(r'지연')),
    ('scope_adjustment', re.compile(r'과업\s*조정')),
    ('staged_opening', re.compile(r'단계적\s*(?:오픈|개방|개통)')),
    ('upper_limit', re.compile(r'이내|한도')),
    ('negotiation_condition', re.compile(r'협의')),
    ('notice_reference', re.compile(r'(?:입찰\s*)?공고문?\s*(?:참조|참고|에\s*따름)')),
)
_FORM = re.compile(r'구분금액비고|제안금액금원|가격제안서|가격산출세부내역|하도급율|하수급사업명|금액단위[:：]?원')


def _compact(value):
    return ''.join(str(value).split())


def _claims(result):
    validation = result.get('validation')
    if isinstance(validation, dict):
        return list(validation.get('claims') or [])
    return list(result.get('claims') or [])


def _vat_states(text):
    compact = _compact(text).upper()
    states = set()
    for match in re.finditer(r'(?:부가가치세|부가세|VAT)(?:[()（）:：]|는|를|가|을|의)*'
                             r'(불포함|미포함|별도|제외|포함|면세)', compact):
        raw = match.group(1)
        states.add('excluded' if raw in {'불포함', '미포함', '별도', '제외'} else raw)
    return states


def audit_result(result):
    """Return diagnostic flags. Zero flags does not establish answer accuracy.

    ``numeric_surface_not_in_claim_quotes`` means that exact numeral spelling
    was not found, not that an answer is necessarily wrong. Currency/duration
    pairs compare the numeral and unit together, removing whitespace only.
    """
    if not isinstance(result, dict):
        raise TypeError('result는 실행 결과 dict여야 합니다.')
    claims = _claims(result)
    flags = []

    def add(code, claim_index=None, **detail):
        flags.append({'code': code, 'claim_index': claim_index,
                      'requires_human_review': True, **detail})

    validation = result.get('validation')
    valid = validation.get('valid') if isinstance(validation, dict) else result.get('valid')
    quotes_verified = bool(claims) and all(claim.get('citations') for claim in claims) and all(
        isinstance(c, dict) and c.get('provenance_verified') is True
        for claim in claims for c in claim.get('citations', []))
    if valid is not True or not quotes_verified:
        add('citation_provenance_not_confirmed',
            explanation='이 진단기는 원문 레지스트리를 다시 검증하지 않습니다. 출처 검사 결과를 확인하세요.')
    for index, claim in enumerate(claims, 1):
        text = _PREFIX.sub('', str(claim.get('text', '')))
        citations = list(claim.get('citations') or [])
        quote = '\n'.join(str(c.get('quote', '')) for c in citations)
        if not citations:
            add('claim_has_no_quote', index)
            continue
        claim_numbers = set(_NUMBERS.findall(text))
        quote_numbers = set(_NUMBERS.findall(quote))
        unsupported = sorted(claim_numbers - quote_numbers)
        if unsupported:
            add('numeric_surface_not_in_claim_quotes', index, tokens=unsupported,
                explanation='숫자 표기·환산을 사람이 대조해야 합니다. 다른 주장의 인용은 사용하지 않았습니다.')
        quote_measures = {(number, _compact(unit)) for number, unit in _MEASURES.findall(quote)}
        unmatched = sorted({(number, _compact(unit)) for number, unit in _MEASURES.findall(text)} - quote_measures)
        if unmatched:
            add('number_unit_pair_not_in_claim_quotes', index,
                pairs=[{'number': number, 'unit': unit} for number, unit in unmatched],
                explanation='같은 숫자라도 금액·기간의 단위가 다른지 확인하세요. 자동 환산하지 않습니다.')
        if _VAT.search(quote) and not _VAT.search(text):
            add('vat_detail_omitted_from_claim', index)
        quote_vat, claim_vat = _vat_states(quote), _vat_states(text)
        if claim_vat and (not quote_vat or not claim_vat.issubset(quote_vat)):
            add('vat_state_not_matched_in_claim_quotes', index,
                claim_states=sorted(claim_vat), quoted_states=sorted(quote_vat))
        for name, pattern in _CONDITIONS:
            if pattern.search(quote) and not pattern.search(text):
                add('quoted_condition_not_repeated_in_claim', index, condition=name,
                    explanation='인용에 있는 조건의 생략 여부를 검토하세요. 이 표면 검사만으로 오류를 확정하지 않습니다.')
        for citation_index, citation in enumerate(citations, 1):
            cited = str(citation.get('quote', ''))
            if _FORM.search(_compact(cited)):
                add('possible_blank_form_or_weak_quote', index, citation_index=citation_index,
                    chunk_id=citation.get('chunk_id'),
                    explanation='양식·산출 내역 표의 인용일 수 있습니다. 실제 사업 조건을 뒷받침하는지 검토하세요.')
    counts = {}
    for flag in flags:
        counts[flag['code']] = counts.get(flag['code'], 0) + 1
    return {'version': 'phase5-surface-audit-v1', 'claim_count': len(claims),
            'flag_count': len(flags), 'flag_counts': counts, 'flags': flags,
            'source_validation_reported': valid is True and quotes_verified,
            'semantic_accuracy': None, 'semantic_entailment_verified': False,
            'automatic_answer_correctness': None, 'api_calls': 0,
            'scope': '주장별 인용의 숫자·단위·VAT·조건 표면 비교. 인용의 관련성·정답성·요약 완전성은 사람 검토가 필요합니다.',
            'zero_flags_means_semantically_correct': False}
