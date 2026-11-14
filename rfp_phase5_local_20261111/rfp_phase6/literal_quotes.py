"""Constrain work-scope citations to exact, chunk-bound source excerpts.

This prevents a model from putting an ellipsis-joined outline in a quotation.
It never repairs such an outline, drops a claim, or weakens EvidenceRegistry.
The same guarded client, counters, cache and request caps remain in use.
"""
from __future__ import annotations

import copy

from rfp_phase4.evidence import ANSWER_SCHEMA
from rfp_phase4.llm import COMPARISON_DOCUMENT_TASK, LLMError


VERSION = 'work-scope-literal-quotes-v1'
QA_TASK = 'qa_work_scope_literal_v1'
MAX_QUOTE_BYTES = 420
MAX_OPTIONS = 256
GUIDE = ('인용 선택 규칙(literal-quotes-v1): citations의 chunk_id와 quote는 응답 스키마에 '
    '주어진 같은 원문 청크의 선택지에서 그대로 선택하세요. 여러 표 제목을 ... 또는 /로 '
    '이어 쓴 요약문을 인용문으로 만들지 마세요. 필요한 근거가 떨어져 있으면 별도 citations로 '
    '선택하세요. 구체적인 대상 업무·기능을 중심으로 핵심 내용을 정리하고 일반 성능·보안·평가 '
    '목록을 사업의 구체적인 대상 업무 대신 나열하지 마세요. 각 주장의 숫자·조건도 선택한 '
    '인용에 포함되어야 하며, 근거가 부족한 항목은 missing_information에 남기세요.')


def literal_options(text):
    """Partition source text into contiguous excerpts without changing a byte.

    Prefer full lines and nearby whitespace boundaries. Every non-whitespace
    character remains covered; this is source segmentation, not extraction of
    facts or inference that an excerpt supports a generated claim.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('인용 선택지를 만들 원문이 없습니다.')
    options = []
    start = 0
    while start < len(text):
        end = start
        used = 0
        while end < len(text) and used + len(text[end].encode('utf-8')) <= MAX_QUOTE_BYTES:
            used += len(text[end].encode('utf-8'))
            end += 1
        if end < len(text):
            minimum = start + (end - start) // 2
            newline = text.rfind('\n', minimum, end)
            if newline >= minimum:
                end = newline + 1
            else:
                spaces = [i for i in range(minimum, end) if text[i].isspace()]
                if spaces:
                    end = spaces[-1] + 1
        quote = text[start:end].strip()
        if quote:
            options.append(quote)
        start = end
    return list(dict.fromkeys(options))


def literal_schema(evidence):
    """Bind each permitted exact excerpt to its actual chunk ID in anyOf."""
    if not evidence:
        raise ValueError('인용 선택지에는 현재 입력 근거가 필요합니다.')
    branches = []
    seen = set()
    count = 0
    for chunk in evidence:
        cid = chunk['chunk_id']
        if cid in seen:
            raise ValueError('중복 청크의 인용 선택지를 만들 수 없습니다.')
        seen.add(cid)
        quotes = literal_options(chunk['text'])
        count += len(quotes)
        if count > MAX_OPTIONS:
            raise ValueError('원문 인용 선택지 수가 상한을 넘었습니다.')
        branches.append({'type': 'object', 'additionalProperties': False,
            'properties': {'chunk_id': {'type': 'string', 'enum': [cid]},
                           'quote': {'type': 'string', 'enum': quotes}},
            'required': ['chunk_id', 'quote']})
    schema = copy.deepcopy(ANSWER_SCHEMA)
    claim = schema['properties']['claims']['items']['properties']
    claim['doc_id'] = {'type': 'string', 'enum': list(dict.fromkeys(c['doc_id'] for c in evidence))}
    claim['citations']['items'] = {'anyOf': branches}
    return schema


def accepts_citations(payload, schema):
    """Check the extra selection contract, including cached/test responses."""
    properties = schema['properties']['claims']['items']['properties']
    docs = set(properties['doc_id']['enum'])
    allowed = {branch['properties']['chunk_id']['enum'][0]: set(branch['properties']['quote']['enum'])
               for branch in properties['citations']['items']['anyOf']}
    try:
        return all(claim['doc_id'] in docs and all(
            cite['chunk_id'] in allowed and cite['quote'] in allowed[cite['chunk_id']]
            for cite in claim['citations']) for claim in payload['claims'])
    except (KeyError, TypeError):
        return False


class LiteralQuoteClient:
    """An adapter around one retained guarded client; no SDK/client creation."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.prepared_requests = 0

    @property
    def config(self):
        return self.delegate.config

    def _request(self, task, question, evidence, schema=None):
        if schema is not None:
            raise ValueError('사업 범위 인용 계약에 다른 응답 스키마를 섞을 수 없습니다.')
        if task not in {'qa', COMPARISON_DOCUMENT_TASK}:
            raise ValueError('지원하지 않는 인용 계약 작업입니다.')
        return {'task': QA_TASK if task == 'qa' else task,
            'question': question + '\n\n' + GUIDE, 'evidence': evidence,
            'schema': literal_schema(evidence)}

    def preflight_requests(self, requests):
        prepared = [self._request(**request) for request in requests]
        self.prepared_requests = len(prepared)
        preflight = getattr(self.delegate, 'preflight_requests', None)
        return preflight(prepared) if callable(preflight) else None

    def generate(self, task, question, evidence, *, schema=None):
        request = self._request(task, question, evidence, schema)
        result = self.delegate.generate(**request)
        if not accepts_citations(result.get('payload'), request['schema']):
            raise LLMError('Model citation is outside the literal selection contract', diagnostic={
                'code': 'invalid_answer_structure', 'request_fingerprint': result.get('request_fingerprint')})
        result = copy.deepcopy(result)
        result['literal_citation_contract'] = {'version': VERSION,
            'quote_options': sum(len(b['properties']['quote']['enum']) for b in
                request['schema']['properties']['claims']['items']['properties']['citations']['items']['anyOf']),
            'selection_valid': True, 'source_correctness_still_requires_registry': True,
            'semantic_correctness_verified': False}
        return result
