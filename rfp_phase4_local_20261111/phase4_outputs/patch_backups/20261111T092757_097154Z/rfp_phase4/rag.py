"""Grounded QA, document comparison and user-only follow-up state."""
from __future__ import annotations

import json
import re
import time

from local_rag.session import SearchSession


DEFAULT_RETRIEVAL = {'method': 'bm25', 'top_k': 8, 'per_doc_top_k': 6,
                     'fetch_k': 40, 'mmr_lambda': .75,
                     'max_context_bytes': 42000, 'max_question_chars': 4000}


def _config(value):
    config = dict(DEFAULT_RETRIEVAL)
    supplied = dict(value or {})
    unknown = set(supplied) - set(config)
    if unknown:
        raise ValueError('Unknown answer retrieval configuration: ' + ', '.join(sorted(unknown)))
    config.update(supplied)
    for name in ['top_k', 'per_doc_top_k', 'fetch_k', 'max_context_bytes', 'max_question_chars']:
        v = config[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(name + ' must be a positive integer')
    if config['method'] not in {'bm25', 'char_tfidf', 'rrf', 'mmr'}:
        raise ValueError('Unknown retrieval method')
    if isinstance(config['mmr_lambda'], bool) or not isinstance(config['mmr_lambda'], (int, float)) or not 0 <= config['mmr_lambda'] <= 1:
        raise ValueError('mmr_lambda must be between zero and one')
    return config


def _view(chunk):
    # This is an input-size estimate, not model tokenization or pricing.
    return {key: chunk.get(key) for key in ['chunk_id', 'doc_id', 'sha256', 'file_name',
            'locator', 'source_method', 'quality_flags', 'text']}


def _size(chunk):
    return len(json.dumps(_view(chunk), ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def _pack(groups, budget):
    """Round-robin document groups before truncation; never cut source text."""
    selected, omitted, seen = [], [], set()
    used = 2
    for offset in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if offset >= len(group):
                continue
            chunk = group[offset]
            cid = chunk['chunk_id']
            if cid in seen:
                continue
            seen.add(cid)
            size = _size(chunk) + 1
            if used + size <= budget:
                selected.append(chunk)
                used += size
            else:
                omitted.append(cid)
    return selected, omitted, used


def _safe(value):
    text = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    for char in ['\\', '|', '*', '_', '`', '[', ']']:
        text = text.replace(char, '\\' + char)
    return text.replace('\n', ' / ')


def _comparison_markdown(validation, selected_doc_ids, catalog):
    """A deterministic document-column table; every factual cell keeps citations."""
    from .evidence import CATEGORIES, CATEGORY_LABELS
    by_doc = {row['doc_id']: row for row in catalog}
    docs = list(selected_doc_ids)
    claims = validation['claims']
    rows, references = [], []
    numbered = {}
    for claim in claims:
        marks = []
        for citation in claim['citations']:
            key = citation['chunk_id'], citation['quote']
            if key not in numbered:
                numbered[key] = len(references) + 1
                references.append(citation)
            marks.append(f'[{numbered[key]}]')
        rows.append((claim['doc_id'], claim['category'], _safe(claim['text']) + ' ' + ''.join(marks)))
    lines = ['| 항목 | ' + ' | '.join(_safe(by_doc.get(doc, {}).get('file_name', doc)) for doc in docs) + ' |',
             '| --- | ' + ' | '.join('---' for _ in docs) + ' |']
    for category in CATEGORIES:
        if not any(row[1] == category for row in rows):
            continue
        cells = ['; '.join(row[2] for row in rows if row[0] == doc and row[1] == category)
                 or '현재 근거에서 미확인' for doc in docs]
        lines.append('| ' + CATEGORY_LABELS[category] + ' | ' + ' | '.join(cells) + ' |')
    if validation['missing_information']:
        lines.extend(['', '추가 확인이 필요한 항목:', ''])
        lines.extend('- ' + _safe(item) for item in validation['missing_information'])
    if references:
        lines.extend(['', '원문 근거:', ''])
        for number, citation in enumerate(references, 1):
            lines.append(f'{number}. ' + _safe(citation['file_name']) + ' · ' + _safe(citation['locator'])
                         + ' — “' + _safe(citation['quote']) + '”')
    lines.extend(['', '인용문과 원문 위치를 대조했습니다. 인용문이 주장 전체를 뒷받침하는지는 별도 검토가 필요합니다.'])
    return '\n'.join(lines)


class AnswerSession:
    """Uses only user history, never previous assistant answers or eval labels."""

    def __init__(self, index, llm, registry, config=None, active_doc_ids=None):
        self.index, self.llm, self.registry = index, llm, registry
        self.config = _config(config)
        self.retrieval_session = SearchSession(index, active_doc_ids=active_doc_ids)

    @property
    def active_doc_ids(self):
        return self.retrieval_session.active_doc_ids

    @property
    def query_history(self):
        return self.retrieval_session.query_history

    def select_documents(self, ids):
        return self.retrieval_session.select_documents(ids)

    def _empty(self, status, execution_status, message, retrieval=None):
        return {'status': status, 'execution_status': execution_status,
                'answer_generated': False, 'claims': [], 'markdown': message,
                'missing_information': [], 'active_doc_ids': self.active_doc_ids,
                'retrieval': retrieval or {}, 'source_chunks': [], 'generation': None,
                'validation': None, 'semantic_entailment_verified': False}

    def ask(self, query, *, feature='qa', candidate_doc_ids=None):
        if not isinstance(query, str):
            raise TypeError('질문은 문자열이어야 합니다.')
        if feature not in {'qa', 'fact', 'comparison', 'followup', 'uncertainty'}:
            raise ValueError('전체 요약은 FullDocumentSummarizer 경로를 사용하세요.')
        if len(query) > self.config['max_question_chars']:
            raise ValueError('질문 길이가 설정 상한을 넘었습니다.')
        started = time.perf_counter()
        search = self.retrieval_session.ask(query, top_k=self.config['top_k'],
            method=self.config['method'], candidate_doc_ids=candidate_doc_ids)
        if search['status'] != 'retrieved':
            return self._empty(search['status'], 'not_requested', search.get('reason') or '문서 범위와 질문을 확인하세요.', search)
        comparison = feature == 'comparison' or bool(re.search(r'비교|차이|두\s*사업', query))
        scope = self.active_doc_ids
        if comparison and len(scope) < 2:
            return self._empty('needs_clarification', 'not_requested',
                               '비교할 문서를 두 개 이상 명시적으로 선택해 주세요.', search)
        groups = []
        if comparison:
            # Each document earns a separate quota BEFORE merged context packing.
            for doc_id in scope:
                hits = self.index.search(search['standalone_query'],
                    top_k=self.config['per_doc_top_k'], allowed_doc_ids=[doc_id],
                    method=self.config['method'], fetch_k=self.config['fetch_k'],
                    mmr_lambda=self.config['mmr_lambda'])
                groups.append(hits)
        else:
            # SearchSession manages scope/history; use configured candidate/MMR values.
            hits = self.index.search(search['standalone_query'], top_k=self.config['top_k'],
                allowed_doc_ids=scope or None, method=self.config['method'],
                fetch_k=self.config['fetch_k'], mmr_lambda=self.config['mmr_lambda'])
            groups = [hits]
        candidate_hits = [hit for group in groups for hit in group]
        # Raises before API dispatch if any hit or source provenance was altered.
        self.registry.validate_chunks(candidate_hits, allowed_doc_ids=scope or None)
        packed, omitted, input_bytes = _pack(groups, self.config['max_context_bytes'])
        context = self.registry.validate_chunks(packed, allowed_doc_ids=scope or None)
        represented = {c['doc_id'] for c in context}
        missing_docs = [doc for doc in scope if doc not in represented]
        retrieval = {**search, 'hits': candidate_hits,
            'comparison': comparison, 'method': self.config['method'],
            'context_chunk_ids': [c['chunk_id'] for c in context],
            'omitted_chunk_ids': omitted, 'context_bytes_estimate': input_bytes,
            'missing_selected_doc_ids': missing_docs,
            'document_quotas_applied_before_packing': comparison,
            'retrieval_latency_ms': (time.perf_counter() - started) * 1000}
        if not context:
            return self._empty('insufficient_evidence', 'not_requested',
                '현재 검색·입력 예산 안에서 인용할 원문을 확보하지 못했습니다.', retrieval)
        enabled = getattr(getattr(self.llm, 'config', None), 'enabled', False)
        if not enabled:
            result = self._empty('not_generated', 'disabled',
                'OpenAI 실행이 꺼져 있습니다. 원문 근거 후보를 확인한 뒤 ENABLE_OPENAI와 해당 실행 옵션을 켜세요.', retrieval)
            result['source_chunks'] = context
            return result
        try:
            generated = self.llm.generate('comparison' if comparison else 'qa',
                                          search['standalone_query'], context)
        except Exception as error:
            # No upstream API body, key, or source snippet is exposed.
            from .llm import LLMError
            safe_detail = str(error) if isinstance(error, LLMError) else ''
            result = self._empty('not_generated', 'error',
                '답변 생성에 실패했습니다. API 설정·사용량 상한·모델 접근 권한을 확인하세요.', retrieval)
            result.update({'source_chunks': context, 'error_type': type(error).__name__,
                           'error_detail': safe_detail,
                           'latency_ms': (time.perf_counter() - started) * 1000})
            return result
        validation = self.registry.validate_answer(generated.get('payload'), context,
                                                   allowed_doc_ids=scope or None)
        generation_report = {key: value for key, value in generated.items() if key != 'payload'}
        if not validation['valid']:
            result = self._empty('not_generated', 'validation_failed',
                '생성 결과의 출처·형식 검사에 실패해 답변을 표시하지 않습니다.', retrieval)
            result.update({'validation': validation, 'generation': generation_report,
                           'source_chunks': context, 'latency_ms': (time.perf_counter() - started) * 1000})
            return result
        # Preserve missing evidence on either selected document; never imply a complete comparison.
        cited_docs = {claim['doc_id'] for claim in validation['claims']}
        missing_comparison = [doc for doc in scope if doc not in cited_docs] if comparison else []
        if missing_docs or missing_comparison:
            validation = dict(validation)
            validation['status'] = 'partial' if validation['claims'] else 'insufficient_evidence'
            validation['missing_information'] = list(validation['missing_information']) + [
                f'선택 문서 {doc}: 현재 입력에서 검증된 비교 근거 미확보'
                for doc in dict.fromkeys(missing_docs + missing_comparison)]
        return {'status': validation['status'], 'execution_status': 'completed',
                'answer_generated': bool(validation['claims']), 'claims': validation['claims'],
                'missing_information': validation['missing_information'],
                'markdown': (_comparison_markdown(validation, scope, self.index.get_catalog())
                             if comparison else self.registry.render(validation)), 'active_doc_ids': scope,
                'source_chunks': context, 'retrieval': retrieval, 'generation': generation_report,
                'validation': validation, 'usage': generated.get('usage', {}),
                'latency_ms': (time.perf_counter() - started) * 1000,
                'semantic_entailment_verified': False}
