"""Source-validated full-input RFP summaries with eleven explicit output fields.

The Phase 4 planner and quote recovery remain the source of truth. This layer
adds field metadata and deterministic presentation, without an extra model
reduction call or an automatic assertion of semantic completeness.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import re

from rfp_phase4.llm import LLMError
from rfp_phase4.summarization import FullDocumentSummarizer

from .fields import FIELDS, FIELD_SCHEMA, FIELD_QUESTION, parse_field_claim, parse_missing_label


TASK = 'phase5_field_summary_map_v1'
_STATE_LABELS = {
    'evidence_found': '근거 추출됨 · 검토 필요',
    'notice_reference': '공고문 등 별도 자료 참조',
    'conflict_candidates': '상충 후보 · 확인 필요',
    'not_found_in_processed_input': '처리한 입력에서 미확인',
    'not_processed': '미실행·미처리',
}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


class _FieldMapLLM:
    """Map identical planning, preflight and generation inputs to one task/schema."""
    def __init__(self, llm):
        self.base = llm
        self.config = llm.config
        self.received_chunk_ids = []
        self.field_contract_failures = []

    def request_input_bytes(self, task, question, evidence, *, schema=None):
        return self.base.request_input_bytes(TASK, question, evidence, schema=FIELD_SCHEMA)

    def preflight_requests(self, requests):
        mapped = [{**request, 'task': TASK, 'schema': FIELD_SCHEMA} for request in requests]
        return self.base.preflight_requests(mapped)

    def usage_report(self):
        return self.base.usage_report()

    def generate(self, task, question, evidence, *, schema=None):
        response = self.base.generate(TASK, question, evidence, schema=FIELD_SCHEMA)
        self.received_chunk_ids.extend(chunk['chunk_id'] for chunk in evidence)
        payload = response.get('payload', {})
        try:
            for claim in payload.get('claims', []):
                parse_field_claim(claim)
        except (ValueError, TypeError, AttributeError):
            # The base client has already counted any paid attempt/cache hit. Do
            # not retry here or pass malformed claims to the accepted batch list.
            self.field_contract_failures.append({
                'chunk_ids': [chunk['chunk_id'] for chunk in evidence],
                'reason': 'field_marker_or_category_invalid',
            })
            raise LLMError('Phase 5 field marker/category validation failed') from None
        return response


class _FieldInputSummarizer(FullDocumentSummarizer):
    def plan(self, doc_id, *, question=FIELD_QUESTION):
        plan = super().plan(doc_id, question=question)
        plan['base_plan_fingerprint'] = plan['plan_fingerprint']
        plan['field_schema_fingerprint'] = _fingerprint(FIELD_SCHEMA)
        plan['task'] = TASK
        plan['field_ids'] = [field['id'] for field in FIELDS]
        plan['plan_fingerprint'] = _fingerprint({
            'base': plan['base_plan_fingerprint'], 'task': TASK,
            'schema': plan['field_schema_fingerprint'],
        })
        fresh_blocks = []
        calls_limit = plan.get('llm_max_calls')
        input_limit = plan.get('llm_max_total_input_bytes')
        if not isinstance(calls_limit, int) or isinstance(calls_limit, bool):
            fresh_blocks.append('llm_max_calls_unavailable')
        elif plan['map_calls'] > calls_limit:
            fresh_blocks.append('max_calls_exceeded')
        if not isinstance(input_limit, int) or isinstance(input_limit, bool):
            fresh_blocks.append('llm_max_total_input_bytes_unavailable')
        elif plan['total_request_bytes'] > input_limit:
            fresh_blocks.append('max_total_input_bytes_exceeded')
        plan['fresh_session_uncached_within_caps'] = not fresh_blocks
        plan['fresh_session_uncached_block_reasons'] = fresh_blocks
        plan['fresh_session_uncached_scope'] = (
            'Fresh session, every request uncached. This informational flag does not change '
            'can_execute (batch/map limits). Execution preflight checks actual remaining '
            'session caps and exact-request cache before any API call.'
        )
        return plan


class FieldSummarizer:
    """A bounded full-document map pass followed by an exact eleven-row view."""
    def __init__(self, index, llm, registry, config=None):
        self.index, self.llm, self.registry = index, llm, registry
        supplied = dict(config or {})
        supplied.setdefault('requested_categories', list(dict.fromkeys(field['category'] for field in FIELDS)))
        self._adapter = _FieldMapLLM(llm)
        self._inner = _FieldInputSummarizer(index, self._adapter, registry, supplied)

    def plan(self, doc_id):
        return self._inner.plan(doc_id, question=FIELD_QUESTION)

    def summarize(self, doc_id):
        self._adapter.received_chunk_ids = []
        self._adapter.field_contract_failures = []
        result = self._inner.summarize(doc_id, question=FIELD_QUESTION)
        # A malformed marker is a received response, while still never a
        # source-validated processed batch. Keep the two coverage meanings apart.
        received = set(result.get('response_received_chunk_ids', [])) | set(self._adapter.received_chunk_ids)
        result['response_received_chunk_ids'] = [cid for cid in result['source_chunk_ids'] if cid in received]
        if result['source_chunk_ids']:
            result['input_chunk_coverage'] = len(result['response_received_chunk_ids']) / len(result['source_chunk_ids'])
        result['field_contract_failures'] = copy.deepcopy(self._adapter.field_contract_failures)
        result = self._field_result(result)
        if result.get('processed_chunk_ids'):
            from .source_audit import audit_summary_sources, render_source_audit
            audit = audit_summary_sources(self.index, self.registry, result)
            result['source_audit'] = audit
            if audit['candidates']:
                labels = list(dict.fromkeys(item['label'] for item in audit['candidates']))
                result['missing_information'].extend('원문 추가 확인 필요: ' + label for label in labels)
                if result.get('claims'):
                    result['status'] = 'partial'
                result['markdown'] += '\n\n' + render_source_audit(audit)
        return result

    def _field_result(self, result):
        # Accepted claims are still the original source-validated payload texts.
        # The display view strips only the transport prefix, after validation.
        rows = {field['id']: {'field_id': field['id'], 'label': field['label'],
                'state': 'not_processed', 'claims': [], 'notes': []} for field in FIELDS}
        for claim in result.get('claims', []):
            parsed = parse_field_claim(claim)
            rows[parsed['field_id']]['claims'].append({
                'text': parsed['text'], 'mode': parsed['mode'],
                'citations': copy.deepcopy(claim['citations']),
            })
        global_notes = []
        for batch in result.get('batch_outcomes', []):
            if not batch.get('valid'):
                continue
            for label in batch.get('batch_missing_information', []):
                parsed = parse_missing_label(label)
                scoped = f'부분 입력 {batch["batch_id"]}에서 미확인: '
                if parsed:
                    rows[parsed['field_id']]['notes'].append(scoped + parsed['text'])
                else:
                    global_notes.append(scoped + str(label))
        any_processed = bool(result.get('processed_chunk_ids'))
        for row in rows.values():
            modes = {claim['mode'] for claim in row['claims']}
            if 'conflict_candidate' in modes:
                row['state'] = 'conflict_candidates'
                row['notes'].append('모델이 구분한 상충 후보입니다. 원문·적용 범위를 대조하기 전에는 충돌로 확정하지 않습니다.')
            elif 'fact' in modes:
                row['state'] = 'evidence_found'
            elif 'notice_reference' in modes:
                row['state'] = 'notice_reference'
            else:
                row['state'] = 'not_found_in_processed_input' if any_processed else 'not_processed'
            if 'notice_reference' in modes:
                row['notes'].append('별도 공고문 등의 실제 내용은 이번 RFP 입력만으로 확인하지 않았습니다.')
            if len({claim['text'] for claim in row['claims']}) > 1:
                row['notes'].append('서로 다른 진술을 모두 보존했습니다. 세부 내역·적용 범위가 다른지 검토하세요.')
            row['notes'] = list(dict.fromkeys(row['notes']))
        result['fields'] = list(rows.values())
        result['field_ids'] = list(rows)
        result['task'] = TASK
        result['field_schema_fingerprint'] = _fingerprint(FIELD_SCHEMA)
        result['semantic_quality_verified'] = False
        result['automatic_conflict_resolution'] = False
        result['field_uncertainty_notes'] = list(dict.fromkeys(global_notes))
        result['phase4_missing_information'] = result.get('missing_information', [])
        missing = []
        for row in rows.values():
            if row['state'] != 'evidence_found':
                missing.append(f'{row["label"]}: {_STATE_LABELS[row["state"]]}')
            missing.extend(f'{row["label"]}: {note}' for note in row['notes'])
        missing.extend(global_notes)
        if result.get('execution_status') == 'incomplete':
            missing.append('일부 원문 청크를 검증된 요약으로 처리하지 못했습니다. 전체 문서 요약으로 확정할 수 없습니다.')
        result['missing_information'] = list(dict.fromkeys(missing))
        if result.get('claims'):
            # Complete processing is necessary, but never called semantic proof.
            result['status'] = ('answered' if result.get('execution_status') == 'completed'
                                and not result['missing_information'] else 'partial')
        result['markdown'] = render_field_summary(result)
        return result


def _escape(value):
    text = html.escape(str(value), quote=False)
    text = re.sub(r'([\\`*_{}\[\]()#!+\-.])', r'\\\1', text)
    return text.replace('|', '&#124;').replace('\r', '').replace('\n', ' / ')


def render_field_summary(result):
    """Render accepted field data; no unchecked model prose is interpolated."""
    lines = ['| 항목 | 내용 | 확인 상태 |', '| --- | --- | --- |']
    citations, by_key = [], {}
    for row in result.get('fields', []):
        items = []
        for claim in row['claims']:
            numbers = []
            for citation in claim['citations']:
                key = (citation.get('doc_id'), citation['chunk_id'], citation['quote'])
                if key not in by_key:
                    by_key[key] = len(citations) + 1
                    citations.append(citation)
                numbers.append(f'[{by_key[key]}]')
            items.append(_escape(claim['text']) + ' ' + ''.join(numbers))
        text = ' / '.join(items) if items else '이번에 처리한 입력에서 근거를 확보하지 못했습니다.'
        if row['state'] == 'not_processed':
            text = '아직 요약을 실행하지 않았거나 이 항목을 처리하지 못했습니다.'
        lines.append(f'| {_escape(row["label"])} | {text} | {_escape(_STATE_LABELS[row["state"]])} |')
    if citations:
        lines.extend(['', '원문 근거:', ''])
        for number, citation in enumerate(citations, 1):
            lines.append(f'{number}. {_escape(citation.get("file_name", ""))} · '
                         f'{_escape(citation.get("locator", ""))} — “{_escape(citation["quote"])}”')
    notes = [f'{row["label"]}: {note}' for row in result.get('fields', []) for note in row['notes']]
    notes.extend(result.get('field_uncertainty_notes', []))
    if result.get('execution_status') not in {'completed', 'incomplete'}:
        notes.insert(0, '요약을 실행하지 못했습니다. 실행 상태와 전체 입력 계획을 확인하세요.')
    elif result.get('execution_status') == 'incomplete':
        notes.insert(0, '일부 입력이 미처리 상태입니다. 이 결과를 전체 문서 요약으로 확정할 수 없습니다.')
    if notes:
        lines.extend(['', '추가 확인 사항:', ''])
        lines.extend('- ' + _escape(note) for note in dict.fromkeys(notes))
    lines.extend(['', '입력 처리율과 인용문 출처를 검사했습니다. 항목의 정확성·조건 보존·요약 완전성은 사람이 원문과 대조해야 합니다.'])
    return '\n'.join(lines)
