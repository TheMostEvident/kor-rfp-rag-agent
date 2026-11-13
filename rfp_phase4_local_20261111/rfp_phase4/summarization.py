"""전체 문서의 모든 청크를 입력하는, 출처 검증 후 결정적으로 합치는 요약.

입력 청크 100% 처리와 문서 의미 100% 요약은 별개다. 이 모듈은 전자를
계수하며 후자는 주장하지 않는다. 계획은 API 호출 없이 만들 수 있다.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping

from .evidence import ANSWER_SCHEMA
from .citation_recovery import validate_with_quote_recovery


TASK = 'full_document_summary_map'
DEFAULT_QUESTION = (
    '이 RFP의 제공된 부분을 한국어로 항목별 정리해 주세요. 사업 개요, 예산과 부가세, '
    '사업기간, 참가자격, 요구사항, 제출 방식과 기한, 평가 기준, 산출물, '
    '계약상 유의사항을 살피세요. 각 사실에는 제공된 원문에서 정확히 복사한 인용문을 붙이세요. '
    '이 입력은 전체 문서의 일부일 수 있으므로 여기서 찾지 못한 항목을 문서 전체에 없다고 '
    '단정하지 마세요. 원문에 쓰인 금액·단위·조건·연도 차이를 그대로 보존하세요.'
)
DEFAULT_CATEGORIES = (
    'overview', 'budget', 'duration', 'eligibility', 'requirements', 'submission',
    'evaluation', 'deliverables', 'risks',
)
CATEGORY_LABELS = {
    'overview': '사업 개요', 'budget': '예산·부가세', 'duration': '사업기간',
    'eligibility': '참가자격', 'requirements': '요구사항', 'submission': '제출 방식·기한',
    'evaluation': '평가 기준', 'deliverables': '산출물', 'risks': '계약상 유의사항',
    'other': '기타',
}


class SummaryError(ValueError):
    """잘못된 문서·배치 설정 또는 확인할 수 없는 입력의 오류."""


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class FullDocumentSummarizer:
    """모든 입력을 배치별로 요약하고 출처가 검증된 주장만 순서대로 합친다.

    ``plan``은 request_input_bytes만 사용하며 네트워크를 호출하지 않는다.
    ``summarize``는 모든 요청의 예산·승인 목록을 먼저 검증한다. 기본 배치 상한은
    30회다. 큰 문서가 이를 넘으면 입력을 자르지 않고 차단 사유와 전체 계획을 돌려준다.
    """

    def __init__(self, index, llm, registry, config=None):
        self.index, self.llm, self.registry = index, llm, registry
        supplied = dict(config or {})
        allowed = {'max_map_calls', 'max_batch_bytes', 'batch_max_bytes', 'requested_categories', 'stop_on_error'}
        if set(supplied) - allowed:
            raise SummaryError('지원하지 않는 전체 문서 요약 설정입니다.')
        llm_config = getattr(llm, 'config', {})
        self.max_map_calls = supplied.get('max_map_calls', 30)
        request_limit = _field(llm_config, 'max_request_bytes', 60000)
        if 'batch_max_bytes' in supplied and 'max_batch_bytes' in supplied:
            raise SummaryError('max_batch_bytes와 이전 별칭을 동시에 설정할 수 없습니다.')
        self.batch_max_bytes = supplied.get('max_batch_bytes', supplied.get('batch_max_bytes', min(35000, request_limit)))
        if isinstance(self.batch_max_bytes, bool) or not isinstance(self.batch_max_bytes, int) or self.batch_max_bytes < 1:
            raise SummaryError('max_batch_bytes는 양의 정수여야 합니다.')
        self.batch_max_bytes = min(self.batch_max_bytes, request_limit)
        self.max_output_tokens = _field(llm_config, 'max_output_tokens', 3000)
        self.requested_categories = tuple(supplied.get('requested_categories', DEFAULT_CATEGORIES))
        self.stop_on_error = supplied.get('stop_on_error', True)
        for value in [self.max_map_calls, self.batch_max_bytes, self.max_output_tokens]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SummaryError('요약 배치·byte·출력 상한은 양의 정수여야 합니다.')
        if not isinstance(self.stop_on_error, bool):
            raise SummaryError('stop_on_error는 bool이어야 합니다.')
        if (not self.requested_categories or len(set(self.requested_categories)) != len(self.requested_categories)
                or any(value not in CATEGORY_LABELS for value in self.requested_categories)):
            raise SummaryError('requested_categories는 중복 없는 지원 항목 목록이어야 합니다.')
        if not callable(getattr(llm, 'request_input_bytes', None)):
            raise SummaryError('LLM의 순수 request_input_bytes 계획 함수가 필요합니다.')
        if not callable(getattr(llm, 'preflight_requests', None)):
            raise SummaryError('모든 배치의 사전 예산 검사 기능이 필요합니다.')

    def _request(self, question, chunks):
        return {'task': TASK, 'question': question, 'evidence': chunks, 'schema': ANSWER_SCHEMA}

    def _bytes(self, question, chunks):
        count = self.llm.request_input_bytes(TASK, question, chunks, schema=ANSWER_SCHEMA)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SummaryError('요청 byte 계산이 잘못되었습니다.')
        return count

    def plan(self, doc_id, *, question=DEFAULT_QUESTION):
        """API 없이 전체 입력 계획을 만든다. 큰 문서도 모든 청크를 계획에 남긴다."""
        if not isinstance(question, str) or not question.strip():
            raise SummaryError('요약 질문을 입력해야 합니다.')
        canonical = list(self.index.resolve_doc_ids([doc_id]))
        if len(canonical) != 1:
            raise SummaryError('전체 요약은 문서 한 개를 명확하게 선택해야 합니다.')
        canonical = canonical[0]
        chunks = list(self.index.get_document_chunks(canonical))
        chunks = list(self.registry.validate_chunks(chunks, allowed_doc_ids=[canonical]))
        ids = [chunk['chunk_id'] for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise SummaryError('전체 문서 입력에 동일한 청크 ID가 중복됐습니다.')
        if any(chunk['doc_id'] != canonical for chunk in chunks):
            raise SummaryError('전체 문서 입력의 문서 범위가 다릅니다.')

        batches, block_reasons = [], []

        def append_batch(items):
            if items:
                batches.append({'batch_id': f'map-{len(batches) + 1:04d}', 'chunks': list(items),
                                'request_bytes': self._bytes(question, items)})

        # Exponential growth followed by binary search keeps planning a large table-rich
        # document close to O(total serialized input) instead of repeatedly encoding every
        # growing prefix. Each batch is a consecutive interval of the original chunk order.
        offset = 0
        while offset < len(chunks):
            first_bytes = self._bytes(question, chunks[offset:offset + 1])
            if first_bytes > self.batch_max_bytes:
                append_batch(chunks[offset:offset + 1])
                block_reasons.append('one_or_more_source_chunks_exceed_request_byte_budget')
                offset += 1
                continue
            remaining = len(chunks) - offset
            fits = 1
            probe = 2
            while probe <= remaining and self._bytes(question, chunks[offset:offset + probe]) <= self.batch_max_bytes:
                fits = probe
                probe *= 2
            low, high = fits + 1, min(probe, remaining)
            while low <= high:
                middle = (low + high) // 2
                if self._bytes(question, chunks[offset:offset + middle]) <= self.batch_max_bytes:
                    fits = middle
                    low = middle + 1
                else:
                    high = middle - 1
            append_batch(chunks[offset:offset + fits])
            offset += fits
        if not chunks:
            block_reasons.append('document_has_no_indexed_chunks')
        if len(batches) > self.max_map_calls:
            block_reasons.append('max_map_calls_exceeded')
        ordered = [chunk['chunk_id'] for batch in batches for chunk in batch['chunks']]
        if ordered != ids:
            raise SummaryError('전체 입력 계획에서 청크 순서 또는 개수가 유실됐습니다.')
        total_bytes = sum(batch['request_bytes'] for batch in batches)
        fingerprint = hashlib.sha256(_json({'doc_id': canonical, 'question': question,
             'source_chunks': [(chunk['chunk_id'], chunk['sha256'], hashlib.sha256(chunk['text'].encode('utf-8')).hexdigest()) for chunk in chunks],
             'batches': [[chunk['chunk_id'] for chunk in batch['chunks']] for batch in batches],
             'requested_categories': self.requested_categories, 'batch_max_bytes': self.batch_max_bytes,
             'max_output_tokens': self.max_output_tokens}).encode('utf-8')).hexdigest()
        return {'doc_id': canonical, 'question': question, 'source_chunk_ids': ids,
                'chunk_count': len(chunks), 'map_calls': len(batches), 'batches': batches,
                'total_request_bytes': total_bytes,
                'uncached_input_bytes_upper_bound': total_bytes,
                'llm_max_calls': _field(getattr(self.llm, 'config', {}), 'max_calls'),
                'llm_max_total_input_bytes': _field(getattr(self.llm, 'config', {}), 'max_total_input_bytes'),
                'whole_session_budget_preflight_required': True,
                'max_output_tokens_upper_bound': len(batches) * self.max_output_tokens,
                'byte_budget_is_token_count': False, 'estimated_price': None,
                'max_map_calls': self.max_map_calls, 'max_batch_bytes': self.batch_max_bytes,
                'batch_max_bytes': self.batch_max_bytes,
                'can_execute': not block_reasons, 'block_reasons': list(dict.fromkeys(block_reasons)),
                'plan_fingerprint': fingerprint, 'api_calls': 0,
                'scope': 'Every indexed chunk appears once, in source order. This is an input plan, not semantic summary coverage.'}

    def _empty_result(self, plan):
        return {'status': 'blocked', 'execution_status': 'not_started', 'doc_id': plan['doc_id'],
                'valid': False, 'claims': [], 'missing_information': [], 'errors': [],
                'answer_generated': False, 'markdown': '',
                'source_chunk_ids': plan['source_chunk_ids'], 'processed_chunk_ids': [],
                'response_received_chunk_ids': [], 'request_attempted_chunk_ids': [],
                'input_chunk_coverage': 0.0, 'validated_chunk_coverage': 0.0,
                'batch_outcomes': [], 'map_calls': plan['map_calls'],
                'plan_fingerprint': plan['plan_fingerprint'], 'preflight': None,
                'aggregation_method': 'deterministic_merge_of_source_validated_map_claims',
                'semantic_coverage_verified': False, 'semantic_entailment_verified': False,
                'answer_correctness_verified': False,
                'usage': {'api_calls': 0, 'live_api_calls': 0, 'mock_calls': 0,
                          'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                          'usage_unknown_attempts': 0},
                'input_coverage_definition': 'Fraction of indexed chunks with a completed model/cache response. Validated coverage separately requires valid source citations.',
                'scope': 'Full-document map input and citation provenance only; semantic accuracy/completeness needs review.'}

    @staticmethod
    def _merge_claims(claims):
        """Only exact duplicate claim text is merged; no uncited synthesis is invented."""
        merged, by_key = [], {}
        for claim in claims:
            key = claim['doc_id'], claim['category'], claim['text']
            if key not in by_key:
                fresh = {**claim, 'citations': [dict(citation) for citation in claim['citations']]}
                by_key[key] = fresh
                merged.append(fresh)
            else:
                target = by_key[key]['citations']
                known = {(citation['chunk_id'], citation['quote']) for citation in target}
                for citation in claim['citations']:
                    pair = citation['chunk_id'], citation['quote']
                    if pair not in known:
                        target.append(dict(citation)); known.add(pair)
        return merged

    def summarize(self, doc_id, question=DEFAULT_QUESTION):
        started = time.perf_counter()
        plan = self.plan(doc_id, question=question)
        result = self._empty_result(plan)
        usage_before = self.llm.usage_report() if callable(getattr(self.llm, 'usage_report', None)) else {}
        if not plan['can_execute']:
            result['execution_status'] = 'plan_blocked'
            result['errors'] = plan['block_reasons']
            result['markdown'] = '전체 문서 요약을 실행하지 않았습니다. 입력을 자르지 않고 계획 상한 초과를 반환했습니다.'
            result['latency_ms'] = (time.perf_counter() - started) * 1000
            return result
        requests = [self._request(question, batch['chunks']) for batch in plan['batches']]
        try:
            result['preflight'] = self.llm.preflight_requests(requests)
            if not isinstance(result['preflight'], dict) or result['preflight'].get('valid') is not True:
                raise SummaryError('전체 요청의 사전 검사가 통과하지 않았습니다.')
        except Exception as error:
            result['execution_status'] = 'preflight_blocked'
            result['errors'] = ['whole_document_preflight_failed:' + type(error).__name__]
            result['markdown'] = '전체 문서 요약을 실행하지 않았습니다. 활성화·문서 승인·전체 요청 예산을 확인하세요.'
            result['latency_ms'] = (time.perf_counter() - started) * 1000
            return result

        collected, processed, responded, attempted, outcomes = [], [], [], [], []
        halted = False
        for batch in plan['batches']:
            chunk_ids = [chunk['chunk_id'] for chunk in batch['chunks']]
            outcome = {'batch_id': batch['batch_id'], 'chunk_ids': chunk_ids,
                       'request_bytes': batch['request_bytes'], 'status': 'skipped',
                       'valid': False, 'usage': {}, 'live_api': False, 'cache_hit': False,
                       'api_called': False}
            if halted:
                outcome['reason'] = 'previous_batch_failed'
                outcomes.append(outcome)
                continue
            attempted.extend(chunk_ids)
            try:
                response = self.llm.generate(TASK, question, batch['chunks'], schema=ANSWER_SCHEMA)
                responded.extend(chunk_ids)
                outcome.update({'usage': response.get('usage', {}), 'live_api': bool(response.get('live_api')),
                                'cache_hit': bool(response.get('cache_hit')),
                                'api_called': bool(response.get('live_api')) and not bool(response.get('cache_hit')),
                                'execution_mode': response.get('execution_mode', 'mock_or_unspecified'),
                                'request_fingerprint': response.get('request_fingerprint')})
                validated = validate_with_quote_recovery(self.registry, response['payload'], batch['chunks'],
                                                          allowed_doc_ids=[plan['doc_id']])
                if validated.get('valid') is not True:
                    outcome.update({'status': 'invalid', 'reason': 'source_citation_validation_failed',
                                    'validation_errors':validated['errors'],
                                    'citation_diagnostics':validated.get('citation_diagnostics',[])})
                    result['errors'].append(batch['batch_id'] + ':source_citation_validation_failed')
                    halted = self.stop_on_error
                else:
                    outcome.update({'status': validated['status'], 'valid': True,
                                    'claim_count': len(validated['claims']),
                                    'batch_missing_information': validated.get('missing_information', [])})
                    processed.extend(chunk_ids)
                    collected.extend(validated['claims'])
            except Exception as error:
                outcome.update({'status': 'error', 'reason': 'batch_failed:' + type(error).__name__})
                result['errors'].append(batch['batch_id'] + ':batch_failed:' + type(error).__name__)
                halted = self.stop_on_error
            outcomes.append(outcome)

        claims = self._merge_claims(collected)
        covered_categories = {claim['category'] for claim in claims}
        missing = [CATEGORY_LABELS[category] + ': 검증된 근거를 확보하지 못했습니다.'
                   for category in self.requested_categories if category not in covered_categories]
        # A category label does not resolve a finer uncertainty such as whether VAT is
        # included. Without a semantic reduction/review step, preserve each map's own
        # uncertainty conservatively and keep its limited-input scope visible.
        partial_batches = [outcome for outcome in outcomes if outcome.get('valid') and outcome['status'] == 'partial']
        for outcome in outcomes:
            if not outcome.get('valid'):
                continue
            for label in outcome.get('batch_missing_information', []):
                missing.append(f'부분 입력 {outcome["batch_id"]}에서 미확인: {label}')
        if partial_batches:
            missing.append('부분 입력의 미확인 사항이 다른 입력에서 해결됐는지는 자동으로 검증하지 않았습니다.')
        missing = list(dict.fromkeys(missing))
        all_valid = len(processed) == len(plan['source_chunk_ids'])
        if not all_valid:
            missing.append('일부 원문 청크를 검증된 요약으로 처리하지 못했습니다. 전체 문서 요약으로 확정할 수 없습니다.')
        if not claims:
            status = 'insufficient_evidence' if all_valid else 'error'
        else:
            status = 'answered' if all_valid and not missing else 'partial'
        result.update({'status': status, 'execution_status': 'completed' if all_valid else 'incomplete',
                       'valid': bool(claims) or all_valid, 'claims': claims, 'missing_information': missing,
                       'processed_chunk_ids': processed, 'response_received_chunk_ids': responded,
                       'request_attempted_chunk_ids': attempted, 'batch_outcomes': outcomes,
                       'partial_batch_count': len(partial_batches),
                       'input_chunk_coverage': len(responded) / plan['chunk_count'],
                       'validated_chunk_coverage': len(processed) / plan['chunk_count'],
                       'answer_generated': any(outcome['valid'] for outcome in outcomes),
                       'elapsed_seconds': time.perf_counter() - started})
        if result['valid']:
            rendered = self.registry.render(result)
            result['markdown'] = rendered + '\n\n입력 청크 처리율은 의미적 요약 완전성이나 사실 정확도를 보장하지 않습니다.'
        else:
            result['markdown'] = '검증된 전체 문서 요약을 만들지 못했습니다. 처리 결과와 누락 청크를 확인하세요.'
        result['latency_ms'] = (time.perf_counter() - started) * 1000
        if callable(getattr(self.llm, 'usage_report', None)):
            after = self.llm.usage_report()
            result['llm_session_usage'] = after
            for name in ['api_calls', 'live_api_calls', 'mock_calls', 'usage_unknown_attempts']:
                result['usage'][name] = (after[name] - usage_before.get(name, 0)
                                         if isinstance(after.get(name), (int, float)) else None)
            for name in ['input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens']:
                key = 'reported_' + name
                result['usage'][name] = (after[key] - usage_before.get(key, 0)
                                        if isinstance(after.get(key), (int, float)) else None)
            result['usage']['scope'] = 'Counters for this summary invocation; failed attempts may have unknown token usage.'
        return result
