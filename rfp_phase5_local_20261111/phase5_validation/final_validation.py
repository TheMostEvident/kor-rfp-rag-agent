"""Phase 5 sample acceptance. No network except an explicitly enabled run_stage.

The runner never changes source/index/config files or rewrites a model answer.
Reference drafts are used only in local human review, never sent to the model.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

VERSION = 'phase5-final-validation-v3'
JUDGEMENTS = ('정확성', '누락없음', '조건보존')
BASE_COLUMNS = ('항목ID', '항목', '기준내용', '필수조건', '실제요약', '확인상태', '기준원문위치', '실제인용', '추가원문확인후보')
CORE_FILES = ('rfp_phase4/llm.py', 'rfp_phase4/evidence.py',
              'rfp_phase4/citation_recovery.py', 'rfp_phase4/summarization.py',
              'rfp_phase5/fields.py', 'rfp_phase5/summary.py',
              'rfp_phase5/source_audit.py', 'rfp_phase5/quality.py',
              'rfp_phase5/runtime.py', 'rfp_phase5/review.py', 'rfp_phase4/runtime.py',
              'local_rag/io.py', 'local_rag/retrieval.py', 'phase5_validation/cache_revalidation.py',
              'tests/test_phase5_summary.py', 'tests/test_phase5_llm_diagnostics.py',
              'tests/test_phase5_saved_diagnostics.py', 'tests/test_phase5_source_audit.py',
              'tests/test_phase4_citation_recovery.py', 'tests/test_phase4_citation_split.py',
              'phase5_validation/test_final_validation.py', 'phase5_validation/test_cache_revalidation.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _csv_text(value):
    text = str(value)
    # CSV is often opened in Excel. Model/source text must stay text.
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def _payload(result):
    claims = [{k: c[k] for k in ('doc_id', 'category', 'text', 'citations')}
              for c in result.get('claims', [])]
    for c in claims:
        c['citations'] = [{k: cit[k] for k in ('chunk_id', 'quote')} for cit in c['citations']]
    missing = list(result.get('missing_information', []))
    return {'status': result.get('status'), 'claims': claims,
            'missing_information': missing}


def technical_check(result, sample, env, plan, request_fingerprints):
    """Revalidate exact quotes and field display, not just saved valid flags."""
    from rfp_phase5.fields import FIELDS, parse_field_claim
    checks = {}
    expected = list(plan['source_chunk_ids'])
    batches = result.get('batch_outcomes') or []
    checks['document_scope'] = result.get('doc_id') == sample['doc_id']
    catalog = {r['doc_id']: r for r in env['index'].get_catalog()}
    checks['source_version'] = catalog.get(sample['doc_id'], {}).get('sha256') == sample['source_sha256']
    checks['plan_matches'] = result.get('plan_fingerprint') == plan['plan_fingerprint']
    checks['all_source_chunks_listed'] = bool(expected) and result.get('source_chunk_ids') == expected
    checks['all_chunks_validated_once'] = (bool(expected) and result.get('processed_chunk_ids') == expected
        and len(expected) == len(set(expected)) and result.get('validated_chunk_coverage') == 1.0
        and result.get('input_chunk_coverage') == 1.0)
    checks['execution_completed'] = result.get('execution_status') == 'completed'
    checks['answer_present'] = result.get('answer_generated') is True and bool(result.get('claims'))
    checks['answer_status_usable'] = result.get('status') in {'answered', 'partial'}
    checks['no_batch_errors'] = not result.get('errors') and not result.get('field_contract_failures')
    checks['live_model_origin'] = bool(batches) and all(b.get('valid') is True
        and b.get('execution_mode') == 'live_openai' and b.get('live_api') is True for b in batches)
    checks['batch_order_preserved'] = ([cid for b in batches for cid in b.get('chunk_ids', [])] == expected
        and [b.get('batch_id') for b in batches] == [b['batch_id'] for b in plan['batches']])
    checks['exact_model_requests_match'] = [b.get('request_fingerprint') for b in batches] == list(request_fingerprints)
    checks['task_matches'] = result.get('task') == plan.get('task')
    validation_errors = []
    try:
        chunks = env['index'].get_document_chunks(sample['doc_id'])
        validation = env['registry'].validate_answer(_payload(copy.deepcopy(result)), chunks,
                                                       allowed_doc_ids=[sample['doc_id']])
        validation_errors = validation.get('errors', [])
        checks['quotes_revalidated'] = validation.get('valid') is True and bool(validation.get('claims'))
        # The displayed fields must be a faithful view of the validated raw claims.
        derived = {f['id']: [] for f in FIELDS}
        for claim in validation.get('claims', []):
            parsed = parse_field_claim(claim)
            derived[parsed['field_id']].append((parsed['text'], parsed['mode'],
                [(c['chunk_id'], c['quote']) for c in claim['citations']]))
        fields = result.get('fields') or []
        checks['eleven_fields_in_order'] = [f.get('field_id') for f in fields] == [f['id'] for f in FIELDS]
        checks['field_display_matches_validated_claims'] = checks['eleven_fields_in_order'] and all(
            f.get('label') == spec['label'] and
            [(c.get('text'), c.get('mode'), [(q.get('chunk_id'), q.get('quote')) for q in c.get('citations', [])])
             for c in f.get('claims', [])] == derived[spec['id']]
            for f, spec in zip(fields, FIELDS))
        checks['no_unprocessed_fields'] = len(fields) == 11 and all(f.get('state') in {
            'evidence_found', 'notice_reference', 'not_found_in_processed_input', 'conflict_candidates'} for f in fields)
        from rfp_phase5.summary import FieldSummarizer
        normalized = FieldSummarizer._field_result(None, copy.deepcopy(result))
        checks['field_states_match_claims'] = [(f.get('field_id'), f.get('state')) for f in fields] == [
            (f['field_id'], f['state']) for f in normalized['fields']]
    except (ValueError, TypeError, KeyError, AttributeError):
        checks['quotes_revalidated'] = False
        checks['field_display_matches_validated_claims'] = False
    return {'passed': all(checks.values()), 'checks': checks,
            'failed_checks': [k for k, v in checks.items() if not v],
            'validation_errors': validation_errors,
            'semantic_quality_verified': False,
            'partial_status_is_not_automatic_failure_or_pass': True}


def make_review_rows(result, sample, references, source_audit=None):
    fields = {f['field_id']: f for f in result.get('fields', [])}
    rows = []
    for ref in [r for r in references['rows'] if r['sample_id'] == sample['sample_id']]:
        field = fields.get(ref['field_id'], {})
        citations = []
        for claim in field.get('claims', []):
            for c in claim.get('citations', []):
                text = str(c.get('locator', '')) + ' | ' + str(c.get('quote', ''))
                if text not in citations:
                    citations.append(text)
        candidates = [c for c in (source_audit or {}).get('candidates', []) if c.get('field_id') == ref['field_id']]
        hints = '\n'.join(' / '.join(dict.fromkeys(str(q.get('locator', '')) for q in c.get('citations', [])))
            + ' | ' + str(c.get('source_quote', '')) for c in candidates)
        omitted = []
        for batch in result.get('batch_outcomes', []):
            for item in batch.get('citation_diagnostics', []):
                if item.get('claim_field_id') == ref['field_id'] and item.get('requires_conditions_review'):
                    for gap in item.get('omitted_source_gaps', []):
                        omitted.append('모델 인용에서 생략된 원문(답변에 자동 추가 안 함): '
                            + str(item.get('chunk_id', '')) + ' | ' + str(gap.get('text', gap.get('preview', ''))))

        for batch in result.get('batch_outcomes', []):
            for item in batch.get('citation_diagnostics', []):
                if (
                    item.get('claim_field_id') == ref['field_id']
                    and item.get('requires_reference_review')
                ):
                    omitted.append(
                        '원래 청크 ID 오류; 현재 배치에서 인용문이 '
                        '정확히 일치하는 청크를 모두 표시: '
                        + str(item.get('original_chunk_id'))
                        + ' -> '
                        + ', '.join(item.get('candidate_chunk_ids', []))
                        + ' | '
                        + str(item.get('model_quote', ''))
                    )
        hints = '\n'.join([hints, *omitted]).strip()
        values = [ref['field_id'], ref['label'], ref['expected'], ref['conditions'],
            '\n'.join(c.get('text', '') for c in field.get('claims', [])) or '검증된 근거를 확보하지 못함',
            field.get('state', 'not_processed'), ref['source_reference'], '\n'.join(citations), hints]
        row = {k: _csv_text(v) for k, v in zip(BASE_COLUMNS, values)}
        row.update({k: '' for k in (*JUDGEMENTS, '메모')})
        rows.append(row)
    if len(rows) != 11 or len({r['항목ID'] for r in rows}) != 11:
        raise ValueError('검토 기준은 문서별 11개 항목이어야 합니다.')
    return rows


def write_review(path, rows):
    """Never replace completed or partially completed reviewer input."""
    path = Path(path)
    if path.exists():
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[*BASE_COLUMNS, *JUDGEMENTS, '메모'])
        writer.writeheader()
        writer.writerows(rows)


def assess_csv(path, expected_rows, reviewer):
    if not Path(path).is_file():
        return {'passed': False, 'status': 'review_missing', 'errors': [], 'reviewed': 0, 'total': 33}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    errors, pending, rejected, count = [], [], [], 0
    if headers != [*BASE_COLUMNS, *JUDGEMENTS, '메모']:
        errors.append('review_columns_changed')
    if len(rows) != len(expected_rows):
        errors.append('review_row_count_changed')
    for row, expected in zip(rows, expected_rows):
        if any(row.get(k) != expected[k] for k in BASE_COLUMNS):
            errors.append('answer_or_reference_cells_changed:' + expected['항목ID'])
        for key in JUDGEMENTS:
            value = str(row.get(key, '')).strip().lower()
            item = {'field_id': expected['항목ID'], 'judgement': key}
            if value in {'통과', 'true'}:
                count += 1
            elif value in {'실패', 'false'}:
                rejected.append(item)
            elif value in {'', '미검토'}:
                pending.append(item)
            else:
                errors.append('invalid_judgement:' + expected['항목ID'] + ':' + key)
    if not isinstance(reviewer, str) or not reviewer.strip():
        pending.append({'field_id': None, 'judgement': '검토자 이름'})
    status = ('invalid_review' if errors else 'review_failed' if rejected else
              'review_pending' if pending else 'passed_human_review')
    return {'passed': status == 'passed_human_review', 'status': status,
            'errors': sorted(set(errors)), 'pending': pending, 'failed': rejected,
            'reviewed': count + len(rejected), 'positive': count, 'total': 33,
            'reviewer': reviewer.strip(), 'checked_at_utc': datetime.now(timezone.utc).isoformat()}


def case_verdict(technical, human):
    return {'accepted': technical.get('passed') is True and human.get('passed') is True,
            'technical_passed': technical.get('passed') is True,
            'human_status': human.get('status'), 'technical_failures': technical.get('failed_checks', []),
            'status': ('passed_sample_case' if technical.get('passed') and human.get('passed') else
                       'technical_failed' if not technical.get('passed') else human.get('status', 'review_pending'))}


class ValidationSuite:
    def __init__(self, env, references, suite_id='phase5_final_v1', *, per_doc_ceiling=900000,
                 suite_calls=75, suite_bytes=2400000):
        from rfp_phase5.runtime import make_components
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', suite_id):
            raise ValueError('SUITE_ID는 영문·숫자·밑줄·하이픈 1~64자로 지정하세요.')
        if any(type(v) is not int or v < 1 for v in (per_doc_ceiling, suite_calls, suite_bytes)):
            raise ValueError('실행 상한은 양의 정수여야 합니다.')
        self.env, self.references = env, copy.deepcopy(references)
        self.root = Path(env['root'])
        self.samples = references['samples']
        if len(self.samples) != 7 or len({s['sample_id'] for s in self.samples}) != 7:
            raise ValueError('검증 문서는 개발 5건·최종 2건으로 고정합니다.')
        if [s['split'] for s in self.samples] != ['dev'] * 5 + ['final'] * 2:
            raise ValueError('검증 문서의 개발·최종 구분이 다릅니다.')
        self.folder = self.root / 'phase5_outputs/final_validation' / suite_id
        self.suite_calls, self.suite_bytes = suite_calls, suite_bytes
        self.plans, self.configs, self.request_fps = {}, {}, {}
        cat = {r['doc_id']: r for r in env['index'].get_catalog()}
        for sample in self.samples:
            sid = sample['sample_id']
            if cat.get(sample['doc_id'], {}).get('sha256') != sample['source_sha256']:
                raise ValueError('표본 문서 버전 불일치: ' + sid)
            component = make_components(env, doc_ids=[sample['doc_id']], enabled=False)
            plan = component['summarizer'].plan(sample['doc_id'])
            if plan['chunk_count'] != sample['expected_chunks']:
                raise ValueError('표본 청크 개수가 기준표와 다릅니다: ' + sid)
            if not plan['can_execute']:
                raise ValueError('문서 전체 입력 계획을 실행할 수 없습니다: ' + sid + ':' + ','.join(plan['block_reasons']))
            if plan['total_request_bytes'] > per_doc_ceiling:
                raise ValueError('문서 전체 입력이 문서별 상한을 넘었습니다: ' + sid)
            local_env = dict(env)
            config = copy.deepcopy(env['phase5_config'])
            current_limit = config['generation']['max_total_input_bytes']
            if plan['total_request_bytes'] > current_limit:
                needed = ((plan['total_request_bytes'] + 9999) // 10000) * 10000
                if needed > per_doc_ceiling:
                    raise ValueError('문서 전체 입력이 문서별 상한을 넘었습니다: ' + sid)
                config['generation']['max_total_input_bytes'] = needed
            local_env['phase5_config'] = config
            # enabled=True creates no SDK client and does not make a request.
            prepared = make_components(local_env, doc_ids=[sample['doc_id']], enabled=True)
            live_plan = prepared['summarizer'].plan(sample['doc_id'])
            self.plans[sid], self.configs[sid] = live_plan, config
            from rfp_phase5.fields import FIELD_SCHEMA, FIELD_QUESTION
            from rfp_phase5.summary import TASK
            self.request_fps[sid] = [prepared['llm']._prepare(TASK, FIELD_QUESTION, b['chunks'], schema=FIELD_SCHEMA)['fingerprint']
                for b in live_plan['batches']]
        self.frozen = {'version': VERSION, 'samples': self.samples,
            'reference_fingerprint': digest(references),
            'code_sha256': {p: file_hash(self.root / p) for p in CORE_FILES},
            'runner_sha256': file_hash(__file__),
            'index_fingerprint': env['index'].get_metadata()['index_fingerprint'],
            'dependencies': env['environment']['current_dependencies'],
            'config_by_sample': self.configs,
            'plan_fingerprints': {s: p['plan_fingerprint'] for s, p in self.plans.items()},
            'suite_calls_cap': suite_calls, 'suite_input_bytes_cap': suite_bytes,
            'per_doc_input_ceiling': per_doc_ceiling}
        self.fingerprint = digest(self.frozen)
        self.folder.mkdir(parents=True, exist_ok=True)
        freeze_path = self.folder / 'frozen.json'
        if freeze_path.exists():
            if read_json(freeze_path) != self.frozen:
                raise ValueError('코드·설정·원문·기준표가 변경됐습니다. 새 SUITE_ID로 개발 평가부터 시작하세요.')
        else:
            write_json(freeze_path, self.frozen)
        write_json(self.folder / 'plan_summary.json', self.plan_summary())

    def assert_frozen(self):
        if read_json(self.folder / 'frozen.json') != self.frozen:
            raise ValueError('고정 실행 기록이 변경됐습니다.')
        if digest(self.references) != self.frozen['reference_fingerprint'] or self.configs != self.frozen['config_by_sample']:
            raise ValueError('평가 중 기준표 또는 설정이 변경됐습니다.')
        current = {p: file_hash(self.root / p) for p in CORE_FILES}
        if current != self.frozen['code_sha256'] or file_hash(__file__) != self.frozen['runner_sha256']:
            raise ValueError('평가 중 코드가 변경됐습니다. 커널 재시작 후 새 SUITE_ID를 사용하세요.')
        from rfp_phase4.runtime import environment_report
        if environment_report(self.root)['current_dependencies'] != self.frozen['dependencies']:
            raise ValueError('평가 중 의존성 버전이 변경됐습니다.')
        from local_rag.io import verify_saved_run
        verify_saved_run(self.root)

    def plan_summary(self):
        return [{'표본': s['sample_id'], '구분': s['split'], '문서명': s['file_name'],
            '청크': self.plans[s['sample_id']]['chunk_count'],
            '캐시없을때_최대요청': self.plans[s['sample_id']]['map_calls'],
            '입력byte': self.plans[s['sample_id']]['total_request_bytes'],
            '문서별_입력상한': self.configs[s['sample_id']]['generation']['max_total_input_bytes']}
            for s in self.samples]

    def _component(self, sample, enabled):
        from rfp_phase5.runtime import make_components
        env = dict(self.env)
        env['phase5_config'] = copy.deepcopy(self.configs[sample['sample_id']])
        return make_components(env, doc_ids=[sample['doc_id']], enabled=enabled)

    def _check(self, result, sample):
        sid = sample['sample_id']
        return technical_check(result, sample, self.env, self.plans[sid], self.request_fps[sid])

    def record_result(self, sample, result, *, origin, session_usage=None):
        from rfp_phase5.review import answer_fingerprint
        from rfp_phase5.quality import audit_result
        sid = sample['sample_id']
        case = self.folder / sid
        if (case / 'result.json').exists():
            raise FileExistsError('기존 결과는 덮어쓰거나 자동 재시도하지 않습니다: ' + sid)
        if case.exists() and any(p.name != 'started.json' for p in case.iterdir()):
            raise RuntimeError('기록이 중단된 폴더입니다. 파일을 보존하고 새 SUITE_ID를 사용하세요: ' + sid)
        case.mkdir(parents=True, exist_ok=True)
        write_json(case / 'result.json', result)
        usage = dict(session_usage or {})
        write_json(case / 'origin.json', {'origin': origin, 'suite_fingerprint': self.fingerprint,
            'answer_fingerprint': answer_fingerprint(result),
            'usage_in_this_suite': usage, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
        (case / 'answer.md').write_text(str(result.get('markdown', '생성 답변 없음')) + '\n', encoding='utf-8')
        write_json(case / 'surface_audit.json', audit_result(result))
        self.assess_case(sample, reviewer='')

    def assess_case(self, sample, reviewer):
        from rfp_phase5.review import answer_fingerprint
        case = self.folder / sample['sample_id']
        if not (case / 'result.json').is_file():
            return {'sample_id': sample['sample_id'], 'split': sample['split'], 'accepted': False,
                    'status': 'not_generated', 'technical_passed': False}
        result = read_json(case / 'result.json')
        origin = read_json(case / 'origin.json')
        if origin['suite_fingerprint'] != self.fingerprint or origin['answer_fingerprint'] != answer_fingerprint(result):
            raise ValueError('평가 결과 또는 실행 기록이 변경됐습니다: ' + sample['sample_id'])
        technical = self._check(result, sample)
        write_json(case / 'technical_check.json', technical)
        source_audit = {}
        if technical['passed']:
            from rfp_phase5.source_audit import audit_summary_sources
            source_audit = audit_summary_sources(self.env['index'], self.env['registry'], result)
            write_json(case / 'source_audit.json', source_audit)
            expected_rows = make_review_rows(result, sample, self.references, source_audit)
            write_review(case / 'comparison.csv', expected_rows)
            human = assess_csv(case / 'comparison.csv', expected_rows, reviewer)
        else:
            human = {'passed': False, 'status': 'technical_failed_before_review', 'reviewed': 0, 'total': 33}
        write_json(case / 'human_review_assessment.json', human)
        verdict = {**case_verdict(technical, human), 'sample_id': sample['sample_id'], 'split': sample['split'],
            'model_status': result.get('status'), 'execution_status': result.get('execution_status'),
            'validated_chunk_coverage': result.get('validated_chunk_coverage'),
            'reviewed_judgements': human.get('reviewed', 0), 'judgements': 33,
            'source_audit_candidates': len(source_audit.get('candidates', [])),
            'citation_gap_review_count': sum(len(d.get('omitted_source_gaps', [])) for b in result.get('batch_outcomes', [])
                for d in b.get('citation_diagnostics', []) if d.get('requires_conditions_review')),
            'result_path': str(case / 'result.json'), 'review_path': str(case / 'comparison.csv')}
        write_json(case / 'acceptance.json', verdict)
        return verdict

    def import_existing(self, split='dev'):
        """Reuse only fully revalidated results of the identical current requests."""
        self.assert_frozen()
        paths = list((self.root / 'phase5_outputs/runs').glob('*/result.json'))
        for previous in (self.root / 'phase5_outputs/final_validation').glob('*/'):
            if previous.resolve() != self.folder.resolve():
                paths.extend(previous.glob('*/result.json'))
        paths = sorted(set(paths), reverse=True)
        imported = []
        for sample in [s for s in self.samples if s['split'] == split]:
            if (self.folder / sample['sample_id'] / 'result.json').exists():
                continue
            for path in paths:
                try:
                    result = read_json(path)
                    if result.get('doc_id') != sample['doc_id'] or not self._check(result, sample)['passed']:
                        continue
                except (ValueError, TypeError, KeyError):
                    continue
                self.record_result(sample, result, origin='reused_existing_exact_requests:' + str(path))
                imported.append(sample['sample_id'])
                break
        return imported

    def used_budget(self):
        calls, size = 0, 0
        for sample in self.samples:
            path = self.folder / sample['sample_id'] / 'origin.json'
            if path.is_file():
                usage = read_json(path).get('usage_in_this_suite', {})
                calls += usage.get('api_calls', 0)
                size += usage.get('attempted_input_bytes', 0)
        return {'api_calls': calls, 'attempted_input_bytes': size}

    def run_stage(self, split, *, enabled=False, reviewer=''):
        if split not in {'dev', 'final'} or type(enabled) is not bool:
            raise ValueError('실행 구분 또는 API 실행 옵션이 올바르지 않습니다.')
        self.assert_frozen()
        current = self.report(reviewer)
        if split == 'final' and not current['dev_passed']:
            return {'status': 'blocked_until_dev_review_passes', 'api_calls': 0,
                    'instruction': '개발 5건의 comparison.csv를 원문과 대조하고 검토자 이름을 입력하세요.'}
        self.import_existing(split)
        existing_failures = [c['sample_id'] for c in current.get('cases', []) if c['split'] == split
            and c.get('status') == 'technical_failed']
        if existing_failures:
            return {'status': 'blocked_by_existing_technical_failure', 'api_calls': 0,
                    'samples': existing_failures, 'automatic_retry': False}
        selected = [s for s in self.samples if s['split'] == split and
                    not (self.folder / s['sample_id'] / 'result.json').exists()]
        interrupted = [s['sample_id'] for s in selected if (self.folder / s['sample_id'] / 'started.json').exists()]
        if interrupted:
            return {'status': 'interrupted_attempt_requires_review', 'api_calls': 0,
                    'samples': interrupted, 'automatic_retry': False}
        if not enabled:
            return {'status': 'dry_run', 'api_calls': 0, 'missing_samples': [s['sample_id'] for s in selected],
                    'instruction': 'API 실행 옵션을 켜면 아직 결과가 없는 문서만 실행합니다.'}
        # Preflight every selected document before the first API request.
        prepared = []
        for sample in selected:
            component = self._component(sample, enabled=True)
            p = self.plans[sample['sample_id']]
            requests = [component['summarizer']._inner._request(p['question'], b['chunks']) for b in p['batches']]
            reservation = component['summarizer']._adapter.preflight_requests(requests)
            prepared.append((sample, component, reservation))
        used = self.used_budget()
        additional_calls = sum(p[2]['uncached_requests'] for p in prepared)
        additional_bytes = sum(p[2]['input_bytes_to_reserve'] for p in prepared)
        if used['api_calls'] + additional_calls > self.suite_calls or used['attempted_input_bytes'] + additional_bytes > self.suite_bytes:
            raise ValueError('검증 전체의 호출·입력 상한을 넘습니다. API 호출 없이 중단했습니다.')
        for sample, component, reservation in prepared:
            self.assert_frozen()
            print('요약 실행:', sample['sample_id'], sample['file_name'], flush=True)
            write_json(self.folder / sample['sample_id'] / 'started.json', {
                'suite_fingerprint': self.fingerprint, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': '완료 기록이 없으면 사용량을 확정할 수 없습니다. 자동 재시도하지 않습니다.'})
            try:
                result = component['summarizer'].summarize(sample['doc_id'])
            except Exception as error:
                from rfp_phase4.llm import safe_llm_diagnostic
                result = {'doc_id': sample['doc_id'], 'status': 'error', 'execution_status': 'incomplete',
                    'answer_generated': False, 'claims': [], 'fields': [],
                    'errors': ['unexpected_runner_exception:' + type(error).__name__],
                    'error_diagnostics': [safe_llm_diagnostic(error)],
                    'usage': {}, 'markdown': '요약을 완료하지 못했습니다. 진단 기록을 확인하세요.'}
            self.record_result(sample, result, origin='current_suite_generation',
                               session_usage=component['llm'].usage_report())
            technical = self._check(result, sample)
            print('실행:', result.get('execution_status'), '/ 출처·전체 입력:', technical['passed'], flush=True)
            if not technical['passed']:
                return {'status': 'stopped_after_technical_failure', 'sample_id': sample['sample_id'],
                        'errors': result.get('errors'), 'error_detail': result.get('error_detail'),
                        'error_diagnostics': result.get('error_diagnostics'),
                        'failed_checks': technical['failed_checks'], 'automatic_retry': False}
        return {'status': 'stage_finished', 'samples_generated': [p[0]['sample_id'] for p in prepared],
                'used_budget': self.used_budget(), 'human_review_pending': True}

    def report(self, reviewer=''):
        cases = [self.assess_case(s, reviewer) for s in self.samples]
        dev_passed = all(c['accepted'] for c in cases if c['split'] == 'dev')
        final_passed = all(c['accepted'] for c in cases if c['split'] == 'final')
        tests_path = self.folder / 'offline_tests.json'
        offline = read_json(tests_path) if tests_path.is_file() else {}
        tests_passed = offline.get('passed') is True and offline.get('suite_fingerprint') == self.fingerprint
        passed = dev_passed and final_passed and tests_passed
        report = {'version': VERSION, 'suite_fingerprint': self.fingerprint,
            'status': 'passed_sample_validation' if passed else 'validation_pending_or_failed',
            'phase5_sample_validation_passed': passed, 'dev_passed': dev_passed,
            'final_passed': final_passed, 'offline_failure_checks_passed': tests_passed,
            'scope': '고정된 공개 RFP 7건·77항목의 요약 검증. 전체 99건·Q&A 전체 성능을 확정하지 않습니다.',
            'all_documents_quality_verified': False, 'cases': cases,
            'budget_this_suite': self.used_budget(), 'output_folder': str(self.folder)}
        write_json(self.folder / 'final_report.json', report)
        return report

    def run_offline_tests(self):
        """Real rejection paths, with sockets blocked. Never calls a model service."""
        test_root = self.root / 'tests'
        validation_tests = Path(__file__).parent / 'test_final_validation.py'
        cache_tests = Path(__file__).parent / 'test_cache_revalidation.py'
        selected = ['test_phase5_summary.py', 'test_phase5_llm_diagnostics.py',
                    'test_phase5_saved_diagnostics.py', 'test_phase5_source_audit.py',
                    'test_phase4_citation_recovery.py', 'test_phase4_citation_split.py']
        program = '''import importlib.util,json,pathlib,sys,unittest
from unittest.mock import patch
root=pathlib.Path(sys.argv[1]); paths=json.loads(sys.argv[2])
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tests'))
suite=unittest.TestSuite()
for i,p in enumerate(paths):
 spec=importlib.util.spec_from_file_location('validation_case_'+str(i),p)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
with patch('socket.socket',side_effect=AssertionError('network prohibited in offline tests')):
 result=unittest.TextTestRunner(verbosity=1).run(suite)
print('VALIDATION_RESULT='+json.dumps({'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'api_calls':0}))
'''
        paths = [str(test_root / n) for n in selected] + [str(validation_tests), str(cache_tests)]
        if not all(Path(p).is_file() for p in paths):
            raise FileNotFoundError('09·10번 패치의 테스트 파일 또는 최종검증 테스트가 없습니다.')
        run = subprocess.run([sys.executable, '-c', program, str(self.root), json.dumps(paths)],
                             capture_output=True, text=True, timeout=90, cwd=self.root)
        lines = [l for l in run.stdout.splitlines() if l.startswith('VALIDATION_RESULT=')]
        if not lines:
            raise RuntimeError('오프라인 테스트를 완료하지 못했습니다. 현재 커널과 09·10번 패치를 확인하세요.')
        result = json.loads(lines[-1].split('=', 1)[1])
        result['passed'] = result['passed'] and run.returncode == 0 and result['skipped'] == 0
        result['suite_fingerprint'] = self.fingerprint
        write_json(self.folder / 'offline_tests.json', result)
        (self.folder / 'offline_tests.log').write_text(run.stderr, encoding='utf-8')
        return result

    def export(self, reviewer=''):
        self.assert_frozen()
        self.report(reviewer)
        target = self.folder.parent / (self.folder.name + '_results.zip')
        # Only this suite's explicit records; no .env, keys, response cache, or corpus.
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.folder.rglob('*')):
                if p.is_file() and p.suffix in {'.json', '.csv', '.md', '.log'}:
                    z.write(p, self.folder.name + '/' + p.relative_to(self.folder).as_posix())
        return target
