"""Phase 1 데이터·정답 근거 검증 도구. API 호출 및 모델 채점은 수행하지 않습니다."""
import json
from collections import Counter
from pathlib import Path

FEATURE_COUNTS = {'fact': 25, 'summary': 20, 'comparison': 20, 'followup': 20, 'uncertainty': 15}
DEV_COUNTS = {'fact': 15, 'summary': 12, 'comparison': 12, 'followup': 12, 'uncertainty': 9}
SUMMARY_FIELDS = {'project_name', 'agency', 'purpose', 'scope_and_requirements', 'budget',
                  'duration', 'qualifications', 'submission_documents', 'submission_deadline',
                  'submission_method', 'evaluation_criteria'}
STATUSES = {'answered', 'partial', 'needs_clarification', 'insufficient_evidence', 'conflicting_evidence'}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_data(root, split='dev'):
    root = Path(root)
    if split not in {'dev', 'holdout', 'smoke.dev'}:
        raise ValueError('split은 dev, holdout, smoke.dev 중 하나입니다.')
    documents = json.loads((root / 'data/documents.json').read_text(encoding='utf-8'))
    sources = {d['doc_id']: d for d in read_jsonl(root / 'data/source_blocks.jsonl')}
    return documents, sources, read_jsonl(root / f'eval/questions.{split}.jsonl')


def validate_bundle(root):
    """정답의 구조·참조·인용 일치만 검사합니다. 의미 정확도나 RAG 성능 점수가 아닙니다."""
    root = Path(root)
    documents, sources, dev = load_data(root)
    holdout = read_jsonl(root / 'eval/questions.holdout.jsonl')
    cases = dev + holdout
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(len(documents) == 100, '원본 파일 수가 100이 아님')
    check(len(sources) == 100, '출처 스냅샷 파일 수가 100이 아님')
    docs = {d['doc_id']: d for d in documents}
    check(set(docs) == set(sources), 'manifest와 source doc_id 불일치')
    check(len({d['sha256'] for d in documents}) == 99, '고유 원문 수가 99가 아님')
    split_hashes = {split: {d['sha256'] for d in documents if d['split'] == split} for split in ['dev', 'holdout']}
    check(not split_hashes['dev'] & split_hashes['holdout'], '개발·최종 분할에 동일 원문이 겹침')
    check(len(cases) == 100, '평가 사례 수가 100이 아님')
    check(len({c['case_id'] for c in cases}) == len(cases), '사례 ID 중복')
    check(dict(Counter(c['feature'] for c in cases)) == FEATURE_COUNTS, '기능별 사례 수 불일치')
    check(dict(Counter(c['feature'] for c in dev)) == DEV_COUNTS, '개발용 기능별 사례 수 불일치')
    check(len(dev) == 60 and len(holdout) == 40, '사례 분할 60/40 불일치')
    check(all(c['split'] == 'dev' for c in dev) and all(c['split'] == 'holdout' for c in holdout), '파일과 사례 split 불일치')
    blocks = {doc_id: {b['block_id']: b for b in s['blocks']} for doc_id, s in sources.items()}
    evidence_count = 0
    fact_count = 0

    def evidence(ev, case):
        nonlocal evidence_count
        evidence_count += 1
        cid = case['case_id']
        did = ev.get('doc_id')
        check(did in case['target_doc_ids'], f'{cid}: 근거 문서가 대상 범위를 벗어남')
        if did not in sources:
            check(False, f'{cid}: 근거 doc_id 없음')
            return
        check(ev.get('sha256') == sources[did]['sha256'], f'{cid}: 원문 SHA256 불일치')
        block = blocks[did].get(ev.get('block_id'))
        if block is None:
            check(False, f'{cid}: 근거 block_id 없음')
            return
        check(ev.get('locator') == block['locator'], f'{cid}: 원문 위치 불일치')
        quote = ev.get('quote')
        check(isinstance(quote, str) and bool(quote) and quote in block['text'], f'{cid}: 근거 인용 불일치')
        check(isinstance(quote, str) and len(quote) <= 1000, f'{cid}: 근거 인용 과다')

    for case in cases:
        cid = case['case_id']
        feature = case['feature']
        check(case.get('expected_status') in STATUSES, f'{cid}: 응답 상태 오류')
        check(bool(case.get('question', '').strip()), f'{cid}: 질문 없음')
        check(bool(case.get('required_behaviors')), f'{cid}: 합격 동작 기준 없음')
        for did in case['target_doc_ids'] + case.get('context', {}).get('selected_doc_ids', []) + case.get('context', {}).get('candidate_doc_ids', []):
            check(did in docs, f'{cid}: 대상 문서 ID 없음')
            if did in docs:
                check(docs[did]['split'] == case['split'], f'{cid}: 문서 분할 위반')
        fact_ids = {f['fact_id'] for f in case.get('required_facts', [])}
        check(len(fact_ids) == len(case.get('required_facts', [])), f'{cid}: 사실 ID 중복')
        fact_count += len(fact_ids)
        for fact in case.get('required_facts', []):
            check(fact.get('doc_id') in case['target_doc_ids'], f'{cid}: 정답 문서 범위 오류')
            check(bool(fact.get('expected_value')), f'{cid}: 정답 값 없음')
            check(bool(fact.get('evidence')), f'{cid}: 정답 사실 근거 없음')
            for ev in fact.get('evidence', []):
                evidence(ev, case)
        for ev in case.get('case_basis_evidence', []):
            evidence(ev, case)
        if feature == 'summary':
            expectations = case.get('summary_field_expectations', {})
            check(set(expectations) == SUMMARY_FIELDS, f'{cid}: 요약 11개 고정 항목 불일치')
            present = {f['field'] for f in case.get('required_facts', [])}
            for key, expectation in expectations.items():
                check(expectation in {'must_include', 'not_established_from_reviewed_text'}, f'{cid}: 요약 항목 상태 오류')
                check((key in present) == (expectation == 'must_include'), f'{cid}: 요약 항목/사실 대응 불일치 {key}')
        if feature == 'comparison':
            check(len(case['target_doc_ids']) >= 2, f'{cid}: 비교 대상 부족')
            check(len({docs[did]['sha256'] for did in case['target_doc_ids']}) == len(case['target_doc_ids']), f'{cid}: 동일 원문 사본끼리 비교')
            pairs = {(f['doc_id'], f['field']) for f in case['required_facts']}
            fields = {f['field'] for f in case['required_facts']}
            check(all((did, field) in pairs for did in case['target_doc_ids'] for field in fields), f'{cid}: 비교표 필수 칸 누락')
        if feature == 'followup':
            turns = case.get('turns', [])
            check(2 <= len(turns) <= 4, f'{cid}: 후속 질문 턴 수 오류')
            for turn in turns:
                check(bool(turn.get('user', '').strip()), f'{cid}: 대화 질문 없음')
                check(turn.get('expected_status') in STATUSES, f'{cid}: 턴 응답 상태 오류')
                check(set(turn.get('required_fact_ids', [])) <= fact_ids, f'{cid}: 턴 사실 참조 오류')
                check(set(turn.get('expected_doc_ids', [])) <= set(case['target_doc_ids']), f'{cid}: 턴 문서 범위 오류')
                check(set(turn.get('candidate_doc_ids', [])) <= set(case['target_doc_ids']), f'{cid}: 턴 후보 문서 범위 오류')
                if turn.get('expected_status') == 'needs_clarification':
                    check(not turn.get('expected_doc_ids'), f'{cid}: 모호성 해소 전 활성 문서가 확정됨')
        fixture = case.get('context', {}).get('synthetic_fixture')
        if fixture:
            check(feature == 'uncertainty', f'{cid}: 합성 오류 사례 유형 오류')
            tool_errors = [r.get('error_code') for r in case.get('context', {}).get('tool_results', []) if r.get('status') == 'error']
            if tool_errors:
                check(case.get('expected_error_code') in tool_errors, f'{cid}: 합성 오류의 별도 오류 코드 불일치')
            else:
                check(case.get('expected_error_code') is None, f'{cid}: 정상 빈 검색을 처리 오류로 분류함')
    smoke = read_jsonl(root / 'eval/questions.smoke.dev.jsonl')
    check(len(smoke) == 10 and dict(Counter(c['feature'] for c in smoke)) == {k: 2 for k in FEATURE_COUNTS}, '빠른 점검용 10사례 구성 오류')
    check({c['case_id'] for c in smoke} <= {c['case_id'] for c in dev}, '빠른 점검에 최종 평가 사례 포함')
    return dict(valid=not errors, errors=errors, document_files=len(documents), unique_documents=len({d['sha256'] for d in documents}),
                cases=len(cases), dev_cases=len(dev), holdout_cases=len(holdout), facts=fact_count,
                evidence_links=evidence_count, feature_counts=dict(Counter(c['feature'] for c in cases)),
                content_disjoint=True if not split_hashes['dev'] & split_hashes['holdout'] else False,
                rag_performance_measured=False,
                check_scope='구조·참조·원문 인용 일치. 의미 지원 여부 검수와 RAG 성능 평가는 별도.')


def runtime_case(case):
    """평가 실행기용 입력. 대화 turns는 실행기가 한 턴씩 보내며 정답 기록은 보내지 않습니다."""
    result = {k: case[k] for k in ['case_id', 'feature', 'question', 'context'] if k in case}
    if 'turns' in case:
        result['turns'] = [{'turn_id': t['turn_id'], 'user': t['user']} for t in case['turns']]
    return result


if __name__ == '__main__':
    report = validate_bundle(Path(__file__).resolve().parent)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['valid'] else 1)
