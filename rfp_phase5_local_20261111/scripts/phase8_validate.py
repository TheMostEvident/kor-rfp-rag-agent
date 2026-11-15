"""Run the bounded Phase 8 exercise. Default: offline, zero API calls."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--doc-id', action='append', default=[])
    parser.add_argument('--keyword', action='append', default=[], help='명시한 검색어마다 문서가 한 건으로 확인되어야 함; 최대 두 건')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--approve-doc-id', action='append', default=[])
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--report-session', type=Path)
    parser.add_argument('--diagnostics-only', action='store_true', help='저장된 검증의 진단만 조회; API 0회')
    args = parser.parse_args(argv)
    from rfp_phase8.validation import run_validation, validation_status
    if args.diagnostics_only and not args.report_session:
        parser.error('--diagnostics-only에는 --report-session이 필요합니다.')
    if args.report_session:
        report = validation_status(args.report_session, include_reviews=not args.diagnostics_only)
        print(json.dumps(report['diagnostics'] if args.diagnostics_only else report,
                         ensure_ascii=False, indent=2))
        return 0
    from rfp_phase5.runtime import bootstrap_phase5
    from rfp_phase6.session import ConversationSession
    env = bootstrap_phase5(args.root)
    try:
        catalog = ConversationSession(env['documents'])
        ids = list(args.doc_id)
        for keyword in args.keyword:
            candidates = catalog.list_documents(keyword)
            if len(candidates) != 1:
                parser.error('각 --keyword는 대표 문서 한 건과 일치해야 합니다. --doc-id로 명시하세요.')
            ids.append(candidates[0]['doc_id'])
        if not ids:
            if args.live:
                parser.error('--live에는 검증할 문서를 --doc-id 또는 --keyword로 지정하세요.')
            ids = [d['doc_id'] for d in catalog.list_documents()[:2]]
        ids = env['registry'].resolve_doc_ids(ids)
        # --keyword is explicit document selection, never a hidden gold target.
        approved = args.approve_doc_id or ids
        print('검증 대상 문서:', ids, flush=True)
        if args.live:
            if not set(ids).issubset(env['registry'].resolve_doc_ids(approved)):
                parser.error('검증 대상 모두를 API 허용 목록에 포함하세요.')
            from rfp_phase5_recovery.runner import load_api_key
            load_api_key(env['root'], args.env_file)
        folder, report = run_validation(env, ids, enabled=args.live, approved_doc_ids=approved if args.live else [])
        print('검증 기록:', folder)
        if args.live:
            from rfp_phase8.pilot import read_record
            for row in report['rows']:
                record = read_record(folder / 'turns' / row['turn_id'])
                print('\n질문:', record['question'])
                print(record['result'].get('markdown', ''))
                print('상태:', record['result']['status'], '/', record['result'].get('execution_status'))
                flagged = (record['result'].get('claim_citation_audit') or {}).get('flagged_claim_indices', [])
                if flagged:
                    print('인용 연결 검토 신호: 문장', ', '.join(str(i + 1) for i in flagged),
                          '— 명칭 누락 가능성을 해당 문장의 인용과 대조하세요.')
                errors = (record['result'].get('validation') or {}).get('errors', [])
                if errors:
                    print('출처·형식 검사 사유:', json.dumps(errors, ensure_ascii=False))
                print('검토 기준:', row['review_hint'])
                print('검토 파일:', folder / 'turns' / row['turn_id'] / 'human_review.json')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['mechanical_checks_passed'] else 1
    finally:
        env['index'].close()


if __name__ == '__main__':
    raise SystemExit(main())
