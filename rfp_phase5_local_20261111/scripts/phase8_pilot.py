"""Phase 8 local pilot. Default: readiness only, zero API calls."""
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
    parser.add_argument('--serve', action='store_true', help='127.0.0.1 로컬 파일럿 화면')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--keyword', default='')
    parser.add_argument('--doc-id', action='append', default=[])
    parser.add_argument('--question', action='append', default=[])
    parser.add_argument('--live', action='store_true', help='명시한 문서의 API 생성; 기본 꺼짐')
    parser.add_argument('--approve-doc-id', action='append', default=[])
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--report-session', type=Path, help='이 세션의 최신 사람 검토를 다시 집계; API 0회')
    parser.add_argument('--diagnostics-only', action='store_true', help='저장된 오류 상세만 조회; API 0회')
    args = parser.parse_args(argv)
    from rfp_phase8.pilot import PilotSession, readiness, session_report, session_diagnostics
    if args.diagnostics_only and not args.report_session:
        parser.error('--diagnostics-only에는 --report-session이 필요합니다.')
    if args.report_session:
        report = session_diagnostics(args.report_session) if args.diagnostics_only else session_report(args.report_session)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    from rfp_phase5.runtime import bootstrap_phase5
    env = bootstrap_phase5(args.root)
    try:
        print(json.dumps(readiness(env), ensure_ascii=False, indent=2))
        if not (args.serve or args.list or args.question):
            return 0
        pilot = PilotSession(env)
        if args.doc_id:
            pilot.select(args.doc_id)
        if args.live:
            approved = args.approve_doc_id or args.doc_id
            if not approved:
                parser.error('--live에는 --doc-id 또는 --approve-doc-id가 필요합니다.')
            from rfp_phase5_recovery.runner import load_api_key
            # Check identity first, then load the existing key without printing it.
            env['registry'].resolve_doc_ids(approved)
            load_api_key(pilot.root, args.env_file)
            pilot.configure_generation(enabled=True, approved_doc_ids=approved)
        if args.list:
            print(json.dumps(pilot.chat.list_documents(args.keyword), ensure_ascii=False, indent=2))
        for question in args.question:
            record = pilot.submit(question)
            print(record['result'].get('markdown', ''))
            print('상태:', record['result']['status'], '/', record['result'].get('execution_status'))
            if record['result'].get('error_diagnostic'):
                print('오류 상세:', json.dumps(record['result']['error_diagnostic'], ensure_ascii=False))
            errors = (record['result'].get('validation') or {}).get('errors', [])
            if errors:
                print('출처·형식 검사 사유:', json.dumps(errors, ensure_ascii=False))
            print('실행 ID:', record['turn_id'])
            if pilot.halted:
                break
        print('실행 기록:', pilot.folder, flush=True)
        if args.serve:
            from rfp_phase8.web import create_server
            server = create_server(pilot, port=args.port)
            print(f'파일럿 화면: http://127.0.0.1:{server.server_port}', flush=True)
            print('종료: Ctrl+C. 브라우저 새로고침은 질문을 재실행하지 않습니다.', flush=True)
            try:
                server.serve_forever(poll_interval=.2)
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        report = session_report(pilot.folder)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report['failed_turns'] or report['incomplete_turns'] else 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print('파일럿 설정·실행 기록을 확인하세요:', type(error).__name__, file=sys.stderr)
        return 2
    finally:
        env['index'].close()


if __name__ == '__main__':
    raise SystemExit(main())
