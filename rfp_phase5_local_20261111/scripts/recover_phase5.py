"""Audit a frozen run, or explicitly retry its failed jobs (default: 0 API calls)."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description='기존 Phase 5 실행을 보존하며 실패 배치만 복구합니다.')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--run-id',default='phase5_final_source_units')
    parser.add_argument('--retry-failed',action='store_true',help='기록된 실패 작업만 재시도합니다.')
    parser.add_argument('--job',action='append',help='재시도 작업 키. 여러 번 지정할 수 있습니다.')
    parser.add_argument('--enable-openai',action='store_true',help='실제 API 재시도를 명시적으로 허용합니다.')
    parser.add_argument('--retry-tag',default='phase5_recovery_v1')
    parser.add_argument('--max-new-calls',type=int,default=1)
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--reviewer',default='')
    parser.add_argument('--resume-after-service-error',action='store_true',help='인증·잔액·연결 문제 해결 후 명시적으로 재개합니다.')
    args=parser.parse_args(argv)
    retry=bool(args.retry_failed or args.job)
    if args.enable_openai and not retry:parser.error('--enable-openai는 --retry-failed 또는 --job과 함께 사용하세요.')
    from rfp_phase5.runtime import bootstrap_phase5
    from rfp_phase5_recovery.runner import open_existing_run,load_api_key
    app=bootstrap_phase5(args.root)
    try:
        runner=open_existing_run(app,args.run_id)
        audit=runner.audit()
        print(json.dumps({k:audit[k] for k in ('revision','completed_jobs','failed_jobs',
            'rejected_claim_count','rejection_reason_counts','budget','api_calls_by_audit')},ensure_ascii=False,indent=2))
        if not retry:
            print('진단만 완료했습니다. 추가 API 호출: 0회')
            print('진단 보고서:',runner.folder/'recovery_audit.json')
            return 0
        jobs=args.job if args.job is not None else [x['job_key'] for x in audit['failed_jobs']]
        plan=runner.retry_failed(jobs,retry_tag=args.retry_tag,max_new_calls=args.max_new_calls,allow_api=False)
        if not args.enable_openai or not plan['selected_failed_jobs']:
            print(json.dumps(plan,ensure_ascii=False,indent=2))
            print('재시도 미실행. API를 허용하려면 --enable-openai를 함께 지정하세요.'
                if plan['selected_failed_jobs'] else '새로 재시도할 작업이 없습니다. 추가 API 호출: 0회')
            return 0
        load_api_key(args.root,args.env_file)
        execution=runner.retry_failed(jobs,retry_tag=args.retry_tag,max_new_calls=args.max_new_calls,allow_api=True,
            resume_after_service_error=args.resume_after_service_error)
        print(json.dumps(execution,ensure_ascii=False,indent=2))
        latest=runner.audit()
        print('남은 실패 배치:',len(latest['failed_jobs']))
        selected=[s['doc_id'] for s in runner.references.get('samples',[])]
        archive=runner.export(selected,args.reviewer)
        print('검토·결과 ZIP:',archive)
        print('최종 보고서:',runner.folder/'final_report.json')
        print('생성 성공 여부와 사람의 내용 검토 완료 여부는 별도입니다.')
        return 2 if latest['failed_jobs'] else 0
    finally:
        app['index'].close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as error:
        # Internal error messages contain fixed guidance/paths, never SDK exception bodies.
        from rfp_phase5_recovery.client import redact
        print(redact(str(error)),file=sys.stderr)
        raise SystemExit(2)
