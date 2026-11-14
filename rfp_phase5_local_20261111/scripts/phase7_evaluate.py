"""Run local Phase7 development measurements; live calls require --live."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--case-id',action='append',help='측정할 dev 사례 ID; 반복 가능')
    p.add_argument('--summary-run',type=Path,help='Phase5 content_documents가 있는 실행 폴더')
    p.add_argument('--live',action='store_true')
    p.add_argument('--approve-doc-id',action='append',default=[])
    p.add_argument('--max-calls',type=int,default=8)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--review-run',type=Path,help='저장된 report/human_review.json을 재집계; API 0회')
    args=p.parse_args(argv)
    from rfp_phase7.evidence import load_dev_cases
    from rfp_phase7.evaluate import run_evaluation,save_report
    from rfp_phase7.review import assess_review
    if args.review_run:
        folder=args.review_run.expanduser().resolve()
        assessed=assess_review(json.loads((folder/'report.json').read_text(encoding='utf-8')),
            json.loads((folder/'human_review.json').read_text(encoding='utf-8')))
        print(json.dumps(assessed,ensure_ascii=False,indent=2));return 0
    root=args.root.expanduser().resolve()
    settings=json.loads((root/'configs/phase7.json').read_text(encoding='utf-8'))
    cases=load_dev_cases(root/settings['development_questions'])
    if args.live and (not args.case_id or not args.approve_doc_id):
        p.error('--live에는 --case-id와 --approve-doc-id를 명시하세요.')
    from rfp_phase5.runtime import bootstrap_phase5
    env=bootstrap_phase5(root)
    try:
        if args.live:
            from rfp_phase5_recovery.runner import load_api_key
            load_api_key(root,args.env_file)
        report=run_evaluation(env,cases,profiles=['phase6_rrf'] if args.live else settings['offline_profiles'],
            enabled=args.live,case_ids=args.case_id,approved_doc_ids=args.approve_doc_id,
            max_calls=args.max_calls,max_input_bytes=settings['live_max_total_input_bytes'],
            summary_folder=args.summary_run,pricing=settings.get('pricing'),
            progress=lambda profile,cid:print(profile,cid,flush=True))
        folder=save_report(root,report)
        print(json.dumps({k:report[k] for k in ['status','coverage','metrics','usage','cost_estimate']},ensure_ascii=False,indent=2))
        print('평가 기록:',folder)
    finally:env['index'].close()
    return 0


if __name__=='__main__':raise SystemExit(main())
