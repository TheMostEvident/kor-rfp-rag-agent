"""Whole-corpus audit, optional new-file intake, and incremental generation."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--parser-root',type=Path)
    p.add_argument('--raw-directory',type=Path)
    p.add_argument('--input-directory',type=Path,help='새 HWP/PDF/HWPX 입력. 새 workspace에 파싱·색인합니다.')
    p.add_argument('--workspace',type=Path)
    p.add_argument('--reparse-raw',action='store_true',help='모든 고유 원본을 캐시 없이 재파싱해 저장 원문과 비교합니다.')
    p.add_argument('--enable-openai',action='store_true')
    p.add_argument('--allow-unreviewed-sources',action='store_true',help='OCR·부분 추출 경고를 확인한 뒤 해당 문서 생성도 명시적으로 허용합니다.')
    p.add_argument('--max-new-calls',type=int,default=20)
    p.add_argument('--max-total-calls',type=int)
    p.add_argument('--max-total-input-bytes',type=int)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--run-id',default='phase5_final_source_units')
    args=p.parse_args(argv)
    from rfp_phase5_corpus.intake import build_workspace
    from rfp_phase5.runtime import bootstrap_phase5
    from rfp_phase5_corpus.workflow import audit_corpus
    from rfp_phase5_corpus.execution import open_corpus_run
    from rfp_phase5_recovery.runner import load_api_key
    root=args.root.resolve()
    intake_path=root/'reports/intake_report.json'
    previous_intake=json.loads(intake_path.read_text(encoding='utf-8')) if intake_path.is_file() else {}
    parser_root=Path(args.parser_root or previous_intake.get('parser_root') or root.parent/'rfp_phase2_local_20261111').resolve()
    if args.input_directory:
        intake=build_workspace(root,args.input_directory,parser_root=parser_root,workspace=args.workspace)
        print(json.dumps(intake,ensure_ascii=False,indent=2));root=Path(intake['workspace'])
        raw_directory=root/'data/raw'
    else:raw_directory=args.raw_directory or (root/'data/raw' if previous_intake else parser_root/'data/raw')
    env=bootstrap_phase5(root)
    try:
        def progress(n,total,row):
            if n%10==0 or n==total or not row['read_success']:
                print(f"원본 [{n}/{total}] {row['file_name']} / 읽기={row['read_success']}",flush=True)
        audit=audit_corpus(env,parser_root=parser_root,raw_directory=raw_directory,
            reparse=args.reparse_raw,progress=progress,allow_unreviewed_sources=args.allow_unreviewed_sources)
        print(json.dumps({k:audit[k] for k in ('original_documents','canonical_documents',
            'readable_original_documents','blocked_original_documents','needs_review_original_documents',
            'unregistered_raw_sources','fresh_parsing_verified')},ensure_ascii=False,indent=2))
        runner=open_corpus_run(env,args.run_id,max_total_calls=args.max_total_calls,
            max_total_input_bytes=args.max_total_input_bytes)
        if args.enable_openai:
            if not audit['eligible_canonical_doc_ids']:raise ValueError('파싱·색인·검색 검사를 통과한 문서가 없습니다.')
            if args.max_new_calls>0:load_api_key(args.root,args.env_file)
            report=runner.run_corpus(audit['eligible_canonical_doc_ids'],allow_api=True,max_new_calls=args.max_new_calls)
            invocation=report.get('invocation')
            report=runner.summarize_corpus('all')
            report['invocation']=invocation
            from rfp_phase5_final.engine import write_json
            write_json(runner.folder/'corpus_execution.json',report)
        else:report=runner.summarize_corpus('all')
        print(json.dumps({k:report[k] for k in ('selected_original_documents','selected_canonical_documents',
            'planned_batches','completed_batches','failed_batches','awaiting_batches','completed_documents',
            'all_selected_generation_completed','budget')},ensure_ascii=False,indent=2))
        print('전체 원본별 검토표:',root/'phase5_outputs/corpus_audit/documents.csv')
        print('전체 문서 생성 상태:',runner.folder/'corpus_execution.json')
        print('추출 문자의 보존·검색 가능 여부와 원본 그림·숫자·표 의미 및 요약 정답은 구분합니다.')
        return 0 if audit['all_existing_documents_text_ready'] and audit['all_raw_files_registered'] else 2
    finally:env['index'].close()


if __name__=='__main__':raise SystemExit(main())
