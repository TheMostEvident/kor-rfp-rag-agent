"""Fast source-registration verification; no LLM, index rebuild or API calls."""
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--parser-root',type=Path)
    parser.add_argument('--raw-directory',type=Path)
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve()
    intake_path=root/'reports/intake_report.json'
    intake=json.loads(intake_path.read_text(encoding='utf-8')) if intake_path.is_file() else {}
    phase2=Path(args.parser_root or intake.get('parser_root') or root.parent/'rfp_phase2_local_20261111').resolve()
    raw=args.raw_directory or (root/'data/raw' if intake else phase2/'data/raw')
    from local_rag.io import load_documents,write_json
    from rfp_phase5_corpus.raw_audit import audit_raw_corpus
    report=audit_raw_corpus(load_documents(root),phase2,raw_directory=raw,reparse=False)
    out=root/'phase5_outputs/corpus_audit/raw_registration_check.json'
    write_json(out,report)
    success=(report['all_registered_sources_read'] and report['all_raw_hashes_match']
        and report['all_snapshots_compatible'] and report['unregistered_source_count']==0
        and not intake.get('failed_files',0))
    print(json.dumps({k:report[k] for k in ('registration_version','document_count','read_success_count',
        'read_failure_count','registered_inventory_files','registered_path_alias_count',
        'ignored_metadata_count','unregistered_source_count','unregistered_known_content_count',
        'all_raw_hashes_match','all_snapshots_compatible','api_calls')},ensure_ascii=False,indent=2))
    if report['unregistered_sources']:
        print('실제 미등록 후보 (최대10개):')
        print(json.dumps(report['unregistered_sources'][:10],ensure_ascii=False,indent=2))
    print('원본 등록 검증:', '통과' if success else '미통과 — 보고서의 누락·변경·신규 파일을 확인하세요.')
    print('진단 보고서:',out)
    print('새 요약 생성·API 호출·기존 인덱스 변경: 없음')
    return 0 if success else 2


if __name__=='__main__':raise SystemExit(main())
