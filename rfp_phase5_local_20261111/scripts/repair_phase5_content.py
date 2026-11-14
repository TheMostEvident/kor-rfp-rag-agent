"""Rebuild saved Phase 5 summaries from original evidence without API calls."""
from pathlib import Path
import argparse
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-id', default='phase5_final_source_units')
    parser.add_argument('--run-folder', type=Path, help='별도 경로의 기존 실행 결과 폴더')
    parser.add_argument('--doc-id', action='append', dest='doc_ids', help='원본/대표 문서 ID, 여러 번 지정 가능')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.run_id):
        parser.error('--run-id는 영문·숫자·밑줄·하이픈으로 지정하세요.')
    from rfp_phase5.runtime import bootstrap_phase5
    from rfp_phase5_content.storage import repair_saved_results
    root = args.root.expanduser().resolve()
    folder = (args.run_folder.expanduser().resolve() if args.run_folder else
              root/'phase5_outputs/final_source_units'/args.run_id)
    if not folder.is_dir():
        parser.error('기존 실행 결과 폴더를 찾을 수 없습니다.')
    env = bootstrap_phase5(root)
    try:
        report = repair_saved_results(env, folder, args.doc_ids)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print('새 요약:', report['output_folder'])
        print('기존 모델 결과·검토표·API 사용량: 보존 / 추가 API 호출: 0회')
        return 0 if report['failed_documents']==0 and report['selected_saved_documents']>0 else 2
    finally:
        env['index'].close()


if __name__ == '__main__':
    raise SystemExit(main())
