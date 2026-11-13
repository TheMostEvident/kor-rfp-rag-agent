"""Phase 2 로컬 실행: 원본 파싱 → 근거 대조 → 품질·실행 기록 저장."""
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone, timedelta
from pathlib import Path

from local_ingest.loader import run_ingestion, write_json, sha256_file
from local_ingest.provenance import verify_evidence
from local_ingest.quality import audit_documents


def read_cases(root):
    cases = []
    for split in ['dev', 'holdout']:
        with (Path(root) / f'eval/questions.{split}.jsonl').open(encoding='utf-8') as handle:
            cases.extend(json.loads(line) for line in handle if line.strip())
    return cases


def record_run(root, documents, quality, evidence):
    root = Path(root).resolve()
    packages = {}
    for name in ['pypdf', 'PyMuPDF']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        'created_at': datetime.now(timezone(timedelta(hours=9), name='Asia/Seoul')).isoformat(),
        'source_files': len(documents), 'unique_contents': quality['unique_contents'],
        'parser_module_sha256': {name: sha256_file(root / 'local_ingest' / (name + '.py')) for name in ['hwp', 'hwpx', 'pdf']},
        'code_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted((root / 'local_ingest').glob('*.py'))},
        'entrypoint_sha256': sha256_file(root / 'run_phase2.py'),
        'source_manifest_sha256': sha256_file(root / 'data/input_manifest.json'),
        'source_annotation_files_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted((root / 'data').glob('*.json')) if p.name != 'input_manifest.json'},
        'processed_files_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted((root / 'data/processed').glob('*.jsonl'))},
        'evaluation_files_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted((root / 'eval').glob('questions.*.jsonl'))},
        'python_version': platform.python_version(), 'dependencies': packages,
        'local_processing': True, 'openai_api_calls': 0, 'remote_file_mutations': 0,
        'evidence_links_checked': evidence['total'], 'phase2_implementation_ready': quality['phase2_implementation_ready'],
        'not_a_rag_performance_measurement': True,
    }
    write_json(root / 'reports/run_manifest.json', manifest)
    return manifest


def run(root, force=False, preparsed_dir=None, progress=None):
    root = Path(root).resolve()
    documents = run_ingestion(root, force=force, preparsed_dir=preparsed_dir, progress=progress)
    evidence = verify_evidence(documents, read_cases(root))
    write_json(root / 'reports/evidence_validation.json', evidence)
    write_json(root / 'reports/evidence_summary.json', {k: v for k, v in evidence.items() if k not in ['records', 'cases']})
    quality = audit_documents(documents, evidence, root=root)
    record_run(root, documents, quality, evidence)
    return documents, evidence, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--force', action='store_true', help='내용·파서가 같은 캐시도 사용하지 않고 원본 재파싱')
    # 개발 중 새 원본을 동일 코드로 이미 처리한 결과만 모듈·옵션 해시 대조 후 재사용합니다.
    parser.add_argument('--preparsed-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    def progress(n, total, document):
        if n % 10 == 0 or document['status'] == 'failed' or n == total:
            print(f"[{n}/{total}] {document['status']}: {document['file_name']}", flush=True)

    documents, evidence, quality = run(args.root, args.force, args.preparsed_dir, progress)
    print(json.dumps({'files': len(documents), 'successful_files': quality['successful_files'],
                      'failed_files': quality['failed_files'], 'evidence_total': evidence['total'],
                      'exact_matches': evidence['exact_matches'], 'normalized_matches': evidence['normalized_matches'],
                      'unmatched': evidence['unmatched'], 'phase2_implementation_ready': quality['phase2_implementation_ready']},
                     ensure_ascii=False, indent=2))
    return 0 if quality['phase2_implementation_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
