"""Read new HWP/PDF/HWPX files and build an isolated, versioned RAG workspace.

The active index, source snapshots, run ledger and response caches are never
overwritten. Failed files remain explicit report entries, not hidden exclusions.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

from local_rag.chunking import build_chunks
from local_rag.quality import validate_chunks
from local_rag.retrieval import LocalSearchIndex
from local_rag.io import write_json,write_jsonl,sha256_file,verify_saved_run


VERSION='rfp-whole-corpus-intake-v1'


def load_parser(parser_root):
    parser_root=Path(parser_root).expanduser().resolve()
    if not (parser_root/'local_ingest/loader.py').is_file():
        raise FileNotFoundError('Phase 2 local_ingest/loader.py가 있는 폴더를 지정하세요.')
    existing=sys.modules.get('local_ingest')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(parser_root):
        raise RuntimeError('다른 버전의 local_ingest가 로드됐습니다. 커널을 재시작하세요.')
    versions={p.name:sha256_file(p) for p in (parser_root/'local_ingest').glob('*.py')}
    if existing is not None and getattr(existing,'_corpus_parser_versions',None)!=versions:
        raise RuntimeError('메모리에 남은 파서 버전을 확인할 수 없습니다. 커널을 재시작하세요.')
    if str(parser_root) not in sys.path:sys.path.insert(0,str(parser_root))
    module=importlib.import_module('local_ingest.loader')
    sys.modules['local_ingest']._corpus_parser_versions=versions
    return module


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':')).encode('utf-8')).hexdigest()


def build_workspace(project_root,raw_directory,*,parser_root=None,workspace=None):
    """Parse every supported input, index successful text, report every failed input.

    Returns a manifest with a new project root. Reusing an identical completed
    workspace verifies its stored files and performs no rebuild or API request.
    """
    project_root=Path(project_root).expanduser().resolve()
    raw_directory=Path(raw_directory).expanduser().resolve()
    if not raw_directory.is_dir():raise FileNotFoundError('입력 문서 폴더가 없습니다.')
    parser_root=Path(parser_root or project_root.parent/'rfp_phase2_local_20261111').resolve()
    loader=load_parser(parser_root)
    sources=[];ignored=[]
    for path in sorted(p for p in raw_directory.rglob('*') if p.is_file()):
        if any(part.startswith('.') for part in path.relative_to(raw_directory).parts):
            ignored.append(path.relative_to(raw_directory).as_posix());continue
        if not path.resolve().is_relative_to(raw_directory):
            raise ValueError('입력 폴더 밖을 가리키는 파일 링크는 읽지 않습니다.')
        rel=path.relative_to(raw_directory).as_posix()
        sources.append({'relative_path':rel,'sha256':sha256_file(path),'size_bytes':path.stat().st_size})
    if not sources:raise ValueError('입력 폴더에 처리할 문서가 없습니다.')
    code={p.name:sha256_file(p) for p in sorted((parser_root/'local_ingest').glob('*.py'))}
    config=json.loads((project_root/'configs/default.json').read_text(encoding='utf-8'))
    runtime={p.relative_to(project_root).as_posix():sha256_file(p)
        for name in ('local_rag','rfp_phase4','rfp_phase5','rfp_phase5_final')
        for p in sorted((project_root/name).glob('*.py'))}
    annotations_path=parser_root/'data/reviewed_transcriptions.json'
    annotations=json.loads(annotations_path.read_text(encoding='utf-8')) if annotations_path.is_file() else []
    contract={'version':VERSION,'builder_sha256':sha256_file(__file__),'sources':sources,'parser_code_sha256':code,
              'runtime_code_sha256':runtime,'configuration':config,'reviewed_annotations':annotations,
              'source_metadata_sha256':sha256_file(project_root/'data/source_documents.jsonl')
                if (project_root/'data/source_documents.jsonl').is_file() else None}
    fingerprint=_fingerprint(contract)
    target=Path(workspace).expanduser().resolve() if workspace else (
        project_root/'phase5_outputs/corpus_workspaces'/fingerprint[:20])
    if (target==project_root or project_root.is_relative_to(target)
            or target==raw_directory or raw_directory.is_relative_to(target)):
        raise ValueError('기존 프로젝트나 입력 폴더를 작업공간으로 덮어쓸 수 없습니다.')
    if target.is_relative_to(raw_directory):raise ValueError('작업공간을 입력 폴더 안에 만들 수 없습니다.')
    report_path=target/'reports/intake_report.json'
    if report_path.is_file():
        old=json.loads(report_path.read_text(encoding='utf-8'))
        if old.get('input_fingerprint')!=fingerprint:raise ValueError('다른 입력 버전의 작업공간입니다. 새 경로를 사용하세요.')
        verify_saved_run(target)
        return dict(old,reused_existing_workspace=True)
    if target.exists() and any(target.iterdir()):
        raise ValueError('비어 있지 않은 미완료 작업공간입니다. 새 경로를 지정하세요.')
    target.mkdir(parents=True,exist_ok=True)
    for name in ('local_rag','rfp_phase4','rfp_phase5','rfp_phase5_final','rfp_phase5_recovery','rfp_phase5_corpus'):
        source=project_root/name
        if source.is_dir():shutil.copytree(source,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(project_root/'configs',target/'configs')
    (target/'data/raw').mkdir(parents=True)
    known={}
    old_sources=project_root/'data/source_documents.jsonl'
    if old_sources.is_file():
        with old_sources.open(encoding='utf-8') as handle:
            for line in handle:
                d=json.loads(line)
                known.setdefault(d.get('sha256'),[]).append({k:d.get(k) for k in
                    ('doc_id','file_name','agency','document_year','source_url','canonical_doc_id')})
    entries=[];used_ids=set()
    for source in sources:
        rel=source['relative_path'];copied=target/'data/raw'/rel
        copied.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw_directory/rel,copied)
        # Recheck the copied bytes; a file changing during intake never silently enters the index.
        if sha256_file(copied)!=source['sha256']:raise ValueError('복사 중 원본 파일이 변경됐습니다: '+rel)
        matches=[r for r in known.get(source['sha256'],[]) if r['doc_id'] not in used_ids]
        old=next((r for r in matches if r['file_name']==copied.name),None)
        if old is None:old=next((r for r in matches if r['doc_id']==r['canonical_doc_id']),None)
        did=old['doc_id'] if old else 'local_'+source['sha256'][:16]+'_'+_fingerprint(rel)[:6]
        used_ids.add(did)
        entries.append({'doc_id':did,
            'name':copied.name,'relative_path':copied.relative_to(target).as_posix(),
            'sha256':source['sha256'],'agency':old['agency'] if old else None,
            'document_year':old['document_year'] if old else None,'split':'unassigned',
            'source_url':old['source_url'] if old else None})
    write_json(target/'data/input_manifest.json',entries)
    approved={(e['doc_id'],e['sha256']) for e in entries}
    exact_annotations=[a for a in annotations if (a.get('doc_id'),a.get('sha256')) in approved]
    write_json(target/'data/reviewed_transcriptions.json',exact_annotations)
    documents=loader.run_ingestion(target,manifest=entries,force=False,extract_pdf_tables=True)
    for document in documents:
        if document.get('status')!='failed' and not any(
                str(b.get('text','')).strip() for b in document.get('blocks',[])):
            document['status']='failed'
            document['error']={'code':'no_extractable_primary_text',
                               'message':'색인할 주 원문이 없습니다. 참조 보기만으로 성공 처리하지 않습니다.'}
    good=[d for d in documents if d.get('status')!='failed']
    failures=[{'doc_id':d['doc_id'],'canonical_doc_id':d.get('canonical_doc_id',d['doc_id']),
        'file_name':d['file_name'],'relative_path':d.get('relative_path'),'sha256':d.get('sha256'),
        'status':'blocked','error':d.get('error',{}),'indexed':False}
        for d in documents if d.get('status')=='failed']
    if not good:
        write_json(target/'reports/intake_failure.json',{'files':len(documents),'failures':failures,'api_calls':0})
        raise ValueError('모든 문서의 파싱이 실패했습니다. intake_failure.json을 확인하세요.')
    write_jsonl(target/'data/source_documents.jsonl',good)
    chunks,chunk_audit=build_chunks(good,config['chunking'])
    integrity=validate_chunks(good,chunks,config['chunking']['chunk_size_bytes'])
    if not integrity['valid']:raise ValueError('새 입력의 청크·원문 범위 검사가 실패했습니다.')
    write_jsonl(target/'data/chunks.jsonl',chunks)
    with LocalSearchIndex(target/'index') as index:build=index.build(chunks,config['retrieval'])
    write_json(target/'reports/chunk_integrity.json',integrity)
    write_json(target/'reports/chunk_audit.json',chunk_audit)
    write_json(target/'reports/index_build.json',build)
    # No old development/holdout answer is silently assigned to new documents.
    write_json(target/'rfp_phase5_final/references.json',{'samples':[],'rows':[]})
    report={'version':VERSION,'workspace':str(target),'parser_root':str(parser_root),'input_fingerprint':fingerprint,
        'input_files':len(documents),'parsed_files':len(good),'failed_files':len(failures),
        'canonical_documents':len({d['canonical_doc_id'] for d in good}),'chunks':len(chunks),
        'all_inputs_parsed':not failures,'failures':failures,'ignored_hidden_files':ignored,
        'format_counts':{k:sum(d.get('detected_format')==k for d in good) for k in ('hwp','hwpx','pdf')},
        'documents_with_warnings':sum(bool(d.get('warnings')) for d in good),
        'version_matched_reviewed_supplements':len(exact_annotations),
        'source_character_coverage':integrity['source_character_coverage'],
        'binary_semantic_completeness_verified':False,'semantic_quality_verified':False,
        'api_calls':0,'existing_project_modified':False,'reused_existing_workspace':False}
    write_json(report_path,report)
    files=[p for base in ('data','index','configs','reports') for p in (target/base).rglob('*')
        if p.is_file() and p.name!='run_manifest.json' and 'cache' not in p.parts and p.suffix!='.lock']
    manifest={'version':VERSION,'files_sha256':{p.relative_to(target).as_posix():sha256_file(p) for p in files},
        'code_sha256':{p.relative_to(target).as_posix():sha256_file(p) for name in ('local_rag','rfp_phase4','rfp_phase5','rfp_phase5_final')
            for p in (target/name).glob('*.py')},'api_calls':0}
    write_json(target/'reports/run_manifest.json',manifest)
    return report
