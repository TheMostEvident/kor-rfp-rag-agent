"""Match every registered source to original bytes; optionally reparse without caches.

No evaluation questions, network services, original files, saved snapshots or
indexes are read as model input or modified. Parser results are used for this
audit only; changed/new documents belong in a separate intake workspace.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unicodedata
import zipfile

REGISTRATION_VERSION='physical-path-registration-v2'

def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def _digest(value):
    encoded=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _path_spelling(path):
    """Candidate key only; Unicode spelling never proves file identity."""
    return unicodedata.normalize('NFC',os.path.normcase(str(path)))


def _registered_path(path,expected_paths,by_spelling):
    if path in expected_paths:return 'exact_path'
    for expected in by_spelling.get(_path_spelling(path),[]):
        try:
            if path.samefile(expected):return 'normalized_spelling_same_file'
        except OSError:continue
    return None


def _metadata_reason(path,raw_dir):
    relative=path.relative_to(raw_dir)
    if '__MACOSX' in relative.parts:return 'macos_metadata_directory'
    if path.name in {'.DS_Store','.gitkeep'}:return 'directory_metadata'
    if path.name.startswith('._'):
        try:
            with path.open('rb') as handle:magic=handle.read(4)
            if magic==b'\x00\x05\x16\x07':return 'appledouble_sidecar'
        except OSError:pass
    return None


def detect_format(path):
    with Path(path).open('rb') as handle:magic=handle.read(8)
    if magic.startswith(b'%PDF-'):return 'pdf'
    if magic==bytes.fromhex('d0cf11e0a1b11ae1'):return 'hwp'
    if magic.startswith(b'PK'):
        with zipfile.ZipFile(path) as archive:
            if any(name.startswith('Contents/section') and name.endswith('.xml') for name in archive.namelist()):return 'hwpx'
    raise ValueError('unsupported_or_damaged_format')


def _source_path(root,document,raw_directory=None):
    relative=document.get('relative_path')
    if raw_directory is not None:
        allowed_root=Path(raw_directory).expanduser().resolve()
        if isinstance(relative,str) and relative.startswith('data/raw/'):
            relative=relative[len('data/raw/'):]
        else:relative=document.get('file_name')
        if not isinstance(relative,str) or not relative:raise ValueError('unsafe_source_path')
        path=(allowed_root/relative).resolve()
        if not path.is_relative_to(allowed_root):raise ValueError('unsafe_source_path')
        return path
    if not isinstance(relative,str) or not relative:
        name=document.get('file_name')
        if not isinstance(name,str) or Path(name).name!=name:raise ValueError('unsafe_source_path')
        relative='data/raw/'+name
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('unsafe_source_path')
    return path


def _parser(root,kind):
    path=root/'local_ingest'/(kind+'.py')
    digest=_sha(path)
    spec=importlib.util.spec_from_file_location('rfp_raw_audit_'+kind+'_'+digest[:16],path)
    if spec is None or spec.loader is None:raise ValueError('parser_unavailable')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module,'parse_'+kind),digest


def _quality(payload):
    blocks=payload.get('blocks')
    if not isinstance(blocks,list):raise ValueError('invalid_parser_blocks')
    characters=sum(len(b.get('text','')) for b in blocks if isinstance(b,dict) and isinstance(b.get('text'),str))
    nonempty=sum(bool(b.get('text','').strip()) for b in blocks if isinstance(b,dict) and isinstance(b.get('text'),str))
    warnings=payload.get('warnings',[])
    codes=Counter(w.get('code','UNKNOWN_WARNING') for w in warnings if isinstance(w,dict))
    tables=payload.get('tables',[])
    table_states=Counter(t.get('structure_status','unknown') for t in tables if isinstance(t,dict))
    stats=payload.get('stats',{})
    return {'primary_block_count':len(blocks),'nonempty_primary_blocks':nonempty,
        'primary_text_characters':characters,'warning_counts':dict(codes),
        'table_count':len(tables),'table_structure_counts':dict(table_states),
        'pages':stats.get('pages'),'empty_text_pages':stats.get('empty_text_pages',[]),
        'low_text_pages':stats.get('low_text_pages',[]),
        'text_extraction_failed_pages':stats.get('text_extraction_failed_pages',[]),
        'ocr_review_needed':bool(codes.get('PDF_OCR_REVIEW_NEEDED') or not nonempty),
        'unreviewed_or_heuristic_tables':sum(n for state,n in table_states.items() if state!='explicit'),
        'semantic_completeness_verified':False}


def audit_raw_corpus(documents,phase2_root,reparse=False,raw_directory=None,progress=None):
    """Audit all original IDs, counting duplicate contents once during reparsing.

    ``read_success`` means that unchanged original bytes have nonempty decoded
    primary text. It does not mean every visual glyph/table was recovered.
    With ``reparse=False``, extraction is checked from the version-matched saved
    snapshot, explicitly labelled as such. No parser cache is used when True.
    """
    if type(reparse)is not bool:raise TypeError('reparse must be bool')
    root=Path(phase2_root).expanduser().resolve()
    rows=[];expected_paths=set();parsed_by_content={};parsers={};reparse_count=0
    ids=Counter(d.get('doc_id') for d in documents)
    raw_dir=Path(raw_directory).expanduser().resolve() if raw_directory is not None else root/'data/raw'
    for position,document in enumerate(documents,1):
        row={'doc_id':document.get('doc_id'),'canonical_doc_id':document.get('canonical_doc_id',document.get('doc_id')),
            'file_name':document.get('file_name'),'expected_sha256':document.get('sha256'),
            'read_success':False,'needs_review':False,'review_codes':[],
            'raw_exists':False,'sha256_matches':False,'reparse_requested':reparse,
            'reparse_executed':False,'snapshot_compatible':False,
            'recognition_verification':'original_reparse' if reparse else 'saved_snapshot_with_raw_hash'}
        rows.append(row)
        try:
            path=_source_path(root,document,raw_directory);expected_paths.add(path)
            row['raw_relative_path']=path.relative_to(raw_dir if raw_directory is not None else root).as_posix()
            if not path.is_file():raise ValueError('raw_source_missing')
            row['raw_exists']=True;actual_sha=_sha(path);row['actual_sha256']=actual_sha
            row['sha256_matches']=bool(document.get('sha256') and actual_sha==document['sha256'])
            if not row['sha256_matches']:raise ValueError('raw_source_changed')
            kind=detect_format(path);row['detected_format']=kind
            row['extension_mismatch']=path.suffix.lower().lstrip('.')!=kind
            if row['extension_mismatch']:row['review_codes'].append('EXTENSION_FORMAT_MISMATCH')
            if document.get('detected_format') not in (None,kind):raise ValueError('snapshot_format_mismatch')
            payload=document
            if reparse:
                if kind not in parsers:parsers[kind]=_parser(root,kind)
                function,parser_sha=parsers[kind]
                key=(actual_sha,parser_sha)
                if key not in parsed_by_content:
                    buffer=io.StringIO()
                    try:
                        with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
                            output=function(path,extract_tables=True) if kind=='pdf' else function(path)
                        if not isinstance(output,dict) or output.get('detected_format')!=kind:raise ValueError('invalid_parser_format')
                        parsed_by_content[key]={'payload':output,'parser_sha256':parser_sha}
                    except Exception as error:
                        parsed_by_content[key]={'error_type':type(error).__name__,'error_message':str(error)}
                    reparse_count+=1;row['reparse_executed']=True
                parsed=parsed_by_content[key]
                if _sha(path)!=actual_sha:raise ValueError('source_changed_during_parse')
                if 'error_type' in parsed:
                    message=parsed.get('error_message','').lower()
                    row['parser_error_type']=parsed['error_type']
                    raise ValueError('encrypted_or_protected_source' if any(x in message for x in ['encrypt','password','암호','배포용']) else 'raw_parse_failed')
                payload=parsed['payload']
                row['parser_module_sha256']=parsed['parser_sha256']
                row['parser_version_matches_snapshot']=parsed['parser_sha256']==document.get('parser_module_sha256')
                row['primary_blocks_match_snapshot']=_digest(payload.get('blocks',[]))==_digest(document.get('blocks',[]))
                row['tables_match_snapshot']=_digest(payload.get('tables',[]))==_digest(document.get('tables',[]))
                row['snapshot_compatible']=all(row[k] for k in ('parser_version_matches_snapshot','primary_blocks_match_snapshot','tables_match_snapshot'))
                if not row['snapshot_compatible']:row['review_codes'].append('REPARSE_SNAPSHOT_DIFFERENCE')
            else:
                if document.get('status')=='failed':raise ValueError('saved_parse_failed')
                parser_path=root/'local_ingest'/(kind+'.py')
                current_parser_sha=_sha(parser_path) if parser_path.is_file() else None
                expected_parser_sha=document.get('parser_module_sha256')
                row['parser_module_sha256']=current_parser_sha
                row['parser_version_matches_snapshot']=bool(current_parser_sha and expected_parser_sha
                    and current_parser_sha==expected_parser_sha)
                row['snapshot_compatible']=row['parser_version_matches_snapshot']
                if not row['snapshot_compatible']:
                    row['review_codes'].append('PARSER_VERSION_UNVERIFIED' if not current_parser_sha or not expected_parser_sha
                        else 'PARSER_VERSION_CHANGED')
            quality=_quality(payload);row['quality']=quality
            if not quality['nonempty_primary_blocks']:raise ValueError('no_extractable_primary_text')
            row['read_success']=True
            if quality['warning_counts']:row['review_codes'].append('PARSER_WARNINGS_REQUIRE_REVIEW')
            if quality['ocr_review_needed']:row['review_codes'].append('OCR_OR_IMAGE_REVIEW_NEEDED')
            if quality['unreviewed_or_heuristic_tables']:row['review_codes'].append('TABLE_REVIEW_NEEDED')
            if quality['text_extraction_failed_pages']:row['review_codes'].append('PAGE_TEXT_EXTRACTION_FAILED')
            if ids[document.get('doc_id')]>1:row['review_codes'].append('DUPLICATE_DOCUMENT_ID')
        except Exception as error:
            known={'unsafe_source_path','raw_source_missing','raw_source_changed','snapshot_format_mismatch',
                'encrypted_or_protected_source','raw_parse_failed','saved_parse_failed','no_extractable_primary_text',
                'invalid_parser_blocks','invalid_parser_format','unsupported_or_damaged_format'}
            known.add('source_changed_during_parse')
            message=str(error)
            row['failure_code']=message if isinstance(error,ValueError) and message in known else 'raw_audit_error'
            row['error_type']=type(error).__name__
            row['review_codes'].append(row['failure_code'])
            row['snapshot_compatible']=False
        row['needs_review']=bool(row['review_codes'])
        if progress is not None:progress(position,len(documents),row)
    unknown=[];ignored=[];aliases=[];registered_inventory_files=0;inventory_files=0
    by_spelling={}
    for path in expected_paths:by_spelling.setdefault(_path_spelling(path),[]).append(path)
    known_hashes={d.get('sha256') for d in documents if d.get('sha256')}
    if raw_dir.is_dir():
        for path in sorted(raw_dir.rglob('*')):
            if not path.is_file():continue
            reason=_metadata_reason(path,raw_dir)
            if reason:
                ignored.append({'relative_path':path.relative_to(raw_dir).as_posix(),'reason':reason});continue
            inventory_files+=1
            resolved=path.resolve()
            registration=_registered_path(resolved,expected_paths,by_spelling)
            if registration:
                registered_inventory_files+=1
                if registration!='exact_path':
                    aliases.append({'relative_path':path.relative_to(raw_dir).as_posix(),'match':registration})
            else:
                entry={'relative_path':path.relative_to(raw_dir if raw_directory is not None else root).as_posix(),'registered_in_snapshot':False}
                try:
                    if not resolved.is_relative_to(raw_dir if raw_directory is not None else root):raise ValueError('unsafe_source_path')
                    digest=_sha(path)
                    entry.update(sha256=digest,detected_format=detect_format(path),
                                 same_content_as_registered_source=digest in known_hashes)
                except Exception:entry['failure_code']='unsupported_or_unreadable_unregistered_source'
                unknown.append(entry)
    return {'version':'phase5-raw-corpus-audit-v1','registration_version':REGISTRATION_VERSION,
        'phase2_root':str(root),'raw_directory':str(raw_dir),
        'document_count':len(documents),'unique_contents':len({d.get('sha256') for d in documents if d.get('sha256')}),
        'read_success_count':sum(r['read_success'] for r in rows),'read_failure_count':sum(not r['read_success'] for r in rows),
        'needs_review_count':sum(r['needs_review'] for r in rows),
        'all_raw_hashes_match':bool(rows) and all(r['sha256_matches'] for r in rows),
        'all_registered_sources_read':bool(rows) and all(r['read_success'] for r in rows),
        'all_snapshots_compatible':bool(rows) and all(r['snapshot_compatible'] for r in rows),
        'extension_mismatch_count':sum(bool(r.get('extension_mismatch')) for r in rows),
        'detected_format_counts':dict(Counter(r['detected_format'] for r in rows if r.get('detected_format'))),
        'duplicate_document_ids':[key for key,count in ids.items() if count>1],
        'unregistered_source_count':len(unknown),'unregistered_sources':unknown,
        'inventory_source_files':inventory_files,'registered_inventory_files':registered_inventory_files,
        'registered_path_alias_count':len(aliases),'registered_path_aliases':aliases,
        'ignored_metadata_count':len(ignored),'ignored_metadata_sources':ignored,
        'unregistered_known_content_count':sum(r.get('same_content_as_registered_source',False) for r in unknown),
        'registration_policy':'exact_path_or_unicode_equivalent_same_physical_file; hash_only_never_registers',
        'reparse_requested':reparse,'unique_contents_reparsed':reparse_count,
        'fresh_parsing_verified':bool(reparse and rows and all(r['read_success'] for r in rows)),
        'parser_cache_used':False,'rows':rows,'api_calls':0,'evaluation_answers_read':False,
        'originals_modified':any(r.get('failure_code')=='source_changed_during_parse' for r in rows),
        'index_modified':False,'semantic_completeness_verified':False}
