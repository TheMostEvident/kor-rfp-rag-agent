"""파일 내용을 기준으로 형식을 판별하고, 문서별 성공·실패와 버전을 보존합니다."""
import hashlib
import importlib
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 'rfp-local-document-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def safe_source_path(root, relative_path):
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError('입력 경로가 지정한 작업 폴더를 벗어납니다.') from error
    return path


def detect_format(path):
    path = Path(path)
    with path.open('rb') as stream:
        magic = stream.read(8)
    if magic.startswith(b'%PDF-'):
        return 'pdf'
    if magic == bytes.fromhex('d0cf11e0a1b11ae1'):
        return 'hwp'
    if magic.startswith(b'PK'):
        try:
            with zipfile.ZipFile(path) as archive:
                if any(n.startswith('Contents/section') and n.endswith('.xml') for n in archive.namelist()):
                    return 'hwpx'
        except zipfile.BadZipFile as error:
            raise ValueError('ZIP/HWPX 컨테이너가 손상되었습니다.') from error
    raise ValueError('지원하는 PDF/HWP5/HWPX 형식으로 판별할 수 없습니다.')


def discover_manifest(root, raw_directory='data/raw'):
    """새 로컬 문서에도 같은 파이프라인을 사용합니다. 기존 평가 문서 ID와는 구별합니다."""
    root = Path(root).resolve()
    rows = []
    directory = safe_source_path(root, raw_directory)
    for path in sorted(p for p in directory.rglob('*') if p.is_file()):
        relative = path.relative_to(root).as_posix()
        path = safe_source_path(root, relative)
        digest = sha256_file(path)
        rows.append({'doc_id': 'local_' + digest[:16] + '_' + hashlib.sha256(relative.encode()).hexdigest()[:6],
                     'name': path.name, 'sha256': digest, 'relative_path': relative,
                     'source_url': None, 'agency': None, 'document_year': None, 'split': 'unassigned'})
    return rows


def _module_for(kind):
    module = importlib.import_module('.' + kind, __package__)
    digest = sha256_file(module.__file__)
    return module, digest


def _validated_payload(payload, kind):
    if not isinstance(payload, dict) or payload.get('detected_format') != kind:
        raise ValueError('파서 결과의 형식 식별자가 일치하지 않습니다.')
    for key in ['blocks', 'tables', 'warnings']:
        if not isinstance(payload.get(key), list):
            raise ValueError('파서 결과에 ' + key + ' 목록이 없습니다.')
    blocks = payload['blocks'] + payload.get('reference_blocks', [])
    if not any(str(b.get('text', '')).strip() for b in blocks):
        raise ValueError('원문 텍스트가 추출되지 않았습니다. 이미지/OCR 확인이 필요합니다.')
    ids = [b.get('block_id') for b in blocks]
    if len(ids) != len(set(ids)) or any(not v for v in ids):
        raise ValueError('파서 블록 ID가 없거나 중복됩니다.')
    if any(not b.get('locator') for b in blocks):
        raise ValueError('원문 위치가 없는 블록이 있습니다.')
    return payload


def _parse_or_cache(path, entry, kind, digest, cache_dir, force, extract_pdf_tables, preparsed_dir):
    module, module_hash = _module_for(kind)
    options = {'extract_pdf_tables': extract_pdf_tables} if kind == 'pdf' else {}
    key = hashlib.sha256((digest + module_hash + json.dumps(options, sort_keys=True)).encode()).hexdigest()
    cache_path = cache_dir / (key + '.json')
    cache_recovery = False
    if not force and cache_path.is_file():
        try:
            stored = json.loads(cache_path.read_text(encoding='utf-8'))
            if stored.get('sha256') == digest and stored.get('parser_module_sha256') == module_hash and stored.get('options') == options:
                return _validated_payload(stored['payload'], kind), 'validated_cache', module_hash
            cache_recovery = True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            cache_recovery = True
    payload = None
    origin = 'parsed_from_original'
    # 개발 중 동일 원본을 새 파서로 이미 처리한 경우에만 재사용합니다.
    if not force and kind == 'pdf' and extract_pdf_tables and preparsed_dir:
        candidate = Path(preparsed_dir) / (entry['doc_id'] + '.json')
        if candidate.is_file():
            try:
                previous = json.loads(candidate.read_text(encoding='utf-8'))
                requested = previous.get('options', {}).get('extract_pdf_tables')
                if requested is None:
                    requested = previous.get('payload', {}).get('stats', {}).get('table_extraction_requested')
                if previous.get('sha256') == digest and previous.get('parser_module_sha256') == module_hash and requested is True:
                    payload = _validated_payload(previous['payload'], kind)
                    origin = 'new_parser_prevalidated_cache'
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                payload = None
    if payload is None:
        function = getattr(module, 'parse_' + kind)
        payload = function(path, extract_tables=extract_pdf_tables) if kind == 'pdf' else function(path)
    payload = _validated_payload(payload, kind)
    if cache_recovery:
        payload = dict(payload, warnings=list(payload['warnings']) + [{'code': 'INVALID_CACHE_REPARSED', 'message': '손상되었거나 현재 파서·옵션과 맞지 않는 캐시 대신 원본을 다시 판독했습니다.'}])
    write_json(cache_path, {'sha256': digest, 'parser_module_sha256': module_hash, 'options': options, 'payload': payload})
    return payload, origin, module_hash


def apply_reviewed_transcriptions(document, entries):
    """원문 버전과 시각 검수 기록이 맞는 전사만 명시적인 별도 출처로 추가합니다."""
    supplements = []
    for entry in entries:
        if entry.get('doc_id') != document['doc_id']:
            continue
        if entry.get('sha256') != document['sha256']:
            document['warnings'].append({'code': 'TRANSCRIPTION_VERSION_MISMATCH', 'message': '전사 검수 버전과 현재 원본 해시가 달라 적용하지 않았습니다.'})
            continue
        if entry.get('review_status') != 'visual_verified' or not entry.get('bbox'):
            raise ValueError('시각 검수와 원문 좌표가 없는 전사를 적용할 수 없습니다.')
        supplements.append({'block_id': 'transcription_' + entry['check_id'], 'kind': 'verified_transcription',
                            'locator': f"PDF p.{entry['page']} / 시각 검수 전사 {entry['check_id']}",
                            'page': entry['page'], 'bbox': entry['bbox'], 'text': entry['text'],
                            'bbox_coordinate_system': entry.get('bbox_coordinate_system', 'unrotated_page_points_top_left'),
                            'source_method': 'reviewed_visual_transcription', 'review_status': 'visual_verified',
                            'review_note': entry.get('review_note', ''), 'review_date': entry.get('review_date')})
    document['supplemental_blocks'] = supplements
    if supplements:
        document['warnings'].append({'code': 'VISUAL_TRANSCRIPTION_APPLIED', 'message': f'자동 텍스트 추출에서 누락된 구간의 시각 검수 전사 {len(supplements)}개를 별도 출처로 보존했습니다.'})


def run_ingestion(root, manifest=None, force=False, extract_pdf_tables=True, preparsed_dir=None, progress=None):
    root = Path(root).resolve()
    if manifest is None:
        manifest = json.loads((root / 'data/input_manifest.json').read_text(encoding='utf-8'))
    if len({e['doc_id'] for e in manifest}) != len(manifest):
        raise ValueError('입력 목록에 중복 doc_id가 있습니다.')
    processed = root / 'data/processed'
    cache = processed / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    prepared = []
    groups = defaultdict(list)
    for entry in manifest:
        try:
            path = safe_source_path(root, entry['relative_path'])
            digest = sha256_file(path)
            groups[digest].append(entry['doc_id'])
            prepared.append((entry, path, digest, None))
        except (OSError, ValueError) as error:
            prepared.append((entry, None, None, error))
    documents = []
    for count, (entry, path, digest, preparation_error) in enumerate(prepared, 1):
        started = time.perf_counter()
        canonical = min(groups[digest]) if digest else entry['doc_id']
        document = {'schema_version': SCHEMA_VERSION, 'doc_id': entry['doc_id'],
                    'content_id': 'sha256:' + digest if digest else None, 'sha256': digest,
                    'expected_sha256': entry.get('sha256'), 'canonical_doc_id': canonical,
                    'is_duplicate_alias': canonical != entry['doc_id'], 'file_name': entry['name'],
                    'relative_path': entry['relative_path'], 'raw_path': entry['relative_path'],
                    'source_url': entry.get('source_url'), 'agency': entry.get('agency'),
                    'document_year': entry.get('document_year'), 'split': entry.get('split'),
                    'blocks': [], 'reference_blocks': [], 'supplemental_blocks': [], 'tables': [], 'warnings': []}
        try:
            if preparation_error:
                raise preparation_error
            kind = detect_format(path)
            payload, origin, module_hash = _parse_or_cache(path, entry, kind, digest, cache, force, extract_pdf_tables, preparsed_dir)
            document.update(payload)
            document['warnings'] = list(payload.get('warnings', []))
            document['parsing_origin'] = origin
            document['parser_module_sha256'] = module_hash
            document['size_bytes'] = path.stat().st_size
            if entry.get('sha256') and digest != entry['sha256']:
                document['warnings'].append({'code': 'SOURCE_HASH_CHANGED', 'message': '입력 목록 작성 이후 원본 내용이 변경되었습니다. 이전 평가 근거를 현재 버전에 적용하지 않습니다.'})
            if path.suffix.lower().lstrip('.') != kind:
                document['warnings'].append({'code': 'EXTENSION_FORMAT_MISMATCH', 'message': '파일 확장자와 실제 형식이 다르므로 실제 형식으로 판독했습니다.'})
            document['status'] = 'parsed_with_warnings' if document['warnings'] else 'parsed'
        except Exception as error:
            document['status'] = 'failed'
            document['error'] = {'code': type(error).__name__, 'message': str(error)}
        document['elapsed_seconds'] = round(time.perf_counter() - started, 4)
        documents.append(document)
        if progress:
            progress(count, len(prepared), document)
    overrides_path = root / 'data/reviewed_transcriptions.json'
    overrides = json.loads(overrides_path.read_text(encoding='utf-8')) if overrides_path.is_file() else []
    for document in documents:
        if document['status'] != 'failed':
            apply_reviewed_transcriptions(document, overrides)
            if document['warnings']:
                document['status'] = 'parsed_with_warnings'
    write_jsonl(processed / 'documents.jsonl', documents)
    indexable = []
    tables = []
    for document in documents:
        if document['status'] == 'failed' or document['is_duplicate_alias']:
            continue
        aliases = sorted(groups[document['sha256']])
        common = {key: document[key] for key in ['doc_id', 'canonical_doc_id', 'sha256', 'file_name', 'detected_format']}
        common['alias_doc_ids'] = aliases
        for block in document['blocks'] + document.get('supplemental_blocks', []):
            if block['text'].strip():
                indexable.append(dict(common, **block, source_kind='original_extracted' if block.get('source_method') != 'reviewed_visual_transcription' else 'reviewed_visual_transcription'))
        tables.extend(dict(common, **table) for table in document['tables'])
    write_jsonl(processed / 'indexable_blocks.jsonl', indexable)
    write_jsonl(processed / 'tables.jsonl', tables)
    return documents


def load_documents(root):
    with (Path(root) / 'data/processed/documents.jsonl').open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]
