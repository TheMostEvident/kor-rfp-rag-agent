"""Local PDF text/provenance reader with explicitly heuristic table candidates.

PDF physical pages are the authoritative text units. Table cells and spans are
geometry candidates, not a promise that all visual tables or glyphs survived.
No network, OCR service, or source-file mutation is performed.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import inspect
import re
import unicodedata

PARSER_VERSION = 'local_pdf_v1'


def _nfc(value):
    return unicodedata.normalize('NFC', value or '')


def _box(value):
    return [round(float(x), 3) for x in value] if value is not None else None


def _warning(code, message, locator=None, **extra):
    result = {'code': code, 'message': message}
    if locator is not None:
        result['locator'] = locator
    result.update(extra)
    return result


def _grid_boundaries(values, tolerance=0.05):
    """Merge near-equal detected edges; never infer missing text from them."""
    groups = []
    for value in sorted(values):
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _span(bounds, low, high):
    if not bounds:
        return 1
    left = min(range(len(bounds)), key=lambda i: abs(bounds[i] - low))
    right = min(range(len(bounds)), key=lambda i: abs(bounds[i] - high))
    return max(1, right - left)


def _table_payload(table, page_number, table_number,
                   coordinate_system='unrotated_page_points_top_left'):
    raw_rows = table.extract()
    rows = [[None if cell is None else _nfc(cell) for cell in row] for row in raw_rows]
    rectangles = [cell for row in table.rows for cell in row.cells if cell is not None]
    xs = _grid_boundaries([v for box in rectangles for v in (box[0], box[2])])
    ys = _grid_boundaries([v for box in rectangles for v in (box[1], box[3])])
    cells = []
    for row_index, row in enumerate(table.rows):
        for column_index, rectangle in enumerate(row.cells):
            if rectangle is None:
                continue
            text = rows[row_index][column_index] if row_index < len(rows) and column_index < len(rows[row_index]) else None
            cells.append({'row': row_index, 'col': column_index,
                          'row_span': _span(ys, rectangle[1], rectangle[3]),
                          'col_span': _span(xs, rectangle[0], rectangle[2]),
                          'text': text, 'bbox': _box(rectangle),
                          'span_basis': 'detected_grid_geometry'})
    warnings = ['PDF table boundaries, cell assignment and merged spans require visual review.']
    nonempty = sum(bool(cell and cell.strip()) for row in rows for cell in row)
    total = sum(len(row) for row in rows)
    if total and nonempty / total < 0.25:
        warnings.append('Sparse candidate: decorative frames or a complex merged table may have been detected.')
    return {'table_id': f'pdf_p{page_number:05d}_t{table_number:03d}',
            'locator': f'PDF p.{page_number} / table {table_number}', 'page': page_number,
            'bbox': _box(table.bbox), 'bbox_coordinate_system': coordinate_system,
            'rows': rows, 'cells': cells, 'method': 'pymupdf_ruled_line_geometry',
            'structure_status': 'heuristic', 'candidate': True, 'warnings': warnings}


def refresh_pdf_table_geometry(payload):
    """Validate derived spans without changing extracted text, rows or bboxes.

    PDF line geometry can accidentally combine a decorative frame with a table.
    A span outside the candidate grid is unknown, rather than clamped to a
    convenient number. Original inferred counts remain as diagnostic metadata.
    This deterministic, idempotent step also upgrades newly generated caches.
    """
    unresolved_tables = 0
    for table in payload['tables']:
        row_count = len(table['rows'])
        column_count = max((len(row) for row in table['rows']), default=0)
        unresolved = 0
        for cell in table['cells']:
            for axis, bound in [('row', row_count), ('col', column_count)]:
                key = axis + '_span'
                span = cell.get(key)
                if span is None:
                    if 'inferred_' + key in cell:
                        unresolved += 1
                    continue
                if (not isinstance(span, int) or span < 1
                        or cell[axis] + span > bound):
                    cell['inferred_' + key] = span
                    cell[key] = None
                    cell['span_unresolved'] = True
                    unresolved += 1
        table['warnings'] = [w for w in table['warnings'] if not w.startswith('Candidate grid conflict:')]
        if unresolved:
            unresolved_tables += 1
            table['structure_status'] = 'unresolved'
            table['warnings'].append(f'Candidate grid conflict: {unresolved} merged-span inference(s) are unresolved; rows and source bboxes are preserved.')
    payload['stats']['unresolved_table_candidates'] = unresolved_tables
    payload['warnings'] = [w for w in payload['warnings'] if w['code'] != 'PDF_TABLE_GRID_UNRESOLVED']
    if unresolved_tables:
        warning = _warning('PDF_TABLE_GRID_UNRESOLVED',
            f'{unresolved_tables}개 표 후보의 병합 범위가 탐지 격자와 일치하지 않아 미확정으로 남겼습니다. 원본 좌표·행·텍스트를 보존했습니다.')
        position = next((i for i,w in enumerate(payload['warnings']) if w['code'] in
                         {'PDF_TABLES_HEURISTIC','PDF_TABLE_EXTRACTION_SKIPPED'}),len(payload['warnings']))
        payload['warnings'].insert(position,warning)
    return payload


def _numeric_gaps(text):
    """Flag explicit numeric-looking slots with no mapped digits; do not fill them."""
    result = []
    expressions = [
        ('budget', r'(?:사업\s*예산|소요\s*예산|사\s*업\s*비)[^\n]{0,100}원'),
        ('duration', r'사업\s*기간[^\n]{0,100}(?:개월|주간|년간|일간)'),
    ]
    for field, expression in expressions:
        for match in re.finditer(expression, text):
            if not re.search(r'\d', match.group(0)):
                result.append({'field_hint': field, 'text_excerpt': match.group(0)})
    return result


def _vector_glyph_candidates(drawings):
    """Small aligned curved paths can be outlined digits/text or ordinary artwork."""
    paths = []
    for drawing in drawings:
        rectangle = drawing.get('rect')
        if rectangle is None:
            continue
        if (0.3 < rectangle.width < 35 and 3 < rectangle.height < 30
                and any(item[0] == 'c' for item in drawing.get('items', []))):
            paths.append(rectangle)
    groups = []
    for rectangle in sorted(paths, key=lambda r: (r.y1, r.x0)):
        group = next((g for g in groups if abs(g[0].y1 - rectangle.y1) <= 2), None)
        if group is None:
            groups.append([rectangle])
        else:
            group.append(rectangle)
    candidates = []
    for group in groups:
        if len(group) < 5:
            continue
        candidates.append({'path_count': len(group),
                           'bbox': _box((min(r.x0 for r in group), min(r.y0 for r in group),
                                         max(r.x1 for r in group), max(r.y1 for r in group)))})
    return len(paths), candidates


def parse_pdf(path, extract_tables=True):
    """Extract physical-page text and table candidates from a local PDF.

    pypdf is required. PyMuPDF is optional for layout/geometry but recommended;
    its absence is an explicit warning rather than a silent table success.
    An empty text page remains in blocks with its true physical-page locator.
    """
    from pypdf import PdfReader

    path = Path(path)
    with path.open('rb') as stream:
        if not stream.read(8).startswith(b'%PDF-'):
            raise ValueError('PDF_SIGNATURE_INVALID: PDF 파일 서명이 아닙니다.')
    result = {'detected_format': 'pdf', 'parser': PARSER_VERSION,
              'blocks': [], 'reference_blocks': [], 'tables': [], 'warnings': [], 'stats': {}}
    layout_document = None
    layout_import_error = None
    try:
        import fitz
        layout_document = fitz.open(path)
    except ImportError:
        layout_import_error = 'PyMuPDF가 설치되지 않아 표·이미지·벡터 배치를 확인하지 못했습니다.'
    except Exception as exc:
        layout_import_error = f'PDF 배치 판독 실패: {type(exc).__name__}: {exc}'
    if layout_import_error:
        result['warnings'].append(_warning('PDF_LAYOUT_UNAVAILABLE', layout_import_error))
    stats = {'pages': 0, 'text_blocks': 0, 'tables': 0, 'text_characters': 0,
             'low_text_pages': [], 'empty_text_pages': [], 'text_extraction_failed_pages': [],
             'image_pages': [], 'mixed_text_image_pages': [], 'vector_glyph_review_pages': [],
             'numeric_gap_review_pages': [], 'table_detection_failed_pages': [],
             'table_extraction_requested': bool(extract_tables),
             'table_detection_scope': 'ruled-line geometry only; no guarantee of completeness',
             'ocr_executed': False}
    try:
        with path.open('rb') as stream:
            try:
                reader = PdfReader(stream, strict=False)
            except Exception as exc:
                raise ValueError(f'PDF_READ_FAILED: {type(exc).__name__}: {exc}') from exc
            if reader.is_encrypted:
                raise ValueError('PDF_ENCRYPTED: 암호화 PDF는 자동으로 처리하지 않습니다.')
            stats['pages'] = len(reader.pages)
            if not stats['pages']:
                raise ValueError('PDF_NO_PAGES: PDF에 물리 페이지가 없습니다.')
            for page_number, source_page in enumerate(reader.pages, 1):
                locator = f'PDF p.{page_number}'
                quality = {'image_count': None, 'drawing_count': None,
                           'small_curved_path_count': None, 'glyph_like_vector_clusters': [],
                           'text_extraction_engine': 'pypdf'}
                try:
                    text = _nfc(source_page.extract_text() or '')
                except Exception as exc:
                    text = ''
                    stats['text_extraction_failed_pages'].append(page_number)
                    result['warnings'].append(_warning('PDF_TEXT_EXTRACTION_FAILED',
                        f'{type(exc).__name__}: {exc}', locator))
                    if layout_document is not None:
                        try:
                            text = _nfc(layout_document[page_number-1].get_text())
                            quality['text_extraction_engine'] = 'pymupdf_fallback'
                            result['warnings'].append(_warning('PDF_TEXT_FALLBACK', 'pypdf 실패 후 PyMuPDF 텍스트를 보존했습니다.', locator))
                        except Exception:
                            pass
                text_characters = len(re.sub(r'\s+', '', text))
                quality['non_whitespace_characters'] = text_characters
                quality['replacement_characters'] = text.count('\ufffd')
                quality['nul_characters'] = text.count('\x00')
                try:
                    page_bbox = [float(source_page.cropbox.left), float(source_page.cropbox.bottom),
                                 float(source_page.cropbox.right), float(source_page.cropbox.top)]
                except Exception:
                    page_bbox = None
                if text_characters < 40:
                    stats['low_text_pages'].append(page_number)
                if not text.strip():
                    stats['empty_text_pages'].append(page_number)
                    result['warnings'].append(_warning('PDF_EMPTY_TEXT_PAGE', '텍스트가 없는 페이지입니다. 빈 면·그림·스캔 여부를 확인해야 합니다.', locator))
                elif text_characters < 40:
                    result['warnings'].append(_warning('PDF_LOW_TEXT_PAGE', '추출 텍스트가 매우 짧아 원본 확인이 필요합니다.', locator))
                if quality['replacement_characters'] or quality['nul_characters']:
                    result['warnings'].append(_warning('PDF_TEXT_CONTROL_OR_REPLACEMENT',
                        '추출문에 NUL 또는 대체문자가 있습니다. 원문을 보존하고 판독 품질을 표시합니다.', locator))
                gaps = _numeric_gaps(text)
                if gaps:
                    stats['numeric_gap_review_pages'].append(page_number)
                    quality['numeric_value_gaps'] = gaps
                    result['warnings'].append(_warning('PDF_NUMERIC_VALUE_UNRESOLVED',
                        '숫자 값이 필요한 예산·기간 문구에 숫자가 추출되지 않았습니다. 벡터 글자·그림 또는 원문 공란을 확인해야 합니다.', locator,
                        field_hints=[g['field_hint'] for g in gaps]))
                if layout_document is not None:
                    try:
                        layout_page = layout_document[page_number - 1]
                        images = layout_page.get_image_info()
                        quality['image_count'] = len(images)
                        quality['rotation_degrees'] = layout_page.rotation
                        quality['layout_page_rect'] = _box(layout_page.rect)
                        drawings = layout_page.get_drawings()
                        quality['drawing_count'] = len(drawings)
                        small_paths, vector_clusters = _vector_glyph_candidates(drawings)
                        quality['small_curved_path_count'] = small_paths
                        quality['glyph_like_vector_clusters'] = vector_clusters
                        if vector_clusters:
                            stats['vector_glyph_review_pages'].append(page_number)
                            result['warnings'].append(_warning('PDF_VECTOR_GLYPH_REVIEW',
                                '작고 나란한 곡선 경로가 있습니다. 윤곽선 글자·숫자가 텍스트에서 누락될 가능성이 있어 시각 검수가 필요합니다. 도식도 이 조건에 해당할 수 있습니다.', locator))
                        if images:
                            stats['image_pages'].append(page_number)
                            if text_characters >= 40:
                                stats['mixed_text_image_pages'].append(page_number)
                            elif text_characters < 40:
                                result['warnings'].append(_warning('PDF_OCR_REVIEW_NEEDED',
                                    '이미지가 있고 추출 텍스트가 적습니다. 로컬 OCR 또는 원본 검토가 필요하며 OCR은 실행하지 않았습니다.', locator))
                        if extract_tables:
                            try:
                                kwargs = {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'}
                                # On a rotated page PyMuPDF's table finder applies a
                                # display transform. Precomputed unrotated paths
                                # cannot safely be reused for that case.
                                if layout_page.rotation == 0 and 'paths' in inspect.signature(layout_page.find_tables).parameters:
                                    kwargs['paths'] = drawings
                                with redirect_stdout(StringIO()):
                                    found = layout_page.find_tables(**kwargs)
                                count = 0
                                for table in found.tables:
                                    # Single-column outlines are not claimed as structured tables.
                                    if table.row_count < 2 or table.col_count < 2:
                                        continue
                                    count += 1
                                    coordinates = ('displayed_page_points_top_left' if layout_page.rotation
                                                   else 'unrotated_page_points_top_left')
                                    result['tables'].append(_table_payload(table, page_number, count, coordinates))
                            except Exception as exc:
                                stats['table_detection_failed_pages'].append(page_number)
                                result['warnings'].append(_warning('PDF_TABLE_DETECTION_FAILED',
                                    f'{type(exc).__name__}: {exc}', locator))
                    except Exception as exc:
                        result['warnings'].append(_warning('PDF_PAGE_LAYOUT_FAILED', f'{type(exc).__name__}: {exc}', locator))
                result['blocks'].append({'block_id': f'pdf_p{page_number:05d}', 'kind': 'page',
                    'locator': locator, 'text': text, 'page': page_number,
                    'bbox': page_bbox, 'quality': quality})
                stats['text_characters'] += len(text)
    finally:
        if layout_document is not None:
            layout_document.close()
    stats['text_blocks'] = len(result['blocks'])
    stats['tables'] = len(result['tables'])
    stats['table_pages'] = sorted({table['page'] for table in result['tables']})
    if extract_tables:
        result['warnings'].append(_warning('PDF_TABLES_HEURISTIC',
            'PDF 표는 선·좌표 기반 후보입니다. 테두리 없는 표, 그림·윤곽선 글자와 복잡한 병합 구조는 자동 추출만으로 완전성을 확정할 수 없습니다.'))
    else:
        result['warnings'].append(_warning('PDF_TABLE_EXTRACTION_SKIPPED', '설정에 따라 PDF 표 탐지를 실행하지 않았습니다.'))
    if not any(block['text'].strip() for block in result['blocks']):
        result['warnings'].append(_warning('PDF_NO_EXTRACTABLE_TEXT', '문서 전체에서 텍스트를 확보하지 못했습니다. 원본 검토 또는 로컬 OCR이 필요합니다.'))
    result['stats'] = stats
    return refresh_pdf_table_geometry(result)
