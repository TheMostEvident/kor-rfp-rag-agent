"""HWPX 본문의 논리 XML 순서·문단·명시 표 좌표를 보존하는 로컬 파서.

페이지 배치를 추정하지 않습니다. reference_blocks는 원문 인용 검증용 섹션 뷰이며
blocks와 함께 중복 인덱싱하면 안 됩니다. 표의 cells에는 병합 span을 보존하고,
rows에는 anchor cell의 텍스트만 넣습니다(병합으로 덮인 칸은 None).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree as ET
import zipfile

PARSER_VERSION = 'local_hwpx_v1'
MAX_ARCHIVE_UNCOMPRESSED = 256 * 1024 * 1024
MAX_SECTION_BYTES = 64 * 1024 * 1024
MAX_GRID_CELLS = 2_000_000
MAX_TABLE_AXIS = 100_000
SECTION_PATTERN = re.compile(r'Contents/section(\d+)\.xml\Z')


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def _integer(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _child(element, name):
    return next((e for e in element if _local(e.tag) == name), None)


def _ancestor(element, parents, name):
    current = parents.get(element)
    while current is not None:
        if _local(current.tag) == name:
            return current
        current = parents.get(current)
    return None


def _inline_text(element):
    """<t>의 실제 문자와 탭/강제 줄바꿈을 보존합니다."""
    parts = [element.text or '']
    for child in element:
        name = _local(child.tag)
        if name == 'tab':
            parts.append('\t')
        elif name in {'lineBreak', 'br'}:
            parts.append('\n')
        elif name == 'nbSpace':
            parts.append('\u00a0')
        else:
            parts.append(_inline_text(child))
        parts.append(child.tail or '')
    return ''.join(parts)


def _section_parts(root, section_name, section_number):
    parents = {child: parent for parent in root.iter() for child in parent}
    paragraphs = [e for e in root.iter() if _local(e.tag) == 'p']
    paragraph_numbers = {e: n for n, e in enumerate(paragraphs, 1)}
    tables_xml = [e for e in root.iter() if _local(e.tag) == 'tbl']
    table_numbers = {e: n for n, e in enumerate(tables_xml, 1)}
    table_ids = {e: f'hwpx:s{section_number}:t{n:05d}' for e, n in table_numbers.items()}
    warnings = []

    # Text events follow XML order. An outer paragraph with a table can have a
    # segment before and after its nested cell paragraphs, without duplicating them.
    segments = []

    def add_text(paragraph, text):
        if paragraph is None or not text:
            return
        if segments and segments[-1][0] is paragraph:
            segments[-1][1].append(text)
        else:
            segments.append((paragraph, [text]))

    def walk(element, paragraph=None):
        name = _local(element.tag)
        if name == 'p':
            paragraph = element
        if name == 't':
            add_text(paragraph, _inline_text(element))
            return
        if name == 'tab':
            add_text(paragraph, '\t')
            return
        if name in {'lineBreak', 'br'}:
            add_text(paragraph, '\n')
            return
        for child in element:
            walk(child, paragraph)

    walk(root)
    blocks = []
    segment_counts = Counter()
    cell_text = {}
    paragraph_with_text = set()
    for paragraph, parts in segments:
        text = _nfc(''.join(parts)).strip()
        if not text:
            continue
        number = paragraph_numbers[paragraph]
        segment_counts[number] += 1
        segment = segment_counts[number]
        paragraph_with_text.add(paragraph)
        block = {
            'block_id': f'hwpx:s{section_number}:p{number:06d}:seg{segment:02d}',
            'kind': 'paragraph',
            'locator': f'{section_name} / paragraph {number} / segment {segment}',
            'section': section_name,
            'paragraph': number,
            'segment': segment,
            'xml_paragraph_id': paragraph.get('id'),
            'text': text,
        }
        cell = _ancestor(paragraph, parents, 'tc')
        if cell is not None:
            table = _ancestor(cell, parents, 'tbl')
            address = _child(cell, 'cellAddr')
            if table in table_ids:
                block['table_id'] = table_ids[table]
            if address is not None:
                block['row'] = _integer(address.get('rowAddr'))
                block['col'] = _integer(address.get('colAddr'))
            cell_text.setdefault(cell, []).append(text)
        blocks.append(block)

    tables = []
    for table in tables_xml:
        number = table_numbers[table]
        table_id = table_ids[table]
        locator = f'{section_name} / table {number}'
        table_warnings = []
        cells = []
        rows_declared = _integer(table.get('rowCnt'), 0)
        cols_declared = _integer(table.get('colCnt'), 0)
        if rows_declared < 0 or cols_declared < 0:
            raise ValueError(f'잘못된 HWPX 표 크기입니다: {locator}')
        for tr in (e for e in table if _local(e.tag) == 'tr'):
            for tc in (e for e in tr if _local(e.tag) == 'tc'):
                address = _child(tc, 'cellAddr')
                span = _child(tc, 'cellSpan')
                row = _integer(address.get('rowAddr')) if address is not None else None
                col = _integer(address.get('colAddr')) if address is not None else None
                row_span = _integer(span.get('rowSpan')) if span is not None and span.get('rowSpan') is not None else 1
                col_span = _integer(span.get('colSpan')) if span is not None and span.get('colSpan') is not None else 1
                if (any(value is None for value in (row, col, row_span, col_span))
                        or row < 0 or col < 0 or row_span < 1 or col_span < 1):
                    table_warnings.append({'code': 'INVALID_OR_MISSING_CELL_COORDINATES',
                                           'message': '행·열 주소 또는 병합 범위를 확정하지 못한 셀이 있습니다.'})
                cell = {
                    'row': row, 'col': col, 'row_span': row_span, 'col_span': col_span,
                    'text': '\n'.join(cell_text.get(tc, [])),
                    'locator': f'{locator} / row {row} col {col}',
                    'nested_table_ids': [table_ids[t] for t in tables_xml
                                         if _ancestor(t, parents, 'tc') is tc],
                }
                size = _child(tc, 'cellSz')
                if size is not None:
                    cell['xml_size'] = {'width': _integer(size.get('width')),
                                        'height': _integer(size.get('height')), 'unit': 'HWPUNIT'}
                cells.append(cell)
        valid = [c for c in cells if all(c[k] is not None for k in ['row', 'col', 'row_span', 'col_span'])
                 and c['row'] >= 0 and c['col'] >= 0 and c['row_span'] >= 1 and c['col_span'] >= 1]
        row_count = max([rows_declared] + [c['row'] + c['row_span'] for c in valid])
        col_count = max([cols_declared] + [c['col'] + c['col_span'] for c in valid])
        if row_count * col_count > MAX_GRID_CELLS or max(row_count, col_count) > MAX_TABLE_AXIS:
            raise ValueError(f'HWPX 표 크기가 처리 한도를 초과합니다: {locator}')
        grid = [[None for _ in range(col_count)] for _ in range(row_count)]
        occupied = set()
        for cell in valid:
            occupied_here = {(r, c) for r in range(cell['row'], cell['row'] + cell['row_span'])
                             for c in range(cell['col'], cell['col'] + cell['col_span'])}
            if occupied & occupied_here:
                table_warnings.append({'code': 'OVERLAPPING_CELL_SPANS',
                                       'message': '명시된 셀 좌표 또는 병합 범위가 중복됩니다.'})
            occupied |= occupied_here
            grid[cell['row']][cell['col']] = cell['text']
        if row_count != rows_declared or col_count != cols_declared:
            table_warnings.append({'code': 'TABLE_DIMENSION_MISMATCH',
                                   'message': '선언된 표 크기와 명시된 셀 좌표 범위가 다릅니다.'})
        if not cells:
            table_warnings.append({'code': 'EMPTY_TABLE_CELLS',
                                   'message': '표 객체에 판독할 수 있는 셀 레코드가 없습니다.'})
        unresolved = any(w['code'] in {'INVALID_OR_MISSING_CELL_COORDINATES', 'OVERLAPPING_CELL_SPANS', 'EMPTY_TABLE_CELLS'}
                         for w in table_warnings)
        tables.append({
            'table_id': table_id, 'locator': locator, 'section': section_name,
            'xml_table_id': table.get('id'),
            'rows': grid, 'cells': cells,
            'row_count': row_count, 'col_count': col_count,
            'declared_row_count': rows_declared, 'declared_col_count': cols_declared,
            'method': 'hwpx_explicit_cellAddr_cellSpan',
            'structure_status': 'unresolved' if unresolved else 'explicit',
            'warnings': table_warnings,
        })
        for warning in table_warnings:
            warnings.append(dict(warning, locator=locator))

    # This compatibility view is reconstructed only from the ORIGINAL XML. It is
    # not read from Phase 1 snapshots and must never be indexed a second time.
    reference_text = _nfc('\n'.join(''.join(e.itertext()) for e in root.iter() if _local(e.tag) == 't'))
    reference = {'block_id': f'hwpx:s{section_number}:reference', 'kind': 'section',
                 'locator': section_name, 'section': section_name, 'text': reference_text,
                 'purpose': 'quotation_validation_only'}
    objects = Counter(_local(e.tag) for e in root.iter()
                      if _local(e.tag) in {'pic', 'ole', 'equation', 'chart'})
    if objects:
        warnings.append({'code': 'NON_TEXT_OBJECTS_PRESENT', 'locator': section_name,
                         'message': '그림·수식·OLE·차트의 내부 의미는 텍스트 파싱만으로 판독하지 않습니다.',
                         'counts': dict(objects)})
    stats = {'paragraph_records': len(paragraphs),
             'empty_paragraph_records': len(paragraphs) - len(paragraph_with_text),
             'text_blocks': len(blocks), 'tables': len(tables),
             'table_cells': sum(len(t['cells']) for t in tables),
             'non_text_objects': dict(objects)}
    return blocks, tables, reference, warnings, stats


def parse_hwpx(path):
    """원본 HWPX를 파싱하고 텍스트·표·추적 가능한 내부 XML 위치를 반환합니다."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [info.filename for info in members]
        if len(names) != len(set(names)):
            raise ValueError('HWPX ZIP에 이름이 중복된 항목이 있어 원문을 확정할 수 없습니다.')
        if sum(info.file_size for info in members) > MAX_ARCHIVE_UNCOMPRESSED:
            raise ValueError('HWPX 압축 해제 크기가 256 MiB 한도를 초과합니다.')
        if any(info.flag_bits & 1 for info in members):
            raise ValueError('암호화된 HWPX ZIP 항목은 지원하지 않습니다.')
        section_names = sorted((name for name in names if SECTION_PATTERN.fullmatch(name)),
                               key=lambda name: int(SECTION_PATTERN.fullmatch(name).group(1)))
        if not section_names:
            raise ValueError('HWPX 본문 Contents/sectionN.xml을 찾지 못했습니다.')
        blocks, tables, references, warnings = [], [], [], []
        statistics = Counter()
        object_counts = Counter()
        for section_name in section_names:
            info = archive.getinfo(section_name)
            if info.file_size > MAX_SECTION_BYTES:
                raise ValueError(f'HWPX 섹션이 64 MiB 한도를 초과합니다: {section_name}')
            raw = archive.read(section_name)
            declaration_check = raw.replace(b'\x00', b'').upper()
            if b'<!DOCTYPE' in declaration_check or b'<!ENTITY' in declaration_check:
                raise ValueError('DOCTYPE/ENTITY를 포함하는 HWPX 본문은 지원하지 않습니다.')
            try:
                section_root = ET.fromstring(raw)
            except ET.ParseError as exc:
                raise ValueError(f'HWPX XML 파싱 실패: {section_name}: {exc}') from exc
            section_number = int(SECTION_PATTERN.fullmatch(section_name).group(1))
            parsed = _section_parts(section_root, section_name, section_number)
            new_blocks, new_tables, reference, new_warnings, stats = parsed
            blocks.extend(new_blocks); tables.extend(new_tables)
            references.append(reference); warnings.extend(new_warnings)
            for key, value in stats.items():
                if key == 'non_text_objects':
                    object_counts.update(value)
                else:
                    statistics[key] += value
        if not blocks:
            raise ValueError('HWPX 본문에서 판독 가능한 문단 텍스트를 찾지 못했습니다. 그림/OCR 확인이 필요합니다.')
        return {
            'detected_format': 'hwpx', 'parser': PARSER_VERSION,
            'blocks': blocks, 'reference_blocks': references,
            'tables': tables, 'warnings': warnings,
            'stats': dict(statistics, sections=len(section_names),
                          non_text_objects=dict(object_counts),
                          text_chars=sum(len(b['text']) for b in blocks),
                          logical_order='xml_document_order', rendered_page_layout=False),
        }
