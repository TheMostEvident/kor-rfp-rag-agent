"""Read-only HWP 5.x paragraph/table extractor, using Python's standard library.

Coordinates and spans come from native table records. HWP page layout, images,
math, and protected documents are not rendered. Source positions are original
BodyText section/paragraph numbers and decompressed record offsets, not pages.

Reference: Hancom HWP 5.0 rev1.3, sections 4.3.7 / 4.3.9.1, tables 65 / 75 / 80.
https://cdn.hancom.com/link/docs/한글문서파일형식_5.0_revision1.3.pdf
The official 6-byte list header and observed 8-byte variant are resolved for an
entire table by explicit cell coordinate/span bounds and overlap validation.
Ambiguity is reported rather than choosing an offset per cell.
"""

import struct
import zlib
import unicodedata
from collections import Counter
from pathlib import Path


class _CompoundFile:
    """Microsoft CFB의 FAT/MiniFAT와 디렉터리 스트림을 읽는 최소 읽기 전용 구현."""

    END = 0xFFFFFFFE
    FREE = 0xFFFFFFFF

    def __init__(self, path):
        self.data = Path(path).read_bytes()
        if len(self.data) < 512 or self.data[:8] != bytes.fromhex("d0cf11e0a1b11ae1"):
            raise ValueError("HWP 5.x OLE 파일이 아닙니다. HWPX/다른 버전은 별도 파서가 필요합니다.")
        u16 = lambda offset: struct.unpack_from("<H", self.data, offset)[0]
        u32 = lambda offset: struct.unpack_from("<I", self.data, offset)[0]
        self.major = u16(26)
        sector_shift, mini_shift = u16(30), u16(32)
        if (self.major, sector_shift) not in {(3, 9), (4, 12)} or mini_shift != 6 or u16(28) != 0xFFFE:
            raise ValueError("지원하지 않는 CFB 헤더입니다.")
        self.sector_size = 1 << sector_shift
        self.mini_size = 1 << mini_shift
        self.sector_count = len(self.data) // self.sector_size - 1
        if len(self.data) % self.sector_size:
            raise ValueError("CFB 파일 크기가 섹터 경계와 맞지 않습니다.")
        self.cutoff = u32(56)
        if self.cutoff != 4096:
            raise ValueError("지원하지 않는 CFB MiniStream 크기입니다.")
        fat_count = u32(44)
        difat = list(struct.unpack_from("<109I", self.data, 76))
        difat_sector = u32(68)
        seen = set()
        for _ in range(u32(72)):
            if difat_sector in seen:
                raise ValueError("CFB DIFAT 순환 체인입니다.")
            seen.add(difat_sector)
            values = self._uints(self._sector(difat_sector))
            difat.extend(values[:-1])
            difat_sector = values[-1]
        fat_sectors = [sector for sector in difat if sector < 0xFFFFFFFA]
        if len(fat_sectors) < fat_count:
            raise ValueError("CFB FAT 섹터 목록이 잘렸습니다.")
        self.fat = []
        for sector in fat_sectors[:fat_count]:
            self.fat.extend(self._uints(self._sector(sector)))
        directory = self._normal_stream(u32(48))
        self.entries = []
        for offset in range(0, len(directory), 128):
            entry = directory[offset:offset + 128]
            if len(entry) != 128:
                raise ValueError("CFB 디렉터리 레코드가 잘렸습니다.")
            name_size = struct.unpack_from("<H", entry, 64)[0]
            if name_size and (name_size > 64 or name_size % 2):
                raise ValueError("CFB 디렉터리 이름 크기 오류입니다.")
            name = entry[:max(0, name_size - 2)].decode("utf-16-le") if name_size else ""
            left, right, child = struct.unpack_from("<3I", entry, 68)
            start = struct.unpack_from("<I", entry, 116)[0]
            size = struct.unpack_from("<Q", entry, 120)[0]
            if self.major == 3:
                size &= 0xFFFFFFFF
            self.entries.append(dict(name=name, kind=entry[66], left=left, right=right, child=child, start=start, size=size))
        if not self.entries or self.entries[0]["kind"] != 5:
            raise ValueError("CFB 루트 스토리지를 찾을 수 없습니다.")
        root = self.entries[0]
        self.mini_stream = self._normal_stream(root["start"], root["size"])
        mini_fat_bytes = self._normal_stream(u32(60), u32(64) * self.sector_size)
        self.mini_fat = self._uints(mini_fat_bytes)
        self.streams = {}
        visited = set()

        def walk(entry_id, prefix):
            if entry_id == self.FREE:
                return
            if entry_id >= len(self.entries) or entry_id in visited:
                raise ValueError("CFB 디렉터리 참조 또는 순환 오류입니다.")
            visited.add(entry_id)
            item = self.entries[entry_id]
            walk(item["left"], prefix)
            name = prefix + item["name"]
            if item["kind"] == 2:
                self.streams[name] = item
            elif item["kind"] == 1:
                walk(item["child"], name + "/")
            walk(item["right"], prefix)

        walk(root["child"], "")

    @staticmethod
    def _uints(data):
        if len(data) % 4:
            raise ValueError("CFB 할당 테이블 크기 오류입니다.")
        return list(struct.unpack("<" + "I" * (len(data) // 4), data))

    def _sector(self, sector):
        if sector < 0 or sector >= self.sector_count:
            raise ValueError("CFB 섹터가 파일 범위를 벗어났습니다.")
        offset = (sector + 1) * self.sector_size
        return self.data[offset:offset + self.sector_size]

    def _chain(self, start, table, block_reader):
        parts, seen = [], set()
        while start != self.END:
            if start in seen or start < 0 or start >= len(table):
                raise ValueError("CFB 스트림의 순환 또는 잘린 체인입니다.")
            seen.add(start)
            parts.append(block_reader(start))
            start = table[start]
        return b"".join(parts)

    def _normal_stream(self, start, size=None):
        if size == 0:
            return b""
        data = self._chain(start, self.fat, self._sector)
        if size is not None and len(data) < size:
            raise ValueError("CFB 스트림이 선언된 크기보다 짧습니다.")
        return data if size is None else data[:size]

    def read(self, name):
        item = self.streams[name]
        if not item["size"]:
            return b""
        if item["size"] >= self.cutoff:
            return self._normal_stream(item["start"], item["size"])

        def mini_block(sector):
            offset = sector * self.mini_size
            if offset + self.mini_size > len(self.mini_stream):
                raise ValueError("CFB MiniStream이 잘렸습니다.")
            return self.mini_stream[offset:offset + self.mini_size]

        data = self._chain(item["start"], self.mini_fat, mini_block)
        if len(data) < item["size"]:
            raise ValueError("CFB 미니 스트림 크기 오류입니다.")
        return data[:item["size"]]


def _decode_paragraph(payload):
    """Return text and number of invalid UTF-16 sequences replaced, excluding controls."""
    if len(payload) % 2:
        raise ValueError("HWP 문단의 UTF-16 길이가 잘렸습니다.")
    units = struct.unpack('<' + 'H' * (len(payload) // 2), payload)
    visible, index = [], 0
    while index < len(units):
        unit = units[index]
        if 1 <= unit <= 23 and unit not in (10, 13):
            if index + 8 > len(units):
                raise ValueError("HWP 인라인 제어문자가 잘렸습니다.")
            visible.append(9 if unit == 9 else 32)
            index += 8
            continue
        if unit in (10, 13):
            visible.append(10)
        elif unit == 24:
            visible.append(45)
        elif unit in (30, 31):
            visible.append(32)
        elif unit >= 32:
            visible.append(unit)
        index += 1
    raw = struct.pack('<' + 'H' * len(visible), *visible)
    try:
        text = raw.decode('utf-16-le')
        replacements = 0
    except UnicodeDecodeError:
        text = raw.decode('utf-16-le', errors='replace')
        replacements = text.count('\ufffd') - visible.count(0xFFFD)
    return unicodedata.normalize('NFC', text.strip()), replacements


def _records(stream):
    """Yield native records with their original decompressed-stream byte offsets."""
    offset = 0
    while offset < len(stream):
        start = offset
        if offset + 4 > len(stream):
            raise ValueError("HWP 레코드 헤더가 잘렸습니다.")
        record = struct.unpack_from('<I', stream, offset)[0]
        tag, level, length = record & 0x3FF, (record >> 10) & 0x3FF, record >> 20
        offset += 4
        if length == 0xFFF:
            if offset + 4 > len(stream):
                raise ValueError("HWP 확장 레코드 길이가 잘렸습니다.")
            length = struct.unpack_from('<I', stream, offset)[0]
            offset += 4
        if length > len(stream) - offset:
            raise ValueError("HWP 레코드 본문이 잘렸습니다.")
        yield tag, level, stream[offset:offset + length], start
        offset += length


def _cell_coordinates(payload, offset, nrows, ncols):
    if len(payload) < offset + 26:
        return None
    col, row, col_span, row_span = struct.unpack_from('<4H', payload, offset)
    if not (0 <= row < nrows and 0 <= col < ncols and
            1 <= row_span <= nrows - row and 1 <= col_span <= ncols - col):
        return None
    width, height = struct.unpack_from('<II', payload, offset + 8)
    return {'row': row, 'col': col, 'row_span': row_span, 'col_span': col_span,
            'width_hwpunit': width, 'height_hwpunit': height}


def _resolve_table(table, block_by_id):
    """Resolve one table's native coordinate records; never infer a missing cell."""
    nrows, ncols = table.get('_nrows', 0), table.get('_ncols', 0)
    raw_cells = table.pop('_raw_cells')
    candidates = []
    if nrows and ncols and nrows * ncols <= 500_000 and raw_cells:
        for offset in (6, 8):
            cells, occupied = [], set()
            for raw in raw_cells:
                cell = _cell_coordinates(raw['payload'], offset, nrows, ncols)
                if cell is None:
                    break
                region = {(r, c) for r in range(cell['row'], cell['row'] + cell['row_span'])
                          for c in range(cell['col'], cell['col'] + cell['col_span'])}
                if region & occupied:
                    break
                occupied.update(region)
                cells.append(cell)
            else:
                candidates.append((offset, cells, occupied))
    coord_signatures = {tuple((c['row'], c['col'], c['row_span'], c['col_span']) for c in cells)
                        for _, cells, _ in candidates}
    table['rows'] = []
    table['cells'] = []
    table['structure_status'] = 'unresolved'
    table['method'] = 'hwp_native_table_records'
    if not candidates or len(coord_signatures) != 1:
        table['warnings'].append('TABLE_COORDINATES_UNRESOLVED')
        table['unresolved_cells'] = [
            {'text': '\n'.join(c['texts']), 'paragraph_block_ids': c['block_ids'],
             'record_offset': c['record_offset']} for c in raw_cells]
    else:
        offset, cells, occupied = candidates[0]
        table['structure_status'] = 'explicit'
        table['list_header_bytes'] = offset
        table['row_count'], table['col_count'] = nrows, ncols
        table['rows'] = [[None for _ in range(ncols)] for _ in range(nrows)]
        for cell, raw in zip(cells, raw_cells):
            cell.update({'text': '\n'.join(raw['texts']), 'paragraph_block_ids': raw['block_ids'],
                         'declared_paragraphs': struct.unpack_from('<H', raw['payload'])[0],
                         'record_offset': raw['record_offset']})
            if raw.get('nested_table_ids'):
                cell['nested_table_ids'] = raw['nested_table_ids']
            table['rows'][cell['row']][cell['col']] = cell['text']
            for block_id in raw['block_ids']:
                block = block_by_id[block_id]
                block.update({'table_id': table['table_id'], 'row': cell['row'], 'col': cell['col']})
            table['cells'].append(cell)
        if len(occupied) != nrows * ncols:
            table['warnings'].append('TABLE_GRID_HAS_UNCOVERED_POSITIONS')
        if offset == 8:
            table['header_variant'] = 'observed_8_byte_list_header'
    for key in list(table):
        if key.startswith('_'):
            del table[key]
    return table


def _parse_section(stream, section):
    blocks, tables, warnings, stack = [], [], [], []
    paragraph = 0
    paragraph_cell = None
    paragraph_table = None
    paragraph_level = 0
    invalid_count = 0
    object_tags = Counter()
    for tag, level, payload, record_offset in _records(stream):
        while stack and level <= stack[-1]['_control_level']:
            stack.pop()
        if tag in (76, 84, 85, 86, 88):
            object_tags[tag] += 1
        if tag == 71 and payload[:4] == b' lbt':
            table_id = f'{section}:table:{len(tables)+1:04d}'
            table = {'table_id': table_id, 'section': section,
                     'locator': f'{section} / table {len(tables)+1} / record offset {record_offset}',
                     'record_offset': record_offset, 'warnings': [],
                     '_control_level': level, '_body_level': None, '_raw_cells': [], '_cell': None}
            if stack:
                table['parent_table_id'] = stack[-1]['table_id']
                if stack[-1].get('_cell') is not None:
                    stack[-1]['_cell'].setdefault('nested_table_ids', []).append(table_id)
            tables.append(table)
            stack.append(table)
        elif tag == 77 and stack and level == stack[-1]['_control_level'] + 1:
            table = stack[-1]
            if len(payload) >= 18:
                table['_nrows'], table['_ncols'] = struct.unpack_from('<HH', payload, 4)
                table['_body_level'] = level
            else:
                table['warnings'].append('TABLE_HEADER_TRUNCATED')
        elif tag == 72 and stack and level == stack[-1].get('_body_level'):
            raw = {'payload': payload, 'record_offset': record_offset, 'texts': [], 'block_ids': []}
            stack[-1]['_raw_cells'].append(raw)
            stack[-1]['_cell'] = raw
        elif tag == 66:
            paragraph += 1
            paragraph_level = level
            paragraph_table = stack[-1] if stack else None
            paragraph_cell = paragraph_table.get('_cell') if paragraph_table else None
        elif tag == 67:
            text, replacements = _decode_paragraph(payload)
            invalid_count += replacements
            locator = f'{section} / paragraph {paragraph}'
            if replacements:
                warnings.append({'code': 'INVALID_UTF16_REPLACED', 'message':
                                 f'{replacements} invalid UTF-16 sequence(s) replaced in extracted text; raw file unchanged.',
                                 'locator': locator, 'count': replacements})
            if text:
                block_id = f'{section}:paragraph:{paragraph}:record:{record_offset}'
                block = {'block_id': block_id, 'kind': 'paragraph', 'locator': locator,
                         'text': text, 'section': section, 'paragraph': paragraph,
                         'record_offset': record_offset, 'record_level': level}
                if paragraph_cell is not None and paragraph_table is not None and paragraph_level >= paragraph_table.get('_body_level', 0):
                    paragraph_cell['texts'].append(text)
                    paragraph_cell['block_ids'].append(block_id)
                    block['table_id'] = paragraph_table['table_id']
                blocks.append(block)
    by_id = {b['block_id']: b for b in blocks}
    tables = [_resolve_table(t, by_id) for t in tables]
    for table in tables:
        for code in table['warnings']:
            warnings.append({'code': code, 'message':
                             'Table coordinates/grid require review; text and native source offsets retained.',
                             'locator': table['locator']})
    return blocks, tables, warnings, {'paragraph_records': paragraph,
                                     'invalid_utf16_sequences': invalid_count,
                                     'nontext_record_counts': dict(object_tags)}


def parse_hwp(path):
    """Parse ordinary HWP5 files without changing their bytes or sending content out."""
    compound = _CompoundFile(Path(path))
    if 'FileHeader' not in compound.streams:
        raise ValueError('HWP FileHeader가 없습니다.')
    header = compound.read('FileHeader')
    if len(header) < 40 or not header.startswith(b'HWP Document File'):
        raise ValueError('HWP 문서 헤더가 올바르지 않습니다.')
    version, flags = struct.unpack_from('<II', header, 32)
    if version >> 24 != 5:
        raise ValueError('이 파서는 HWP 5.x만 지원합니다.')
    if flags & ((1 << 1) | (1 << 2) | (1 << 4)):
        raise ValueError('암호/배포용/DRM HWP 문서는 지원하지 않습니다.')
    sections = sorted((name for name in compound.streams
                       if name.startswith('BodyText/Section') and name.split('Section')[-1].isdigit()),
                      key=lambda name: int(name.split('Section')[-1]))
    if not sections:
        raise ValueError('HWP 본문 Section 스트림이 없습니다.')
    blocks, tables, warnings = [], [], []
    stats = {'sections': len(sections), 'paragraph_records': 0, 'invalid_utf16_sequences': 0}
    object_counts = Counter()
    for section in sections:
        stream = compound.read(section)
        if flags & 1:
            inflater = zlib.decompressobj(-15)
            stream = inflater.decompress(stream, 32 * 1024 * 1024)
            if not inflater.eof or inflater.unconsumed_tail:
                raise ValueError('HWP Section 압축이 손상되었거나 32 MiB 제한을 초과했습니다.')
        b, t, w, s = _parse_section(stream, section)
        blocks.extend(b); tables.extend(t); warnings.extend(w)
        stats['paragraph_records'] += s['paragraph_records']
        stats['invalid_utf16_sequences'] += s['invalid_utf16_sequences']
        object_counts.update(s['nontext_record_counts'])
    if not blocks:
        raise ValueError('HWP 본문 텍스트를 추출하지 못했습니다. 그림 문서는 OCR 검토가 필요합니다.')
    warnings.append({'code': 'HWP_LAYOUT_NOT_RENDERED', 'message':
                     'Source positions are section/paragraph/record offsets, not physical pages. Images, diagrams, equations and visual reading order are not rendered.'})
    if object_counts:
        warnings.append({'code': 'NON_TEXT_OBJECTS_PRESENT', 'message':
                         'Native drawing/image/equation records exist; textual extraction does not verify their visual meaning.',
                         'record_counts': dict(object_counts)})
    stats.update({'text_blocks': len(blocks), 'tables': len(tables),
                  'explicit_tables': sum(t['structure_status'] == 'explicit' for t in tables),
                  'unresolved_tables': sum(t['structure_status'] == 'unresolved' for t in tables),
                  'text_characters': sum(len(b['text']) for b in blocks),
                  'nontext_record_counts': dict(object_counts)})
    return {'detected_format': 'hwp', 'parser': 'local_hwp_v1', 'blocks': blocks,
            'reference_blocks': [], 'tables': tables, 'warnings': warnings, 'stats': stats}


