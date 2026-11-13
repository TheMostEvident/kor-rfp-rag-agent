"""Meaningful HWP record/control/table corruption and provenance regression tests."""
import struct
import unittest
from unittest.mock import patch

from local_ingest.hwp import _decode_paragraph, _parse_section, _records, parse_hwp


def record(tag, level, payload=b'', extended=False):
    size = 0xFFF if extended else len(payload)
    head = struct.pack('<I', tag | (level << 10) | (size << 20))
    return head + (struct.pack('<I', len(payload)) if extended else b'') + payload


def para(text, level):
    return record(66, level, b'\0'*24) + record(67, level+1, text.encode('utf-16-le'))


def table_header(nrows, ncols, level=1):
    return (record(71, level, b' lbt') +
            record(77, level+1, struct.pack('<IHHH4H', 0, nrows, ncols, 0, 0, 0, 0, 0)))


def cell(row, col, text, row_span=1, col_span=1, level=2, header_bytes=8):
    header = struct.pack('<II', 1, 0x01000020) if header_bytes == 8 else struct.pack('<HI', 1, 0)
    attributes = struct.pack('<4HII4HH', col, row, col_span, row_span, 1000, 1000, 0, 0, 0, 0, 1)
    return record(72, level, header + attributes) + para(text, level)


class HwpTests(unittest.TestCase):
    def test_unicode_controls_and_literal_replacement(self):
        text, count = _decode_paragraph('한글😀�'.encode('utf-16-le'))
        self.assertEqual(text, '한글😀�')
        self.assertEqual(count, 0)
        controls = struct.pack('<8H', 9, 0, 0, 0, 0, 0, 0, 9)
        text, count = _decode_paragraph('A'.encode('utf-16-le') + controls + 'B'.encode('utf-16-le'))
        self.assertEqual(text, 'A\tB')
        self.assertEqual(count, 0)

    def test_invalid_surrogate_is_counted_without_losing_neighbors(self):
        text, count = _decode_paragraph('A'.encode('utf-16-le') + b'\x00\xd8' + 'B'.encode('utf-16-le'))
        self.assertEqual((text, count), ('A�B', 1))

    def test_truncated_control_and_odd_utf16_are_errors(self):
        for payload in [b'A', struct.pack('<H', 9)]:
            with self.assertRaises(ValueError):
                _decode_paragraph(payload)

    def test_extended_records_and_truncation(self):
        data = record(67, 2, b'abcd', extended=True)
        self.assertEqual(list(_records(data)), [(67, 2, b'abcd', 0)])
        for invalid in [data[:3], data[:6], data[:-1]]:
            with self.assertRaises(ValueError):
                list(_records(invalid))

    def test_explicit_merged_cell_and_paragraph_provenance(self):
        data = (para('before', 0) + table_header(2, 2) +
                cell(0, 0, 'merged', col_span=2) + cell(1, 0, 'left') +
                cell(1, 1, 'right') + para('after', 0))
        blocks, tables, warnings, stats = _parse_section(data, 'BodyText/Section0')
        self.assertEqual([b['paragraph'] for b in blocks], [1, 2, 3, 4, 5])
        self.assertEqual(tables[0]['rows'], [['merged', None], ['left', 'right']])
        self.assertEqual(tables[0]['cells'][0]['col_span'], 2)
        self.assertEqual(tables[0]['structure_status'], 'explicit')
        self.assertEqual((blocks[2]['row'], blocks[2]['col']), (1, 0))
        self.assertNotIn('table_id', blocks[-1])

    def test_official_six_byte_list_header_is_supported(self):
        blocks, tables, _, _ = _parse_section(table_header(1, 1) + cell(0, 0, 'six', header_bytes=6), 'BodyText/Section0')
        self.assertEqual(tables[0]['list_header_bytes'], 6)
        self.assertEqual(tables[0]['rows'], [['six']])

    def test_out_of_bounds_table_is_not_guessed(self):
        blocks, tables, warnings, _ = _parse_section(table_header(1, 1) + cell(0, 2, 'retained'), 'BodyText/Section0')
        self.assertEqual(blocks[0]['text'], 'retained')
        self.assertEqual(tables[0]['structure_status'], 'unresolved')
        self.assertEqual(tables[0]['rows'], [])
        self.assertNotIn('col', blocks[0])
        self.assertTrue(any(w['code'] == 'TABLE_COORDINATES_UNRESOLVED' for w in warnings))

    def test_overlapping_native_cells_are_not_fabricated(self):
        _, tables, _, _ = _parse_section(table_header(1, 2) + cell(0, 0, 'a', col_span=2) + cell(0, 1, 'b'), 'BodyText/Section0')
        self.assertEqual(tables[0]['structure_status'], 'unresolved')
        self.assertEqual([c['text'] for c in tables[0]['unresolved_cells']], ['a', 'b'])

    def test_nested_table_has_distinct_source_cells(self):
        data = (table_header(1, 1) + cell(0, 0, 'outer') +
                table_header(1, 1, level=3) + cell(0, 0, 'inner', level=4) + para('end', 0))
        blocks, tables, _, _ = _parse_section(data, 'BodyText/Section0')
        self.assertEqual(tables[0]['rows'], [['outer']])
        self.assertEqual(tables[1]['rows'], [['inner']])
        self.assertEqual(tables[1]['parent_table_id'], tables[0]['table_id'])
        self.assertEqual(tables[0]['cells'][0]['nested_table_ids'], [tables[1]['table_id']])

    def test_protected_files_fail_before_body_processing(self):
        header = bytearray(40)
        header[:17] = b'HWP Document File'
        struct.pack_into('<II', header, 32, 0x05000000, 2)
        fake = type('FakeCompound', (), {'streams': {'FileHeader': {}}, 'read': lambda self, name: bytes(header)})
        with patch('local_ingest.hwp._CompoundFile', return_value=fake()):
            with self.assertRaisesRegex(ValueError, '암호'):
                parse_hwp('does_not_need_to_exist.hwp')


if __name__ == '__main__':
    unittest.main()
