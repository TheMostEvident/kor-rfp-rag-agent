"""HWPX의 원문 순서, 중첩 표, 병합 좌표, 실패 경계에 대한 로컬 검증."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / 'local_ingest/hwpx.py'
SPEC = importlib.util.spec_from_file_location('phase2_hwpx_under_test', MODULE_PATH)
hwpx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hwpx)


def section(body):
    return ('<hs:sec xmlns:hs="urn:s" xmlns:hp="urn:p">' + body + '</hs:sec>').encode()


def p(text):
    return f'<hp:p><hp:run><hp:t>{text}</hp:t></hp:run></hp:p>'


def cell(text, row, col, row_span=1, col_span=1):
    return (f'<hp:tc><hp:subList>{p(text)}</hp:subList>'
            f'<hp:cellAddr rowAddr="{row}" colAddr="{col}"/>'
            f'<hp:cellSpan rowSpan="{row_span}" colSpan="{col_span}"/></hp:tc>')


class HwpxTests(unittest.TestCase):
    def parse(self, body=None, entries=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'document.hwpx'
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
                if entries is None:
                    archive.writestr('Contents/section0.xml', section(body))
                else:
                    for name, content in entries:
                        archive.writestr(name, content)
            return hwpx.parse_hwpx(path)

    def test_paragraph_text_and_source_compatibility_view(self):
        result = self.parse(p('A<hp:tab/>B<hp:lineBreak/>C') + p('서울'))
        self.assertEqual([b['text'] for b in result['blocks']], ['A\tB\nC', '서울'])
        self.assertEqual(result['reference_blocks'][0]['text'], 'ABC\n서울')
        self.assertEqual(result['blocks'][0]['paragraph'], 1)

    def test_nested_paragraph_text_is_not_duplicated_and_after_text_follows_table(self):
        table = '<hp:tbl rowCnt="1" colCnt="1"><hp:tr>' + cell('cell', 0, 0) + '</hp:tr></hp:tbl>'
        body = '<hp:p><hp:run><hp:t>before</hp:t>' + table + '<hp:t>after</hp:t></hp:run></hp:p>'
        result = self.parse(body)
        self.assertEqual([b['text'] for b in result['blocks']], ['before', 'cell', 'after'])
        self.assertEqual(result['blocks'][0]['paragraph'], result['blocks'][2]['paragraph'])
        self.assertEqual(result['blocks'][0]['segment'], 1)
        self.assertEqual(result['blocks'][2]['segment'], 2)
        self.assertEqual(result['tables'][0]['rows'], [['cell']])
        self.assertEqual(result['blocks'][1]['row'], 0)
        self.assertEqual(result['blocks'][1]['col'], 0)

    def test_merged_and_empty_cell_positions_are_preserved(self):
        table = ('<hp:tbl rowCnt="2" colCnt="3"><hp:tr>' + cell('merged', 0, 0, 2, 2)
                 + cell('', 0, 2) + '</hp:tr><hp:tr>' + cell('last', 1, 2) + '</hp:tr></hp:tbl>')
        result = self.parse('<hp:p><hp:run>' + table + '</hp:run></hp:p>')
        table = result['tables'][0]
        self.assertEqual(table['rows'], [['merged', None, ''], [None, None, 'last']])
        self.assertEqual(table['cells'][0]['row_span'], 2)
        self.assertEqual(table['cells'][0]['col_span'], 2)
        self.assertEqual(table['structure_status'], 'explicit')

    def test_nested_tables_keep_separate_cell_ownership(self):
        inner = '<hp:tbl rowCnt="1" colCnt="1"><hp:tr>' + cell('inside', 0, 0) + '</hp:tr></hp:tbl>'
        outer_cell = ('<hp:tc><hp:subList>' + p('outer') + '<hp:p><hp:run>' + inner
                      + '</hp:run></hp:p></hp:subList><hp:cellAddr rowAddr="0" colAddr="0"/>'
                      '<hp:cellSpan rowSpan="1" colSpan="1"/></hp:tc>')
        outer = '<hp:tbl rowCnt="1" colCnt="1"><hp:tr>' + outer_cell + '</hp:tr></hp:tbl>'
        result = self.parse('<hp:p><hp:run>' + outer + '</hp:run></hp:p>')
        self.assertEqual([b['text'] for b in result['blocks']], ['outer', 'inside'])
        self.assertEqual(len(result['tables']), 2)
        self.assertEqual(result['tables'][0]['cells'][0]['text'], 'outer')
        self.assertEqual(result['tables'][0]['cells'][0]['nested_table_ids'], [result['tables'][1]['table_id']])

    def test_missing_coordinates_are_not_guessed(self):
        body = '<hp:p><hp:run><hp:tbl rowCnt="1" colCnt="1"><hp:tr><hp:tc><hp:subList>'
        body += p('unlocated') + '</hp:subList></hp:tc></hp:tr></hp:tbl></hp:run></hp:p>'
        table = self.parse(body)['tables'][0]
        self.assertEqual(table['structure_status'], 'unresolved')
        self.assertIsNone(table['cells'][0]['row'])
        self.assertEqual(table['rows'], [[None]])

    def test_overlapping_spans_warn(self):
        table = '<hp:tbl rowCnt="1" colCnt="1"><hp:tr>' + cell('A', 0, 0) + cell('B', 0, 0) + '</hp:tr></hp:tbl>'
        result = self.parse('<hp:p><hp:run>' + table + '</hp:run></hp:p>')
        self.assertEqual(result['tables'][0]['structure_status'], 'unresolved')
        self.assertTrue(any(w['code'] == 'OVERLAPPING_CELL_SPANS' for w in result['warnings']))

    def test_invalid_span_is_not_silently_changed_to_one(self):
        table = '<hp:tbl rowCnt="1" colCnt="1"><hp:tr>' + cell('A', 0, 0, 'invalid', 1) + '</hp:tr></hp:tbl>'
        result = self.parse('<hp:p><hp:run>' + table + '</hp:run></hp:p>')
        self.assertEqual(result['tables'][0]['structure_status'], 'unresolved')
        self.assertIsNone(result['tables'][0]['cells'][0]['row_span'])

    def test_zero_column_count_cannot_bypass_dimension_limit(self):
        table = '<hp:tbl rowCnt="999999999" colCnt="0"/>'
        with self.assertRaisesRegex(ValueError, '표 크기'):
            self.parse(p('text') + '<hp:p><hp:run>' + table + '</hp:run></hp:p>')

    def test_numeric_section_order_and_deterministic_identifiers(self):
        entries = [('Contents/section10.xml', section(p('ten'))),
                   ('Contents/section2.xml', section(p('two'))),
                   ('Contents/section0.xml', section(p('zero')))]
        first = self.parse(entries=entries)
        second = self.parse(entries=entries)
        self.assertEqual([b['text'] for b in first['blocks']], ['zero', 'two', 'ten'])
        self.assertEqual(first, second)

    def test_absent_body_and_image_only_body_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'sectionN.xml'):
            self.parse(entries=[('mimetype', b'application/hwp+zip')])
        with self.assertRaisesRegex(ValueError, 'OCR'):
            self.parse('<hp:p><hp:run><hp:pic/></hp:run></hp:p>')

    def test_xml_failure_is_not_reported_as_missing_information(self):
        with self.assertRaisesRegex(ValueError, 'XML 파싱 실패'):
            self.parse(entries=[('Contents/section0.xml', b'<broken>')])

    def test_entity_declarations_are_rejected_without_external_access(self):
        raw = b'<!DOCTYPE a [<!ENTITY x "text">]><a>&x;</a>'
        with self.assertRaisesRegex(ValueError, 'DOCTYPE/ENTITY'):
            self.parse(entries=[('Contents/section0.xml', raw)])

    def test_grid_size_limit(self):
        table = '<hp:tbl rowCnt="9999999" colCnt="9999999"><hp:tr>' + cell('A', 0, 0) + '</hp:tr></hp:tbl>'
        with self.assertRaisesRegex(ValueError, '표 크기'):
            self.parse('<hp:p><hp:run>' + table + '</hp:run></hp:p>')

    def test_duplicate_archive_members_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with self.assertRaisesRegex(ValueError, '중복'):
                self.parse(entries=[('Contents/section0.xml', section(p('A'))),
                                    ('Contents/section0.xml', section(p('B')))])


if __name__ == '__main__':
    unittest.main()
