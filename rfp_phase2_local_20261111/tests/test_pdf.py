"""Real local PDF fixtures exercise page identity, tables and missing-text flags."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase2_pdf_reader', ROOT / 'local_ingest/pdf.py')
pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdf)


class PDFParsingTests(unittest.TestCase):
    def setUp(self):
        import fitz
        self.fitz = fitz
        self.directory = tempfile.TemporaryDirectory(prefix='rfp-pdf-test-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.pdf'

    def test_physical_pages_and_blank_page_are_preserved(self):
        doc = self.fitz.open()
        page = doc.new_page()
        page.insert_text((40, 50), 'Budget 250000000 VAT included')
        doc.new_page()
        doc.save(self.path);doc.close()
        result = pdf.parse_pdf(self.path, extract_tables=False)
        self.assertEqual([b['locator'] for b in result['blocks']], ['PDF p.1', 'PDF p.2'])
        self.assertEqual([b['page'] for b in result['blocks']], [1, 2])
        self.assertIn('250000000', result['blocks'][0]['text'])
        self.assertEqual(result['blocks'][1]['text'], '')
        self.assertEqual(result['stats']['empty_text_pages'], [2])
        self.assertIn('PDF_EMPTY_TEXT_PAGE', {w['code'] for w in result['warnings']})

    def test_table_geometry_and_merged_cells_remain_heuristic(self):
        doc = self.fitz.open();page = doc.new_page()
        page.draw_rect(self.fitz.Rect(40, 40, 240, 120))
        page.draw_line(self.fitz.Point(40, 80), self.fitz.Point(240, 80))
        page.draw_line(self.fitz.Point(140, 80), self.fitz.Point(140, 120))
        page.insert_text((50, 65), 'Merged header')
        page.insert_text((50, 105), 'Left')
        page.insert_text((150, 105), 'Right')
        doc.save(self.path);doc.close()
        result = pdf.parse_pdf(self.path)
        self.assertEqual(len(result['tables']), 1)
        table = result['tables'][0]
        self.assertEqual(table['structure_status'], 'heuristic')
        self.assertEqual(table['page'], 1)
        self.assertEqual(table['rows'][0][0], 'Merged header')
        self.assertIsNone(table['rows'][0][1])
        self.assertEqual(table['rows'][1], ['Left', 'Right'])
        header = next(c for c in table['cells'] if c['row'] == 0 and c['col'] == 0)
        self.assertEqual(header['col_span'], 2)
        self.assertEqual(header['row_span'], 1)
        self.assertEqual(header['bbox'], [40.0, 40.0, 240.0, 80.0])

    def test_image_only_page_is_not_silently_successful_text(self):
        image = self.fitz.Pixmap(self.fitz.csRGB,self.fitz.IRect(0,0,40,40),False)
        image.clear_with(255)
        doc = self.fitz.open();page = doc.new_page()
        page.insert_image(self.fitz.Rect(40, 40, 400, 400), stream=image.tobytes('png'))
        doc.save(self.path);doc.close()
        result = pdf.parse_pdf(self.path)
        self.assertEqual(result['blocks'][0]['quality']['image_count'], 1)
        self.assertIn('PDF_OCR_REVIEW_NEEDED', {w['code'] for w in result['warnings']})
        self.assertFalse(result['stats']['ocr_executed'])
        self.assertEqual(result['stats']['text_characters'], 0)

    def test_missing_numeric_fields_are_flagged_without_filling_values(self):
        gaps = pdf._numeric_gaps('사업기간( )계약체결일로부터 개월 이내\n사업예산( ) 원 이내 부가세 포함')
        self.assertEqual({g['field_hint'] for g in gaps}, {'budget', 'duration'})
        self.assertEqual(pdf._numeric_gaps('사업기간 : 16개월\n사업예산 : 71,620,000,000원'), [])

    def test_inconsistent_merged_span_is_unknown_not_clamped(self):
        payload={'tables':[{'rows':[['A','B'],['C','D']],
            'cells':[{'row':0,'col':0,'row_span':3,'col_span':1,'text':'A','bbox':[0,0,10,30]}],
            'structure_status':'heuristic','warnings':[]}], 'stats':{},
            'warnings':[{'code':'PDF_TABLES_HEURISTIC','message':'Review'}]}
        result=pdf.refresh_pdf_table_geometry(payload)
        cell=result['tables'][0]['cells'][0]
        self.assertIsNone(cell['row_span'])
        self.assertEqual(cell['inferred_row_span'],3)
        self.assertEqual(cell['bbox'],[0,0,10,30])
        self.assertEqual(result['tables'][0]['rows'],[['A','B'],['C','D']])
        self.assertEqual(result['tables'][0]['structure_status'],'unresolved')
        first=repr(result)
        self.assertEqual(repr(pdf.refresh_pdf_table_geometry(result)),first)

    def test_table_switch_keeps_text_and_reports_omitted_detection(self):
        doc = self.fitz.open();page = doc.new_page()
        page.insert_text((40, 50), 'A local extraction test')
        doc.save(self.path);doc.close()
        enabled = pdf.parse_pdf(self.path)
        disabled = pdf.parse_pdf(self.path, extract_tables=False)
        self.assertEqual(enabled['blocks'][0]['text'], disabled['blocks'][0]['text'])
        self.assertEqual(disabled['tables'], [])
        self.assertIn('PDF_TABLE_EXTRACTION_SKIPPED', {w['code'] for w in disabled['warnings']})

    def test_non_pdf_signature_is_rejected(self):
        self.path.write_bytes(b'not a pdf document')
        with self.assertRaisesRegex(ValueError, 'PDF_SIGNATURE_INVALID'):
            pdf.parse_pdf(self.path)

    def test_rotated_page_retains_physical_identity_and_declares_display_coordinates(self):
        doc = self.fitz.open();page = doc.new_page(width=300, height=500)
        for rectangle in [(40,40,140,80),(140,40,240,80),(40,80,140,120),(140,80,240,120)]:
            page.draw_rect(self.fitz.Rect(rectangle))
        for point, text in [((50,65),'A'),((150,65),'B'),((50,105),'C'),((150,105),'D')]:
            page.insert_text(point,text)
        page.set_rotation(90)
        doc.save(self.path);doc.close()
        result = pdf.parse_pdf(self.path)
        self.assertEqual(result['blocks'][0]['page'],1)
        self.assertEqual(result['blocks'][0]['quality']['rotation_degrees'],90)
        self.assertEqual(result['tables'][0]['bbox_coordinate_system'],'displayed_page_points_top_left')
        self.assertEqual(result['tables'][0]['bbox'],[380.0,40.0,460.0,240.0])


if __name__ == '__main__':
    unittest.main()
