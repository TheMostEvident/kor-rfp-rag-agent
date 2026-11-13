import importlib.util
import json
from pathlib import Path
import shutil
import unittest

MODULE=Path(__file__).resolve().parents[1]/'local_ingest/ocr.py'
spec=importlib.util.spec_from_file_location('phase2_local_ocr',MODULE)
ocr=importlib.util.module_from_spec(spec);spec.loader.exec_module(ocr)


def critical_pdf():
    bundle=Path(__file__).resolve().parents[1]
    manifest=bundle/'data/input_manifest.json'
    if manifest.exists():
        for doc in json.loads(manifest.read_text(encoding='utf-8')):
            if doc.get('doc_id')=='1XuHMGElCN1wjEBBucRF55JWKMCePUmUM':
                path=bundle/doc['relative_path']
                if path.is_file():return path
    return Path(__file__).resolve().parents[3]/'drive_rename/raw/1XuHMGElCN1wjEBBucRF55JWKMCePUmUM'


class FakePage:
    def get_drawings(self):
        return [
            {'type':'f','rect':Rect(1,10,5,20),'items':[('c',)]},
            {'type':'f','rect':Rect(6,10,9,20),'items':[('l',)]},
            {'type':'f','rect':Rect(10,18,12,22),'items':[('c',)]},
            {'type':'f','rect':Rect(1,30,100,31),'items':[('re',)]},
            {'type':'s','rect':Rect(1,40,10,41),'items':[('l',)]},
        ]


class Rect(list):
    def __init__(self,*args):super().__init__(args)
    @property
    def width(self):return self[2]-self[0]
    @property
    def height(self):return self[3]-self[1]


class VectorOCRTests(unittest.TestCase):
    def test_group_retains_straight_glyph_and_comma_but_not_table_line(self):
        result=ocr._candidate_regions(FakePage())
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['path_count'],3)
        self.assertEqual(result[0]['bbox'],[1,10,12,22])

    def test_actual_critical_page_is_local_unreviewed_supplement(self):
        raw=critical_pdf()
        if not raw.exists() or not shutil.which('tesseract'):
            self.skipTest('Original critical PDF or local tesseract unavailable')
        result=ocr.inspect_vector_regions(raw,pages=[3],max_regions=20)
        tokens=[token for block in result['blocks'] for token in block['numeric_candidates']]
        self.assertIn('71,620,000,000',tokens)
        self.assertIn('16',tokens)
        self.assertTrue(all(block['review_status']=='unreviewed' for block in result['blocks']))
        self.assertTrue(all(block['page']==3 and len(block['bbox'])==4 for block in result['blocks']))
        self.assertEqual(result['stats']['physical_pages'],[3])

    def test_invalid_page_rejected_without_guessing(self):
        raw=critical_pdf()
        if not raw.exists():self.skipTest('Original unavailable')
        with self.assertRaises(ValueError):ocr.inspect_vector_regions(raw,pages=[0],run_ocr=False)


if __name__=='__main__':unittest.main()
