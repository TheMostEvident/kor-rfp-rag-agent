import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_phase5_corpus.workflow import audit_corpus


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.env={'root':self.root,'documents':[{'doc_id':'good','canonical_doc_id':'good','file_name':'good.pdf'}]}
        self.raw={'rows':[{'doc_id':'good','read_success':True,'snapshot_compatible':True,
            'sha256_matches':True,'needs_review':False,'review_codes':[],'detected_format':'pdf'}],
            'unregistered_sources':[],'unregistered_source_count':0,'fresh_parsing_verified':False}
        self.index={'documents':[{'doc_id':'good','text_pipeline_ready':True,'chunk_count':1,
                    'source_character_coverage':1,'review_reasons':[]}],'canonical_document_count':1}

    def audit(self,**kwargs):
        with patch('rfp_phase5_corpus.workflow.audit_raw_corpus',return_value=self.raw),\
             patch('rfp_phase5_corpus.workflow.audit_indexed_corpus',return_value=self.index):
            return audit_corpus(self.env,parser_root=self.root,**kwargs)

    def test_success_is_text_readiness_not_semantic_completeness(self):
        report=self.audit()
        self.assertEqual(report['original_documents'],1)
        self.assertTrue(report['all_existing_documents_text_ready'])
        self.assertFalse(report['all_document_semantic_quality_verified'])

    def test_failed_intake_files_are_included_in_the_complete_document_csv(self):
        folder=self.root/'reports';folder.mkdir()
        (folder/'intake_report.json').write_text(json.dumps({'failures':[{
            'doc_id':'bad','file_name':'bad.hwp','relative_path':'data/raw/bad.hwp'}]}))
        self.raw['unregistered_sources']=[{'relative_path':'bad.hwp'}]
        self.raw['unregistered_source_count']=1
        report=self.audit()
        self.assertEqual(report['original_documents'],2)
        self.assertEqual(report['blocked_original_documents'],1)
        self.assertFalse(report['all_existing_documents_text_ready'])
        self.assertTrue(report['all_raw_files_registered'])
        self.assertIn('bad.hwp',(self.root/'phase5_outputs/corpus_audit/documents.csv').read_text())

    def test_ocr_warning_requires_explicit_generation_acknowledgment(self):
        self.raw['rows'][0].update(needs_review=True,review_codes=['OCR_OR_IMAGE_REVIEW_NEEDED'])
        report=self.audit()
        self.assertTrue(report['all_existing_documents_text_ready'])
        self.assertEqual(report['eligible_canonical_doc_ids'],[])
        self.assertTrue(report['documents'][0]['generation_requires_visual_review'])
        accepted=self.audit(allow_unreviewed_sources=True)
        self.assertEqual(accepted['eligible_canonical_doc_ids'],['good'])
        self.assertEqual(accepted['documents'][0]['status'],'needs_review')

    def test_changed_parser_never_enters_generation_eligible_set(self):
        self.raw['rows'][0].update(snapshot_compatible=False,review_codes=['PARSER_VERSION_CHANGED'])
        report=self.audit(allow_unreviewed_sources=True)
        self.assertEqual(report['eligible_canonical_doc_ids'],[])
        self.assertEqual(report['blocked_original_documents'],1)

    def test_new_unregistered_files_remain_visible(self):
        self.raw['unregistered_sources']=[{'relative_path':'new.pdf'}]
        self.raw['unregistered_source_count']=1
        self.assertFalse(self.audit()['all_raw_files_registered'])


if __name__=='__main__':unittest.main()
