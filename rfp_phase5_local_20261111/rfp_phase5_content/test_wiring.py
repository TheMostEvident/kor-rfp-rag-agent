"""Existing notebook runner entrypoints use correction without paid retries."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rfp_phase5_final'))
from test_catalog_audit import env as small_env
from rfp_phase5_corpus.execution import CorpusRun
from rfp_phase5_recovery.test_recovery import FakeClient
from rfp_phase5_content.storage import load_saved_result


class ContentWiringTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.env=small_env();self.env['root']=self.root
        for name in ('rfp_phase4.llm','rfp_phase4.evidence','rfp_phase5.fields'):
            source=Path(importlib.import_module(name).__file__)
            target=self.root/(name.replace('.','/')+'.py')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        self.client=FakeClient()
        self.runner=CorpusRun(self.env,'content_wiring',client=self.client)
        for patcher in [patch.dict('os.environ',{'OPENAI_API_KEY':'offline-test-only'}),
                        patch('socket.socket.connect',side_effect=AssertionError('network prohibited')),
                        patch('socket.socket.connect_ex',side_effect=AssertionError('network prohibited'))]:
            patcher.start();self.addCleanup(patcher.stop)

    def test_default_result_is_corrected_and_generation_snapshot_stays_raw(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.runner.run_corpus(['doc-a'],allow_api=True,max_new_calls=1)
        requests=len(self.client.requests)
        old_latest=(self.runner.folder/'documents/doc-a/latest.json').read_bytes()
        raw,raw_folder=load_saved_result(self.runner.folder,'doc-a')
        original={p:p.read_bytes() for p in raw_folder.iterdir() if p.is_file()}
        result=self.runner.result('copy-a')
        self.assertIn('content_storage',result)
        self.assertIn('content_audit',result)
        self.assertNotIn('content_storage',raw)
        self.assertEqual(result['doc_id'],'doc-a')
        self.assertEqual(len(self.client.requests),requests)
        self.assertEqual((self.runner.folder/'documents/doc-a/latest.json').read_bytes(),old_latest)
        for path,expected in original.items():self.assertEqual(path.read_bytes(),expected)
        _,corrected_folder=load_saved_result(self.runner.folder,'doc-a',content=True)
        self.assertNotEqual(corrected_folder,raw_folder)

    def test_report_and_export_leave_original_report_archive_and_ledger_untouched(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.runner.run_corpus(['doc-a'],allow_api=True,max_new_calls=1)
        raw_report=self.runner.folder/'final_report.json'
        raw_report.write_text('{"original":"report"}\n',encoding='utf-8')
        original_zip=self.runner.folder.parent/(self.runner.folder.name+'_results.zip')
        original_zip.write_bytes(b'original zip retained')
        journal=self.runner.folder/'journal.json';old_journal=journal.read_bytes()
        requests=len(self.client.requests)
        report=self.runner.report(['doc-a'])
        self.assertEqual(report['version'],'phase5-content-report-v1')
        self.assertFalse(report['phase5_sample_validation_passed'])
        self.assertFalse(report['cases'][0]['review']['human_review_completed'])
        target=self.runner.export(['doc-a'])
        self.assertTrue(target.name.endswith('_content_results.zip'))
        with zipfile.ZipFile(target) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue(any('/content_documents/' in name for name in archive.namelist()))
        self.assertEqual(raw_report.read_text(),'{"original":"report"}\n')
        self.assertEqual(original_zip.read_bytes(),b'original zip retained')
        self.assertEqual(journal.read_bytes(),old_journal)
        self.assertEqual(len(self.client.requests),requests)

    def test_pending_generation_stays_pending_in_corrected_view(self):
        result=self.runner.result('doc-a')
        self.assertEqual(result['execution_status'],'awaiting_api')
        self.assertFalse(result['answer_generated'])
        self.assertEqual(result['checked_chunk_coverage'],0)
        self.assertEqual(self.client.requests,[])


if __name__=='__main__':unittest.main()
