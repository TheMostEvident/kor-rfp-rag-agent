"""Read actual temporary PDF bytes; no network, model, or previous index mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
import shutil

from rfp_phase5_corpus.intake import build_workspace,load_parser
from rfp_phase5.runtime import bootstrap_phase5
from rfp_phase5_final.catalog_audit import build_catalog


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.raw=self.base/'raw';self.raw.mkdir()
        self.project=Path(__file__).resolve().parents[1]
        self.parser=self.project.parent/'rfp_phase2_local_20261111'
        # Developer verification may use the complete read-only original package.
        if not (self.parser/'local_ingest/loader.py').is_file():
            self.parser=Path(__file__).resolve().parents[4]/'output/rfp_phase2_local_20261111'
        if not (self.parser/'local_ingest/loader.py').is_file():
            self.skipTest('Phase2 parser package is required for native-PDF intake integration')
        self.net=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        self.net.start();self.addCleanup(self.net.stop)

    def pdf(self,name='proposal.pdf'):
        import fitz
        path=self.raw/name
        with fitz.open() as document:
            page=document.new_page();page.insert_text((60,70),'Request for proposal: alpha platform implementation. Budget 123456 USD.')
            page.insert_text((60,100),'The contract duration is 12 months including stabilization and acceptance.')
            document.save(path)
        return path

    def build(self,**kwargs):
        return build_workspace(self.project,self.raw,parser_root=self.parser,workspace=self.base/'workspace',**kwargs)

    def test_native_pdf_can_be_parsed_indexed_and_loaded_as_new_rag_project(self):
        source=self.pdf();before=hashlib.sha256(source.read_bytes()).hexdigest()
        report=self.build()
        self.assertTrue(report['all_inputs_parsed']);self.assertEqual(report['format_counts']['pdf'],1)
        self.assertEqual(report['api_calls'],0);self.assertGreater(report['chunks'],0)
        env=bootstrap_phase5(report['workspace'])
        try:
            self.assertEqual(len(build_catalog(env)),1)
            self.assertIn('123456',env['chunks'][0]['text'])
        finally:env['index'].close()
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)

    def test_identical_input_reuses_verified_workspace_without_rebuild(self):
        self.pdf();first=self.build();second=self.build()
        self.assertEqual(first['input_fingerprint'],second['input_fingerprint'])
        self.assertTrue(second['reused_existing_workspace'])

    def test_one_corrupt_file_is_reported_and_does_not_discard_other_files(self):
        self.pdf();(self.raw/'broken.hwp').write_bytes(b'not a valid document')
        report=self.build()
        self.assertFalse(report['all_inputs_parsed'])
        self.assertEqual(report['input_files'],2);self.assertEqual(report['parsed_files'],1)
        self.assertEqual(report['failed_files'],1);self.assertFalse(report['failures'][0]['indexed'])

    def test_changed_raw_files_cannot_reuse_an_old_workspace(self):
        self.pdf();self.build();(self.raw/'new.pdf').write_bytes(b'%PDF-broken')
        with self.assertRaises(ValueError):self.build()

    def test_active_project_or_raw_directory_cannot_be_overwritten(self):
        self.pdf()
        for workspace in (self.project,self.raw,self.raw/'inside'):
            with self.assertRaises(ValueError):
                build_workspace(self.project,self.raw,parser_root=self.parser,workspace=workspace)

    def test_all_failed_inputs_have_explicit_failure_report(self):
        (self.raw/'broken.pdf').write_bytes(b'not a pdf')
        with self.assertRaises(ValueError):self.build()
        failure=json.loads((self.base/'workspace/reports/intake_failure.json').read_text())
        self.assertEqual(failure['files'],1);self.assertEqual(len(failure['failures']),1)

    def test_reference_only_document_is_isolated_before_chunk_build(self):
        self.pdf();(self.raw/'reference_only.hwpx').write_bytes(b'not an actual fixture document')
        loader=load_parser(self.parser);original=loader.run_ingestion
        def parse_with_reference_only(*args,**kwargs):
            values=original(*args,**kwargs)
            for d in values:
                if d['file_name']=='reference_only.hwpx':
                    d.update(status='parsed',blocks=[],reference_blocks=[{'text':'reference view only'}])
            return values
        with patch.object(loader,'run_ingestion',side_effect=parse_with_reference_only):report=self.build()
        self.assertEqual(report['parsed_files'],1);self.assertEqual(report['failed_files'],1)
        self.assertEqual(report['failures'][0]['error']['code'],'no_extractable_primary_text')

    def test_parser_code_drift_requires_restart_instead_of_mislabeled_versions(self):
        fake=SimpleNamespace(__file__=str(self.parser/'local_ingest/__init__.py'),_corpus_parser_versions={})
        with patch.dict(sys.modules,{'local_ingest':fake}):
            with self.assertRaisesRegex(RuntimeError,'재시작'):load_parser(self.parser)

    def test_only_sha_and_id_matched_reviewed_transcriptions_are_preserved(self):
        source=self.pdf();digest=hashlib.sha256(source.read_bytes()).hexdigest()
        template=self.base/'template';template.mkdir()
        for name in ('configs','local_rag','rfp_phase4','rfp_phase5','rfp_phase5_final'):
            shutil.copytree(self.project/name,template/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (template/'data').mkdir()
        (template/'data/source_documents.jsonl').write_text(json.dumps({'doc_id':'known','canonical_doc_id':'known',
            'sha256':digest,'file_name':source.name,'agency':'Test agency','document_year':2026,'source_url':None})+'\n')
        item={'doc_id':'known','sha256':digest,'review_status':'visual_verified','bbox':[50,50,300,90],
            'check_id':'fixture-only','page':1,'text':'123456 USD','review_note':'synthetic test fixture'}
        annotations=[item,dict(item,doc_id='different_document'),dict(item,sha256='0'*64)]
        annotation_path=(self.parser/'data/reviewed_transcriptions.json').resolve()
        read_text=Path.read_text
        def override(path,*args,**kwargs):
            return json.dumps(annotations) if path.resolve()==annotation_path else read_text(path,*args,**kwargs)
        with patch.object(Path,'read_text',override):
            report=build_workspace(template,self.raw,parser_root=self.parser,workspace=self.base/'workspace')
        self.assertEqual(report['version_matched_reviewed_supplements'],1)
        doc=json.loads((Path(report['workspace'])/'data/source_documents.jsonl').read_text())
        self.assertEqual(doc['doc_id'],'known');self.assertEqual(len(doc['supplemental_blocks']),1)
        self.assertEqual(doc['supplemental_blocks'][0]['text'],'123456 USD')


if __name__=='__main__':unittest.main()
