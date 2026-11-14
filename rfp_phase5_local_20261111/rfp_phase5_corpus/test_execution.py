"""Whole-corpus execution checks use synthetic sources and no model service."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rfp_phase5_final'))
from test_catalog_audit import env as small_env, corpus, Index
from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase5_final.engine import FinalConfig
from rfp_phase5_recovery.test_recovery import FakeClient
from rfp_phase5_corpus.execution import CorpusRun, open_corpus_run


class CorpusExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.env=small_env();self.env['root']=self.root
        for name in ['rfp_phase4.llm','rfp_phase4.evidence','rfp_phase5.fields']:
            source=Path(importlib.import_module(name).__file__)
            target=self.root/(name.replace('.','/')+'.py')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        self.fake=FakeClient()
        for patcher in [patch.dict('os.environ',{'OPENAI_API_KEY':'offline-corpus-test-only'}),
                        patch('socket.socket.connect',side_effect=AssertionError('network prohibited')),
                        patch('socket.socket.connect_ex',side_effect=AssertionError('network prohibited'))]:
            patcher.start();self.addCleanup(patcher.stop)

    def runner(self,config=None,env=None):
        return CorpusRun(env or self.env,'test',config=config,client=self.fake)

    def execute(self,runner,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return runner.run_corpus(*args,**kwargs)

    def test_default_selects_whole_corpus_and_calls_no_api(self):
        report=self.execute(self.runner())
        self.assertEqual(report['selected_canonical_documents'],2)
        self.assertEqual(report['selected_original_documents'],3)
        self.assertTrue(report['all_canonical_documents_selected'])
        self.assertEqual(report['awaiting_batches'],2)
        self.assertFalse(report['all_selected_generation_completed'])
        self.assertEqual(self.fake.requests,[])

    def test_partial_progress_is_allowed_when_entire_remainder_exceeds_budget(self):
        runner=self.runner(FinalConfig(max_total_calls=1))
        self.assertFalse(runner.preflight('all')['within_remaining_budget'])
        report=self.execute(runner,allow_api=True,max_new_calls=20)
        self.assertEqual(len(self.fake.requests),1)
        self.assertEqual(report['completed_documents'],1)
        self.assertEqual(report['awaiting_batches'],1)
        self.assertFalse(report['invocation']['whole_remaining_corpus_must_fit'])

    def test_per_invocation_limit_is_independent_of_durable_total_limit(self):
        runner=self.runner()
        first=self.execute(runner,allow_api=True,max_new_calls=1)
        self.assertEqual(first['invocation']['api_calls_used_or_reserved_this_invocation'],1)
        self.assertEqual(first['budget']['max_total_calls'],100)
        second=self.execute(runner,allow_api=True,max_new_calls=1)
        self.assertEqual(second['invocation']['api_calls_used_or_reserved_this_invocation'],1)
        self.assertEqual(len(self.fake.requests),2)
        self.assertTrue(second['all_selected_generation_completed'])
        self.assertFalse(second['all_documents_semantic_quality_verified'])

    def test_completed_jobs_and_written_reviews_survive_resume(self):
        runner=self.runner();self.execute(runner,allow_api=True,max_new_calls=1)
        job=next(v for v in runner._journal()['jobs'].values() if v['status']=='completed')
        saved_job=copy.deepcopy(job);outcome=runner.folder/job['outcome_file'];raw=outcome.read_bytes()
        reviews={p:p.read_bytes() for p in runner.folder.rglob('comparison.csv')}
        self.execute(runner,allow_api=True,max_new_calls=1)
        self.assertEqual(outcome.read_bytes(),raw)
        self.assertIn(saved_job,list(runner._journal()['jobs'].values()))
        for p,data in reviews.items():self.assertEqual(p.read_bytes(),data)
        again=self.execute(runner,allow_api=True,max_new_calls=20)
        self.assertEqual(again['invocation']['api_calls_used_or_reserved_this_invocation'],0)

    def test_zero_new_call_limit_is_cache_only(self):
        report=self.execute(self.runner(),allow_api=True,max_new_calls=0)
        self.assertEqual(report['budget']['calls_used_or_reserved'],0)
        self.assertEqual(self.fake.requests,[])

    def test_explicit_total_limit_increase_reuses_old_completed_jobs(self):
        first=self.runner(FinalConfig(max_total_calls=1))
        self.execute(first,allow_api=True,max_new_calls=1)
        reopened=open_corpus_run(self.env,'test',client=self.fake,max_total_calls=3)
        report=self.execute(reopened,allow_api=True,max_new_calls=1)
        self.assertEqual(len(self.fake.requests),2)
        self.assertEqual(report['budget']['calls_used_or_reserved'],2)
        self.assertEqual(report['budget']['max_total_calls'],3)

    def test_one_document_plan_failure_does_not_abort_other_documents(self):
        runner=self.runner();original=runner.plan
        def bad_one(doc_id):
            if doc_id=='doc-a':raise ValueError('synthetic unreadable document')
            return original(doc_id)
        with patch.object(runner,'plan',side_effect=bad_one):
            report=self.execute(runner,allow_api=True,max_new_calls=20)
        rows={r['doc_id']:r for r in report['documents']}
        self.assertEqual(rows['doc-a']['status'],'document_error')
        self.assertEqual(rows['doc-b']['status'],'completed')
        self.assertEqual(len(self.fake.requests),1)
        self.assertEqual(report['document_errors'],1)

    def test_failed_generation_does_not_abort_later_documents(self):
        self.fake.fail_on=1
        report=self.execute(self.runner(),allow_api=True,max_new_calls=20)
        self.assertEqual(report['failed_batches'],1)
        self.assertEqual(report['completed_documents'],1)
        self.assertEqual(len(self.fake.requests),2)
        self.assertFalse(report['all_selected_generation_completed'])

    def test_service_failure_stops_further_paid_calls_until_explicit_resume(self):
        self.fake.fail_on=1;self.fake.mode='network'
        report=self.execute(self.runner(),allow_api=True,max_new_calls=20)
        self.assertEqual(len(self.fake.requests),1)
        self.assertEqual(report['failed_batches'],1)
        self.assertEqual(report['awaiting_batches'],1)

    def test_original_alias_selection_deduplicates_without_widening_scope(self):
        report=self.execute(self.runner(),['doc-a','copy-a'],allow_api=True,max_new_calls=20)
        self.assertEqual(report['selected_doc_ids'],['doc-a'])
        self.assertEqual(report['selected_original_documents'],2)
        self.assertFalse(report['all_canonical_documents_selected'])
        self.assertEqual(len(self.fake.requests),1)

    def test_all_scope_has_no_seven_or_five_document_limit(self):
        template_doc=corpus()[0][1];template_chunk=corpus()[1][2]
        documents=[];chunks=[]
        for i in range(99):
            did=f'doc-{i:03d}';sha=f'{i+1:064x}'
            doc=copy.deepcopy(template_doc);doc.update(doc_id=did,canonical_doc_id=did,sha256=sha,file_name=did+'.pdf')
            chunk=copy.deepcopy(template_chunk);chunk.update(doc_id=did,canonical_doc_id=did,sha256=sha,
                chunk_id=did+':c0',file_name=did+'.pdf')
            documents.append(doc);chunks.append(chunk)
        env={'root':self.root,'documents':documents,'chunks':chunks,
             'registry':EvidenceRegistry(documents,chunks),'index':Index(documents,chunks)}
        report=self.execute(self.runner(env=env),max_new_calls=0)
        self.assertEqual(report['selected_canonical_documents'],99)
        self.assertEqual(report['planned_batches'],99)
        self.assertEqual(len(report['documents']),99)
        self.assertEqual(self.fake.requests,[])

    def test_invalid_call_options_fail_before_any_request(self):
        runner=self.runner()
        for kwargs in ({'max_new_calls':-1},{'max_new_calls':True},{'allow_api':'yes'},
                       {'resume_after_service_error':'yes'},{'retry_jobs':['job'],'retry_tag':''}):
            with self.assertRaises((ValueError,TypeError)):self.execute(runner,**kwargs)
        self.assertEqual(self.fake.requests,[])

    def test_new_workspace_can_start_with_empty_reference_samples(self):
        refs=self.root/'rfp_phase5_final/references.json';refs.parent.mkdir(parents=True,exist_ok=True)
        refs.write_text(json.dumps({'samples':[],'rows':[]}),encoding='utf-8')
        self.env['phase5_config']={'generation':{'model':'gpt-5.4-mini','max_output_tokens':6000},
                                   'summary':{'max_batch_bytes':35000}}
        fresh=open_corpus_run(self.env,'fresh_hwp_pdf_workspace',client=self.fake)
        report=self.execute(fresh,max_new_calls=0)
        self.assertEqual(fresh.references,{'samples':[],'rows':[]})
        self.assertEqual(report['selected_canonical_documents'],2)
        self.assertFalse(report['all_documents_semantic_quality_verified'])
        self.assertEqual(self.fake.requests,[])


if __name__=='__main__':unittest.main()
