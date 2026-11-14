"""Failure recovery tests use synthetic documents and a disconnected test client."""
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rfp_phase5_final'))
from test_catalog_audit import env as small_env
from rfp_phase4.llm import LLMError
from rfp_phase5_final.engine import FinalRun, FinalConfig, TASK, QUESTION, FIELD_SCHEMA, read_json, file_hash
from rfp_phase5_final.catalog_audit import get_validated_document_chunks
from rfp_phase5_final.transport import ShortCitationLLM
from rfp_phase5_recovery.client import RecoveryLLM
from rfp_phase5_recovery.runner import RecoveryRun, rejection_reason


class FakeClient:
    def __init__(self,fail_on=None,mode='empty_missing'):
        self.responses=self;self.requests=[];self.fail_on=fail_on;self.mode=mode

    def create(self,**body):
        self.requests.append(copy.deepcopy(body))
        data=json.loads(body['input'][1]['content'])
        unit=next(x for x in data['evidence'] if x['citable'])
        claim={'doc_id':'D001','category':'budget','text':'[budget_vat|fact] 테스트 원문 연결',
               'citations':[{'chunk_id':unit['chunk_id'],'quote':'SOURCE'}]}
        payload={'status':'answered','claims':[claim],'missing_information':[]}
        if self.fail_on==len(self.requests):
            if self.mode=='network':raise RuntimeError('test-only transport failure')
            if self.mode=='empty_missing':payload.update(status='partial',missing_information=['',os.environ['OPENAI_API_KEY']])
            elif self.mode=='status':payload['status']='insufficient_evidence'
            elif self.mode=='alias':claim['citations'][0]['chunk_id']='E99999'
            elif self.mode=='field':claim['category']='other'
        return SimpleNamespace(status='completed',output=[],output_text=json.dumps(payload,ensure_ascii=False),
            usage=SimpleNamespace(input_tokens=10,output_tokens=5,total_tokens=15,
                                  input_tokens_details=SimpleNamespace(cached_tokens=0)))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.env=small_env();self.env['root']=self.root
        for name in ['rfp_phase4.llm','rfp_phase4.evidence','rfp_phase5.fields']:
            module=importlib.import_module(name);p=self.root/(name.replace('.','/')+'.py')
            p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(module.__file__,p)
        self.secret='dummy-scope-secret-1234'
        p=patch.dict(os.environ,{'OPENAI_API_KEY':self.secret});p.start();self.addCleanup(p.stop)
        p=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'));p.start();self.addCleanup(p.stop)
        p=patch('socket.socket.connect_ex',side_effect=AssertionError('network prohibited'));p.start();self.addCleanup(p.stop)

    def quiet(self,callback,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return callback(*args,**kwargs)

    def old_run(self):
        client=FakeClient(fail_on=2)
        runner=FinalRun(self.env,'test',client=client)
        self.quiet(runner.run,['doc-a','doc-b'],allow_api=True)
        self.assertEqual(len(client.requests),2)
        return runner

    def recovery(self,client):return RecoveryRun(self.env,'test',client=client)

    def test_requests_and_cache_fingerprints_are_unchanged(self):
        chunks=get_validated_document_chunks(self.env,'doc-b');allowed={'doc-b':'b'*64}
        config=FinalConfig().llm_config();client=FakeClient()
        old=ShortCitationLLM(self.root/'cache',config=config,allowed_sources=allowed,client=client)
        new=RecoveryLLM(self.root/'cache',config=config,allowed_sources=allowed,client=client)
        before=old._prepare(TASK,QUESTION,chunks,FIELD_SCHEMA)
        after=new._prepare(TASK,QUESTION,chunks,FIELD_SCHEMA)
        self.assertEqual(before['exact'],after['exact']);self.assertEqual(before['fingerprint'],after['fingerprint'])

    def test_default_audit_and_retry_plan_make_no_requests(self):
        self.old_run();client=FakeClient();runner=self.recovery(client)
        audit=runner.audit();plan=runner.retry_failed()
        self.assertEqual(audit['completed_jobs'],1);self.assertEqual(len(audit['failed_jobs']),1)
        self.assertFalse(plan['retry_executed']);self.assertEqual(client.requests,[])
        self.assertEqual(runner.budget_usage()['calls_used_or_reserved'],2)

    def test_only_failed_job_is_retried_and_success_files_are_preserved(self):
        old=self.old_run();journal=old._journal()
        key=next(k for k,j in journal['jobs'].items() if j['status']=='completed')
        saved=copy.deepcopy(journal['jobs'][key]);path=old.folder/saved['outcome_file'];before=path.read_bytes()
        client=FakeClient();runner=self.recovery(client)
        result=self.quiet(runner.retry_failed,allow_api=True)
        self.assertEqual(result['api_calls_this_recovery'],1);self.assertEqual(len(client.requests),1)
        self.assertEqual(runner.budget_usage()['calls_used_or_reserved'],3)
        self.assertEqual(runner._journal()['jobs'][key],saved);self.assertEqual(path.read_bytes(),before)
        self.assertEqual(runner.result('doc-b')['execution_status'],'completed')
        self.assertFalse(runner.result('doc-b')['semantic_quality_verified'])

    def test_repeating_successful_recovery_is_free(self):
        self.old_run();client=FakeClient();runner=self.recovery(client)
        self.quiet(runner.retry_failed,allow_api=True)
        again=runner.retry_failed(allow_api=True)
        self.assertEqual(again['api_calls_this_recovery'],0);self.assertEqual(len(client.requests),1)

    def test_failed_retry_keeps_private_raw_output_and_detailed_reason(self):
        self.old_run();client=FakeClient(fail_on=1);runner=self.recovery(client)
        result=self.quiet(runner.retry_failed,allow_api=True)
        self.assertEqual(result['api_calls_this_recovery'],1)
        failures=runner.audit()['failed_jobs'];self.assertEqual(len(failures),1)
        detail=failures[0]['recovery_diagnostic']
        self.assertEqual(detail['reason'],'invalid_missing_information_labels')
        self.assertTrue(detail['diagnostic_saved'])
        raw=Path(detail['diagnostic_file']).read_text()
        self.assertNotIn(self.secret,raw);self.assertIn('[API_KEY_REDACTED]',raw)
        self.assertEqual(json.loads(raw)['empty_missing_label_count'],1)
        self.assertEqual(Path(detail['diagnostic_file']).stat().st_mode&0o777,0o600)

    def test_retry_tag_does_not_repeat_a_failed_request(self):
        self.old_run();client=FakeClient(fail_on=1);runner=self.recovery(client)
        self.quiet(runner.retry_failed,allow_api=True)
        again=runner.retry_failed(allow_api=True)
        self.assertFalse(again['retry_executed']);self.assertTrue(again['retry_tag_already_used'])
        self.assertEqual(len(client.requests),1)

    def test_status_contract_is_still_rejected_not_silently_normalized(self):
        self.old_run();runner=self.recovery(FakeClient(fail_on=1,mode='status'))
        self.quiet(runner.retry_failed,allow_api=True)
        fail=runner.audit()['failed_jobs'][0]
        self.assertEqual(fail['recovery_diagnostic']['reason'],'status_claim_count_mismatch')
        self.assertEqual(fail['status'],'validation_failed')

    def test_unknown_source_alias_is_still_rejected_and_diagnosed(self):
        self.old_run();runner=self.recovery(FakeClient(fail_on=1,mode='alias'))
        self.quiet(runner.retry_failed,allow_api=True)
        fail=runner.audit()['failed_jobs'][0]
        self.assertEqual(fail['recovery_diagnostic']['reason'],'source_unit_alias_not_allowed')
        self.assertEqual(fail['status'],'validation_failed')

    def test_field_category_validation_is_not_relaxed(self):
        self.old_run();runner=self.recovery(FakeClient(fail_on=1,mode='field'))
        self.quiet(runner.retry_failed,allow_api=True)
        result=runner.result('doc-b')
        self.assertFalse(result['claims']);self.assertEqual(len(result['rejected_claims']),1)
        self.assertEqual(runner.audit()['rejection_reason_counts'],{'field_category_mismatch':1})

    def test_new_unprocessed_document_cannot_be_included_in_failed_retry(self):
        runner=FinalRun(self.env,'test',client=FakeClient(fail_on=1))
        self.quiet(runner.run,['doc-b'],allow_api=True)
        client=FakeClient();recovery=self.recovery(client)
        with self.assertRaises(ValueError):recovery.retry_failed(['doc-a:B0001:missing'],allow_api=True)
        self.assertEqual(client.requests,[])

    def test_call_limit_and_job_scope_are_checked_before_request(self):
        old=self.old_run();client=FakeClient();runner=self.recovery(client)
        runner.config=FinalConfig(max_total_calls=2)
        with self.assertRaises(ValueError):runner.retry_failed(allow_api=True)
        self.assertEqual(client.requests,[])
        self.assertEqual(old.budget_usage()['calls_used_or_reserved'],2)

    def test_frozen_files_and_existing_reviews_remain_unchanged(self):
        old=self.old_run();before={p:p.read_bytes() for p in old.folder.rglob('comparison.csv')}
        frozen=(old.folder/'frozen.json').read_bytes()
        runner=self.recovery(FakeClient());self.quiet(runner.retry_failed,allow_api=True)
        self.assertEqual((old.folder/'frozen.json').read_bytes(),frozen)
        for p,data in before.items():self.assertEqual(p.read_bytes(),data)
        runner.assert_frozen()

    def test_service_failure_requires_explicit_resume_and_new_retry_tag(self):
        self.old_run();client=FakeClient(fail_on=1,mode='network');runner=self.recovery(client)
        first=self.quiet(runner.retry_failed,allow_api=True)
        self.assertTrue(first['execution']['service_halted'])
        runner.client_injection=FakeClient()
        blocked=self.quiet(runner.retry_failed,allow_api=True,retry_tag='intentional_second_attempt')
        self.assertEqual(blocked['api_calls_this_recovery'],0)
        self.assertEqual(runner.client_injection.requests,[])
        resumed=self.quiet(runner.retry_failed,allow_api=True,retry_tag='intentional_second_attempt',
                           resume_after_service_error=True)
        self.assertEqual(resumed['api_calls_this_recovery'],1)
        self.assertEqual(len(runner.client_injection.requests),1)

    def test_diagnostic_disk_failure_does_not_trigger_a_request_retry(self):
        chunks=get_validated_document_chunks(self.env,'doc-b');fake=FakeClient(fail_on=1)
        client=RecoveryLLM(self.root/'cache',config=FinalConfig().llm_config(),
            allowed_sources={'doc-b':'b'*64},client=fake)
        with patch('rfp_phase5_recovery.client.os.open',side_effect=OSError('test-only disk error')):
            with self.assertRaises(LLMError) as caught:
                client.generate(TASK,QUESTION,chunks,schema=FIELD_SCHEMA)
        self.assertTrue(caught.exception.recovery_diagnostic['diagnostic_write_failed'])
        self.assertEqual(len(fake.requests),1)
        self.assertEqual(client.usage_report()['failed_attempts'],1)


if __name__=='__main__':unittest.main()
