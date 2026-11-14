"""Corpus-wide execution: all documents remain selected while each run is bounded.

This module extends the frozen engine rather than rewriting its implementation,
request identity, completed jobs or review files. Summary counts are operational
coverage, never semantic accuracy or original HWP/PDF extraction completeness.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import zipfile

from rfp_phase5_final.catalog_audit import build_catalog, select_documents
from rfp_phase5_final.engine import FinalConfig, read_json, write_json
from rfp_phase5_recovery.runner import RecoveryRun, FAILED_STATES


VERSION='phase5-corpus-execution-v1'
COMPLETE_STATES=frozenset({'completed','completed_with_rejections'})


class CorpusRun(RecoveryRun):
    """Process a corpus incrementally, not only when all remaining work fits."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._corpus_errors={}

    def result(self, doc_id):
        """Return the corrected document view; preserve frozen model output."""
        from rfp_phase5_content.pipeline import rebuild_document_summary
        from rfp_phase5_content.storage import save_content_result
        from rfp_phase5_final.catalog_audit import get_validated_document_chunks
        raw=super().result(doc_id)
        document=self.env['registry'].documents_by_id[raw['doc_id']]
        chunks=get_validated_document_chunks(self.env,raw['doc_id'])
        corrected=rebuild_document_summary(raw,document,chunks,self.env['registry'])
        return save_content_result(self.folder,raw,corrected)

    def report(self, doc_ids, reviewer=''):
        """Report derived content and its own review without rewriting final_report."""
        from rfp_phase5_content.storage import assess_content_review, load_saved_result
        selected=select_documents(self.env,doc_ids);cases=[]
        for did in selected:
            try:
                result=self.result(did)
                _,folder=load_saved_result(self.folder,did,content=True)
                review=assess_content_review(folder,result,reviewer)
                cases.append({'doc_id':did,'execution_status':result['execution_status'],
                    'status':result['status'],'input_chunk_coverage':result['input_chunk_coverage'],
                    'checked_chunk_coverage':result['checked_chunk_coverage'],
                    'rejected_claim_count':len(result.get('rejected_claims',[])),
                    'content_audit':result.get('content_audit',{}),'review':review,
                    'result_path':str(folder/'result.json'),'review_path':str(folder/'review.csv')})
            except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
                cases.append({'doc_id':did,'execution_status':'content_repair_failed',
                    'status':'not_generated','error':self._error(error),
                    'review':{'case_accepted':False,'human_review_completed':False}})
        sample_ids={s['doc_id'] for s in (self.references or {}).get('samples',[])}
        by_id={c['doc_id']:c for c in cases}
        sample_passed=bool(sample_ids) and sample_ids.issubset(by_id) and all(
            by_id[did]['review'].get('case_accepted') is True for did in sample_ids)
        technical_path=self.folder/'all_document_plans.json'
        catalog_path=self.folder/'catalog_audit.json'
        technical=read_json(technical_path) if technical_path.exists() else {}
        catalog=read_json(catalog_path) if catalog_path.exists() else {}
        tests_path=self.folder/'offline_tests.json'
        tests=read_json(tests_path) if tests_path.exists() else {}
        reports_bound=all(r.get('run_fingerprint')==self.fingerprint for r in (technical,catalog,tests))
        report={'version':'phase5-content-report-v1','run_fingerprint':self.fingerprint,
            'phase5_sample_validation_passed':sample_passed and reports_bound
                and technical.get('all_document_id_transport_passed') is True
                and catalog.get('passed') is True and tests.get('passed') is True,
            'all_document_id_transport_passed':technical.get('all_document_id_transport_passed',False),
            'sample_review_passed':sample_passed,'cases':cases,'budget':self.budget_usage(),
            'api_calls_by_report':0,'all_documents_semantic_quality_verified':False,
            'scope':'문서 전체 맥락으로 보정한 요약과 해당 보정본의 별도 검토 기록. 원래 생성 기록은 보존합니다.'}
        write_json(self.folder/'content_report.json',report)
        return report

    def export(self, doc_ids, reviewer=''):
        """Export raw and corrected revisions to a separate content archive."""
        self.report(doc_ids,reviewer)
        target=self.folder.parent/(self.folder.name+'_content_results.zip')
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.folder.rglob('*')):
                if path.is_file() and path.suffix in {'.json','.csv','.md','.log'}:
                    archive.write(path,self.folder.name+'/'+path.relative_to(self.folder).as_posix())
        return target

    @staticmethod
    def _error(error):
        # Do not copy exception text, response bodies or source content to reports.
        return {'code':'document_processing_failed','exception_type':type(error).__name__}

    def summarize_corpus(self,requested='all'):
        """Metadata-only document/batch coverage. No API/SDK/key access."""
        selected=select_documents(self.env,requested)
        originals=build_catalog(self.env)
        aliases={did:[r['original_doc_id'] for r in originals if r['canonical_doc_id']==did]
                 for did in selected}
        names={r['canonical_doc_id']:r.get('canonical_file_name',r['file_name']) for r in originals}
        journal=self._journal();rows=[]
        for did in selected:
            row={'doc_id':did,'original_doc_ids':aliases[did],'file_name':names[did],
                 'chunk_count':0,'planned_batches':0,'completed_batches':0,
                 'failed_batches':0,'awaiting_batches':0,'rejected_claim_count':0,
                 'input_chunk_coverage':0.0,'checked_chunk_coverage':0.0,
                 'status':'awaiting_api','errors':[]}
            try:
                plan=self.plan(did)
                row.update(chunk_count=plan['chunk_count'],planned_batches=plan['batch_count'],
                           plan_executable=plan['can_execute'],plan_block_reasons=plan['block_reasons'])
                received=set();checked=set()
                for batch in plan['batches']:
                    job=journal['jobs'].get(batch['job_key'],{})
                    outcome=self._load_outcome(job,batch)
                    state=job.get('status','awaiting_api')
                    if state in COMPLETE_STATES and (outcome is None or not outcome.get('response_received')
                                                     or not outcome.get('claims_checked')):
                        raise ValueError('Completed journal job has no checked outcome')
                    if state in COMPLETE_STATES:row['completed_batches']+=1
                    elif state in FAILED_STATES or state=='running':row['failed_batches']+=1
                    else:row['awaiting_batches']+=1
                    if outcome:
                        if outcome.get('response_received'):received.update(batch['source_chunk_ids'])
                        if outcome.get('claims_checked'):checked.update(batch['source_chunk_ids'])
                        row['rejected_claim_count']+=len(outcome.get('rejected_claims',[]))
                if row['chunk_count']:
                    row['input_chunk_coverage']=len(received)/row['chunk_count']
                    row['checked_chunk_coverage']=len(checked)/row['chunk_count']
                row['status']=('plan_blocked' if not plan['can_execute'] else
                    'incomplete' if row['failed_batches'] else
                    'completed_with_rejections' if row['completed_batches']==row['planned_batches'] and row['rejected_claim_count'] else
                    'completed' if row['planned_batches'] and row['completed_batches']==row['planned_batches'] else
                    'partially_processed' if row['completed_batches'] else 'awaiting_api')
                if did in self._corpus_errors:
                    row['errors'].append(self._corpus_errors[did])
                    row['status']='document_error'
            except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
                row['status']='document_error';row['errors'].append(self._error(error))
            rows.append(row)
        report={'version':VERSION,'run_fingerprint':self.fingerprint,'selected_doc_ids':selected,
            'selected_canonical_documents':len(selected),
            'selected_original_documents':sum(len(aliases[d]) for d in selected),
            'all_canonical_documents_selected':set(selected)=={r['canonical_doc_id'] for r in originals},
            'documents':rows,'planned_batches':sum(r['planned_batches'] for r in rows),
            'completed_batches':sum(r['completed_batches'] for r in rows),
            'failed_batches':sum(r['failed_batches'] for r in rows),
            'awaiting_batches':sum(r['awaiting_batches'] for r in rows),
            'completed_documents':sum(r['status'] in COMPLETE_STATES for r in rows),
            'document_errors':sum(r['status']=='document_error' for r in rows),
            'all_selected_generation_completed':bool(rows) and all(r['status'] in COMPLETE_STATES for r in rows),
            'budget':self.budget_usage(),'api_calls_by_summary':0,
            'all_documents_semantic_quality_verified':False,'raw_file_parse_quality_verified':False,
            'scope':'전체 선택 문서의 입력 계획과 실제 처리 상태. 원본 바이너리 판독·의미적 정답 평가는 별도입니다.'}
        write_json(self.folder/'corpus_execution.json',report)
        return report

    def run_corpus(self,requested='all',*,allow_api=False,max_new_calls=20,
                   retry_jobs=(),retry_tag='',resume_after_service_error=False):
        if type(allow_api)is not bool:raise TypeError('allow_api는 bool이어야 합니다.')
        if type(max_new_calls)is not int or max_new_calls<0:raise ValueError('이번 추가 호출 상한은 0 이상의 정수여야 합니다.')
        if type(resume_after_service_error)is not bool:raise TypeError('서비스 재개 옵션은 bool이어야 합니다.')
        if not isinstance(retry_jobs,(list,tuple)) or any(not isinstance(k,str) for k in retry_jobs):
            raise TypeError('retry_jobs는 작업 키 목록이어야 합니다.')
        if retry_jobs and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',retry_tag):
            raise ValueError('실패 작업 재시도에는 명시적인 retry_tag가 필요합니다.')
        self.assert_frozen()
        selected=select_documents(self.env,requested)
        initial=self.budget_usage();old_config=self.config
        invocation_cap=min(old_config.max_total_calls,initial['calls_used_or_reserved']+max_new_calls)
        self.config=replace(old_config,max_total_calls=max(1,invocation_cap))
        self._corpus_errors={};outcomes=[]
        try:
            for did in selected:
                try:
                    outcome=super().run([did],allow_api=allow_api and max_new_calls>0,
                        retry_jobs=retry_jobs,retry_tag=retry_tag,
                        resume_after_service_error=resume_after_service_error)
                    # Resume a known service failure once. A new failure must stop
                    # subsequent requests until a later explicit invocation.
                    resume_after_service_error=False
                    outcomes.append({'doc_id':did,'status':'processed',
                        'budget_halted':outcome['budget_halted'],'service_halted':outcome['service_halted']})
                except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
                    detail=self._error(error);self._corpus_errors[did]=detail
                    outcomes.append({'doc_id':did,'status':'document_error','error':detail})
        finally:
            self.config=old_config
        report=self.summarize_corpus(selected)
        after=self.budget_usage()
        report['invocation']={'allow_api':allow_api,'max_new_calls':max_new_calls,
            'api_calls_used_or_reserved_this_invocation':after['calls_used_or_reserved']-initial['calls_used_or_reserved'],
            'budget_before':initial,'budget_after':after,'document_outcomes':outcomes,
            'partial_progress_allowed':True,'whole_remaining_corpus_must_fit':False}
        write_json(self.folder/'corpus_execution.json',report)
        return report


def open_corpus_run(env,run_id='phase5_final_source_units',*,client=None,
                    max_total_calls=None,max_total_input_bytes=None):
    """Open the old frozen run, or start a new run for a newly built corpus."""
    root=Path(env['root']);folder=root/'phase5_outputs/final_source_units'/run_id
    frozen_path=folder/'frozen.json';report_path=folder/'final_report.json'
    budget=read_json(report_path).get('budget',{}) if report_path.exists() else {}
    if frozen_path.exists():
        settings=read_json(frozen_path)['model_settings']
    else:
        generation=env.get('phase5_config',{}).get('generation',{})
        summary=env.get('phase5_config',{}).get('summary',{})
        settings={'model':generation.get('model','gpt-5.4-mini'),
            'max_output_tokens':generation.get('max_output_tokens',6000),
            'reasoning_effort':generation.get('reasoning_effort','none'),
            'max_request_bytes':generation.get('max_request_bytes',60000),
            'batch_max_bytes':summary.get('max_batch_bytes',35000)}
    config=FinalConfig(**settings,
        max_total_calls=max_total_calls if max_total_calls is not None else budget.get('max_total_calls',100),
        max_total_input_bytes=max_total_input_bytes if max_total_input_bytes is not None else budget.get('max_total_input_bytes',6000000))
    refs_path=root/'rfp_phase5_final/references.json'
    references=read_json(refs_path) if refs_path.exists() else None
    return CorpusRun(env,run_id,config,references=references,client=client)
