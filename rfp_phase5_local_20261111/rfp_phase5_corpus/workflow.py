"""One whole-corpus view across raw recognition, source/index checks and generation."""
from pathlib import Path
import csv,json

from rfp_phase5_final.engine import write_json
from .raw_audit import audit_raw_corpus
from .index_audit import audit_indexed_corpus


def audit_corpus(env,*,parser_root,raw_directory=None,reparse=False,output_directory=None,progress=None,
                 allow_unreviewed_sources=False):
    if type(allow_unreviewed_sources)is not bool:raise TypeError('시각 검토 승인 옵션은 bool이어야 합니다.')
    raw=audit_raw_corpus(env['documents'],parser_root,reparse=reparse,
        raw_directory=raw_directory,progress=progress)
    indexed=audit_indexed_corpus(env)
    by_original={r['doc_id']:r for r in raw['rows']}
    by_canonical={r['doc_id']:r for r in indexed['documents']}
    rows=[]
    for source in env['documents']:
        did=source['doc_id'];canonical=source.get('canonical_doc_id',did)
        original=by_original[did];text=by_canonical.get(canonical,{})
        ready=original['read_success'] and original['snapshot_compatible'] and text.get('text_pipeline_ready',False)
        rows.append({'doc_id':did,'canonical_doc_id':canonical,'file_name':source.get('file_name'),
            'detected_format':original.get('detected_format'),'raw_read_success':original['read_success'],
            'raw_sha_matches':original['sha256_matches'],'snapshot_compatible':original['snapshot_compatible'],
            'raw_reparse_executed':original.get('reparse_executed',False),
            'chunk_count':text.get('chunk_count',0),'source_character_coverage':text.get('source_character_coverage',0),
            'retrieval_ready':text.get('text_pipeline_ready',False),'text_pipeline_ready':ready,
            'status':'blocked' if not ready else 'needs_review' if original['needs_review'] or text.get('review_reasons') else 'read_ready',
            'review_codes':original['review_codes']+text.get('review_reasons',[]),
            'failure_code':original.get('failure_code'),'semantic_quality_verified':False})
    intake_path=Path(env['root'])/'reports/intake_report.json'
    intake=json.loads(intake_path.read_text(encoding='utf-8')) if intake_path.is_file() else {}
    failures=intake.get('failures',[])
    for failure in failures:
        if any(r['doc_id']==failure['doc_id'] for r in rows):continue
        rows.append({'doc_id':failure['doc_id'],'canonical_doc_id':failure.get('canonical_doc_id',failure['doc_id']),
            'file_name':failure['file_name'],'detected_format':None,'raw_read_success':False,
            'raw_sha_matches':False,'snapshot_compatible':False,'raw_reparse_executed':False,
            'chunk_count':0,'source_character_coverage':0,'retrieval_ready':False,'text_pipeline_ready':False,
            'status':'blocked','review_codes':['RAW_PARSE_FAILED'],
            'failure_code':'raw_parse_failed','semantic_quality_verified':False})
    failed_paths={str(f.get('relative_path','')).removeprefix('data/raw/') for f in failures}
    unknown=[r for r in raw['unregistered_sources'] if r.get('relative_path') not in failed_paths]
    for row in rows:
        review_required=any(c in row['review_codes'] for c in
            ('PAGE_TEXT_EXTRACTION_FAILED','OCR_OR_IMAGE_REVIEW_NEEDED'))
        row['generation_requires_visual_review']=review_required
        row['generation_eligible']=row['text_pipeline_ready'] and (not review_required or allow_unreviewed_sources)
    eligible=sorted({r['canonical_doc_id'] for r in rows if r['generation_eligible']})
    # Any broken alias/source of the same content remains a visible blocker.
    eligible=[d for d in eligible if all(r['generation_eligible'] for r in rows if r['canonical_doc_id']==d)]
    folder=Path(output_directory) if output_directory else Path(env['root'])/'phase5_outputs/corpus_audit'
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'raw_sources.json',raw);write_json(folder/'source_index_retrieval.json',indexed)
    report={'version':'phase5-whole-corpus-v1','original_documents':len(rows),
        'canonical_documents':indexed['canonical_document_count'],'eligible_canonical_doc_ids':eligible,
        'readable_original_documents':sum(r['text_pipeline_ready'] for r in rows),
        'blocked_original_documents':sum(not r['text_pipeline_ready'] for r in rows),
        'needs_review_original_documents':sum(r['status']=='needs_review' for r in rows),
        'unregistered_raw_sources':len(unknown),'known_failed_files_not_indexed':len(failures),
        'all_existing_documents_text_ready':bool(rows) and all(r['text_pipeline_ready'] for r in rows),
        'all_raw_files_registered':len(unknown)==0,
        'fresh_parsing_verified':raw.get('fresh_parsing_verified',False),
        'unreviewed_sources_explicitly_allowed':allow_unreviewed_sources,
        'documents':rows,'api_calls':0,'all_document_semantic_quality_verified':False,
        'raw_report':str(folder/'raw_sources.json'),'index_report':str(folder/'source_index_retrieval.json')}
    write_json(folder/'corpus_audit.json',report)
    with (folder/'documents.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=list(rows[0]) if rows else []
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:('; '.join(v) if isinstance(v,list) else v) for k,v in row.items()})
    return report
