"""Whole-corpus source-to-index and real excerpt retrieval checks; no model calls.

Extracted-character preservation is deliberately distinct from binary rendering
completeness, OCR quality, table meaning, and generated-answer correctness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import re
import sqlite3
import time

from rfp_phase4.evidence import _indexable
from rfp_phase5_final.catalog_audit import select_documents, _chunks_fingerprint


VERSION='phase5-corpus-index-audit-v1'
SUPPORTED_FORMATS=frozenset({'hwp','hwpx','pdf'})


def search_corpus(env,query,doc_ids=None,top_k=8,method='rrf',*,prefer_literal=True):
    """Search all canonical documents by default; explicit selections never widen.

    RRF combines the existing BM25 and character lexical rankings. It is not an
    OpenAI embedding or a calibrated confidence. All returned source fields
    are validated before retrieval scores/metadata are returned. Long literal
    excerpts (40+ normalized characters) prefer contiguous source matches;
    ordinary questions keep RRF ranking. A literal-only hit has score=None,
    never an invented confidence. The raw BM25 audit disables this preference.
    """
    if not isinstance(query,str):raise TypeError('query must be a string')
    if type(top_k)is not int or not 0<=top_k<=100:raise ValueError('top_k must be between 0 and 100')
    if type(prefer_literal)is not bool:raise TypeError('prefer_literal must be a bool')
    if method not in {'bm25','char_tfidf','rrf','mmr'}:raise ValueError('Unsupported retrieval method')
    selected=select_documents(env,'all' if doc_ids is None else doc_ids)
    limit=env['index'].get_metadata().get('config',{}).get('query_max_chars',4000)
    if len(query)>limit:raise ValueError('query exceeds the index character limit')
    if not query.strip() or top_k==0:return []
    hits=env['index'].search(query,top_k=top_k,allowed_doc_ids=selected,
        method=method,fetch_k=max(40,top_k))
    normalized_query=_normalized(query)
    if prefer_literal and len(normalized_query)>=40 and len(re.findall(r'[가-힣A-Za-z0-9]',normalized_query))>=12:
        metadata=env['index'].get_metadata()
        cache_key=(metadata.get('index_fingerprint'),id(env['chunks']),len(env['chunks']))
        cache=env.get('_corpus_literal_lookup')
        if cache is None or cache['key']!=cache_key:
            if _chunks_fingerprint(env['chunks'])!=metadata.get('chunks_fingerprint'):
                raise ValueError('literal lookup source chunks differ from the persisted index')
            cache={'key':cache_key,'rows':[(c['doc_id'],_normalized(c['text']),c) for c in env['chunks']]}
            env['_corpus_literal_lookup']=cache
        scope=set(selected);ranked={h['chunk_id']:h for h in hits};literal=[]
        for did,text,chunk in cache['rows']:
            if did in scope and normalized_query in text:
                hit=copy.deepcopy(ranked.get(chunk['chunk_id'],chunk))
                hit['original_retrieval_rank']=hit.get('rank')
                hit['literal_match']=True
                if chunk['chunk_id'] not in ranked:
                    hit.update(score=None,method=method,score_is_confidence=False,
                        score_kind='literal_contiguous_source_match_not_confidence')
                literal.append(hit)
                if len(literal)==top_k:break
        if literal:
            literal_ids={h['chunk_id'] for h in literal}
            hits=(literal+[h for h in hits if h['chunk_id'] not in literal_ids])[:top_k]
            for rank,hit in enumerate(hits,1):hit['rank']=rank
    env['registry'].validate_chunks(hits,allowed_doc_ids=selected)
    return copy.deepcopy(hits)


def _normalized(value):return re.sub(r'\s+',' ',value).strip()
def _nonwhite(value):return sum(not x.isspace() for x in value)
def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def _coverage(document,chunks):
    """Union original character ranges; overlaps cannot hide a missing range."""
    spans=defaultdict(list)
    for chunk in chunks:
        for span in chunk.get('source_spans',[]):
            spans[(span.get('collection'),span.get('block_id'))].append(
                (span.get('char_start'),span.get('char_end')))
    total=covered=0;missing=[];eligible_blocks=0;excluded_blocks=0
    for collection in ('blocks','supplemental_blocks'):
        for block in document.get(collection,[]):
            text=block.get('text','')
            if not isinstance(text,str):raise ValueError('source_block_text_not_string')
            if not _indexable(block,collection):excluded_blocks+=1;continue
            count=_nonwhite(text)
            if not count:continue
            eligible_blocks+=1;total+=count
            merged=[]
            for left,right in sorted(spans.get((collection,block.get('block_id')),[])):
                if (type(left)is not int or type(right)is not int
                        or not 0<=left<right<=len(text)):
                    raise ValueError('invalid_source_coverage_range')
                if merged and left<=merged[-1][1]:merged[-1][1]=max(right,merged[-1][1])
                else:merged.append([left,right])
            preserved=sum(_nonwhite(text[a:b]) for a,b in merged)
            covered+=preserved
            if preserved!=count:
                missing.append({'collection':collection,'block_id':block.get('block_id'),
                    'missing_non_whitespace_characters':count-preserved})
    return {'eligible_source_blocks':eligible_blocks,'excluded_unreviewed_or_reference_blocks':excluded_blocks,
        'eligible_source_non_whitespace_characters':total,
        'covered_source_non_whitespace_characters':covered,
        'source_character_coverage':covered/total if total else 0.0,
        'uncovered_source_blocks':missing,
        'source_character_coverage_is_binary_completeness':False}


def _warning_summary(document):
    counts=Counter(w.get('code','unknown_warning') if isinstance(w,dict) else 'unstructured_warning'
        for w in document.get('warnings',[]))
    stats=document.get('stats') or {}
    texts=[b.get('text','') for name in ('blocks','supplemental_blocks')
        for b in document.get(name,[]) if isinstance(b,dict) and isinstance(b.get('text'),str)]
    return {'parser_warning_counts':dict(sorted(counts.items())),
        'replacement_character_count':sum(t.count('\ufffd') for t in texts),
        'unexpected_control_character_count':sum(sum(ord(c)<32 and c not in '\n\r\t' for c in t) for t in texts),
        'ocr_executed':stats.get('ocr_executed'),
        'ocr_review_pages':len({w.get('locator') for w in document.get('warnings',[])
            if isinstance(w,dict) and w.get('code')=='PDF_OCR_REVIEW_NEEDED'}),
        'numeric_gap_review_pages':list(stats.get('numeric_gap_review_pages',[])),
        'vector_glyph_review_pages':list(stats.get('vector_glyph_review_pages',[])),
        'unresolved_table_candidates':stats.get('unresolved_table_candidates',stats.get('unresolved_tables',0)),
        'image_page_count':len(stats.get('image_pages',[])),
        'rendered_page_layout':stats.get('rendered_page_layout'),
        'visual_content_completeness_verified':False}


def _probe_queries(chunks,all_texts,limit):
    candidates=[c for c in chunks if len(re.findall(r'[가-힣A-Za-z0-9]',c.get('text','')))>=12]
    if not candidates:return []
    positions=sorted(set(round(i*(len(candidates)-1)/max(1,limit-1)) for i in range(limit)))
    probes=[];seen=set()
    for pos in positions:
        chunk=candidates[pos];text=_normalized(chunk['text']);choices=[]
        for start in dict.fromkeys((0,max(0,len(text)//2-80),max(0,len(text)-160))):
            query=text[start:start+160]
            if len(re.findall(r'[가-힣A-Za-z0-9]',query))<12 or query in seen:continue
            docs=[]
            for did,values in all_texts.items():
                if any(query in value for value in values):docs.append(did)
                if len(docs)>1:break
            choices.append((len(docs)==1,query))
            if len(docs)==1:break
        if not choices:continue
        unique,query=next((x for x in choices if x[0]),choices[0])
        probes.append({'chunk_id':chunk['chunk_id'],'query':query,'unique_document_excerpt':unique})
        seen.add(query)
    return probes


def audit_indexed_corpus(env,*,probes_per_document=3,retrieval_method='rrf'):
    """Return one canonical row per source and a mapping for every original ID.

    Each row validates all indexed chunks against source spans, merges coverage
    ranges, and issues source-text queries both scoped and unscoped. Excerpts
    are selected from source text, never fabricated from a document ID/title.
    ``recognized_document_count`` counts technical text pipeline readiness;
    warnings still yield ``needs_review`` rather than a visual/semantic pass.
    """
    if type(probes_per_document)is not int or not 1<=probes_per_document<=10:
        raise ValueError('probes_per_document must be an integer between 1 and 10')
    if retrieval_method not in {'bm25','char_tfidf','rrf','mmr'}:raise ValueError('Unsupported retrieval method')
    started=time.perf_counter();documents=list(env['documents']);chunks=list(env['chunks'])
    index=env['index'];registry=env['registry'];errors=[]
    originals={d['doc_id']:d for d in documents}
    canonical={d['doc_id']:d for d in documents if d.get('canonical_doc_id',d['doc_id'])==d['doc_id']}
    grouped=defaultdict(list)
    for chunk in chunks:grouped[chunk.get('doc_id')].append(chunk)
    catalog_rows=index.get_catalog();catalog={d['doc_id']:d for d in catalog_rows}
    metadata=index.get_metadata()
    global_checks={
        'index_chunk_count_matches_snapshot':metadata.get('chunk_count')==len(chunks),
        'index_canonical_count_matches_snapshot':metadata.get('canonical_document_count')==len(canonical),
        'catalog_doc_ids_unique':len(catalog_rows)==len(catalog),
        'source_chunk_ids_unique':len(chunks)==len({c.get('chunk_id') for c in chunks}),
        'index_alias_map_matches_originals':metadata.get('alias_map')=={
            d['doc_id']:d.get('canonical_doc_id',d['doc_id']) for d in documents},
    }
    all_texts={did:[_normalized(c.get('text','')) for c in cs] for did,cs in grouped.items()}
    original_mapping=[]
    for document in documents:
        did=document['doc_id'];target=document.get('canonical_doc_id',did);item_errors=[]
        if target not in canonical:item_errors.append('canonical_document_missing')
        elif document.get('sha256')!=canonical[target].get('sha256'):item_errors.append('alias_source_sha_mismatch')
        try:
            if index.resolve_doc_ids([did])!=[target] or registry.resolve_doc_ids([did])!=[target]:
                item_errors.append('original_id_resolution_mismatch')
        except (ValueError,KeyError,TypeError):item_errors.append('original_id_resolution_failed')
        original_mapping.append({'original_doc_id':did,'canonical_doc_id':target,
            'is_duplicate_alias':did!=target,'file_name':document.get('file_name'),
            'detected_format':document.get('detected_format'),'mapped':not item_errors,'errors':item_errors})
    rows=[]
    for did,document in canonical.items():
        blocked=[];review=[];fmt=document.get('detected_format')
        row={'doc_id':did,'file_name':document.get('file_name'),'detected_format':fmt,
            'sha256':document.get('sha256'),'parser':document.get('parser'),
            'parsing_origin':document.get('parsing_origin'),**_warning_summary(document)}
        if fmt not in SUPPORTED_FORMATS:blocked.append('unsupported_detected_format')
        if document.get('status') in {'error','failed','unsupported','encrypted'}:blocked.append('parser_failed')
        if document.get('expected_sha256') not in (None,document.get('sha256')):blocked.append('source_sha_mismatch')
        for key in ('agency','document_year'):
            if document.get(key) in (None,''):review.append('metadata_'+key+'_missing')
        if row['parser_warning_counts']:review.append('parser_warnings_require_review')
        if row['replacement_character_count'] or row['unexpected_control_character_count']:
            review.append('extracted_text_contains_control_or_replacement_characters')
        try:
            expected=grouped.get(did,[]);selected=index.get_document_chunks(did)
            if not expected or [c.get('chunk_id') for c in selected]!=[c.get('chunk_id') for c in expected]:
                raise ValueError('indexed_chunk_order_or_count_mismatch')
            registered=registry.validate_chunks(selected,allowed_doc_ids=[did])
            if registered!=expected:raise ValueError('source_index_chunk_content_mismatch')
            if did not in catalog or catalog[did].get('sha256')!=document.get('sha256'):
                raise ValueError('catalog_source_sha_mismatch')
            coverage=_coverage(document,registered);row.update(coverage)
            if coverage['excluded_unreviewed_or_reference_blocks']:
                review.append('unreviewed_or_reference_source_blocks_not_indexed')
            if not coverage['eligible_source_non_whitespace_characters']:blocked.append('no_indexable_source_text')
            elif coverage['source_character_coverage']!=1.0:blocked.append('extracted_source_not_fully_indexed')
            probes=_probe_queries(registered,all_texts,probes_per_document);probe_rows=[];baseline_rows=[]
            if not probes:blocked.append('no_nonempty_source_retrieval_probe')
            for probe in probes:
                query=probe['query'];scoped=search_corpus(env,query,doc_ids=[did],top_k=5,method=retrieval_method)
                unscoped=search_corpus(env,query,doc_ids=None,top_k=5,method=retrieval_method)
                baseline=search_corpus(env,query,doc_ids=[did],top_k=5,method='bm25',prefer_literal=False)
                scope_safe=all(h.get('doc_id')==did for h in scoped)
                excerpt_found=any(query in _normalized(h.get('text','')) for h in scoped)
                global_doc_found=any(h.get('doc_id')==did for h in unscoped)
                if not scoped or not scope_safe or not excerpt_found:blocked.append('scoped_source_excerpt_not_retrieved')
                if probe['unique_document_excerpt'] and not global_doc_found:review.append('unique_excerpt_global_retrieval_missed')
                if not probe['unique_document_excerpt']:review.append('retrieval_probe_text_not_unique_to_document')
                baseline_found=any(query in _normalized(h.get('text','')) for h in baseline)
                if not baseline_found:review.append('baseline_bm25_excerpt_missed')
                baseline_rows.append({'source_chunk_id':probe['chunk_id'],'query_sha256':_digest(query),
                    'scoped_hit_count':len(baseline),'scoped_source_excerpt_found':baseline_found,
                    'scoped_document_scope_preserved':all(h.get('doc_id')==did for h in baseline),
                    'self_chunk_in_scoped_top_5':any(h.get('chunk_id')==probe['chunk_id'] for h in baseline)})
                probe_rows.append({'source_chunk_id':probe['chunk_id'],'query_sha256':_digest(query),
                    'query_characters':len(query),'unique_document_excerpt':probe['unique_document_excerpt'],
                    'scoped_hit_count':len(scoped),'scoped_source_excerpt_found':excerpt_found,
                    'scoped_document_scope_preserved':scope_safe,
                    'literal_match_preferred':any(h.get('literal_match') for h in scoped),
                    'self_chunk_in_scoped_top_5':any(h.get('chunk_id')==probe['chunk_id'] for h in scoped),
                    'unscoped_self_document_in_top_5':global_doc_found})
            row.update(chunk_count=len(registered),source_spans_valid=True,
                index_content_matches_snapshot=True,retrieval_probes=probe_rows,
                baseline_bm25_probes=baseline_rows)
        except (ValueError,TypeError,KeyError,AttributeError,RuntimeError,OSError,sqlite3.Error) as error:
            blocked.append('source_or_index_validation_failed')
            row.update(validation_exception_type=type(error).__name__,source_spans_valid=False)
        blocked=list(dict.fromkeys(blocked));review=list(dict.fromkeys(review))
        row.update(status='blocked' if blocked else 'needs_review' if review else 'read_ready',
            text_pipeline_ready=not blocked,block_reasons=blocked,review_reasons=review,
            binary_content_completeness_verified=False,semantic_quality_verified=False)
        rows.append(row)
    if len(originals)!=len(documents):errors.append('duplicate_original_document_id')
    if set(catalog)!=set(canonical) or set(grouped)!=set(canonical):errors.append('document_catalog_chunk_scope_mismatch')
    if not all(x['mapped'] for x in original_mapping):errors.append('original_document_mapping_failed')
    errors.extend(key for key,passed in global_checks.items() if not passed)
    status_counts=dict(Counter(r['status'] for r in rows))
    return {'version':VERSION,'original_document_count':len(documents),'canonical_document_count':len(rows),
        'recognized_document_count':sum(r['text_pipeline_ready'] for r in rows),
        'all_text_pipelines_ready':bool(rows) and not errors and all(r['text_pipeline_ready'] for r in rows),
        'status_counts':status_counts,'original_mapping':original_mapping,'documents':rows,'errors':errors,
        'global_checks':global_checks,
        'retrieval_method':retrieval_method,'comparison_baseline_method':'bm25',
        'literal_excerpt_preference_enabled':True,'baseline_bm25_is_raw_ranker':True,
        'baseline_bm25_scoped_excerpt_misses':sum(not p['scoped_source_excerpt_found']
            for d in rows for p in d.get('baseline_bm25_probes',[])),
        'format_counts':dict(Counter(d.get('detected_format') for d in documents)),
        'api_calls':0,'semantic_quality_verified':False,'binary_content_completeness_verified':False,
        'retrieval_probe_is_semantic_accuracy':False,'elapsed_seconds':round(time.perf_counter()-started,3),
        'scope':'전체 원본·대표 연결, 추출 원문 문자 보존, 인덱스 일치와 실제 본문 구간 검색 검사. 바이너리의 그림·표 의미 및 실제 요약 정답성은 별도입니다.'}
