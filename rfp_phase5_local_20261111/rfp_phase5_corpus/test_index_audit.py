"""Corpus checks use source excerpts, not ID-only mocks or real model calls."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rfp_phase5_final'))
from test_catalog_audit import env as small_env
from test_catalog_audit import Index
from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase5_corpus.index_audit import audit_indexed_corpus,search_corpus


def setup_env():
    value=small_env()
    for d in value['documents']:
        d.update(detected_format='pdf',agency='테스트기관',document_year=2026,
            status='parsed',warnings=[],stats={},parser='synthetic_parser')
    value['index'].leak=False;value['index'].drop_global=False
    def search(query,top_k=5,allowed_doc_ids=None,method='bm25',**kwargs):
        index=value['index']
        if (allowed_doc_ids is None or set(allowed_doc_ids)=={'doc-a','doc-b'}) and index.drop_global:return []
        hits=[copy.deepcopy(c) for c in value['chunks']
            if query in ' '.join(c['text'].split())
            and (allowed_doc_ids is None or c['doc_id'] in allowed_doc_ids)]
        if index.leak and allowed_doc_ids is not None:
            hits.append({'doc_id':'unexpected','chunk_id':'leaked','text':query})
        for rank,h in enumerate(hits,1):h.update(rank=rank,score=0.25,method=method,score_is_confidence=False)
        return hits[:top_k]
    value['index'].search=search
    return value


def long_source_env():
    value=setup_env();search=value['index'].search
    text='이 사업의 상세 원문은 장문의 정확한 인용 구간 검색에서 순위 때문에 누락되지 않아야 하며 조건과 근거를 그대로 보존합니다.'
    value['documents'][0]['blocks'][0]['text']=text
    chunk=value['chunks'][0];chunk.update(text=text,text_bytes=len(text.encode()))
    chunk['source_spans'][0].update(char_end=len(text),chunk_end=len(text))
    value['index']=Index(value['documents'],value['chunks'])
    value['index'].leak=False;value['index'].drop_global=False;value['index'].search=search
    value['registry']=EvidenceRegistry(value['documents'],value['chunks'])
    return value,text


class IndexedCorpusTests(unittest.TestCase):
    def setUp(self):
        self.env=setup_env()
        p=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        p.start();self.addCleanup(p.stop)

    def row(self,report,did='doc-a'):return next(r for r in report['documents'] if r['doc_id']==did)

    def test_every_original_id_and_real_text_retrieval_are_checked(self):
        report=audit_indexed_corpus(self.env)
        self.assertEqual(report['original_document_count'],3)
        self.assertEqual(report['canonical_document_count'],2)
        self.assertEqual(report['recognized_document_count'],2)
        self.assertEqual(len(report['original_mapping']),3)
        self.assertEqual(report['status_counts'],{'read_ready':2})
        self.assertTrue(report['all_text_pipelines_ready'])
        self.assertFalse(report['semantic_quality_verified'])
        self.assertFalse(report['binary_content_completeness_verified'])
        row=self.row(report)
        self.assertEqual(row['source_character_coverage'],1.0)
        self.assertTrue(all(x['scoped_source_excerpt_found'] for x in row['retrieval_probes']))

    def test_scoped_search_and_global_retrieval_are_distinguished(self):
        self.env['index'].drop_global=True
        report=audit_indexed_corpus(self.env)
        self.assertTrue(report['all_text_pipelines_ready'])
        self.assertEqual(self.row(report)['status'],'needs_review')
        self.assertIn('unique_excerpt_global_retrieval_missed',self.row(report)['review_reasons'])

    def test_scope_leak_blocks_only_affected_document(self):
        original=self.env['index'].search
        def search(query,**kwargs):
            hits=original(query,**kwargs)
            if kwargs.get('allowed_doc_ids')==['doc-a']:
                hits.append({'doc_id':'doc-b','chunk_id':'other','text':query})
            return hits
        self.env['index'].search=search
        report=audit_indexed_corpus(self.env)
        self.assertEqual(self.row(report)['status'],'blocked')
        self.assertEqual(self.row(report,'doc-b')['status'],'read_ready')

    def test_omitted_index_chunk_is_blocked(self):
        self.env['index'].chunks.pop(0)
        row=self.row(audit_indexed_corpus(self.env))
        self.assertEqual(row['status'],'blocked')
        self.assertFalse(row['source_spans_valid'])

    def test_valid_chunks_cannot_hide_uncovered_source_block(self):
        self.env['documents'][0]['blocks'].append({'block_id':'new','text':'미처리 원문 내용입니다.',
                                                'locator':'PDF p.99','page':99})
        row=self.row(audit_indexed_corpus(self.env))
        self.assertIn('extracted_source_not_fully_indexed',row['block_reasons'])
        self.assertLess(row['source_character_coverage'],1)

    def test_pdf_ocr_vector_warnings_are_not_green_semantic_success(self):
        d=self.env['documents'][0]
        d['warnings']=[{'code':'PDF_OCR_REVIEW_NEEDED','locator':'PDF p.2'},
                       {'code':'PDF_NUMERIC_VALUE_UNRESOLVED','locator':'PDF p.3'}]
        d['stats']={'ocr_executed':False,'numeric_gap_review_pages':[3],
                    'vector_glyph_review_pages':[2,3],'image_pages':[2]}
        row=self.row(audit_indexed_corpus(self.env))
        self.assertTrue(row['text_pipeline_ready']);self.assertEqual(row['status'],'needs_review')
        self.assertFalse(row['ocr_executed']);self.assertEqual(row['ocr_review_pages'],1)
        self.assertFalse(row['visual_content_completeness_verified'])

    def test_nonreviewed_ocr_supplement_is_not_counted_as_preserved_source(self):
        d=self.env['documents'][0]
        d['supplemental_blocks']=[{'block_id':'ocr','text':'untrusted OCR',
                                 'source_method':'ocr','review_status':'unreviewed'}]
        row=self.row(audit_indexed_corpus(self.env))
        self.assertEqual(row['source_character_coverage'],1)
        self.assertEqual(row['excluded_unreviewed_or_reference_blocks'],1)
        self.assertEqual(row['status'],'needs_review')

    def test_detected_binary_format_takes_precedence_over_extension(self):
        d=self.env['documents'][0];d['file_name']='mislabeled.hwp'
        # File identity changes need a fresh registry/index in a real ingestion.
        d['detected_format']='exe'
        row=self.row(audit_indexed_corpus(self.env))
        self.assertEqual(row['status'],'blocked')
        self.assertIn('unsupported_detected_format',row['block_reasons'])

    def test_alias_sha_mismatch_is_reported(self):
        self.env['documents'][-1]['sha256']='f'*64
        report=audit_indexed_corpus(self.env)
        alias=report['original_mapping'][-1]
        self.assertFalse(alias['mapped']);self.assertFalse(report['all_text_pipelines_ready'])

    def test_mutated_source_chunk_is_not_silently_read(self):
        self.env['chunks'][0]['text']='mutated text'
        row=self.row(audit_indexed_corpus(self.env))
        self.assertEqual(row['status'],'blocked')

    def test_source_text_not_document_id_is_used_as_query(self):
        seen=[];original=self.env['index'].search
        def spy(query,**kwargs):seen.append(query);return original(query,**kwargs)
        self.env['index'].search=spy
        audit_indexed_corpus(self.env)
        self.assertTrue(seen)
        self.assertTrue(all('사업비와 계약기간' in query for query in seen))
        self.assertTrue(all(query not in {'doc-a','doc-b','copy-a'} for query in seen))

    def test_probe_count_and_method_options_are_validated(self):
        for kwargs in ({'probes_per_document':0},{'probes_per_document':True},{'retrieval_method':'unknown'}):
            with self.assertRaises(ValueError):audit_indexed_corpus(self.env,**kwargs)

    def test_index_metadata_mismatch_does_not_claim_full_corpus_ready(self):
        self.env['index'].metadata['chunk_count']+=1
        report=audit_indexed_corpus(self.env)
        self.assertFalse(report['all_text_pipelines_ready'])
        self.assertIn('index_chunk_count_matches_snapshot',report['errors'])

    def test_search_backend_exception_is_isolated_to_document(self):
        original=self.env['index'].search
        def failing(query,**kwargs):
            if 'doc-a' in query:raise RuntimeError('synthetic backend failure')
            return original(query,**kwargs)
        self.env['index'].search=failing
        report=audit_indexed_corpus(self.env)
        self.assertEqual(self.row(report)['status'],'blocked')
        self.assertEqual(self.row(report,'doc-b')['status'],'read_ready')

    def test_search_defaults_to_all_documents_and_rrf(self):
        requests=[];original=self.env['index'].search
        def spy(query,**kwargs):requests.append(kwargs);return original(query,**kwargs)
        self.env['index'].search=spy
        hits=search_corpus(self.env,'doc-b 원문 0: 사업비와 계약기간을 확인합니다.')
        self.assertEqual(requests[0]['method'],'rrf')
        self.assertEqual(set(requests[0]['allowed_doc_ids']),{'doc-a','doc-b'})
        self.assertEqual(hits[0]['doc_id'],'doc-b')
        self.assertFalse(hits[0]['score_is_confidence'])

    def test_search_aliases_preserve_selected_scope(self):
        hits=search_corpus(self.env,'doc-a 원문 0: 사업비와 계약기간을 확인합니다.',doc_ids=['copy-a'])
        self.assertTrue(hits);self.assertTrue(all(h['doc_id']=='doc-a' for h in hits))
        with self.assertRaises(ValueError):search_corpus(self.env,'test',doc_ids=[])
        with self.assertRaises(ValueError):search_corpus(self.env,'test',doc_ids=['unknown'])

    def test_search_rejects_corrupted_hit_source_fields(self):
        original=self.env['index'].search
        def corrupt(query,**kwargs):
            hits=original(query,**kwargs)
            if hits:hits[0]['text']='fabricated source text'
            return hits
        self.env['index'].search=corrupt
        with self.assertRaises(ValueError):search_corpus(self.env,'doc-a 원문 0: 사업비와 계약기간을 확인합니다.')

    def test_search_returns_independent_metadata_and_scores(self):
        query='doc-a 원문 0: 사업비와 계약기간을 확인합니다.'
        hits=search_corpus(self.env,query)
        self.assertEqual(hits[0]['score'],0.25);self.assertEqual(hits[0]['rank'],1)
        self.assertEqual(hits[0]['sha256'],'a'*64)
        self.assertTrue(hits[0]['source_spans'])
        hits[0]['source_spans'][0]['locator']='changed by caller'
        self.assertNotEqual(search_corpus(self.env,query)[0]['source_spans'][0]['locator'],'changed by caller')

    def test_search_validates_inputs_without_widening_scope(self):
        with self.assertRaises(TypeError):search_corpus(self.env,None)
        for kwargs in ({'top_k':True},{'top_k':101},{'method':'not-real'}):
            with self.assertRaises(ValueError):search_corpus(self.env,'test',**kwargs)
        with self.assertRaises(ValueError):search_corpus(self.env,'x'*4001)
        self.assertEqual(search_corpus(self.env,'  '),[])

    def test_long_exact_excerpt_is_found_even_when_rrf_omits_its_chunk(self):
        value,query=long_source_env();wrong=copy.deepcopy(value['chunks'][1])
        wrong.update(score=0.8,rank=1,method='rrf',score_is_confidence=False)
        value['index'].search=lambda *args,**kwargs:[copy.deepcopy(wrong)]
        hit=search_corpus(value,query,top_k=1)[0]
        self.assertEqual(hit['chunk_id'],value['chunks'][0]['chunk_id'])
        self.assertTrue(hit['literal_match']);self.assertIsNone(hit['score'])
        self.assertFalse(hit['score_is_confidence'])

    def test_normal_question_keeps_rrf_ranking_and_score(self):
        value,_=long_source_env();hit=copy.deepcopy(value['chunks'][1])
        hit.update(score=0.8,rank=1,method='rrf',score_is_confidence=False)
        value['index'].search=lambda *args,**kwargs:[copy.deepcopy(hit)]
        result=search_corpus(value,'이 사업의 수행기간은 몇 개월인가요?')[0]
        self.assertEqual(result['chunk_id'],hit['chunk_id']);self.assertEqual(result['score'],0.8)
        self.assertNotIn('literal_match',result)

    def test_literal_preference_never_widens_explicit_document_scope(self):
        value,query=long_source_env()
        hits=search_corpus(value,query,doc_ids=['doc-b'])
        self.assertTrue(all(h['doc_id']=='doc-b' for h in hits))
        self.assertFalse(any(h.get('literal_match') for h in hits))

    def test_literal_preference_preserves_existing_rrf_score(self):
        value,query=long_source_env();a,b=copy.deepcopy(value['chunks'][:2])
        a.update(score=0.23,rank=2,method='rrf',score_is_confidence=False)
        b.update(score=0.91,rank=1,method='rrf',score_is_confidence=False)
        value['index'].search=lambda *args,**kwargs:[copy.deepcopy(b),copy.deepcopy(a)]
        hit=search_corpus(value,query,top_k=2)[0]
        self.assertEqual(hit['score'],0.23);self.assertEqual(hit['original_retrieval_rank'],2)

    def test_raw_bm25_comparison_does_not_receive_literal_repair(self):
        value,query=long_source_env();wrong=copy.deepcopy(value['chunks'][1])
        wrong.update(score=0.8,rank=1,method='bm25',score_is_confidence=False)
        original=value['index'].search
        value['index'].search=lambda q,**kwargs:([copy.deepcopy(wrong)] if q==query else original(q,**kwargs))
        report=audit_indexed_corpus(value)
        self.assertTrue(report['all_text_pipelines_ready'])
        self.assertTrue(report['baseline_bm25_scoped_excerpt_misses'])
        self.assertTrue(report['baseline_bm25_is_raw_ranker'])

    def test_literal_lookup_rejects_stale_snapshot_fingerprint(self):
        value,query=long_source_env();value['index'].metadata['chunks_fingerprint']='wrong'
        with self.assertRaises(ValueError):search_corpus(value,query)


if __name__=='__main__':unittest.main()
