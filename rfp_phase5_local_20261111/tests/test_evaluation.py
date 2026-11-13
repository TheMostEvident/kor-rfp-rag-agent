import copy
import unittest

from local_rag.evaluation import (audit_chunk_coverage, evaluate_retrieval,
                                evaluate_summary_coverage)

SHA = 'a' * 64


def document(doc_id='d1', text='예산 100원 부가세 포함', *, canonical=None, sha=SHA):
    return {'doc_id': doc_id, 'canonical_doc_id': canonical or doc_id, 'sha256': sha,
            'file_name': doc_id+'.hwp', 'blocks': [{'block_id':'b1','locator':'Section0 / p1',
            'section':'s0','text':text}], 'reference_blocks':[], 'supplemental_blocks':[]}


def evidence(doc_id='d1', text='예산 100원 부가세 포함', *, sha=SHA, start=0, end=None,
             block_id='b1', reference=False, alternatives=None, case_id='F001'):
    candidate={'block_id':block_id,'locator':'Section0 / p1','char_start':start,
               'char_end':len(text) if end is None else end,'reference_view':reference}
    return {'records':[{'case_id':case_id,'fact_id':'f1','evidence_kind':'fact',
            'evidence_index':1,'doc_id':doc_id,'gold_sha256':sha,'source_sha256':sha,
            'quote':text,'match_mode':'exact','location_resolved':alternatives is None,
            'candidates':alternatives or [candidate]}]}


def chunk(text='예산 100원 부가세 포함', *, cid='c1', doc_id='d1', sha=SHA, start=0,
          end=None, block_id='b1', locator='Section0 / p1', collection='blocks'):
    end = start+len(text) if end is None else end
    return {'chunk_id':cid,'doc_id':doc_id,'sha256':sha,'text':text,
            'source_spans':[{'block_id':block_id,'locator':locator,'char_start':start,
             'char_end':end,'chunk_start':0,'chunk_end':len(text),'collection':collection}]}


def case(feature='fact', *, selected=None, split='dev'):
    return {'case_id':'F001','feature':feature,'split':split,'question':'이 사업 예산은?',
            'context':{'selected_doc_ids':['d1'] if selected is None else selected},
            'target_doc_ids':['gold-only-target'],
            'required_facts':[{'fact_id':'f1','doc_id':'d1','expected_value':'SECRET GOLD VALUE',
                              'evidence':[{}]}]}


class FakeIndex:
    def __init__(self,hits):self.hits=hits;self.calls=[];self.document_calls=[]
    def search(self, query, **kwargs):
        self.calls.append((query, kwargs));return self.hits[:kwargs['top_k']]
    def get_document_chunks(self,doc_id):
        self.document_calls.append(doc_id);return [h for h in self.hits if h['doc_id']==doc_id]


class SourceCoverageTests(unittest.TestCase):
    def test_exact_offsets_and_complete_source_coverage(self):
        d=document();c=chunk();a=audit_chunk_coverage([d],[c],evidence())
        self.assertTrue(a['valid']);self.assertEqual(a['fully_covered_links'],1)
        self.assertEqual(a['records'][0]['alternatives'][0]['chunk_coverage']['c1'],[[0,11]])

    def test_evidence_spanning_chunks_requires_union_not_merely_touch(self):
        text='가나다라마바사';d=document(text=text)
        chunks=[chunk(text[:3],cid='a'),chunk(text[3:],cid='b',start=3)]
        a=audit_chunk_coverage([d],chunks,evidence(text=text));self.assertTrue(a['valid'])
        result=evaluate_retrieval(FakeIndex(chunks),[case()],a,top_ks=(1,2))
        self.assertEqual(result['metrics_at_k']['1']['micro_evidence_span_recall'],0)
        self.assertEqual(result['metrics_at_k']['2']['micro_evidence_span_recall'],1)
        self.assertAlmostEqual(result['metrics_at_k']['1']['mean_evidence_character_coverage'],3/7)

    def test_overlap_is_not_double_counted(self):
        text='가나다라마바사';d=document(text=text)
        chunks=[chunk(text[:5],cid='a'),chunk(text[2:5],cid='b',start=2)]
        a=audit_chunk_coverage([d],chunks,evidence(text=text))
        result=evaluate_retrieval(FakeIndex(chunks),[case()],a,top_ks=(2,))
        self.assertAlmostEqual(result['metrics_at_k']['2']['mean_evidence_character_coverage'],5/7)
        self.assertEqual(result['metrics_at_k']['2']['micro_evidence_span_recall'],0)

    def test_whitespace_does_not_fake_missing_substantive_characters(self):
        text='가 나\n다';d=document(text=text)
        a=audit_chunk_coverage([d],[chunk(text)],evidence(text='가나다',end=len(text)))
        self.assertTrue(a['valid']);self.assertEqual(a['records'][0]['alternatives'][0]['required_non_whitespace_characters'],3)

    def test_hwp_reference_maps_across_primary_paragraphs(self):
        d=document(text='예산 100원');d['blocks'][0]['section']='s0'
        d['blocks'].append({'block_id':'b2','locator':'Section0 / p2','section':'s0','text':'부가세 포함'})
        ref='예산\n100원\n부가세 포함';d['reference_blocks']=[{'block_id':'r','locator':'s0','section':'s0','text':ref}]
        e=evidence(text=ref,block_id='r',reference=True)
        chunks=[chunk('예산 100원',cid='a'),chunk('부가세 포함',cid='b',block_id='b2',locator='Section0 / p2')]
        a=audit_chunk_coverage([d],chunks,e);self.assertTrue(a['valid'])
        self.assertEqual(len(a['records'][0]['alternatives'][0]['source_fragments']),2)

    def test_reference_mismatch_is_reported_not_approximately_repaired(self):
        d=document(text='예산 100원');d['reference_blocks']=[{'block_id':'r','locator':'s0','section':'s0','text':'예산 900원'}]
        a=audit_chunk_coverage([d],[chunk('예산 100원')],evidence(text='예산 900원',block_id='r',reference=True))
        self.assertEqual(a['unmapped_links'],1);self.assertFalse(a['valid'])
        self.assertIn('identical non-whitespace',a['records'][0]['reason'])

    def test_ambiguous_source_candidates_are_all_retained(self):
        d=document(text='금액 금액');c=chunk('금액 금액')
        alternatives=[{'block_id':'b1','locator':'Section0 / p1','char_start':s,'char_end':s+2,'reference_view':False} for s in [0,3]]
        a=audit_chunk_coverage([d],[c],evidence(text='금액',alternatives=alternatives))
        self.assertTrue(a['valid']);self.assertEqual(a['source_ambiguous_links'],1)
        self.assertEqual(len(a['records'][0]['alternatives']),2)
        self.assertTrue(a['records'][0]['all_alternatives_covered'])

    def test_other_document_with_identical_text_cannot_supply_coverage(self):
        d=document();other=document('d2');a=audit_chunk_coverage([d,other],[chunk(doc_id='d2')],evidence())
        self.assertFalse(a['valid']);self.assertEqual(a['fully_covered_links'],0)

    def test_changed_source_hash_never_reuses_old_gold_match(self):
        a=audit_chunk_coverage([document(sha='b'*64)],[chunk(sha='b'*64)],evidence())
        self.assertEqual(a['unmapped_links'],1);self.assertFalse(a['valid'])

    def test_incorrect_chunk_hash_is_source_issue(self):
        a=audit_chunk_coverage([document()],[chunk(sha='b'*64)],evidence())
        self.assertGreater(a['chunk_source_issue_count'],0);self.assertFalse(a['valid'])

    def test_source_and_chunk_text_disagreement_is_not_labelled_relevant(self):
        a=audit_chunk_coverage([document()],[chunk('예산 999원 부가세 포함')],evidence())
        self.assertGreater(a['chunk_source_issue_count'],0);self.assertEqual(a['fully_covered_links'],0)

    def test_synthetic_metadata_prefix_cannot_be_original_evidence(self):
        c=chunk();c['text']='발주기관\n'+c['text']
        for span in c['source_spans']:span['chunk_start']+=5;span['chunk_end']+=5
        a=audit_chunk_coverage([document()],[c],evidence())
        self.assertGreater(a['chunk_source_issue_count'],0)
        self.assertEqual(a['fully_covered_links'],0)

    def test_same_hash_alias_uses_canonical_chunks_only(self):
        docs=[document('d1'),document('alias',canonical='d1')]
        a=audit_chunk_coverage(docs,[chunk()],evidence(doc_id='alias'))
        self.assertTrue(a['valid']);self.assertEqual(a['records'][0]['canonical_doc_id'],'d1')

    def test_reference_view_is_not_allowed_as_indexable_source_span(self):
        d=document();d['reference_blocks']=[{'block_id':'r','locator':'s0','section':'s0','text':d['blocks'][0]['text']}]
        c=chunk(block_id='r',locator='s0',collection='reference_blocks')
        a=audit_chunk_coverage([d],[c],evidence());self.assertGreater(a['chunk_source_issue_count'],0)

    def test_unreviewed_ocr_cannot_be_promoted_by_hash_match(self):
        d=document();d['supplemental_blocks']=[{'block_id':'o','locator':'PDF p.1','text':d['blocks'][0]['text'],
                 'review_status':'unreviewed','source_method':'local_ocr'}]
        c=chunk(block_id='o',locator='PDF p.1',collection='supplemental_blocks')
        a=audit_chunk_coverage([d],[c],evidence());self.assertGreater(a['chunk_source_issue_count'],0)


class RetrievalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.hits=[chunk()];self.alignment=audit_chunk_coverage([document()],self.hits,evidence())

    def test_only_question_and_user_selected_scope_reach_search(self):
        index=FakeIndex(self.hits);c=case(selected=['user-selected-only'])
        evaluate_retrieval(index,[c],self.alignment,top_ks=(1,))
        query,kwargs=index.calls[0]
        self.assertEqual(query,c['question']);self.assertNotIn('SECRET',query)
        self.assertEqual(kwargs['allowed_doc_ids'],['user-selected-only'])
        self.assertNotEqual(kwargs['allowed_doc_ids'],c['target_doc_ids'])

    def test_global_protocol_ignores_even_user_context_filter(self):
        index=FakeIndex(self.hits)
        evaluate_retrieval(index,[case()],self.alignment,top_ks=(1,),scope_mode='global')
        self.assertIsNone(index.calls[0][1]['allowed_doc_ids'])

    def test_search_configuration_matches_interactive_route(self):
        index=FakeIndex(self.hits)
        result=evaluate_retrieval(index,[case()],self.alignment,top_ks=(5,20),
                                  search_config={'fetch_k':40,'mmr_lambda':.6})
        self.assertEqual(index.calls[0][1]['fetch_k'],40)
        self.assertEqual(index.calls[0][1]['mmr_lambda'],.6)
        self.assertEqual(result['effective_search_config'],
                         {'fetch_k':40,'mmr_lambda':.6,'method':'rrf','top_k':20})

    def test_fetch_budget_never_underflows_requested_top_k(self):
        index=FakeIndex(self.hits)
        evaluate_retrieval(index,[case()],self.alignment,top_ks=(20,),search_config={'fetch_k':5})
        self.assertEqual(index.calls[0][1]['fetch_k'],20)

    def test_invalid_search_configuration_is_rejected(self):
        for config in [{'fetch_k':0},{'fetch_k':True},{'mmr_lambda':1.5},{'mmr_lambda':float('nan')}]:
            with self.assertRaises(ValueError):
                evaluate_retrieval(FakeIndex(self.hits),[case()],self.alignment,search_config=config)

    def test_missing_alignment_labels_stay_in_denominator(self):
        a=copy.deepcopy(self.alignment);a['records']=[]
        result=evaluate_retrieval(FakeIndex(self.hits),[case()],a,top_ks=(1,))
        self.assertEqual(result['metrics_at_k']['1']['evidence_links'],1)
        self.assertEqual(result['metrics_at_k']['1']['micro_evidence_span_recall'],0)

    def test_holdout_requires_explicit_frozen_run_opt_in(self):
        with self.assertRaises(ValueError):evaluate_retrieval(FakeIndex(self.hits),[case(split='holdout')],self.alignment)
        result=evaluate_retrieval(FakeIndex(self.hits),[case(split='holdout')],self.alignment,top_ks=(1,),allow_holdout=True)
        self.assertEqual(result['case_count'],1)

    def test_summary_and_dialogues_are_not_called_answer_accuracy(self):
        index=FakeIndex(self.hits)
        result=evaluate_retrieval(index,[case('summary'),case('followup')],self.alignment)
        self.assertEqual(result['case_count'],0);self.assertEqual(index.calls,[])
        self.assertIsNone(result['metrics_at_k']['5']['micro_evidence_span_recall'])

    def test_first_overlap_mrr_is_separate_from_complete_proof_recall(self):
        d=document(text='가나다라');hits=[chunk('가',cid='a')]
        a=audit_chunk_coverage([d],hits,evidence(text='가나다라'))
        result=evaluate_retrieval(FakeIndex(hits),[case()],a,top_ks=(1,))
        self.assertEqual(result['mrr_first_source_overlap'],1)
        self.assertEqual(result['metrics_at_k']['1']['micro_evidence_span_recall'],0)

    def test_summary_route_fetches_all_actual_user_selected_document_chunks(self):
        index=FakeIndex(self.hits)
        result=evaluate_summary_coverage(index,[case('summary')],self.alignment)
        self.assertEqual(index.document_calls,['d1']);self.assertEqual(result['all_evidence_covered_cases'],1)

    def test_invalid_scope_and_topk_are_explicit_errors(self):
        with self.assertRaises(ValueError):evaluate_retrieval(FakeIndex(self.hits),[case()],self.alignment,scope_mode='gold')
        with self.assertRaises(ValueError):evaluate_retrieval(FakeIndex(self.hits),[case()],self.alignment,top_ks=(0,))


if __name__=='__main__':unittest.main()
