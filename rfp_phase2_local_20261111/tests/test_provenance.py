import copy
import unittest

from local_ingest.provenance import verify_evidence


HASH = 'a' * 64


def document(doc_id='d1', text='사업예산 100원', locator='PDF p.1', **extra):
    return dict(doc_id=doc_id, sha256=HASH, file_name=doc_id+'.pdf',
                blocks=[dict(block_id='p1', locator=locator, text=text)], **extra)


def evidence(doc_id='d1', quote='사업예산 100원', locator='PDF p.1', sha256=HASH):
    return dict(doc_id=doc_id, quote=quote, locator=locator, sha256=sha256, block_id='old')


def case(ev=None, case_id='F001'):
    return dict(case_id=case_id, required_facts=[dict(fact_id='f1', evidence=[ev or evidence()])])


class ProvenanceTests(unittest.TestCase):
    def test_exact_match_has_raw_span_and_expected_locator(self):
        doc=document(text='앞 사업예산 100원 뒤')
        result=verify_evidence([doc],[case()]);record=result['records'][0]
        self.assertEqual(result['exact_matches'],1)
        self.assertEqual(record['current_locator'],'PDF p.1')
        self.assertEqual(doc['blocks'][0]['text'][record['char_start']:record['char_end']],'사업예산 100원')
        self.assertTrue(record['location_resolved'])

    def test_correct_locator_preferred_over_same_text_on_other_page(self):
        doc=document();doc['blocks'].append(dict(block_id='p2',locator='PDF p.2',text='사업예산 100원'))
        result=verify_evidence([doc],[case(evidence(locator='PDF p.2'))])
        self.assertEqual(result['records'][0]['block_id'],'p2')
        self.assertEqual(result['ambiguous_matches'],0)

    def test_never_borrows_other_document(self):
        result=verify_evidence([document(doc_id='d2')],[case()])
        self.assertEqual(result['missing_document'],1)
        self.assertEqual(result['exact_matches'],0)

    def test_changed_source_hash_rejects_identical_quote(self):
        result=verify_evidence([document()],[case(evidence(sha256='b'*64))])
        self.assertEqual(result['hash_mismatch'],1)
        self.assertEqual(result['unmatched'],1)

    def test_whitespace_match_maps_back_to_original_characters(self):
        doc=document(text='앞 사업예산\n 100원 뒤')
        result=verify_evidence([doc],[case()]);record=result['records'][0]
        self.assertEqual(result['normalized_matches'],1)
        self.assertFalse(record['raw_exact'])
        raw=doc['blocks'][0]['text'][record['char_start']:record['char_end']]
        self.assertEqual(raw,'사업예산\n 100원')
        self.assertEqual(record['matched_text'],raw)

    def test_normalized_original_locator_before_exact_other_location(self):
        doc=document(text='사업예산\n100원')
        doc['blocks'].append(dict(block_id='p2',locator='PDF p.2',text='사업예산 100원'))
        result=verify_evidence([doc],[case()]);record=result['records'][0]
        self.assertEqual(record['current_locator'],'PDF p.1')
        self.assertEqual(record['match_mode'],'normalized_whitespace')

    def test_ambiguous_relocated_quote_does_not_select_arbitrary_position(self):
        doc=document();doc['blocks'].append(dict(block_id='p2',locator='PDF p.2',text='사업예산 100원'))
        result=verify_evidence([doc],[case(evidence(locator='old section'))]);record=result['records'][0]
        self.assertEqual(result['exact_matches'],1)
        self.assertEqual(result['ambiguous_matches'],1)
        self.assertEqual(record['candidate_count'],2)
        self.assertIsNone(record['current_locator'])
        self.assertIsNone(record['char_start'])

    def test_overlapping_occurrences_remain_ambiguous(self):
        result=verify_evidence([document(text='banana')],[case(evidence(quote='ana'))])
        self.assertEqual(result['records'][0]['candidate_count'],2)
        self.assertEqual(result['resolved_matches'],0)

    def test_reference_view_exact_match_is_explicit(self):
        doc=document(text='사업예산')
        doc['reference_blocks']=[dict(block_id='ref1',locator='Contents/section0.xml',text='사업예산\n100원')]
        result=verify_evidence([doc],[case(evidence(quote='사업예산\n100원',locator='Contents/section0.xml'))])
        self.assertEqual(result['exact_matches'],1)
        self.assertEqual(result['reference_view_matches'],1)

    def test_same_physical_text_in_compatibility_view_does_not_invent_ambiguity(self):
        doc=document();doc['reference_blocks']=copy.deepcopy(doc['blocks'])
        result=verify_evidence([doc],[case()])
        self.assertEqual(result['ambiguous_matches'],0)

    def test_does_not_stitch_independent_blocks_into_quote(self):
        doc=document(text='사업예산')
        doc['blocks'].append(dict(block_id='p2',locator='PDF p.2',text='100원'))
        result=verify_evidence([doc],[case()])
        self.assertEqual(result['unmatched'],1)

    def test_alias_requires_explicit_same_hash_canonical_source(self):
        alias=dict(doc_id='alias',sha256=HASH,file_name='사본.pdf',canonical_doc_id='d1',blocks=[])
        result=verify_evidence([document(),alias],[case(evidence(doc_id='alias'))]);record=result['records'][0]
        self.assertEqual(record['doc_id'],'alias')
        self.assertEqual(record['source_doc_id'],'d1')
        self.assertEqual(record['file_name'],'사본.pdf')
        canonical=document();canonical['sha256']='b'*64
        result=verify_evidence([canonical,alias],[case(evidence(doc_id='alias'))])
        self.assertEqual(result['alias_source_mismatch'],1)

    def test_no_evidence_cases_are_not_automatic_performance_success(self):
        result=verify_evidence([],[dict(case_id='U001',required_facts=[])])
        self.assertEqual(result['total'],0)
        self.assertIsNone(result['coverage']['found_fraction'])
        self.assertIsNone(result['all_links_found'])
        self.assertEqual(result['cases_without_evidence'],['U001'])

    def test_case_basis_links_count_independently_and_inputs_are_not_mutated(self):
        doc=document();c=case();c['case_basis_evidence']=[evidence()]
        original=copy.deepcopy((doc,c));result=verify_evidence([doc],[c])
        self.assertEqual(result['total'],2)
        self.assertEqual((doc,c),original)

    def test_empty_whitespace_evidence_is_invalid(self):
        result=verify_evidence([document()],[case(evidence(quote=' \n\t'))])
        self.assertEqual(result['invalid_evidence'],1)

    def test_candidate_list_limit_does_not_hide_actual_ambiguity(self):
        result=verify_evidence([document(text='word word word')],[case(evidence(quote='word'))],max_candidates=1)
        record=result['records'][0]
        self.assertEqual(record['candidate_count'],3)
        self.assertEqual(len(record['candidates']),1)
        self.assertTrue(record['candidates_truncated'])
        self.assertFalse(record['location_resolved'])

    def test_duplicate_doc_ids_rejected(self):
        with self.assertRaises(ValueError):verify_evidence([document(),document()],[case()])


if __name__=='__main__':
    unittest.main()
