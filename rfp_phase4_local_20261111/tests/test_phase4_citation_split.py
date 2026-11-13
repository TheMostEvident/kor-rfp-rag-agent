"""A joined noncontiguous quote becomes two exact quotes, never a fake substring."""
import copy
import unittest

from test_phase4_evidence import block,document,chunk,answer
from rfp_phase4.evidence import EvidenceRegistry
from rfp_phase4.citation_recovery import validate_with_quote_recovery


HEADER='<부가세 포함, 금액단위 : 원>'
BODY='구       분\n금      액\n비     고\n용 역 비\n부가가치세 포함\n장  비\n구입비\nHW 구매 부문\n부가가치세 포함'
GAP='발주자\n한국주택금융공사\n직전도급자\n-\n하도급\n주는자\n사업명\n계약금액\n상  호\n대표자'


class CitationSplitTests(unittest.TestCase):
    def check(self,source,quote,*,did='d1',allowed=None):
        b=block(text=source);d=document(source=b);c=chunk(b)
        reg=EvidenceRegistry([d],[c]);payload=answer(quote=quote,did=did)
        original=copy.deepcopy(payload)
        before=reg.validate_answer(payload,[c],allowed)
        after=validate_with_quote_recovery(reg,payload,[c],allowed)
        self.assertEqual(payload,original)
        return before,after,d,c

    def test_reported_header_plus_table_body_is_two_exact_citations(self):
        before,r,d,c=self.check(HEADER+'\n'+GAP+'\n'+BODY+'\nSW 구매 부문\n부가가치세 포함',HEADER+'\n'+BODY)
        self.assertFalse(before['valid']);self.assertTrue(r['valid'])
        citations=r['claims'][0]['citations']
        self.assertEqual(len(citations),2)
        self.assertEqual([v['quote'] for v in citations],[HEADER,BODY])
        self.assertEqual(r['citation_recovery']['split_citations'],1)
        self.assertEqual(r['citation_recovery']['additional_api_calls'],0)
        for v in citations:
            self.assertIn(v['quote'],c['text'])
            for span in v['source_ranges']:
                self.assertEqual(d['blocks'][0]['text'][span['char_start']:span['char_end']],
                                 c['text'][span['chunk_start']:span['chunk_end']])
        self.assertNotIn(GAP,' '.join(v['quote'] for v in citations))

    def test_whitespace_inside_each_fragment_is_restored(self):
        _,r,_,_=self.check(HEADER+'\n'+GAP+'\n'+BODY,HEADER.replace(' ','')+'\n'+'\n'.join(s.replace(' ','') for s in BODY.splitlines()))
        self.assertTrue(r['valid']);self.assertEqual(r['claims'][0]['citations'][1]['quote'],BODY)

    def test_changed_numbers_vat_units_punctuation_and_words_are_rejected(self):
        source=HEADER+'\n'+GAP+'\n사업예산은 688백만원이며 부가세 포함입니다.'
        for tail in ['사업예산은 699백만원이며 부가세 포함입니다.','사업예산은 688백만원이며 부가세 별도입니다.',
                     '사업예산은 688달러이며 부가세 포함입니다.','사업예산은 688백만원이며 부가세 포함입니다!',
                     '사업예산은 688백만원으로 확정되었습니다.']:
            self.assertFalse(self.check(source,HEADER+'\n'+tail)[1]['valid'])

    def test_intra_line_omission_and_negation_removal_are_rejected(self):
        source=HEADER+'\n'+GAP+'\n사업예산은 688백만원이며 부가세 포함이 아닙니다.'
        r=self.check(source,HEADER+'\n사업예산은 688백만원이며 부가세 포함입니다.')[1]
        self.assertFalse(r['valid'])

    def test_reordered_fragments_are_rejected(self):
        self.assertFalse(self.check(HEADER+'\n'+GAP+'\n'+BODY,BODY+'\n'+HEADER)[1]['valid'])

    def test_duplicate_fragment_placements_are_ambiguous(self):
        source=HEADER+'\n'+GAP+'\n'+BODY+'\n여백\n'+BODY
        r=self.check(source,HEADER+'\n'+BODY)[1]
        self.assertFalse(r['valid'])
        self.assertEqual(r['citation_diagnostics'][0]['split_check'],'ambiguous_fragment_mapping')

    def test_more_than_one_omitted_block_is_not_accepted(self):
        a='사업예산은 688백만원입니다.';b='사업기간은 계약일부터 5개월입니다.';c='안정화 기간은 1개월 포함입니다.'
        source=a+'\n생략 구간 하나\n'+b+'\n생략 구간 둘\n'+c
        self.assertFalse(self.check(source,a+'\n'+b+'\n'+c)[1]['valid'])

    def test_short_isolated_labels_do_not_enable_split(self):
        source='예산\n다른 문장\n부가세는 포함이며 예산은 688백만원입니다.'
        self.assertFalse(self.check(source,'예산\n부가세는 포함이며 예산은 688백만원입니다.')[1]['valid'])

    def test_scope_and_document_checks_still_apply(self):
        source=HEADER+'\n'+GAP+'\n'+BODY;quote=HEADER+'\n'+BODY
        self.assertFalse(self.check(source,quote,allowed=[])[1]['valid'])
        self.assertFalse(self.check(source,quote,did='other')[1]['valid'])

    def test_true_contiguous_quote_is_not_split(self):
        _,r,_,_=self.check(HEADER+'\n'+BODY,HEADER+'\n'+BODY)
        self.assertTrue(r['valid']);self.assertEqual(r['citation_recovery']['split_citations'],0)
        self.assertEqual(len(r['claims'][0]['citations']),1)

    def test_replaying_revalidated_output_keeps_original_usage(self):
        import test_phase4_citation_recovery as fixtures
        from rfp_phase4.citation_recovery import replay_saved_result
        fixture=fixtures.ReplayTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        first=replay_saved_result(fixture.root,fixture.path,registry=fixture.reg)
        second=replay_saved_result(fixture.root,first['revalidated_result_path'],registry=fixture.reg)
        self.assertEqual(second['generation']['original_usage']['api_calls'],1)
        self.assertEqual(second['generation']['usage']['api_calls'],0)


if __name__=='__main__':unittest.main()
