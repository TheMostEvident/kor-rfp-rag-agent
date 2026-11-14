"""Raw corpus checks use temporary originals and explicit parser fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rfp_phase5_corpus.raw_audit import audit_raw_corpus,detect_format


class RawAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'phase2';(self.root/'data/raw').mkdir(parents=True)
        (self.root/'local_ingest').mkdir()
        self.payload={'detected_format':'pdf','blocks':[{'block_id':'p1','locator':'PDF p.1','text':'사업비 100원','page':1}],
                      'tables':[],'warnings':[],'stats':{'pages':1,'empty_text_pages':[]}}
        self.parser(self.payload)

    def parser(self,payload,error=None):
        source="import json\nfrom pathlib import Path\ndef parse_pdf(path,extract_tables=True):\n"
        source+="    counter=Path(path).parent/'parse-count.txt'\n    counter.write_text(counter.read_text()+'1' if counter.exists() else '1')\n"
        source+=("    raise ValueError("+repr(error)+")\n") if error else ("    return json.loads("+repr(json.dumps(payload,ensure_ascii=False))+")\n")
        (self.root/'local_ingest/pdf.py').write_text(source)

    def document(self,name='one.pdf',data=b'%PDF-1.7\nfixture',did='doc-1'):
        path=self.root/'data/raw'/name;path.write_bytes(data)
        return {'doc_id':did,'canonical_doc_id':did,'file_name':name,'relative_path':'data/raw/'+name,
            'sha256':hashlib.sha256(data).hexdigest(),'detected_format':'pdf','status':'parsed',
            'blocks':self.payload['blocks'],'tables':self.payload['tables'],'warnings':self.payload['warnings'],
            'stats':self.payload['stats'],'parser_module_sha256':hashlib.sha256((self.root/'local_ingest/pdf.py').read_bytes()).hexdigest()}

    def test_snapshot_check_reads_actual_bytes_without_running_parser(self):
        doc=self.document();before=(self.root/'data/raw/one.pdf').read_bytes()
        result=audit_raw_corpus([doc],self.root)
        self.assertTrue(result['all_registered_sources_read']);self.assertTrue(result['all_raw_hashes_match'])
        self.assertEqual(result['unique_contents_reparsed'],0)
        self.assertFalse((self.root/'data/raw/parse-count.txt').exists())
        self.assertEqual((self.root/'data/raw/one.pdf').read_bytes(),before)
        self.assertEqual(result['rows'][0]['recognition_verification'],'saved_snapshot_with_raw_hash')
        self.assertFalse(result['semantic_completeness_verified'])

    def test_reparse_checks_original_output_and_ignores_no_parser_cache(self):
        doc=self.document();result=audit_raw_corpus([doc],self.root,reparse=True)
        self.assertEqual((self.root/'data/raw/parse-count.txt').read_text(),'1')
        self.assertTrue(result['rows'][0]['primary_blocks_match_snapshot'])
        self.assertTrue(result['rows'][0]['tables_match_snapshot'])
        self.assertTrue(result['all_snapshots_compatible']);self.assertFalse(result['parser_cache_used'])

    def test_snapshot_only_audit_detects_changed_or_missing_parser_without_executing_it(self):
        doc=self.document();parser=self.root/'local_ingest/pdf.py'
        parser.write_text(parser.read_text()+'\n# new parser revision\n')
        changed=audit_raw_corpus([doc],self.root)
        self.assertTrue(changed['rows'][0]['read_success'])
        self.assertFalse(changed['all_snapshots_compatible'])
        self.assertIn('PARSER_VERSION_CHANGED',changed['rows'][0]['review_codes'])
        parser.unlink()
        missing=audit_raw_corpus([doc],self.root)
        self.assertTrue(missing['rows'][0]['read_success'])
        self.assertFalse(missing['all_snapshots_compatible'])
        self.assertIn('PARSER_VERSION_UNVERIFIED',missing['rows'][0]['review_codes'])
        self.assertFalse((self.root/'data/raw/parse-count.txt').exists())

    def test_duplicate_original_ids_share_only_same_content_parse(self):
        a=self.document();b=self.document(name='copy.pdf',did='doc-2');b['canonical_doc_id']='doc-1'
        result=audit_raw_corpus([a,b],self.root,reparse=True)
        self.assertEqual(result['document_count'],2);self.assertEqual(result['unique_contents_reparsed'],1)
        self.assertEqual(result['read_success_count'],2)
        self.assertEqual((self.root/'data/raw/parse-count.txt').read_text(),'1')

    def test_missing_and_changed_originals_are_isolated(self):
        a=self.document();b=self.document(name='two.pdf',did='doc-2');c=self.document(name='three.pdf',did='doc-3')
        (self.root/'data/raw/one.pdf').unlink();(self.root/'data/raw/two.pdf').write_bytes(b'changed')
        result=audit_raw_corpus([a,b,c],self.root,reparse=True)
        self.assertEqual([r.get('failure_code') for r in result['rows']],['raw_source_missing','raw_source_changed',None])
        self.assertEqual(result['read_success_count'],1)

    def test_extension_does_not_override_pdf_magic(self):
        doc=self.document(name='wrong.hwp');result=audit_raw_corpus([doc],self.root)
        self.assertTrue(result['all_registered_sources_read']);self.assertEqual(result['extension_mismatch_count'],1)
        self.assertEqual(result['rows'][0]['detected_format'],'pdf')

    def test_new_and_unsupported_raw_files_are_reported(self):
        doc=self.document();(self.root/'data/raw/new.pdf').write_bytes(b'%PDF-1.7 new')
        (self.root/'data/raw/unsupported.bin').write_bytes(b'not a supported document')
        result=audit_raw_corpus([doc],self.root)
        self.assertEqual(result['unregistered_source_count'],2)
        self.assertTrue(any(r.get('failure_code') for r in result['unregistered_sources']))

    def test_encrypted_parser_failure_is_not_called_success(self):
        self.parser(self.payload,error='PDF_ENCRYPTED');doc=self.document()
        result=audit_raw_corpus([doc],self.root,reparse=True)
        self.assertEqual(result['rows'][0]['failure_code'],'encrypted_or_protected_source')
        self.assertFalse(result['all_registered_sources_read'])

    def test_image_only_pdf_requires_ocr_and_has_no_read_success(self):
        image={'detected_format':'pdf','blocks':[{'block_id':'p1','locator':'PDF p.1','text':'','page':1}],
               'tables':[],'warnings':[{'code':'PDF_OCR_REVIEW_NEEDED'}],'stats':{'pages':1,'empty_text_pages':[1]}}
        self.parser(image);doc=self.document();result=audit_raw_corpus([doc],self.root,reparse=True)
        self.assertFalse(result['rows'][0]['read_success']);self.assertTrue(result['rows'][0]['quality']['ocr_review_needed'])
        self.assertEqual(result['rows'][0]['failure_code'],'no_extractable_primary_text')

    def test_readable_pdf_with_uncertain_tables_is_readable_but_needs_review(self):
        self.payload['tables']=[{'structure_status':'heuristic','rows':[['100원']]}]
        self.parser(self.payload);doc=self.document();result=audit_raw_corpus([doc],self.root,reparse=True)
        self.assertTrue(result['rows'][0]['read_success']);self.assertTrue(result['rows'][0]['needs_review'])
        self.assertEqual(result['rows'][0]['quality']['unreviewed_or_heuristic_tables'],1)

    def test_new_parser_output_cannot_be_silently_equated_to_old_snapshot(self):
        doc=self.document();new=json.loads(json.dumps(self.payload));new['blocks'][0]['text']='사업비 200원'
        self.parser(new);result=audit_raw_corpus([doc],self.root,reparse=True)
        self.assertTrue(result['rows'][0]['read_success']);self.assertFalse(result['all_snapshots_compatible'])
        self.assertFalse(result['rows'][0]['primary_blocks_match_snapshot'])
        self.assertFalse(result['rows'][0]['parser_version_matches_snapshot'])

    def test_path_traversal_is_rejected_before_reading(self):
        doc=self.document();doc['relative_path']='../external.pdf'
        result=audit_raw_corpus([doc],self.root)
        self.assertEqual(result['rows'][0]['failure_code'],'unsafe_source_path')

    def test_separate_raw_workspace_and_progress_callback(self):
        doc=self.document();new=self.root.parent/'intake/data/raw';new.mkdir(parents=True)
        (new/'one.pdf').write_bytes((self.root/'data/raw/one.pdf').read_bytes())
        (new/'new.pdf').write_bytes(b'%PDF-1.7 new')
        progress=[];result=audit_raw_corpus([doc],self.root,raw_directory=new,
            progress=lambda n,total,row:progress.append((n,total,row['doc_id'])))
        self.assertTrue(result['all_registered_sources_read']);self.assertEqual(result['unregistered_source_count'],1)
        self.assertEqual(progress,[(1,1,'doc-1')]);self.assertEqual(result['raw_directory'],str(new.resolve()))


if __name__=='__main__':unittest.main()
