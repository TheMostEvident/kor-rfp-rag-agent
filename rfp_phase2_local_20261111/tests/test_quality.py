"""실제 문서 없이 품질 집계·버전·표 좌표·완료 게이트를 독립 검증합니다."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from local_ingest.quality import audit_documents, validate_source_checks


HASH = 'a' * 64


def document(**changes):
    doc = {'doc_id': 'd1', 'file_name': 'sample.pdf', 'sha256': HASH,
           'expected_sha256': HASH, 'status': 'parsed', 'detected_format': 'pdf',
           'is_duplicate_alias': False,
           'blocks': [{'block_id': 'p1', 'kind': 'page', 'locator': 'PDF p.1',
                       'page': 1, 'text': '자동 추출된 본문'}],
           'reference_blocks': [], 'supplemental_blocks': [], 'tables': [], 'warnings': []}
    doc.update(changes)
    return doc


def evidence_report(**changes):
    report = {'total': 2, 'exact_matches': 2, 'normalized_matches': 0,
              'unmatched': 0, 'hash_mismatch': 0, 'missing_document': 0,
              'invalid_evidence': 0, 'records': []}
    report.update(changes)
    return report


def explicit_table(**changes):
    table = {'table_id': 't1', 'locator': 'Contents/section0.xml / table 1',
             'structure_status': 'explicit', 'rows': [['가', None], ['나', '다']],
             'cells': [{'row': 0, 'col': 0, 'row_span': 1, 'col_span': 2, 'text': '가'},
                       {'row': 1, 'col': 0, 'row_span': 1, 'col_span': 1, 'text': '나'},
                       {'row': 1, 'col': 1, 'row_span': 1, 'col_span': 1, 'text': '다'}]}
    table.update(changes)
    return table


class QualityTests(unittest.TestCase):
    def test_valid_merged_explicit_table_has_no_false_overlap(self):
        doc = document(detected_format='hwpx', tables=[explicit_table()])
        report = audit_documents([doc], evidence_report())
        self.assertEqual(report['source_structure_issues'], [])
        self.assertEqual(report['table_counts_by_format_and_status'], {'hwpx:explicit': 1})
        self.assertTrue(report['phase2_implementation_ready'])

    def test_duplicate_explicit_cell_coordinates_are_rejected(self):
        table = explicit_table()
        table['cells'].append(copy.deepcopy(table['cells'][0]))
        report = audit_documents([document(tables=[table])], evidence_report())
        self.assertTrue(any(i['code'] == 'EXPLICIT_CELL_OVERLAP' for i in report['source_structure_issues']))
        self.assertFalse(report['phase2_implementation_ready'])

    def test_invalid_explicit_cell_extents_are_rejected(self):
        cases = [dict(row=-1), dict(col=-1), dict(row_span=0), dict(col_span=0),
                 dict(row=2), dict(col=2), dict(row_span=3), dict(col_span=3)]
        for change in cases:
            with self.subTest(change=change):
                table = explicit_table()
                table['cells'][0].update(change)
                report = audit_documents([document(tables=[table])], evidence_report())
                self.assertTrue(any(i['code'] == 'EXPLICIT_CELL_RANGE_INVALID' for i in report['source_structure_issues']))
                self.assertFalse(report['phase2_implementation_ready'])

    def test_pdf_candidates_are_not_counted_as_explicit_tables(self):
        table = explicit_table(structure_status='heuristic', locator='PDF p.1', page=1)
        report = audit_documents([document(tables=[table])], evidence_report())
        self.assertEqual(report['table_counts_by_format_and_status'], {'pdf:heuristic': 1})
        self.assertTrue(report['pdf_tables_are_heuristic_candidates'])
        self.assertNotIn('pdf:explicit', report['table_counts_by_format_and_status'])

    def test_pdf_heuristic_tables_still_require_visual_review(self):
        table = explicit_table(structure_status='heuristic', locator='PDF p.1', page=1)
        doc = document(tables=[table], warnings=[{'code': 'PDF_TABLES_HEURISTIC', 'message': '시각 검수 필요'}])
        report = audit_documents([doc], evidence_report())
        self.assertTrue(report['visual_or_ocr_review_items_remain'])

    def test_duplicate_alias_does_not_double_count_tables_or_body(self):
        main = document(tables=[explicit_table()])
        alias = copy.deepcopy(main)
        alias.update(doc_id='d2', is_duplicate_alias=True)
        report = audit_documents([main, alias], evidence_report())
        self.assertEqual(report['document_files'], 2)
        self.assertEqual(report['unique_contents'], 1)
        self.assertEqual(report['canonical_text_blocks'], 1)
        self.assertEqual(sum(report['table_counts_by_format_and_status'].values()), 1)

    def test_three_transcriptions_do_not_raise_automatic_evidence_counts(self):
        supplements = [{'block_id': 'v'+str(n), 'kind': 'verified_transcription',
                        'locator': 'PDF p.1 / 검수 '+str(n), 'page': 1,
                        'text': '시각 전사 '+str(n), 'source_method': 'reviewed_visual_transcription'} for n in range(3)]
        doc = document(supplemental_blocks=supplements)
        report = audit_documents([doc], evidence_report())
        self.assertEqual(report['reviewed_transcription_blocks'], 3)
        self.assertEqual(report['canonical_text_blocks'], 1)
        self.assertEqual(report['automatic_evidence_check']['total'], 2)
        self.assertEqual(report['automatic_evidence_check']['exact_matches'], 2)

    def test_visual_regression_counts_auto_and_transcription_separately(self):
        supplement = {'block_id': 'v1', 'locator': 'PDF p.1 / 검수', 'page': 1, 'text': '전사된 예산 100원'}
        doc = document(supplemental_blocks=[supplement])
        check = {'check_id': 'VIS1', 'doc_id': 'd1', 'page': 1, 'sha256': HASH, 'expected_text': '예산 100원'}
        report = validate_source_checks([doc], [check])
        self.assertEqual(report['automatic_text_matches'], 0)
        self.assertEqual(report['reviewed_transcription_matches'], 1)
        self.assertTrue(report['all_covered'])
        check['sha256'] = 'b' * 64
        report = validate_source_checks([doc], [check])
        self.assertFalse(report['all_covered'])
        self.assertEqual(report['reviewed_transcription_matches'], 0)

    def test_missing_gold_evidence_prevents_ready_gate(self):
        report = audit_documents([document()], evidence_report(exact_matches=1, unmatched=1, missing_document=1))
        self.assertFalse(report['phase2_implementation_ready'])

    def test_hash_mismatched_gold_evidence_prevents_ready_gate(self):
        report = audit_documents([document()], evidence_report(exact_matches=1, unmatched=1, hash_mismatch=1))
        self.assertFalse(report['phase2_implementation_ready'])

    def test_input_version_mismatch_prevents_ready_even_if_quotes_happen_to_match(self):
        doc = document(expected_sha256='b' * 64)
        report = audit_documents([doc], evidence_report())
        self.assertFalse(report['all_input_sha256_match'])
        self.assertFalse(report['phase2_implementation_ready'])

    def test_empty_corpus_does_not_claim_phase2_ready(self):
        report = audit_documents([], evidence_report(total=0, exact_matches=0))
        self.assertFalse(report['phase2_implementation_ready'])

    def test_source_file_bytes_are_rechecked_and_unchanged_by_audit(self):
        with tempfile.TemporaryDirectory(prefix='rfp-quality-') as temp:
            root = Path(temp)
            source = root / 'data/raw/source.pdf'
            source.parent.mkdir(parents=True)
            original = b'%PDF-1.7\noriginal local bytes\n'
            source.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            doc = document(sha256=digest, expected_sha256=digest, relative_path='data/raw/source.pdf')
            report = audit_documents([doc], evidence_report(), root=root)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report['source_structure_issues'], [])
            self.assertTrue(report['all_input_sha256_match'])
            source.write_bytes(original+b'changed')
            report = audit_documents([doc], evidence_report(), root=root)
            self.assertTrue(any(i['code'] == 'RAW_CHANGED_DURING_RUN' for i in report['source_structure_issues']))
            self.assertFalse(report['phase2_implementation_ready'])

    def test_missing_source_location_prevents_ready(self):
        doc = document()
        del doc['blocks'][0]['locator']
        report = audit_documents([doc], evidence_report())
        self.assertFalse(report['all_source_locations_present'])
        self.assertFalse(report['phase2_implementation_ready'])


if __name__ == '__main__':
    unittest.main()
