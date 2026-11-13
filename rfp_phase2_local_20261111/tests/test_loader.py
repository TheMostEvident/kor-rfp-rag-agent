"""실제 RFP 재파싱 없이 파일·캐시·별칭·전사의 중요한 무결성을 검증합니다."""
import copy
import hashlib
import json
import socket
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from local_ingest import loader


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rfp-local-loader-')
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / 'project'
        (self.root / 'data/raw').mkdir(parents=True)
        self.module_hash = 'a' * 64
        self.calls = []
        self.module = types.SimpleNamespace(parse_pdf=self.fake_pdf)

    def fake_pdf(self, path, extract_tables=True):
        self.calls.append((Path(path).name, extract_tables))
        return {'detected_format': 'pdf', 'parser': 'test-local-pdf',
                'blocks': [{'block_id': 'page1', 'kind': 'page', 'locator': 'PDF p.1',
                            'text': Path(path).read_bytes().decode('ascii')}],
                'reference_blocks': [], 'tables': [], 'warnings': [],
                'stats': {'table_extraction_requested': bool(extract_tables)}}

    def entry(self, doc_id='d1', content=b'%PDF-1.7\nSource A\n', filename=None):
        filename = filename or (doc_id + '.pdf')
        path = self.root / 'data/raw' / filename
        path.write_bytes(content)
        return {'doc_id': doc_id, 'name': filename,
                'relative_path': path.relative_to(self.root).as_posix(),
                'sha256': hashlib.sha256(content).hexdigest(),
                'agency': 'First Agency', 'document_year': 2025, 'split': 'dev'}

    def run_manifest(self, manifest, **kwargs):
        with patch.object(loader, '_module_for', side_effect=lambda kind: (self.module, self.module_hash)):
            return loader.run_ingestion(self.root, manifest=manifest, **kwargs)

    def read_jsonl(self, relative):
        return [json.loads(line) for line in (self.root / relative).read_text().splitlines()]

    def test_duplicate_bytes_parse_once_and_index_canonical_once(self):
        later = self.entry('z_alias', filename='copy.pdf')
        canonical = self.entry('a_main', filename='main.pdf')
        documents = self.run_manifest([later, canonical])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([d['parsing_origin'] for d in documents],
                         ['parsed_from_original', 'validated_cache'])
        self.assertTrue(documents[0]['is_duplicate_alias'])
        self.assertFalse(documents[1]['is_duplicate_alias'])
        self.assertEqual(documents[0]['canonical_doc_id'], 'a_main')
        rows = self.read_jsonl('data/processed/indexable_blocks.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['doc_id'], 'a_main')
        self.assertEqual(rows[0]['alias_doc_ids'], ['a_main', 'z_alias'])
        self.assertEqual(documents[0]['file_name'], 'copy.pdf')
        self.assertEqual(documents[1]['file_name'], 'main.pdf')

    def test_changed_raw_bytes_reparse_and_preserve_expected_hash(self):
        entry = self.entry()
        first = self.run_manifest([entry])[0]
        (self.root / entry['relative_path']).write_bytes(b'%PDF-1.7\nNew content\n')
        second = self.run_manifest([entry])[0]
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertEqual(second['expected_sha256'], first['sha256'])
        self.assertTrue(any(w['code'] == 'SOURCE_HASH_CHANGED' for w in second['warnings']))
        self.assertIn('New content', second['blocks'][0]['text'])

    def test_parser_module_hash_change_invalidates_cache(self):
        entry = self.entry()
        first = self.run_manifest([entry])[0]
        self.module_hash = 'b' * 64
        second = self.run_manifest([entry])[0]
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(first['parser_module_sha256'], second['parser_module_sha256'])
        self.assertEqual(second['parsing_origin'], 'parsed_from_original')

    def test_pdf_table_option_has_separate_cache(self):
        entry = self.entry()
        without = self.run_manifest([entry], extract_pdf_tables=False)[0]
        with_tables = self.run_manifest([entry], extract_pdf_tables=True)[0]
        self.assertEqual([called[1] for called in self.calls], [False, True])
        self.assertFalse(without['stats']['table_extraction_requested'])
        self.assertTrue(with_tables['stats']['table_extraction_requested'])

    def test_missing_file_does_not_abort_valid_documents_or_output(self):
        missing = self.entry('missing')
        (self.root / missing['relative_path']).unlink()
        valid = self.entry('valid')
        documents = self.run_manifest([missing, valid])
        self.assertEqual(documents[0]['status'], 'failed')
        self.assertIn(documents[1]['status'], ['parsed', 'parsed_with_warnings'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(loader.load_documents(self.root)), 2)
        self.assertEqual(len(self.read_jsonl('data/processed/indexable_blocks.jsonl')), 1)

    def test_relative_path_escape_fails_one_document_without_reading_it(self):
        outside = self.parent / 'outside.pdf'
        outside.write_bytes(b'%PDF-1.7\nOutside\n')
        evil = {'doc_id': 'outside', 'name': 'outside.pdf', 'relative_path': '../outside.pdf'}
        valid = self.entry('inside')
        documents = self.run_manifest([evil, valid])
        self.assertEqual(documents[0]['status'], 'failed')
        self.assertIsNone(documents[0]['sha256'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0], 'inside.pdf')

    def test_absolute_and_symlink_escape_rejected_by_source_guard(self):
        outside = self.parent / 'outside.pdf'
        outside.write_bytes(b'%PDF-1.7\nOutside\n')
        with self.assertRaises(ValueError):
            loader.safe_source_path(self.root, str(outside))
        link = self.root / 'data/raw/escape.pdf'
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            loader.safe_source_path(self.root, link.relative_to(self.root))

    def test_discovery_never_hashes_a_symlink_target_outside_root(self):
        outside = self.parent / 'outside.pdf'
        outside.write_bytes(b'%PDF-1.7\nOutside\n')
        (self.root / 'data/raw/escape.pdf').symlink_to(outside)
        self.entry('inside')
        original_hash = loader.sha256_file

        def guarded_hash(path):
            self.assertTrue(Path(path).resolve().is_relative_to(self.root),
                            'discover_manifest read bytes outside the work root')
            return original_hash(path)

        with patch.object(loader, 'sha256_file', side_effect=guarded_hash):
            try:
                loader.discover_manifest(self.root)
            except ValueError:
                pass  # 거절 또는 건너뛰기 모두 작업 폴더 밖의 바이트를 읽지 않으면 됩니다.

    def test_version_mismatched_transcription_not_applied(self):
        document = {'doc_id': 'd1', 'sha256': 'a' * 64, 'blocks': [], 'warnings': []}
        entry = {'doc_id': 'd1', 'sha256': 'b' * 64, 'review_status': 'visual_verified',
                 'bbox': [1, 2, 3, 4], 'check_id': 'check1', 'page': 1, 'text': '100원'}
        loader.apply_reviewed_transcriptions(document, [entry])
        self.assertEqual(document['supplemental_blocks'], [])
        self.assertEqual(document['warnings'][0]['code'], 'TRANSCRIPTION_VERSION_MISMATCH')

    def test_unreviewed_transcription_is_rejected(self):
        document = {'doc_id': 'd1', 'sha256': 'a' * 64, 'blocks': [], 'warnings': []}
        entry = {'doc_id': 'd1', 'sha256': 'a' * 64, 'review_status': 'draft',
                 'bbox': [1, 2, 3, 4], 'check_id': 'check1', 'page': 1, 'text': '100원'}
        with self.assertRaises(ValueError):
            loader.apply_reviewed_transcriptions(document, [entry])

    def test_valid_transcription_remains_separate_from_parser_blocks(self):
        document = {'doc_id': 'd1', 'sha256': 'a' * 64,
                    'blocks': [{'block_id': 'p1', 'text': '원문', 'locator': 'PDF p.1'}], 'warnings': []}
        original = copy.deepcopy(document['blocks'])
        entry = {'doc_id': 'd1', 'sha256': 'a' * 64, 'review_status': 'visual_verified',
                 'bbox': [1, 2, 3, 4], 'check_id': 'check1', 'page': 1, 'text': '100원'}
        loader.apply_reviewed_transcriptions(document, [entry])
        self.assertEqual(document['blocks'], original)
        self.assertEqual(document['supplemental_blocks'][0]['source_method'], 'reviewed_visual_transcription')

    def test_cached_text_does_not_cache_old_document_metadata(self):
        entry = self.entry()
        first = self.run_manifest([entry])[0]
        first['blocks'][0]['text'] = 'caller-side mutation'
        updated = dict(entry, name='renamed.pdf', agency='New Agency', document_year=2026, split='holdout')
        second = self.run_manifest([updated])[0]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(second['parsing_origin'], 'validated_cache')
        self.assertEqual(second['file_name'], 'renamed.pdf')
        self.assertEqual(second['agency'], 'New Agency')
        self.assertEqual(second['document_year'], 2026)
        self.assertEqual(second['split'], 'holdout')
        self.assertNotEqual(second['blocks'][0]['text'], 'caller-side mutation')

    def test_duplicate_doc_id_is_rejected(self):
        entry = self.entry()
        with self.assertRaises(ValueError):
            self.run_manifest([entry, entry])

    def test_corrupt_cache_is_reparsed_from_original(self):
        entry = self.entry()
        self.run_manifest([entry])
        cache_file = next((self.root / 'data/processed/cache').glob('*.json'))
        cache_file.write_text('{not valid json')
        second = self.run_manifest([entry])[0]
        self.assertNotEqual(second['status'], 'failed')
        self.assertEqual(len(self.calls), 2)

    def test_preparsed_text_only_pdf_cannot_satisfy_table_requested_run(self):
        entry = self.entry()
        preparsed = self.root / 'preparsed'
        preparsed.mkdir()
        payload = self.fake_pdf(self.root / entry['relative_path'], extract_tables=False)
        self.calls.clear()
        (preparsed / (entry['doc_id'] + '.json')).write_text(json.dumps({
            'sha256': entry['sha256'], 'parser_module_sha256': self.module_hash,
            'options': {'extract_pdf_tables': False}, 'payload': payload}))
        result = self.run_manifest([entry], preparsed_dir=preparsed, extract_pdf_tables=True)[0]
        self.assertTrue(result['stats']['table_extraction_requested'])
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0][1])

    def test_loader_pipeline_requires_no_network(self):
        entry = self.entry()
        with patch.object(socket, 'socket', side_effect=AssertionError('network prohibited')):
            with patch.object(socket, 'create_connection', side_effect=AssertionError('network prohibited')):
                documents = self.run_manifest([entry])
        self.assertNotEqual(documents[0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
