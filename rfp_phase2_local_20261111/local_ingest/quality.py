"""파싱 성공, 원문 인용 복구, 표 구조 및 남은 검수를 구분해 기록합니다."""
import json
import re
from collections import Counter
from pathlib import Path

from .loader import write_json, sha256_file, safe_source_path


def _compact(text):
    return re.sub(r'\s+', '', text)


def validate_source_checks(documents, checks):
    by_id = {d['doc_id']: d for d in documents}
    rows = []
    for check in checks:
        document = by_id.get(check['doc_id'])
        row = {'check_id': check['check_id'], 'doc_id': check['doc_id'], 'page': check.get('page'),
               'automatic_text_match': False, 'reviewed_transcription_match': False,
               'source_hash_matches': bool(document and document['sha256'] == check['sha256'])}
        if row['source_hash_matches']:
            query = _compact(check['expected_text'])
            for field, result_key in [('blocks', 'automatic_text_match'), ('supplemental_blocks', 'reviewed_transcription_match')]:
                row[result_key] = any(query in _compact(b['text']) and b.get('page') == check.get('page') for b in document.get(field, []))
        rows.append(row)
    return {'total': len(rows), 'automatic_text_matches': sum(r['automatic_text_match'] for r in rows),
            'reviewed_transcription_matches': sum(r['reviewed_transcription_match'] for r in rows),
            'all_covered': all(r['source_hash_matches'] and (r['automatic_text_match'] or r['reviewed_transcription_match']) for r in rows),
            'records': rows, 'scope': '별도 시각 검수 회귀검사. 자동 파싱 인용 복구율과 합산하지 않음.'}


def validate_table_samples(documents):
    by_id = {d['doc_id']: d for d in documents}
    specifications = [
        ('BOHUN_INTERNAL_USER', '1EMKe6hKnikAMXKJFufR5rU-nUBDguhJJ', 4, ['내부사용자']),
        ('KSD_BRIEFING_REMARK', '1XuHMGElCN1wjEBBucRF55JWKMCePUmUM', 3, ['제안요청설명회', '12층']),
        ('KSD_CONTRACT_REMARK', '1XuHMGElCN1wjEBBucRF55JWKMCePUmUM', 3, ['계약체결', '16개월']),
    ]
    results = []
    for check_id, doc_id, page, expected in specifications:
        hits = []
        for table in by_id.get(doc_id, {}).get('tables', []):
            if table.get('page') != page:
                continue
            for row_index, row in enumerate(table['rows']):
                text = _compact(' '.join(cell or '' for cell in row))
                if all(_compact(part) in text for part in expected):
                    hits.append({'table_id': table['table_id'], 'row': row_index, 'text': text})
        results.append({'check_id': check_id, 'doc_id': doc_id, 'page': page, 'passed': bool(hits), 'matches': hits})
    return {'total': len(results), 'passed': sum(r['passed'] for r in results), 'all_passed': all(r['passed'] for r in results),
            'records': results, 'scope': '시각 검수한 특정 표 행/비고 셀의 회귀검사. 전체 PDF 표의 정확도는 아님.'}


def audit_documents(documents, evidence_report=None, root=None):
    canonical = [d for d in documents if not d.get('is_duplicate_alias')]
    warnings = [w for d in documents for w in d.get('warnings', [])]
    tables = [(d, t) for d in canonical for t in d.get('tables', [])]
    issues = []
    for document in documents:
        for block in document.get('blocks', []) + document.get('reference_blocks', []) + document.get('supplemental_blocks', []):
            if not block.get('block_id') or not block.get('locator'):
                issues.append({'doc_id': document['doc_id'], 'code': 'BLOCK_SOURCE_MISSING'})
        if root and document.get('sha256'):
            current = sha256_file(safe_source_path(root, document['relative_path']))
            if current != document['sha256']:
                issues.append({'doc_id': document['doc_id'], 'code': 'RAW_CHANGED_DURING_RUN'})
    for document, table in tables:
        if not table.get('locator') or not table.get('table_id'):
            issues.append({'doc_id': document['doc_id'], 'code': 'TABLE_SOURCE_MISSING'})
        if table.get('structure_status') != 'explicit':
            continue
        rows = table.get('rows', [])
        width = max((len(r) for r in rows), default=0)
        occupied = set()
        for cell in table.get('cells', []):
            r, c = cell.get('row'), cell.get('col')
            rs, cs = cell.get('row_span', 1), cell.get('col_span', 1)
            valid = all(isinstance(v, int) for v in [r, c, rs, cs]) and r >= 0 and c >= 0 and rs > 0 and cs > 0 and r + rs <= len(rows) and c + cs <= width
            if not valid:
                issues.append({'doc_id': document['doc_id'], 'table_id': table['table_id'], 'code': 'EXPLICIT_CELL_RANGE_INVALID'})
                continue
            slots = {(i, j) for i in range(r, r + rs) for j in range(c, c + cs)}
            if occupied & slots:
                issues.append({'doc_id': document['doc_id'], 'table_id': table['table_id'], 'code': 'EXPLICIT_CELL_OVERLAP'})
            occupied |= slots
    queue = []
    general = {'HWP_LAYOUT_NOT_RENDERED', 'NON_TEXT_OBJECTS_PRESENT', 'PDF_TABLES_HEURISTIC',
               'HWPX_LAYOUT_NOT_RENDERED', 'HWPX_NON_TEXT_OBJECTS', 'XML_READING_ORDER', 'VISUAL_TRANSCRIPTION_APPLIED'}
    for document in canonical:
        for warning in document.get('warnings', []):
            code = warning.get('code', 'UNKNOWN_WARNING')
            if code in general:
                continue
            priority = 'high' if any(token in code for token in ['NUMERIC', 'OCR', 'EMPTY', 'UNRESOLVED', 'FAILED', 'SOURCE_HASH']) else 'review'
            queue.append({'doc_id': document['doc_id'], 'file_name': document['file_name'], 'code': code,
                          'priority': priority, 'locator': warning.get('locator'), 'message': warning.get('message')})
    for document, table in tables:
        if table.get('structure_status') == 'unresolved':
            queue.append({'doc_id': document['doc_id'], 'file_name': document['file_name'],
                          'code': 'TABLE_STRUCTURE_UNRESOLVED', 'priority': 'review',
                          'locator': table['locator'], 'table_id': table['table_id'],
                          'message': '표 또는 표 후보의 행·열/병합 범위를 확정하지 않았습니다. 원문 좌표와 텍스트를 보존하고 재검수합니다.'})
    statuses = Counter(d['status'] for d in documents)
    report = {
        'document_files': len(documents), 'unique_contents': len({d['sha256'] for d in documents if d.get('sha256')}),
        'duplicate_alias_files': sum(d.get('is_duplicate_alias', False) for d in documents),
        'successful_files': sum(d['status'] != 'failed' for d in documents), 'failed_files': statuses.get('failed', 0),
        'status_counts': dict(statuses), 'detected_format_counts': dict(Counter(d.get('detected_format', 'unidentified') for d in documents)),
        'canonical_text_blocks': sum(len(d.get('blocks', [])) for d in canonical),
        'canonical_text_characters': sum(len(b.get('text', '')) for d in canonical for b in d.get('blocks', [])),
        'reference_view_blocks': sum(len(d.get('reference_blocks', [])) for d in canonical),
        'reviewed_transcription_blocks': sum(len(d.get('supplemental_blocks', [])) for d in canonical),
        'pdf_physical_pages_unique_contents': sum(d.get('stats', {}).get('pages', len(d.get('blocks', []))) for d in canonical if d.get('detected_format') == 'pdf'),
        'table_counts_by_format_and_status': dict(Counter(d['detected_format'] + ':' + t.get('structure_status', 'unknown') for d, t in tables)),
        'pdf_tables_are_heuristic_candidates': True,
        'warning_counts': dict(Counter(w.get('code', 'UNKNOWN_WARNING') for w in warnings)),
        'source_structure_issues': issues, 'review_queue_entries': len(queue), 'review_queue': queue,
        'all_input_sha256_match': all(not d.get('expected_sha256') or d.get('sha256') == d['expected_sha256'] for d in documents),
        'all_source_locations_present': not any(i['code'].endswith('SOURCE_MISSING') for i in issues),
        'automatic_evidence_check': {k: v for k, v in (evidence_report or {}).items() if k not in {'records', 'cases'}},
        'openai_api_calls': 0, 'remote_file_mutations': 0,
        'validation_scope': '본문 추출·원문 위치·문서 버전·평가 근거 복구 및 명시 표 좌표 무결성. 모든 그림/표의 시각적 의미 완전성은 아님.',
        'phase2_implementation_ready': bool(documents) and statuses.get('failed', 0) == 0 and not issues
            and all(not d.get('expected_sha256') or d.get('sha256') == d['expected_sha256'] for d in documents)
            and (evidence_report is None or evidence_report.get('unmatched') == 0),
        'visual_or_ocr_review_items_remain': bool(queue) or any(t.get('structure_status') != 'explicit' for _, t in tables),
        'table_candidates_or_unresolved_requiring_review': sum(t.get('structure_status') != 'explicit' for _, t in tables),
    }
    if root:
        root = Path(root)
        checks_path = root / 'eval/phase2_source_checks.json'
        if checks_path.is_file():
            report['visual_transcription_regression'] = validate_source_checks(documents, json.loads(checks_path.read_text(encoding='utf-8')))
        report['table_sample_regression'] = validate_table_samples(documents)
        if report.get('visual_transcription_regression', {}).get('all_covered') is False or not report['table_sample_regression']['all_passed']:
            report['phase2_implementation_ready'] = False
        numeric_review_path = root / 'data/numeric_page_reviews.json'
        if numeric_review_path.is_file():
            raw_reviews = json.loads(numeric_review_path.read_text(encoding='utf-8'))
            by_id = {d['doc_id']: d for d in documents}
            valid_reviews = {(r['doc_id'], r['page']): r for r in raw_reviews
                             if r.get('review_status') == 'visual_verified'
                             and r['doc_id'] in by_id and by_id[r['doc_id']]['sha256'] == r['sha256']}
            numeric_flags = {(d['doc_id'], int(w['locator'].split('p.')[-1])) for d in canonical
                             for w in d.get('warnings', []) if w.get('code') == 'PDF_NUMERIC_VALUE_UNRESOLVED'
                             and str(w.get('locator', '')).startswith('PDF p.')}
            covered = numeric_flags & set(valid_reviews)
            report['numeric_priority_review'] = {'flagged_pages': len(numeric_flags),
                                                'version_matched_reviewed_pages': len(covered),
                                                'unreviewed_pages': len(numeric_flags - covered),
                                                'classification_counts': dict(Counter(valid_reviews[key]['classification'] for key in covered))}
            for item in queue:
                if item['code'] != 'PDF_NUMERIC_VALUE_UNRESOLVED':
                    continue
                key = (item['doc_id'], int(item['locator'].split('p.')[-1]))
                if key in valid_reviews:
                    item['review_status'] = 'visual_verified'
                    item['review_classification'] = valid_reviews[key]['classification']
                    item['priority'] = 'reviewed'
            if numeric_flags - covered:
                report['phase2_implementation_ready'] = False
        report['review_queue_open_entries'] = sum(item.get('priority') != 'reviewed' for item in queue)
        write_json(root / 'reports/quality_report.json', report)
        write_json(root / 'reports/review_queue.json', queue)
    return report
