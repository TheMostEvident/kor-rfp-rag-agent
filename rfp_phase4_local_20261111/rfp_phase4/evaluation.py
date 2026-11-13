"""Dev-only diagnostics for grounded answers; automatic checks are not truth grading.

Questions and user-visible context are the only generation inputs. Gold facts,
expected document IDs, source alignment and review notes are consumed after an
answer has returned. The previously seen holdout is deliberately unsupported.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import time
import unicodedata


_FEATURES = {'fact', 'summary', 'comparison', 'followup', 'uncertainty'}
_STATUS_EVALUABLE = {'completed', 'not_requested', 'partial', 'incomplete'}
_NUMBERS = re.compile(r'(?<!\w)\d[\d,]*(?:\.\d+)?')


def _compact(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(text)) if not c.isspace())


def _union_size(intervals):
    merged = []
    for a, b in sorted(intervals):
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def _canonical(doc_id, alignment):
    return (alignment or {}).get('document_aliases', {}).get(doc_id, doc_id)


def _validated_view(result):
    validation = result.get('validation')
    if isinstance(validation, dict):
        return validation
    value = result.get('validated_answer')
    return value if isinstance(value, dict) else result


def _claims(result):
    return list(_validated_view(result).get('claims', result.get('claims', [])) or [])


def _source_maps(documents, alignment):
    blocks, hashes = {}, {}
    for document in documents or []:
        did = _canonical(document.get('canonical_doc_id', document['doc_id']), alignment)
        hashes[did] = document.get('sha256')
        if document.get('is_duplicate_alias'):
            continue
        for collection in ['blocks', 'supplemental_blocks']:
            for block in document.get(collection, []):
                blocks[(did, collection, block['block_id'])] = block.get('text', '')
    return blocks, hashes


def _citation_ranges(citation):
    value = citation.get('source_ranges', citation.get('source_spans', []))
    return value if isinstance(value, list) else []


def _label_coverage(label, citations, blocks, hashes, alignment):
    """Intersect quoted raw-source ranges; citing an entire chunk earns no credit."""
    if not label or not label.get('mapped'):
        return None, False
    did = _canonical(label.get('canonical_doc_id', label.get('doc_id')), alignment)
    relevant = [c for c in citations
                if _canonical(c.get('doc_id'), alignment) == did
                and (did not in hashes or c.get('sha256') == hashes[did])]
    alternatives, unavailable = [], False
    for alternative in label.get('alternatives', []):
        covered, required, missing = 0, 0, False
        for fragment in alternative.get('source_fragments', []):
            key = did, fragment.get('collection', 'blocks'), fragment.get('block_id')
            text = blocks.get(key)
            if text is None:
                # Do not substitute a text overlap heuristic for source coverage.
                missing = True
                break
            a, b = fragment.get('char_start'), fragment.get('char_end')
            if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text):
                missing = True
                break
            positions = [p for p in range(a, b) if not text[p].isspace()]
            required += len(positions)
            spans = []
            for citation in relevant:
                for span in _citation_ranges(citation):
                    if span.get('collection', 'blocks') != key[1] or span.get('block_id') != key[2]:
                        continue
                    left, right = span.get('char_start'), span.get('char_end')
                    if isinstance(left, int) and isinstance(right, int):
                        spans.append((left, right))
            covered += sum(any(left <= p < right for left, right in spans) for p in positions)
        if missing:
            unavailable = True
        elif required:
            alternatives.append((covered / required, covered == required))
    if any(value[1] for value in alternatives):
        return 1.0, True
    if unavailable:
        return None, False
    if not alternatives:
        return None, False
    return max(value[0] for value in alternatives), any(value[1] for value in alternatives)


def _core_payload(result):
    """Strip enriched metadata before a second independent registry validation."""
    source = _validated_view(result)
    return {'status': source.get('status', result.get('status')),
            'claims': [{'doc_id': claim.get('doc_id'), 'category': claim.get('category'),
                        'text': claim.get('text'),
                        'citations': [{'chunk_id': c.get('chunk_id'), 'quote': c.get('quote')}
                                      for c in claim.get('citations', [])]}
                       for claim in _claims(result)],
            'missing_information': list(source.get('missing_information', result.get('missing_information', [])) or [])}


def _context_chunks(result, registry):
    context = result.get('source_chunks', result.get('context_chunks'))
    if isinstance(context, list) and context:
        return context
    identifiers = result.get('processed_chunk_ids', result.get('source_chunk_ids'))
    if registry is not None and isinstance(identifiers, list):
        return [registry.get_chunk(cid) for cid in dict.fromkeys(identifiers)]
    return context


def evaluate_answer_case(case, result, alignment=None, *, source_documents=None, registry=None):
    """Return status/provenance/source-span diagnostics plus an unfilled review.

    A correct quotation does not establish that the associated model claim is
    entailed by it. Source coverage therefore remains separate from answer
    correctness, which is deliberately ``None`` until reviewed by a person.
    """
    if case.get('split') != 'dev':
        raise ValueError('Phase 4 evaluation accepts dev cases only; the prior holdout is already seen.')
    if not isinstance(result, dict):
        raise TypeError('Answer result must be a dict.')
    view = _validated_view(result)
    basis = 'pipeline_validation_and_enriched_source_ranges'
    context = _context_chunks(result, registry)
    independently_checked = False
    if registry is not None and isinstance(context, list):
        scope = result.get('active_doc_ids', result.get('selected_doc_ids'))
        if scope is None:
            scope = case.get('context', {}).get('selected_doc_ids')
        view = registry.validate_answer(_core_payload(result), context, allowed_doc_ids=scope or None)
        independently_checked = True
        basis = 'independent_registry_validation'
    valid = view.get('valid') is True
    claims = list(view.get('claims', result.get('claims', []))) if valid else []
    citations = [citation for claim in claims for citation in claim.get('citations', [])]
    declared_claims = _claims(result)
    declared_citations = [citation for claim in declared_claims for citation in claim.get('citations', [])]
    blocks, hashes = _source_maps(source_documents, alignment)
    parent_id = case.get('_alignment_case_id', case.get('case_id'))
    label_map = {label.get('link_id'): label for label in (alignment or {}).get('records', [])}
    evidence_rows = []
    literal_rows = []
    for position, fact in enumerate(case.get('required_facts', []), 1):
        fact_id = fact.get('fact_id', f'f{position}')
        for evidence_index, evidence in enumerate(fact.get('evidence', []), 1):
            link_id = f'{parent_id}|fact|{fact_id}|{evidence_index}'
            label = label_map.get(link_id)
            ratio, complete = _label_coverage(label, citations, blocks, hashes, alignment)
            evidence_rows.append({'link_id': link_id, 'mapped': bool(label and label.get('mapped')),
                                  'source_character_coverage': ratio,
                                  'fully_quoted_required_source': complete if ratio is not None else None})
        did = _canonical(fact.get('doc_id'), alignment)
        claim_text = ' '.join(str(c.get('text', '')) for c in claims
                              if _canonical(c.get('doc_id'), alignment) == did)
        expected = str(fact.get('expected_value', ''))
        tokens = list(dict.fromkeys(match.replace(',', '') for match in _NUMBERS.findall(expected)))
        found = [token for token in tokens if token in _compact(claim_text).replace(',', '')]
        literal_rows.append({'fact_id': fact_id, 'field': fact.get('field'),
                             'expected_phrase_present': bool(expected.strip()) and _compact(expected) in _compact(claim_text),
                             'expected_numeric_tokens': tokens, 'present_numeric_tokens': found,
                             'purpose': 'Literal diagnostic only; unit conversion, negation and semantic correctness are not graded.'})
    measured = [row for row in evidence_rows if row['source_character_coverage'] is not None]
    status = result.get('status', view.get('status'))
    execution = result.get('execution_status', 'unverified')
    declared_ok = bool(declared_claims) and all(bool(c.get('citations')) for c in declared_claims)
    return {
        'case_id': case.get('case_id'), 'feature': case.get('feature'),
        'expected_status': case.get('expected_status'), 'actual_status': status,
        'execution_status': execution, 'status_match': status == case.get('expected_status'),
        'status_label_evaluable': execution in _STATUS_EVALUABLE,
        'status_match_is_task_accuracy': False,
        'citation_source_validity': {'validation_passed': valid,
            'independently_checked': independently_checked, 'audit_basis': basis,
            'declared_claim_count': len(declared_claims), 'validated_claim_count': len(claims),
            'declared_citation_count': len(declared_citations), 'validated_citation_count': len(citations),
            'all_declared_claims_have_citations': declared_ok if declared_claims else None,
            'validated_citation_fraction': len(citations) / len(declared_citations) if declared_citations else None,
            'semantic_entailment_verified': False},
        'gold_source_span_recovery': {
            'required_evidence_links': len(evidence_rows), 'measured_links': len(measured),
            'fully_quoted_links': sum(row['fully_quoted_required_source'] is True for row in measured),
            'recall': (sum(row['fully_quoted_required_source'] is True for row in measured) / len(evidence_rows)
                       if evidence_rows and len(measured) == len(evidence_rows) else None),
            'mean_source_character_coverage': statistics.mean(row['source_character_coverage'] for row in measured)
                if measured and len(measured) == len(evidence_rows) else None,
            'measurement_available': bool(evidence_rows) and len(measured) == len(evidence_rows),
            'source_unit': 'non-whitespace Unicode characters in gold raw source ranges',
            'scope': 'Quoted source recovery only; a correct source quote can accompany an incorrect claim.',
            'evidence_results': evidence_rows},
        'literal_diagnostics': literal_rows,
        'answer_correctness': None, 'semantic_entailment_score': None,
        'human_review': {'status': 'not_reviewed', 'reviewer': None,
            'answer_correct': None, 'each_claim_entailed_by_its_quotes': None,
            'units_dates_vat_and_conditions_preserved': None,
            'required_facts_complete': None, 'abstention_or_clarification_appropriate': None,
            'summary_omissions_checked': None, 'notes': '',
            'question': case.get('question'), 'validated_claims': claims,
            'expected_facts_for_review_only': case.get('required_facts', []),
            'required_behaviors_for_review_only': case.get('required_behaviors', [])},
    }


def _explicitly_offline(target):
    if getattr(target, 'is_mock', False) is True:
        return True
    llm = getattr(target, 'llm', None)
    if getattr(llm, 'is_mock', False) is True:
        return True
    config = getattr(llm, 'config', None)
    if isinstance(config, dict):
        return config.get('enabled') is False
    return getattr(config, 'enabled', None) is False


def _blocked_result(reason):
    return {'status': 'insufficient_evidence', 'execution_status': 'blocked',
            'claims': [], 'validation': {'valid': False, 'claims': []},
            'reason': reason, 'answer_generated': False}


def _mode(result):
    generation = result.get('generation')
    direct = generation if isinstance(generation, dict) else result
    direct_live = direct.get('live_api')
    cached = direct.get('cache_hit') is True
    nested = list(result.get('batch_outcomes', [])) + list(result.get('document_results', []))
    nested_modes = [_mode(value) for value in nested if isinstance(value, dict)]
    if (direct_live is True and not cached) or 'live_api' in nested_modes:
        return 'live_api'
    if result.get('execution_status') in {'disabled', 'blocked'}:
        return result['execution_status']
    if cached:
        return 'cached_from_live_api' if direct_live is True else 'cached_from_mock_or_unverified'
    if 'cached_from_live_api' in nested_modes:
        return 'cached_from_live_api'
    if result.get('is_mock') is True or direct_live is False or nested_modes:
        return 'mock_or_cached_not_live'
    return 'unverified_execution'


def _usage(result):
    generation = result.get('generation')
    if isinstance(generation, dict) and isinstance(generation.get('usage'), dict):
        value = generation['usage']
    else:
        value = result.get('usage', {})
    if not value:
        nested = list(result.get('batch_outcomes', [])) + list(result.get('document_results', []))
        if nested:
            entries = [_usage(item) for item in nested if isinstance(item, dict)]
            return {key: sum(item.get(key, 0) for item in entries)
                    for key in ['input_tokens', 'output_tokens', 'total_tokens']}
    return {key: value[key] for key in ['input_tokens', 'output_tokens', 'total_tokens']
            if isinstance(value, dict) and isinstance(value.get(key), (int, float))
            and not isinstance(value.get(key), bool) and value[key] >= 0}


def _turn_case(case, turn):
    chosen = set(turn.get('required_fact_ids', []))
    result = dict(case)
    result['case_id'] = f"{case['case_id']}/{turn.get('turn_id', 'turn')}"
    result['_alignment_case_id'] = case['case_id']
    result['question'] = turn['user']
    result['expected_status'] = turn.get('expected_status', case.get('expected_status'))
    result['required_facts'] = [fact for fact in case.get('required_facts', [])
                                if fact.get('fact_id') in chosen] if 'required_fact_ids' in turn else []
    return result


def run_dev_evaluation(cases, answer_session_factory, summary_factory=None,
                       output_dir=None, max_cases=5, allow_live=False, *,
                       alignment=None, source_documents=None, registry=None,
                       allowed_doc_ids=None):
    """Run a bounded dev sample using only question/user context as model input.

    Fact/comparison/uncertainty call ``session.ask``; a follow-up keeps one
    session and passes only ``turn.user``. Summary always uses the full-document
    route. ``allow_live=False`` blocks enabled or unknown clients before calls.
    Explicit test doubles may set ``is_mock=True`` and are labelled separately.
    """
    cases = list(cases)
    if any(case.get('split') != 'dev' for case in cases):
        raise ValueError('Phase 4 dev runner rejects holdout/unspecified splits before any factory or API call.')
    if any(case.get('feature') not in _FEATURES for case in cases):
        raise ValueError('Unsupported evaluation feature.')
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError('max_cases must be a positive integer.')
    selected = cases[:max_cases]
    if allowed_doc_ids is not None:
        approved = {_canonical(value, alignment) for value in allowed_doc_ids}
        for case in selected:
            scope = case.get('context', {}).get('selected_doc_ids', [])
            candidates = case.get('context', {}).get('candidate_doc_ids', [])
            if any(_canonical(value, alignment) not in approved for value in list(scope) + list(candidates)):
                raise ValueError('A dev case user context is outside the explicitly approved document scope.')
    rows = []
    for case in selected:
        started = time.perf_counter()
        context = case.get('context') or {}
        scope = list(context.get('selected_doc_ids') or [])
        candidates = context.get('candidate_doc_ids')
        turn_rows = []
        try:
            if case['feature'] == 'summary':
                if summary_factory is None:
                    result = _blocked_result('No full-document summary factory was supplied; top-k is not substituted.')
                elif not scope:
                    result = {'status': 'needs_clarification', 'execution_status': 'completed',
                              'claims': [], 'validation': {'valid': True, 'claims': []},
                              'answer_generated': False, 'reason': 'A user-selected document is required for full summary.'}
                else:
                    summarizer = summary_factory()
                    if not allow_live and not _explicitly_offline(summarizer):
                        result = _blocked_result('Live or unknown summary client requires allow_live=True.')
                    else:
                        summaries = [summarizer.summarize(doc_id, question=case['question']) for doc_id in scope]
                        if len(summaries) == 1:
                            result = summaries[0]
                        else:
                            audited_summaries = [value for value in summaries
                                                 if _validated_view(value).get('valid') is True]
                            combined = [claim for value in audited_summaries for claim in _claims(value)]
                            all_valid = len(audited_summaries) == len(summaries)
                            ok = bool(combined) or all_valid
                            status = ('answered' if all_valid and all(value.get('status') == 'answered' for value in summaries)
                                      else 'partial' if combined else 'insufficient_evidence')
                            execution = ('completed' if all(value.get('execution_status') == 'completed' for value in summaries)
                                         else 'incomplete' if combined else 'blocked')
                            missing = [str(item) for value in summaries for item in value.get('missing_information', [])]
                            if not all_valid:
                                missing.append('하나 이상의 선택 문서 요약을 검증된 결과로 완료하지 못했습니다.')
                            result = {'status': status,
                                'execution_status': execution,
                                'claims': combined,
                                'missing_information': missing,
                                'validation': {'valid': ok, 'status': status,
                                               'claims': combined, 'missing_information': missing},
                                'document_results': summaries,
                                'source_chunks': [chunk for value in audited_summaries for chunk in value.get('source_chunks', [])],
                                'source_chunk_ids': list(dict.fromkeys(cid for value in summaries
                                    for cid in value.get('source_chunk_ids', []))),
                                'processed_chunk_ids': list(dict.fromkeys(cid for value in audited_summaries
                                    for cid in value.get('processed_chunk_ids', value.get('source_chunk_ids', []))))}
            else:
                session = answer_session_factory(active_doc_ids=scope)
                if not allow_live and not _explicitly_offline(session):
                    result = _blocked_result('Live or unknown answer client requires allow_live=True.')
                elif case['feature'] == 'followup':
                    turns = case.get('turns') or []
                    if not turns:
                        raise ValueError('A followup case requires user turns.')
                    results = []
                    for turn in turns:
                        value = session.ask(turn['user'], feature='qa', candidate_doc_ids=candidates)
                        results.append(value)
                        score = evaluate_answer_case(_turn_case(case, turn), value, alignment,
                            source_documents=source_documents, registry=registry)
                        score['execution_mode'] = _mode(value)
                        score['usage'] = _usage(value)
                        score['expected_doc_ids_for_scoring_only'] = turn.get('expected_doc_ids', [])
                        turn_rows.append(score)
                    result = dict(results[-1])
                    # Each turn has already been audited in its own actual scope.
                    # A subsequent local clarification with no model payload does
                    # not erase genuine citations from an earlier answered turn.
                    merged = [claim for score in turn_rows
                              for claim in score.get('human_review', {}).get('validated_claims', [])]
                    audited = all(score['citation_source_validity']['validation_passed']
                                  for score in turn_rows
                                  if score['citation_source_validity']['declared_claim_count'])
                    result['claims'] = merged
                    result['validation'] = {'valid': audited, 'claims': merged,
                        'status': result.get('status'), 'missing_information': result.get('missing_information', []),
                        'audit_basis': 'Union of individually audited turn claims; no cross-turn scope inference.'}
                    result['source_chunks'] = [chunk for value in results for chunk in value.get('source_chunks', [])]
                    result['usage'] = {key: sum(row['usage'].get(key, 0) for row in turn_rows)
                                       for key in ['input_tokens', 'output_tokens', 'total_tokens']}
                    result.pop('generation', None)
                else:
                    feature = 'comparison' if case['feature'] == 'comparison' else 'qa'
                    result = session.ask(case['question'], feature=feature, candidate_doc_ids=candidates)
            score = evaluate_answer_case(case, result, alignment,
                source_documents=source_documents, registry=None if turn_rows else registry)
            if not turn_rows:
                score['execution_mode'] = _mode(result)
            else:
                turn_modes = Counter(row['execution_mode'] for row in turn_rows)
                score['turn_execution_modes'] = dict(turn_modes)
                score['execution_mode'] = ('live_api' if turn_modes.get('live_api') else
                    'cached_from_live_api' if turn_modes.get('cached_from_live_api') else 'mock_or_cached_followup_not_live')
            score['usage'] = _usage(result)
            if turn_rows:
                score['turn_results'] = turn_rows
                score['all_turn_statuses_match'] = all(row['status_label_evaluable'] and row['status_match'] for row in turn_rows)
                score['last_turn_status_match_only_is_dialogue_success'] = False
        except Exception as error:
            failed = _blocked_result('Evaluation execution failed.')
            failed['execution_status'] = 'error'
            score = evaluate_answer_case(case, failed, alignment, source_documents=source_documents)
            score.update({'execution_mode': 'error', 'error_type': type(error).__name__,
                     'error_message': 'Evaluation execution failed. Inspect the local controlled client logs; raw exception text is not saved.',
                     'turn_results': turn_rows, 'status_match': False, 'usage': {}})
        score['latency_ms'] = (time.perf_counter() - started) * 1000
        score['generation_input_policy'] = 'Only question/user turns and context selected/candidate IDs; no gold values, target IDs or expected future scope.'
        rows.append(score)
    modes = Counter(row['execution_mode'] for row in rows)
    measured = [row['gold_source_span_recovery'] for row in rows
                if row.get('gold_source_span_recovery', {}).get('measurement_available')]
    denominator = sum(row['required_evidence_links'] for row in measured)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split': 'dev',
        'supplied_cases': len(cases), 'evaluated_cases': len(rows), 'max_cases': max_cases,
        'selection_policy': 'First max_cases in supplied order; no gold-driven case selection.',
        'feature_counts': dict(Counter(row['feature'] for row in rows)),
        'execution_modes': dict(modes), 'live_execution_explicitly_allowed': bool(allow_live),
        'live_api_executed': any(mode == 'live_api' for mode in modes),
        'live_api_quality_measured': False,
        'live_quality_measurement_policy': 'Source diagnostics can run on live outputs; semantic answer quality requires the separate human review.',
        'automatic_answer_accuracy': None, 'human_semantic_review_required': True,
        'status_label_evaluable_cases': sum(row.get('status_label_evaluable') is True for row in rows),
        'status_label_match_count': sum(row.get('status_match') is True and row.get('status_label_evaluable') is True for row in rows),
        'status_label_match_is_answer_accuracy': False,
        'quoted_gold_source_span_recall': sum(row['fully_quoted_links'] for row in measured) / denominator if denominator else None,
        'quoted_gold_source_evidence_links_measured': denominator,
        'quoted_gold_source_evidence_links_total': sum(row.get('gold_source_span_recovery', {}).get('required_evidence_links', 0) for row in rows),
        'latency_ms': {'p50': statistics.median(row['latency_ms'] for row in rows) if rows else None,
                       'scope': 'End-to-end local evaluation execution; interpret with execution_modes and cache status.'},
        'usage_totals': {key: sum(row.get('usage', {}).get(key, 0) for row in rows)
                         for key in ['input_tokens', 'output_tokens', 'total_tokens']},
        'limitations': ['A valid source quote does not prove that a model claim follows from it.',
            'Gold-source recovery and literal token probes are not semantic answer correctness.',
            'No automatic LLM judge was used; review templates remain unfilled.',
            'Previously seen holdout data is rejected rather than reused for Phase 4 tuning.',
            'Mock/disabled/cached/unverified execution must not be presented as live model quality.'],
        'case_results': rows}
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'dev_answer_evaluation.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        review = [row['human_review'] | {'case_id': row['case_id']}
                  for row in rows if 'human_review' in row]
        (destination / 'human_review.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in review), encoding='utf-8')
    return report
