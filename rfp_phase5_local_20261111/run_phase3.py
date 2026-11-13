"""Reproducible local chunking/retrieval experiments. No API or network calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from local_rag.chunking import build_chunks
from local_rag.evaluation import audit_chunk_coverage, evaluate_retrieval, evaluate_summary_coverage
from local_rag.io import (load_documents, load_chunks, read_jsonl, write_json, write_jsonl,
                          sha256_file, configuration_hash, verify_saved_run)
from local_rag.quality import validate_chunks
from local_rag.retrieval import LocalSearchIndex


ROOT = Path(__file__).resolve().parent


def _now():
    return datetime.now(timezone.utc).isoformat()


def _config(root):
    return json.loads((Path(root) / 'configs/default.json').read_text(encoding='utf-8'))


def refresh_manifest(root):
    root = Path(root).resolve()
    files = ['data/source_documents.jsonl', 'data/documents_manifest.json',
             'data/source_evidence_validation.json', 'data/chunks.jsonl',
             'configs/default.json', 'eval/questions.dev.jsonl', 'eval/questions.holdout.jsonl',
             'reports/evidence_chunk_alignment.json', 'reports/chunk_integrity.json',
             'reports/chunk_audit.json', 'reports/index_build.json']
    files.extend(str(p.relative_to(root)) for p in sorted((root / 'index').glob('*')) if p.is_file())
    code = ['run_phase3.py'] + [str(p.relative_to(root)) for p in sorted((root / 'local_rag').glob('*.py'))]
    saved = {'created_at_utc': _now(), 'python': platform.python_version(),
             'files_sha256': {p: sha256_file(root / p) for p in files},
             'code_sha256': {p: sha256_file(root / p) for p in code},
             'api_calls': 0, 'network_calls': 0}
    write_json(root / 'reports/run_manifest.json', saved)
    return saved


def build_pipeline(root=ROOT, force=False):
    """Build only chunks/index. Never parses/evaluates dev/holdout questions or invokes an LLM."""
    root = Path(root).resolve()
    if not force and (root / 'reports/run_manifest.json').exists():
        verify_saved_run(root)
        return json.loads((root / 'reports/phase3_summary.json').read_text(encoding='utf-8'))
    started = time.perf_counter()
    config = _config(root)
    docs = load_documents(root)
    chunks, chunk_audit = build_chunks(docs, config['chunking'])
    quality = validate_chunks(docs, chunks, config['chunking']['chunk_size_bytes'])
    write_json(root / 'reports/chunk_integrity.json', quality)
    if not quality['valid']:
        raise ValueError('Source integrity check failed; see reports/chunk_integrity.json')
    evidence = json.loads((root / 'data/source_evidence_validation.json').read_text(encoding='utf-8'))
    alignment = audit_chunk_coverage(docs, chunks, evidence)
    write_json(root / 'reports/evidence_chunk_alignment.json', alignment)
    if not alignment['valid']:
        raise ValueError('Evidence alignment failed; see reports/evidence_chunk_alignment.json')
    write_jsonl(root / 'data/chunks.jsonl', chunks)
    write_json(root / 'reports/chunk_audit.json', chunk_audit)
    with LocalSearchIndex(root / 'index') as index:
        index_build = index.build(chunks, config['retrieval'])
    write_json(root / 'reports/index_build.json', index_build)
    # A rebuild invalidates measurements from another configuration.
    for path in (root / 'reports').glob('*evaluation*.json'):
        path.unlink()
    for name in ['frozen_configuration.json', 'selection.json']:
        (root / 'reports' / name).unlink(missing_ok=True)
    summary = {
        'phase3_scope': 'chunking_and_retrieval_baseline', 'built_at_utc': _now(),
        'source_files': quality['source_files'], 'unique_documents': quality['unique_documents'],
        'chunks': len(chunks), 'chunk_config': config['chunking'], 'chunk_audit': chunk_audit,
        'source_character_coverage': quality['source_character_coverage'],
        'evidence_alignment': {k: alignment[k] for k in ['total_links', 'covered_links', 'uncovered_links',
                               'source_ambiguous_links', 'chunk_source_issue_count', 'valid']},
        'local_methods': ['bm25', 'char_tfidf', 'rrf', 'mmr'],
        'semantic_embedding_measured': False, 'openai_live_measured': False, 'openai_api_calls': 0,
        'answer_generation_measured': False, 'five_feature_task_accuracy_measured': False,
        'dev_evaluation_summary': None, 'holdout_evaluation_summary': None,
        'build_seconds': time.perf_counter() - started,
    }
    write_json(root / 'reports/phase3_summary.json', summary)
    refresh_manifest(root)
    return summary


def _compact_report(report):
    return {key: report[key] for key in ['method', 'scope_protocol', 'case_count', 'excluded_case_count',
                                        'metrics_at_k', 'mrr_first_source_overlap', 'latency_ms', 'scope']}


def evaluate_split(root=ROOT, split='dev', *, allow_holdout=False):
    root = Path(root).resolve()
    if split not in {'dev', 'holdout'}:
        raise ValueError('split must be dev or holdout')
    if split == 'dev' and (root / 'reports/frozen_configuration.json').exists():
        raise ValueError('This experiment is frozen. Start a separate dev experiment before changing measurements.')
    if split == 'holdout':
        if not allow_holdout or not (root / 'reports/frozen_configuration.json').is_file():
            raise ValueError('Freeze the configuration and explicitly allow holdout first.')
        if (root / 'reports/holdout_evaluation.json').exists():
            raise ValueError('This frozen holdout run already exists. Do not silently rerun/tune it.')
    verify_saved_run(root)
    config = _config(root)
    if split == 'holdout':
        frozen = json.loads((root / 'reports/frozen_configuration.json').read_text(encoding='utf-8'))
        checks = {'configuration_sha256': configuration_hash(config),
                  'run_manifest_sha256': sha256_file(root / 'reports/run_manifest.json'),
                  'dev_report_sha256': sha256_file(root / 'reports/dev_evaluation.json'),
                  'selection_sha256': sha256_file(root / 'reports/selection.json'),
                  'holdout_questions_sha256': sha256_file(root / 'eval/questions.holdout.jsonl')}
        if any(frozen.get(key) != actual for key, actual in checks.items()):
            raise ValueError('Frozen configuration, inputs, code, or development results changed.')
    cases = read_jsonl(root / f'eval/questions.{split}.jsonl')
    if any(case.get('split') != split for case in cases):
        raise ValueError('Question split metadata does not match the requested evaluation split.')
    alignment = json.loads((root / 'reports/evidence_chunk_alignment.json').read_text(encoding='utf-8'))
    reports = []
    with LocalSearchIndex(root / 'index') as index:
        index.get_metadata()  # Exclude disk load/checksum time from per-query measurements.
        for scope in config['evaluation']['scope_modes']:
            for method in config['evaluation']['methods']:
                report = evaluate_retrieval(index, cases, alignment,
                    top_ks=config['evaluation']['top_ks'], method=method, scope_mode=scope,
                    search_config=config['search'], allow_holdout=allow_holdout)
                reports.append(report)
        summary_route = evaluate_summary_coverage(index, cases, alignment, allow_holdout=allow_holdout)
    saved = {'split': split, 'created_at_utc': _now(), 'configuration_sha256': configuration_hash(config),
             'retrieval_reports': reports, 'summary_input_coverage': summary_route,
             'holdout_policy': 'Frozen configuration; no holdout questions or outcomes used for tuning.',
             'limitations': ['Character TF-IDF is lexical, not a semantic embedding model.',
                 'Positive fact/comparison retrieval only; no generated answer or abstention accuracy.',
                 'Gold quotations verify evidence recovery, not whole-document semantic coverage.',
                 'All 99 documents are indexed; queries/labels split by source content for dev/holdout.',
                 'Integrity alignment checks all source evidence before retrieval; not an external benchmark.'],
             'api_calls': 0}
    write_json(root / f'reports/{split}_evaluation.json', saved)
    compact = [_compact_report(report) for report in reports]
    summary = json.loads((root / 'reports/phase3_summary.json').read_text(encoding='utf-8'))
    summary[f'{split}_evaluation_summary'] = compact
    write_json(root / 'reports/phase3_summary.json', summary)
    return saved


def select_on_dev(root=ROOT):
    """Predeclare the default method using dev top-5 context evidence recall."""
    root = Path(root).resolve()
    if (root / 'reports/frozen_configuration.json').exists() or (root / 'reports/holdout_evaluation.json').exists():
        raise ValueError('Cannot reselect a method after freezing or measuring holdout.')
    verify_saved_run(root)
    path = root / 'reports/dev_evaluation.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('split') != 'dev':
        raise ValueError('Method selection requires a development report, never holdout results.')
    config = _config(root)
    if result.get('configuration_sha256') != configuration_hash(config):
        raise ValueError('Development report belongs to a different configuration. Evaluate dev first.')
    eligible = [r for r in result['retrieval_reports'] if r['scope_protocol'] == 'context']
    # Fixed tie order favours the simpler fusion before optional diversification.
    tie_order = {'rrf': 3, 'bm25': 2, 'char_tfidf': 1, 'mmr': 0}
    best = max(eligible, key=lambda r: (r['metrics_at_k']['5']['micro_evidence_span_recall'],
               r['metrics_at_k']['10']['micro_evidence_span_recall'], tie_order[r['method']]))
    config['search']['method'] = best['method']
    write_json(root / 'configs/default.json', config)
    selection = {'selected_at_utc': _now(), 'selected_default_method': best['method'],
                 'criterion': 'Dev context micro evidence-span recall@5, then @10, then fixed tie order.',
                 'dev_metrics': best['metrics_at_k'], 'holdout_outcomes_used': False,
                 'other_hyperparameters_tuned': False}
    write_json(root / 'reports/selection.json', selection)
    refresh_manifest(root)
    return selection


def freeze_and_evaluate_holdout(root=ROOT):
    root = Path(root).resolve()
    if (root / 'reports/holdout_evaluation.json').exists():
        raise ValueError('Holdout already measured. Inspect saved report; do not tune on it.')
    if (root / 'reports/frozen_configuration.json').exists():
        raise ValueError('A frozen experiment already exists. Resume it without overwriting its freeze record.')
    verify_saved_run(root)
    if not (root / 'reports/selection.json').is_file():
        raise ValueError('Select a configuration on dev first.')
    config = _config(root)
    # Written before the first holdout questions are read by this pipeline.
    frozen = {'frozen_at_utc': _now(), 'configuration': config,
              'configuration_sha256': configuration_hash(config),
              'run_manifest_sha256': sha256_file(root / 'reports/run_manifest.json'),
              'dev_report_sha256': sha256_file(root / 'reports/dev_evaluation.json'),
              'selection_sha256': sha256_file(root / 'reports/selection.json'),
              'holdout_questions_sha256': sha256_file(root / 'eval/questions.holdout.jsonl'),
              'planned_comparisons': {'scope_modes': config['evaluation']['scope_modes'],
                 'methods': config['evaluation']['methods'], 'top_ks': config['evaluation']['top_ks']},
              'holdout_question_outcomes_previously_used_for_tuning': False}
    write_json(root / 'reports/frozen_configuration.json', frozen)
    return evaluate_split(root, 'holdout', allow_holdout=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build', action='store_true')
    group.add_argument('--evaluate-dev', action='store_true')
    group.add_argument('--select-on-dev', action='store_true')
    group.add_argument('--freeze-and-evaluate-holdout', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.build:
        result = build_pipeline(ROOT, force=args.force)
        print(json.dumps({k: result[k] for k in ['source_files','unique_documents','chunks','evidence_alignment','build_seconds']}, ensure_ascii=False))
    elif args.evaluate_dev:
        result = evaluate_split(ROOT, 'dev')
        print(json.dumps([{'method': r['method'], 'scope': r['scope_protocol'], 'cases': r['case_count'],
            'recall@5': r['metrics_at_k']['5']['micro_evidence_span_recall'],
            'recall@10': r['metrics_at_k']['10']['micro_evidence_span_recall'],
            'recall@20': r['metrics_at_k']['20']['micro_evidence_span_recall']} for r in result['retrieval_reports']], ensure_ascii=False))
    elif args.select_on_dev:
        print(json.dumps(select_on_dev(ROOT), ensure_ascii=False))
    else:
        result = freeze_and_evaluate_holdout(ROOT)
        print(json.dumps([{'method': r['method'], 'scope': r['scope_protocol'], 'cases': r['case_count'],
            'recall@5': r['metrics_at_k']['5']['micro_evidence_span_recall'],
            'recall@10': r['metrics_at_k']['10']['micro_evidence_span_recall'],
            'recall@20': r['metrics_at_k']['20']['micro_evidence_span_recall']} for r in result['retrieval_reports']], ensure_ascii=False))


if __name__ == '__main__':
    main()
