"""Local notebook wiring and isolated run records; no implicit model calls."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from rfp_phase4.runtime import bootstrap, source_allowlist


def bootstrap_phase5(root):
    root = Path(root).expanduser().resolve()
    config_path = root / 'configs/phase5.json'
    if not config_path.is_file():
        raise FileNotFoundError('Phase 5 ZIP을 압축 해제한 폴더를 ROOT로 선택하세요.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    env = bootstrap(root)
    env['phase5_config'] = config
    return env


def make_components(env, *, doc_ids, enabled=False, client=None, cache_dir=None):
    """Construct clients without SDK import/network. Only explicit documents allowed.

    Existing Phase 4 exact-request cache can be reused on an in-place local
    upgrade. Phase 5 field requests use their own task/schema and cache key.
    Injected clients are offline test doubles, as enforced by Phase 4.
    """
    if not isinstance(enabled, bool):
        raise TypeError('enabled에는 True 또는 False를 입력하세요.')
    from rfp_phase4.llm import LLMConfig, OpenAIGroundedLLM
    from rfp_phase4.rag import AnswerSession
    from .summary import FieldSummarizer
    config = env['phase5_config']
    approved = source_allowlist(env['index'], doc_ids)
    generation = replace(LLMConfig(**config['generation']), enabled=enabled)
    cache = Path(cache_dir) if cache_dir is not None else env['root'] / 'phase4_outputs/generation_cache'
    llm = OpenAIGroundedLLM(cache, config=generation, allowed_sources=approved, client=client)
    qa = AnswerSession(env['index'], llm, env['registry'],
                       config=config['retrieval'], active_doc_ids=list(approved))
    summarizer = FieldSummarizer(env['index'], llm, env['registry'], config=config['summary'])
    return {'llm': llm, 'qa': qa, 'summarizer': summarizer,
            'approved_sources': approved, 'effective_enabled': generation.enabled}


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def _cost_usage(usage):
    # Summary invocation counters use input_tokens; the client uses reported_*.
    converted = dict(usage or {})
    for key in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens'):
        if 'reported_' + key not in converted and key in converted:
            converted['reported_' + key] = converted[key]
    return converted


def save_run(root, result, *, operation, usage=None, pricing=None):
    """Save one result, diagnostics and a blank human review; never overwrite a run.

    ``usage`` is an optional cumulative session snapshot, kept separately from
    this result's invocation usage. Cost estimates here use invocation usage.
    No credentials or process environment are collected.
    """
    from .quality import audit_result
    from .review import build_review
    from rfp_phase4.costs import estimate_usage_cost
    if operation not in {'qa', 'fact', 'comparison', 'followup', 'uncertainty', 'summary'}:
        raise ValueError('지원하지 않는 실행 종류입니다.')
    if not isinstance(result, dict):
        raise TypeError('result는 실행 결과 dict여야 합니다.')
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8]
    quality = audit_result(result)
    review = build_review(result, case_id=run_id)
    invocation = _cost_usage(result.get('usage', {}))
    # Some Q&A versions keep usage under generation; never substitute cumulative
    # session totals for an individual answer's billed-token estimate.
    if not invocation and isinstance(result.get('generation'), dict):
        invocation = _cost_usage(result['generation'].get('usage', {}))
    record = {'run_id': run_id, 'created_at_utc': now.isoformat(),
              'operation': operation, 'status': result.get('status'),
              'execution_status': result.get('execution_status'),
              'answer_generated': bool(result.get('answer_generated')),
              'invocation_usage': invocation, 'session_usage_snapshot': usage,
              'human_review_status': 'pending', 'semantic_accuracy': None,
              'quality_is_semantic_verdict': False}
    if pricing is not None:
        record['cost_estimate'] = estimate_usage_cost(invocation, pricing)
        record['cost_estimate_scope'] = 'This result invocation only; no currency spending cap is enforced.'
    # Serialize every artifact before creating the destination. Invalid data
    # therefore fails without leaving a half-created apparent successful run.
    files = {'result.json': _json(result), 'quality.json': _json(quality),
             'human_review.json': _json(review), 'run.json': _json(record),
             'answer.md': str(result.get('markdown') or '답변이 생성되지 않았습니다.') + '\n'}
    folder = Path(root).expanduser().resolve() / 'phase5_outputs/runs' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (folder / name).write_text(content, encoding='utf-8')
    return folder
