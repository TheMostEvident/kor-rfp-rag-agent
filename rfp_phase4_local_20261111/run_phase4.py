"""Local CLI companion to notebook 05. API calls require --enable-openai."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from rfp_phase4.runtime import bootstrap, source_allowlist


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    plan = sub.add_parser('plan-summary')
    plan.add_argument('--doc-id', required=True)
    ask = sub.add_parser('ask')
    ask.add_argument('question')
    ask.add_argument('--doc-id', action='append', required=True)
    ask.add_argument('--comparison', action='store_true')
    ask.add_argument('--enable-openai', action='store_true')
    args = parser.parse_args()
    env = bootstrap(args.root)
    index = env['index']
    try:
        if args.command == 'inspect':
            print(json.dumps({'environment': env['environment'],
                              'documents': len(index.get_catalog()), 'chunks': len(env['chunks']),
                              'openai_default_enabled': env['config']['generation']['enabled'],
                              'live_api_measured_in_this_package': False}, ensure_ascii=False, indent=2))
            return
        from rfp_phase4.llm import LLMConfig, OpenAIGroundedLLM
        requested = args.doc_id if isinstance(args.doc_id, list) else [args.doc_id]
        allowed = source_allowlist(index, requested)
        config = LLMConfig(**env['config']['generation'])
        config = replace(config, enabled=bool(getattr(args, 'enable_openai', False)))
        llm = OpenAIGroundedLLM(env['root'] / 'phase4_outputs/answer_cache',
                               config=config, allowed_sources=allowed)
        if args.command == 'plan-summary':
            from rfp_phase4.summarization import FullDocumentSummarizer
            summarizer = FullDocumentSummarizer(index, llm, env['registry'], env['config']['summary'])
            summary_plan = summarizer.plan(args.doc_id)
            summary_plan['batches'] = [{key: value for key, value in b.items() if key != 'chunks'}
                                       for b in summary_plan['batches']]
            print(json.dumps(summary_plan, ensure_ascii=False, indent=2))
        else:
            from rfp_phase4.rag import AnswerSession
            session = AnswerSession(index, llm, env['registry'],
                config=env['config']['retrieval'], active_doc_ids=requested)
            result = session.ask(args.question, feature='comparison' if args.comparison else 'qa')
            print(result['markdown'])
            print(json.dumps({'status': result['status'], 'execution_status': result['execution_status'],
                              'usage': llm.usage_report()}, ensure_ascii=False, indent=2))
    finally:
        index.close()


if __name__ == '__main__':
    main()
