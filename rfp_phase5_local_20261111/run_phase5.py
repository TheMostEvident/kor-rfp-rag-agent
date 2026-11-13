"""Local Phase 5 companion: inspect and plan are free of API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfp_phase5.runtime import bootstrap_phase5, make_components, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    for command in ('plan-summary', 'summary', 'ask'):
        p = sub.add_parser(command)
        p.add_argument('--doc-id', action='append', required=True)
        if command != 'plan-summary':
            p.add_argument('--enable-openai', action='store_true')
        if command == 'ask':
            p.add_argument('question')
    args = parser.parse_args()
    env = bootstrap_phase5(args.root)
    try:
        if args.command == 'inspect':
            print(json.dumps({'environment': env['environment'],
                              'canonical_documents': len(env['index'].get_catalog()),
                              'chunks': len(env['chunks']), 'phase': 5,
                              'live_api_calls': 0, 'semantic_quality_verified': False},
                             ensure_ascii=False, indent=2))
            return
        if args.command in {'summary', 'plan-summary'} and len(args.doc_id) != 1:
            parser.error('전체 문서 요약은 --doc-id 하나를 지정하세요.')
        components = make_components(env, doc_ids=args.doc_id,
                                     enabled=bool(getattr(args, 'enable_openai', False)))
        if args.command == 'plan-summary':
            plan = components['summarizer'].plan(args.doc_id[0])
            plan['batches'] = [{key: value for key, value in b.items() if key != 'chunks'}
                               for b in plan['batches']]
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        result = (components['summarizer'].summarize(args.doc_id[0]) if args.command == 'summary'
                  else components['qa'].ask(args.question))
        print(result.get('markdown', ''))
        folder = save_run(env['root'], result,
                          operation='summary' if args.command == 'summary' else 'qa',
                          usage=components['llm'].usage_report(),
                          pricing=env['phase5_config']['pricing'])
        print(json.dumps({'status': result.get('status'),
                          'execution_status': result.get('execution_status'),
                          'run_directory': str(folder),
                          'usage': components['llm'].usage_report()}, ensure_ascii=False, indent=2))
    finally:
        env['index'].close()


if __name__ == '__main__':
    main()
