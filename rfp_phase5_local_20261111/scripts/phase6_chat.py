"""Local Phase6 conversation; default is source retrieval with zero model calls."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--list',action='store_true',help='모든 원본·대표 ID 목록')
    p.add_argument('--keyword',default='')
    p.add_argument('--doc-id',action='append',default=[],help='처음 선택할 문서; 반복 가능')
    p.add_argument('--question',action='append',default=[],help='순서대로 처리할 질문; 반복 가능')
    p.add_argument('--filters',default=None,help='agency/document_year/detected_format JSON; {}로 해제')
    p.add_argument('--live',action='store_true',help='선택한 문서의 실제 API 답변 생성')
    p.add_argument('--approve-doc-id',action='append',default=[],help='API 전송을 허용할 문서 ID; 기본은 --doc-id')
    p.add_argument('--env-file',type=Path)
    p.add_argument('--state-file',type=Path,help='사용자 질문·문서 상태만 저장/복원; phase6_outputs/ 사용 권장')
    p.add_argument('--fresh',action='store_true',help='저장된 대화 상태를 읽지 않음')
    args=p.parse_args(argv)
    from rfp_phase5.runtime import bootstrap_phase5
    from rfp_phase6.runtime import make_conversation
    root=args.root.expanduser().resolve()
    settings=json.loads((root/'configs/phase6.json').read_text(encoding='utf-8'))
    env=bootstrap_phase5(root)
    try:
        approved=args.approve_doc_id or args.doc_id
        if args.live:
            if not approved:p.error('--live 실행에는 --doc-id 또는 --approve-doc-id가 필요합니다.')
            from rfp_phase5_recovery.runner import load_api_key
            load_api_key(root,args.env_file)
        chat=make_conversation(env,enabled=args.live,approved_doc_ids=approved,
            config=settings['conversation'],retrieval=settings['retrieval'])
        if args.state_file and args.state_file.is_file() and not args.fresh:chat.restore(args.state_file)
        if args.filters is not None:chat.set_filters(json.loads(args.filters))
        if args.doc_id:chat.select_documents(args.doc_id)
        if args.list or not args.question:
            rows=chat.list_documents(args.keyword,filtered=bool(chat.filters),include_aliases=True)
            print(json.dumps(rows,ensure_ascii=False,indent=2))
            print('원본 ID:',len(rows),'/ 전체 대표 문서:',len(chat.list_documents()))
        for q in args.question:
            result=chat.ask(q);route=result.get('conversation') or result
            print('\n사용자:',q)
            print('선택 범위:',route.get('effective_doc_ids',[]))
            if route.get('standalone_query'):print('독립 검색질문:',route['standalone_query'])
            print(result.get('markdown',''))
            if result.get('candidate_documents'):
                print(json.dumps([{'번호':i,'doc_id':d['doc_id'],'사업명':d['project_title']}
                    for i,d in enumerate(result['candidate_documents'],1)],ensure_ascii=False,indent=2))
            if not args.live and result.get('source_chunks'):
                print('원문 근거 후보(최대3개):')
                for chunk in result['source_chunks'][:3]:
                    print(chunk['file_name'],chunk['locator'],'\n',chunk['text'][:550])
            print('상태:',result['status'],'/',result.get('execution_status'))
        if args.state_file:
            chat.save(args.state_file);print('사용자 대화 상태 저장:',args.state_file)
        usage=chat.llm.usage_report()
        print('API 활성화:',chat.llm.config.enabled,'/ 실제 API 호출:',usage['live_api_calls'],
              '/ 캐시:',usage['cache_hits'],'/ 대역:',usage['mock_calls'])
        return 0
    except (ValueError,TypeError,KeyError) as error:
        print('실행 설정 또는 대화 상태를 확인하세요:',type(error).__name__,file=sys.stderr)
        return 2
    finally:env['index'].close()


if __name__=='__main__':raise SystemExit(main())
