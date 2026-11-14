"""Wire conversation plans to the existing source-validated QA implementation."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from rfp_phase4.llm import LLMConfig, OpenAIGroundedLLM
from rfp_phase4.rag import AnswerSession
from .session import ConversationSession, ConversationConfig


class ScopedIndex:
    """Enforce a single turn's canonical scope even if a lower layer changes it."""
    def __init__(self,index,registry,doc_ids):
        self.index=index;self.registry=registry
        self.ids=registry.resolve_doc_ids(doc_ids)
        if not self.ids:raise ValueError('범위 없는 검색은 허용하지 않습니다.')

    def resolve_doc_ids(self,ids):
        chosen=self.registry.resolve_doc_ids(ids)
        if not chosen or not set(chosen).issubset(self.ids):raise ValueError('선택 문서 밖으로 범위를 넓힐 수 없습니다.')
        return chosen

    def get_catalog(self):return [copy.deepcopy(r) for r in self.index.get_catalog() if r['doc_id'] in self.ids]

    def search(self,query,**kwargs):
        requested=kwargs.pop('allowed_doc_ids',None)
        selected=self.ids if requested is None else self.resolve_doc_ids(requested)
        hits=self.index.search(query,allowed_doc_ids=selected,**kwargs)
        self.registry.validate_chunks(hits,allowed_doc_ids=selected)
        return hits


class _FixedSelection:
    """Disable lower-layer title selection/rewrite after Phase6 resolves it."""
    def __init__(self,ids,query):self.active_doc_ids=list(ids);self.query_history=[];self.query=query

    def select_documents(self,ids):
        if list(ids)!=self.active_doc_ids:raise ValueError('턴 중 문서 선택 변경을 거절했습니다.')
        return list(self.active_doc_ids)

    def ask(self,query,**kwargs):
        return {'status':'retrieved','standalone_query':self.query,'query':self.query,
            'active_doc_ids':list(self.active_doc_ids),'hits':[],'query_history':[],
            'selection_reason':'phase6_fixed_scope','scope_preflight_only':True}


class GroundedBackend:
    def __init__(self,env,llm,retrieval=None):
        self.env=env;self.llm=llm
        self.retrieval=dict(retrieval or {});self.retrieval.setdefault('method','rrf')

    def __call__(self,query,*,doc_ids,feature):
        if len(query)>self.retrieval.get('max_question_chars',4000):
            raise ValueError('검색질문이 설정 상한을 넘었습니다.')
        index=ScopedIndex(self.env['index'],self.env['registry'],doc_ids)
        answer=AnswerSession(index,self.llm,self.env['registry'],config=self.retrieval,
            active_doc_ids=index.ids)
        answer.retrieval_session=_FixedSelection(index.ids,query)
        # Intent was resolved from the current user turn, excluding titles.
        # The fixed retrieval session still supplies the full original query
        # for search and generation. Do not re-infer comparison from a title.
        result=answer.ask('비교 요청' if feature=='comparison' else '문서 질문',feature=feature)
        if result.get('execution_status')=='disabled':
            result['status']='retrieved' if result.get('source_chunks') else 'insufficient_evidence'
            result['markdown']='선택한 문서에서 원문 근거 후보를 찾았습니다. 실제 답변 생성은 꺼져 있습니다.'
        return result


def make_conversation(env,*,enabled=False,approved_doc_ids=None,client=None,
                      config=None,retrieval=None,generation=None):
    """No SDK/key/network use until an explicitly enabled turn calls generate.

    One LLM instance is retained for the entire conversation, so usage limits
    are cumulative across turns and scope switches rather than reset per call.
    """
    if type(enabled)is not bool:raise TypeError('enabled는 True/False여야 합니다.')
    registry=env['registry']
    approved=registry.resolve_doc_ids(approved_doc_ids) if approved_doc_ids else []
    if enabled and not approved:raise ValueError('API에 전송할 문서 ID를 명시적으로 허용하세요.')
    settings=dict(generation or env.get('phase5_config',{}).get('generation',{}))
    settings.pop('enabled',None)
    llm_config=replace(LLMConfig(**settings),enabled=enabled)
    allowed={did:registry.source_hashes[did] for did in approved}
    llm=OpenAIGroundedLLM(Path(env['root'])/'phase6_outputs/generation_cache',
        config=llm_config,allowed_sources=allowed,client=client)
    backend=GroundedBackend(env,llm,retrieval=retrieval)

    def dispatch(query,*,doc_ids,feature):
        if enabled and not set(doc_ids).issubset(approved):
            return {'status':'not_generated','execution_status':'not_approved',
                'answer_generated':False,'claims':[],'active_doc_ids':list(doc_ids),
                'markdown':'선택한 문서를 API 허용 목록에 추가한 뒤 실행해 주세요.',
                'source_chunks':[]}
        return backend(query,doc_ids=doc_ids,feature=feature)

    conversation=ConversationSession(env['documents'],backend=dispatch,
        config=ConversationConfig(**config) if isinstance(config,dict) else config)
    conversation.llm=llm  # Usage inspection only; never used for query rewriting.
    return conversation
