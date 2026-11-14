"""Content queries for an already selected document scope; never add facts.

The generation question retains the business title and user context. Lexical
search uses the question's subject instead: repeating a title otherwise ranks
covers and blank proposal forms above the paragraph answering the question.
"""
from __future__ import annotations

import re
import unicodedata

from local_rag.retrieval import normalize_retrieval_text
from .resolution import title_variants


QUERY_VERSION = 'phase6-content-query-v3'
_TOPICS = (
    ('duration', r'기간|개월|며칠|몇\s*(?:달|일|주)|안정화|언제까지',
     '사업기간 과업기간 수행기간 용역기간 계약기간 개월'),
    ('budget', r'예산|금액|용역비|사업비|부가세|부가가치세|VAT|세금|얼마',
     '사업예산 사업비 용역비 소요예산 부가세 부가가치세'),
    ('eligibility', r'자격|공동수급|공동도급|하도급|참가조건',
     '참가자격 입찰참가자격 공동수급 공동도급 하도급'),
    ('submission', r'제출|마감|접수|서류|방문|우편',
     '제출기한 제출마감 제출서류 제출방법 제출방식 접수'),
    ('evaluation', r'평가|배점|낙찰|협상|점수',
     '평가기준 평가항목 배점 낙찰자 선정방식'),
    ('requirements', r'요구사항|필수|과업(?!기간)|지원|기능|GPU',
     '주요 과업 필수 요구사항 수행범위'),
)


def _without_titles(text, titles, agencies=()):
    """Remove selected project names only, including their spaced variants."""
    text = unicodedata.normalize('NFC', text)
    names = {v for title in titles for v in title_variants(title)}
    names.update(''.join(c for c in str(a).casefold() if c.isalnum())
                 for a in agencies if a and len(''.join(c for c in str(a) if c.isalnum()))>=3)
    names.discard('')
    for name in sorted(names, key=len, reverse=True):
        pattern = r'[\W_]*'.join(re.escape(c) for c in name)
        text = re.sub(pattern, ' ', text, flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()


def query_topics(text):
    return [name for name,pattern,_ in _TOPICS if re.search(pattern,text,re.I)]


def plan_content_queries(question, titles, *, previous_question=None, agencies=()):
    """Return bounded search variants made only from user text and synonyms.

    Previous text is supplied only after the conversation planner has checked
    document and topic continuity. It is an original user question, never an
    assistant answer or a recursively rewritten question. No duration, amount,
    answer label or source-derived fact is inserted by these expansions.
    """
    current = _without_titles(question, titles, agencies)
    previous = _without_titles(previous_question, titles, agencies) if previous_question else ''
    content = ' '.join(x for x in (current, previous) if x)
    topics = [(name, terms) for name, pattern, terms in _TOPICS
              if re.search(pattern, content, flags=re.I)]
    if re.search(r'안정화|모니터링|시범운영', current):
        topics = [(name, terms + ' 안정화 시스템모니터링 시범운영'
                   if name == 'duration' else terms) for name, terms in topics]
    if len(topics) == 1:
        # Keep literal user wording as well: expansion alone can overemphasize
        # the many generic "contract period" forms in a long RFP.
        queries = [content, topics[0][1] + ' ' + content]
    elif topics:
        # Give each requested subject a place before the general query. This
        # prevents a budget-heavy document from crowding out its period clause.
        queries = [terms for _, terms in topics] + [content]
    else:
        queries = [content] if content else []
    queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    return {'version': QUERY_VERSION, 'content_question': content,
            'queries': queries, 'topics': [name for name, _ in topics],
            'project_title_used_as_scope_only': True,
            'facts_added_by_rewrite': False}


_PERIOD_LABEL = re.compile(r'(?:사업|과업|용역|계약|수행)(?:수행)?\s*기간')
_TIME_VALUE = re.compile(r'\d+(?:\.\d+)?\s*(?:개\s*월|달|일|년|주)|'
                         r'(?:19|20)?\d{2}\s*[./-]\s*\d{1,2}')
_STABILITY = re.compile(r'안정화|모니터링|시범운영')
_BUDGET_LABEL = re.compile(r'(?:사업|과업|용역|개발|구입|구매)\s*(?:예산|금액|비)|'
                           r'소요\s*예산|총\s*예산|추정\s*가격')
_MONEY_VALUE = re.compile(r'\d[\d,.]*\s*(?:조|억|천만|백만|만|천)?\s*원|'
                          r'\d[\d,.]*\s*(?:조|억|백만|천만)')


def prioritize_content_candidates(hits, *, topics, content_question):
    """Prefer populated budget/period clauses over headings and empty forms.

    This is a ranking hint on retrieved source text, never a claim extraction,
    an answer or a semantic validity verdict. Lower-priority candidates are
    retained until the normal top_k and context-byte limits are applied.
    """
    if not {'duration', 'budget'}.intersection(topics):
        return hits
    stability_requested = bool(_STABILITY.search(content_question))

    def priority(chunk):
        text = normalize_retrieval_text(chunk['text'])
        concrete_period = 'duration' in topics and any(
            _TIME_VALUE.search(text[m.end():m.end() + 160]) for m in _PERIOD_LABEL.finditer(text))
        concrete_budget = 'budget' in topics and any(
            _MONEY_VALUE.search(text[m.end():m.end() + 140]) for m in _BUDGET_LABEL.finditer(text))
        relevant_condition = 'duration' in topics and stability_requested and bool(_STABILITY.search(text))
        return relevant_condition, int(concrete_period) + int(concrete_budget)

    return sorted(hits, key=priority, reverse=True)
