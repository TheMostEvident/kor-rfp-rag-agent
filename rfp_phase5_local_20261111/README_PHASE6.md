# Phase 6 — 후속 질문·대화 맥락 처리

목표는 올바른 문서를 유지하고, 사업을 바꾸면 이전 문서의 내용이 섞이지 않는 대화입니다.
Phase 5 원문·청크·인덱스와 기존 가상환경을 사용합니다. 새 라이브러리 설치나 인덱스 재구축은 없습니다.

## 1. 먼저 API 없이 실행

프로젝트 최상위에서 pull한 뒤 기존 `.venv` 커널로 다음 노트북을 실행합니다.

```text
rfp_phase5_local_20261111/notebooks/13_younggwan_phase6_conversation.ipynb
```

첫 Run All의 `RUN_LIVE_QA=False`는 실제 답변을 생성하지 않습니다.
문서 목록과 검색 근거, 독립 검색질문, 모호한 지칭의 확인 질문, 문서 전환을 점검합니다.
원문 후보가 출력되는 상태와 실제 생성 답변이 있는 상태를 구별해 표시합니다.

CLI에서도 사용할 수 있습니다.

```bash
python rfp_phase5_local_20261111/scripts/phase6_chat.py --list
python rfp_phase5_local_20261111/scripts/phase6_chat.py --keyword 검색어 --list

python rfp_phase5_local_20261111/scripts/phase6_chat.py \
  --doc-id 선택한문서ID \
  --question "사업예산과 부가세 조건은?" \
  --question "그건 몇 개월이야? 안정화도 포함돼?"
```

## 2. 대화 중 문서·필터 선택

```python
chat.list_documents("검색어", include_aliases=True)  # 모든 원본/대표 ID
chat.select_documents([doc_id])
result = chat.ask("예산은 얼마야?")
result = chat.ask("그건 몇 개월짜리야?")

chat.select_documents([first_id, second_id])
chat.ask("두 사업의 예산을 비교해 줘")
chat.ask("그건 몇 개월이야?")  # 어느 사업인지 확인 요청
chat.ask("1번")                 # 후보 첫 문서로 원래 질문을 이어 처리
chat.ask("두 번째 사업의 사업기간은?")

chat.set_filters({"agency": agency_name, "document_year": year})
chat.set_filters({})  # 필터 해제
chat.reset()         # 문서·필터·지칭·이력 초기화
```

Phase 5 요약에서 이어 질문하려면 `chat.select_documents([summary_result['doc_id']])`로
요약 대상 ID를 선택합니다. 이전 요약문을 다시 사실 근거로 넣지 않고 같은 문서의 원문을 검색합니다.

- 전체 목록의 원본 ID와 사본 ID를 모두 받으며, 같은 원문은 대표 문서로 연결합니다.
- 문서 ID는 숫자형 순번이 아닌 실제 ID입니다. 확인 질문이 나온 뒤의 `1번` 등만 후보 순번으로 해석합니다.
- 현재 선택이 없으면 전체 문서를 자동 검색하지 않고 대상을 확인합니다. 전체 선택은 `chat.select_documents('all')`로 명시합니다.
- 여러 문서에서 단수 지칭이 모호하면 확인합니다. `첫 번째/두 번째`, `앞 사업/뒤 사업`, `둘 다/각각`을 구별합니다.
- 문서 또는 필터가 바뀌면 이전 범위의 질문 이력을 지웁니다. 지원 필터는 기관·문서연도·형식입니다.
- 독립 검색질문은 현재 사용자 질문, 대상 사업명, 필요한 경우 같은 범위의 직전 **원래 사용자 질문**으로 만듭니다.
- 이전 assistant 답변·숫자·인용을 다음 검색질문이나 사실 근거로 사용하지 않습니다. 매 턴 원문을 새로 검색합니다.
- 문서 본문의 다른 사업명이나 명령문은 선택 상태를 바꾸지 못합니다. 원문 범위 밖의 검색 결과는 생성 전에 거절합니다.

## 3. 실제 OpenAI 답변

노트북의 `LIVE_APPROVED_DOC_IDS`가 전송할 문서인지 확인한 뒤 `RUN_LIVE_QA=True`로 실행합니다.
`.env`는 기존처럼 프로젝트 또는 상위 폴더에서 읽습니다. 키를 노트북 코드에 붙여넣지 않습니다.

CLI는 다음처럼 `--live`를 명시합니다.

```bash
python rfp_phase5_local_20261111/scripts/phase6_chat.py \
  --live --doc-id 선택한공개문서ID \
  --question "사업예산과 부가세 포함 여부는?" \
  --question "그건 몇 개월짜리야?" \
  --question "그 안에 안정화도 포함돼?"
```

CLI의 기본 API 허용 목록은 `--doc-id`입니다. 나중에 전환할 다른 문서는 `--approve-doc-id`로 명시적으로 추가합니다.
허용하지 않은 문서로 전환하면 키나 SDK를 호출하지 않고 `not_approved`로 중단합니다.
생성 클라이언트를 대화 내에서 공유하므로 여러 턴과 문서 전환의 사용량이 누적됩니다.
프로세스를 새로 시작하면 사용량 상한은 새 실행에 적용됩니다. 통화 단위 비용 상한은 아닙니다.
사용량과 실제 실행 여부를 확인할 수 있습니다.

```python
chat.llm.usage_report()
```

## 4. 사용자 질문 상태 저장

```python
chat.save(ROOT / "phase6_outputs/conversation_state.json")
chat.restore(ROOT / "phase6_outputs/conversation_state.json")
```

사용자 질문·문서 ID·필터·지칭 상태만 저장합니다. 원문·assistant 답변·키·클라이언트는 저장하지 않습니다.
원문 SHA와 ID를 포함한 목록 지문이 달라지면 상태를 복원하지 않습니다.
`phase6_outputs/`는 Git에서 제외합니다. 공개 문서 여부와 관계없이 대화 이력을 자동 공개하지 않습니다.
CLI의 `--state-file`도 같은 형식이며 `--fresh`는 저장된 이력을 읽지 않고 새로 시작합니다.

## 5. 검증과 완료 기준

```bash
cd rfp_phase5_local_20261111
python -m unittest rfp_phase6.test_session -v
```

회귀 테스트는 독립적으로 만든 합성 문서와 대역 응답을 사용합니다.
실제 모델 답변의 의미적 정답 검증으로 표시하지 않습니다.

| 시나리오 | 기대 동작 |
| --- | --- |
| 같은 문서의 예산 → 기간 → 안정화 | 같은 문서 유지, 매 턴 원문 검색 |
| 문서 A → B 전환 후 후속 질문 | A의 질문·숫자·근거 혼입 없음 |
| 두 문서 비교 후 “그건” | 대상 확인, 확인 전 API 호출 없음 |
| 첫째/둘째 선택 또는 후보 번호 응답 | 해당 문서로 원래 질문 해소 |
| 기관/연도 필터 유지와 충돌 | 필터 유지, 충돌 시 확인 |
| 잘못된 ID·범위 밖 결과·잘못된 인용 | 전체 검색이나 무근거 답변으로 대체하지 않음 |
| 상태 복원 또는 새 대화 | 원문 버전 일치 확인, 세션 간 이력 격리 |

로컬 기본 실행과 합성 테스트가 통과하면 대화 제어·검색 경로를 확인한 것입니다.
실제 생성까지 검증할 때는 선택한 공개 문서에서 연속 질문을 실행하고 문서 유지·인용·숫자·조건을 원문과 대조합니다.
기존 Phase 5 전체 문서 요약과 인덱스를 재생성할 필요는 없습니다.

Phase 7은 품질·속도·비용 평가, Phase 8은 운영·파일럿 단계입니다.
