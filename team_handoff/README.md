# 팀 공유용 RFP RAG 구현

유영관의 Phase 1–8 작업 중 RAG 실행에 필요한 구현을 `rfp_rag` Python 패키지로 추렸습니다.
실험 순서·패치 이력·중복 노트북·실제 문서·저장 색인·API 실행 로그는 포함하지 않습니다.
기존 생성 모델과 BM25 + 문자 TF-IDF/RRF 검색을 유지했습니다. 모델 비교와 의미 검색 추가는 Task 2입니다.

## 빠른 실행

이 폴더(`team_handoff`)에서 Python 3.11 또는 3.12와 uv를 사용합니다.

```bash
uv sync --extra notebook
uv run python -m unittest discover -s tests -t tests -q
uv run rfp-rag build --demo --workspace workspaces/demo --config configs/rag.yaml
uv run rfp-rag catalog --workspace workspaces/demo
uv run rfp-rag search --workspace workspaces/demo --doc-id DEMO_A --question "사업 예산과 기간은?"
uv run rfp-rag ask --workspace workspaces/demo --doc-id DEMO_A --question "예산은?" --question "그 기간에 안정화도 포함돼?"
uv run rfp-rag summarize --workspace workspaces/demo --doc-id DEMO_A
uv run rfp-rag validate --workspace workspaces/demo --doc-id DEMO_A
```

기본 실행은 실제 API를 호출하지 않습니다. 가상 문서의 질의응답·요약·검증 경로를 확인하며,
생성되지 않은 결과와 부분 처리 상태를 그대로 표시합니다. 가상 예제는 실제 성능 평가 자료가 아닙니다.

실제 답변 생성:

```bash
uv sync --extra notebook --extra openai
cp .env.example .env
# .env 파일에 키를 입력합니다. 키를 노트북 셀이나 Git에 넣지 않습니다.
uv run rfp-rag ask --workspace workspaces/demo --doc-id DEMO_A --question "완료 검수 조건은?" --live --env-file .env
```

`--live`는 지정한 문서의 근거를 API로 전송합니다. 선택 문서 허용 목록과 호출·입력량 상한을 적용합니다.
CLI의 한 번 실행이 한 세션이며, 새 프로세스는 새 호출 예산을 시작합니다.
실제 생성 모델은 workspace 구축 시 사용한 설정에 고정됩니다. 바꾸려면 새 workspace를 만드세요.

## 포함한 노트북과 Python 구현

| 실행 노트북 | 역할 | 실제 구현 |
| --- | --- | --- |
| `notebooks/01_younggwan_build_and_search.ipynb` | 원문 → 청크·색인 → 문서 내 검색 | `workspace.py`, `retrieval/`, `workflows.search` |
| `notebooks/02_younggwan_grounded_conversation.ipynb` | 근거 답변, 비교·후속 질문의 문서 범위 유지 | `generation/`, `conversation/`, `workflows.submit_questions` |
| `notebooks/03_younggwan_document_summary.ipynb` | 문서 전체 11항목 요약과 검토표 | `summary_full/`, `summary_content/`, `workflows.summarize` |
| `notebooks/04_younggwan_pilot_validation.ipynb` | 파일럿 6개 사례와 출처·실행 검사 | `pilot/validation.py`, `workflows.validate` |

노트북은 업무 함수 호출과 결과 확인만 담당합니다. `.py`가 구현의 기준입니다.
노트북 출력과 실행 횟수는 지웠고, 기본 `RUN_LIVE=False`입니다.
02번은 같은 커널에서 동일 배치 ID를 재실행하면 기존 제출을 반환합니다.
04번도 같은 커널·문서·실행 여부·배치 ID에서는 저장된 검증 결과를 다시 조회합니다.
재검증은 배치 ID를 바꿔 명시적으로 시작합니다. 새 커널은 새 세션과 예산을 시작합니다.

## 실제 원문 연결

파싱 담당자가 생성한 **원문 블록 JSONL**을 사용합니다. PDF/HWP 바이너리를 읽는 파서는 이 기여 범위에 넣지 않았습니다.
기존 프로젝트의 `rfp_phase5_local_20261111/data/source_documents.jsonl`을 그대로 입력할 수도 있습니다.

```bash
uv run rfp-rag build --source /absolute/path/source_documents.jsonl --workspace workspaces/team-v1 --config configs/rag.yaml
uv run rfp-rag catalog --workspace workspaces/team-v1
```

각 행은 아래 구조의 문서입니다. `sha256`은 실제 자료에서는 원본 파일의 SHA-256입니다.
동일 원본의 사본은 `canonical_doc_id`와 `is_duplicate_alias`로 연결합니다.
`locator`, `page` 또는 `section`, `block_id`를 실제 위치 그대로 보존해야 합니다.
없는 페이지·원문 위치를 임의로 만들지 않습니다. 가상 예제의 SHA는 가상 본문에서 계산했습니다.

```json
{"doc_id":"A","canonical_doc_id":"A","sha256":"64자리 원본 SHA-256","file_name":"제안요청서.pdf","detected_format":"pdf","status":"parsed","blocks":[{"block_id":"p1","text":"원문 텍스트","page":1,"locator":"PDF p.1","kind":"paragraph"}],"supplemental_blocks":[],"reference_blocks":[],"tables":[]}
```

표는 원문 위치가 있는 `blocks`에 포함되어야 합니다. `tables`만 전달하면 본문 검색에 들어가지 않습니다.
검토되지 않은 OCR·참조용 중복 뷰는 검색에서 제외됩니다.
구축은 네트워크를 사용하지 않으며 OpenAI 임베딩을 호출하지 않습니다.
원문·설정·구현·라이브러리 버전이 바뀌면 기존 색인을 덮어쓰지 않고 새 workspace를 요구합니다.
새 작업 폴더의 청크와 인덱스는 현재 환경에서 재구축하므로 과거 pickle 색인을 옮길 필요가 없습니다.

## 팀 저장소에 반영할 범위

팀 저장소의 기능 브랜치에 **이 폴더 전체**를 `contributions/younggwan_rag/`로 넣으면 독립 실행할 수 있습니다.
기존 저장소의 과거 Phase 폴더, 패치 노트북과 결과 파일을 전부 PR할 필요는 없습니다.
팀 루트가 아니라 해당 기여 폴더에서 `uv sync`를 실행합니다.
현재 검증 환경은 PyPI 접속이 HTTP 403으로 차단되어 `uv.lock`을 생성하지 못했습니다.
인터넷 연결이 가능한 개발 환경의 첫 `uv sync`에서 생성되는 `uv.lock`을 검토해 팀 반영 시 `pyproject.toml`과 함께 커밋하세요.
현재 환경의 실행 검증과 새 가상환경의 의존성 해결 검증은 구분합니다. 검증 범위는 `VALIDATION.md`에 기록했습니다.

팀 그래프에 연결할 때는 별도 통합 PR에서 다음 경계를 맞춥니다.

| 경계 | 현재 준비한 내용 | 통합 시 확인 |
| --- | --- | --- |
| 파싱 → 청킹 | 원문 JSONL 입력 계약과 SHA·위치 검사 | 팀 `ParsedDocument`에서 원문 위치를 유지하는 변환 |
| 검색 → 에이전트 | `workflows.search` + `adapters.to_langchain_documents` | 팀 `retrieve(query, config, filters)`에 연결 |
| LLM 생성 | 기존 출처 제한·구조화 출력·사용량을 유지한 OpenAI 클라이언트 | 팀 `get_llm(config)`/Ollama와의 어댑터는 후속 통합 |
| 에이전트 → API | 구조화 결과와 `markdown`, 검증된 `claims[].citations` | 팀 LangGraph `build_graph`와 `AskResponse(answer,sources)`에 연결 |

이 패키지는 팀 LangGraph의 대체 구현을 자동 설치하거나 다른 담당자의 파일을 수정하지 않습니다.
추가한 LangChain 변환은 `uv sync --extra team`으로 사용할 수 있습니다.

## 로컬 화면과 검토

```bash
uv run rfp-rag serve --workspace workspaces/demo --port 8765
```

표시된 `http://127.0.0.1:8765`에서 문서를 선택해 사용합니다.
API 생성을 켜려면 프로세스의 `OPENAI_API_KEY` 환경변수 또는 workspace 부모 폴더의 `.env`를 사용합니다.
모든 결과는 workspace 아래에 저장됩니다. 기존 `phase*_outputs` 하위 경로와 데이터 형식은 검토 도구 호환을 위해 유지했습니다.

기계 검사는 문서 범위·원문 인용·실행 상태를 검사합니다. 문장의 모든 의미가 인용으로 뒷받침되는지는 별도 검토합니다.
`semantic_accuracy=null`과 `review_pending`을 자동 통과로 바꾸지 않습니다.
미처리 청크가 남은 요약을 문서 전체 생성 완료로 표시하지 않습니다.

## 출처 및 제외 항목

기준 구현: `TheMostEvident/kor-rfp-rag-agent-openai@f99e7942`.
모듈별 원본·이동 경로와 원본 SHA-256은 `SOURCE_MAP.json`에 기록했습니다.
모듈 import를 `rfp_rag` 아래로 옮기고, 코드 지문이 설치 패키지를 검사하도록 경로를 수정했습니다.
원래 저장된 요약 실행 지문과 새 패키지의 지문은 다릅니다. 과거 실행을 새 실행으로 덮어쓰지 않습니다.

Phase 1 수용 기준 노트북, 중간 패치·진단 노트북, 같은 내용의 수정본, 실제 문서와 평가 정답,
인덱스·캐시·실행 로그·사람 검토 기록·키 파일은 이번 팀 기여에서 제외했습니다.
