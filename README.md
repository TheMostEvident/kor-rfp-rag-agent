+# RFP RAG 에이전트

공공 입찰 RFP(제안요청서, hwp·pdf) 문서를 검색해서 질문에 답하고, 조건에 맞는 공고를 찾고, 공고의 핵심 내용을 요약해 주는 LangGraph 기반 RAG 에이전트입니다.

- 팀 노션: [RFP RAG 에이전트](https://app.notion.com/p/413232ca6c7b812099f3f7e9436693d2) (작업 방식, 기록 규칙, 문제 해결은 노션 「🛠️ 개발 가이드」 참고)

## 초기 설정

```bash
# 1. uv 설치 (macOS/Linux, Windows는 https://docs.astral.sh/uv/ 참고)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 레포 받기 + 패키지 설치
git clone https://github.com/yulesn/rfp-agent-part3-team1.git
cd rfp-agent-part3-team1
uv sync

# 3. API 키 설정: .env 를 열어 OPENAI_API_KEY 입력
cp .env.example .env

# 4. 데이터: RFP 원본과 메타데이터 파일을 data/raw/ 에 넣기 (git에 올리지 않음)

# 5. 확인
uv run pytest
```

VS Code·Jupyter 에서는 인터프리터/커널로 `.venv` 를 선택하세요.

## uv 규칙

| 언제 | 명령 |
| --- | --- |
| `git pull` 한 뒤 | `uv sync` |
| 패키지 추가 | `uv add 패키지명` (개발용은 `uv add --dev`) → `pyproject.toml`, `uv.lock` 함께 커밋 |
| 패키지 삭제 | `uv remove 패키지명` |
| 코드 실행 | `uv run python ...`, `uv run pytest` |

`pip install` 은 쓰지 않습니다.

## 실행

```bash
uv run python scripts/build_index.py      # 문서 → 벡터DB 생성 (문서·파싱·청킹·임베딩 설정이 바뀔 때만)
uv run python scripts/ask.py "질문"        # 터미널에서 질문
uv run streamlit run app/streamlit_app.py # 데모 UI
uv run python scripts/evaluate.py         # 평가
```

다른 설정으로 실험할 때는 `configs/default.yaml` 을 복사해서 `--config configs/새설정.yaml` 로 실행합니다.

## 프로젝트 구조

```
├── configs/default.yaml      # 실험 설정 (청크 크기, 모델명, top_k 등)
├── data/                     # 데이터 (git 제외): raw/, processed/, vectorstore/
├── src/rfp_agent/
│   ├── config.py, llm.py     # 설정 불러오기, LLM 생성
│   ├── ingest/               # 문서 파싱·메타데이터·청킹
│   ├── retrieval/            # 임베딩·벡터DB(Chroma)·검색
│   ├── agent/                # LangGraph 에이전트 (state, prompts, nodes, graph)
│   │   └── tools/            #   에이전트 도구 (도구 1개 = 파일 1개)
│   └── evaluation/           # 평가 지표
├── scripts/                  # build_index, ask, evaluate
├── eval/questions.jsonl      # 평가용 질문·정답
├── app/streamlit_app.py      # 데모 UI
├── notebooks/                # 개인 실험 노트북 (이름: 01_이름_내용.ipynb)
└── tests/
```

벡터DB는 별도 서버 없이 Chroma 를 `data/vectorstore/` 에 파일로 저장합니다.

## 파트별 담당 파일

각 파일 맨 위 주석에 작성할 내용이 적혀 있습니다. `TODO` 부분을 채워 주세요.

| 파트 | 파일 |
| --- | --- |
| 문서 파싱 | `ingest/schema.py`, `pdf.py`, `hwp.py`, `metadata.py`, `loader.py`, `chunking.py` |
| 검색 | `retrieval/embeddings.py`, `vectorstore.py`, `retriever.py`, `scripts/build_index.py` |
| 에이전트·도구 | `agent/state.py`, `prompts.py`, `nodes.py`, `graph.py`, `scripts/ask.py` |
| | `agent/tools/search_rfp.py` (RFP 본문 검색), `filter_metadata.py` (발주기관·금액·마감일 조건 검색), `summarize_rfp.py` (공고 1건 요약) |
| 평가·데모 | `eval/questions.jsonl`, `evaluation/metrics.py`, `scripts/evaluate.py`, `app/streamlit_app.py` |
| 공통 | `config.py`, `llm.py`, `configs/default.yaml`, `tests/` |

## 파트 간 약속

바꿔야 하면 팀에 먼저 공유해 주세요.

| 경계 | 형식 |
| --- | --- |
| 파싱 → 검색 | `ParsedDocument(doc_id, text, metadata)` (`ingest/schema.py`) |
| 청킹 → 벡터DB | `Document(page_content, metadata)`, metadata 에 원본 메타데이터 포함 |
| 검색 → 에이전트 | `retrieve(query, config, filters=None) -> list[Document]` (`retrieval/retriever.py`) |
| 에이전트 → UI·평가 | `build_graph(config).invoke({"messages": [("user", 질문)]})` (`agent/graph.py`) |

- 숫자·모델명은 코드에 쓰지 말고 `configs/*.yaml` 에서 불러오기
- LLM 은 `get_llm(config)`, 임베딩은 `get_embeddings(config)` 로만 생성
- `.env`, `data/` 는 커밋하지 않기

---
# Korean RFP RAG Agent (`kor-rfp-rag-agent`)

A Retrieval-Augmented Generation (RAG) pipeline designed to parse, index, and query complex Korean Request for Proposal (RFP / 제안요청서) documents. Built across an iterative 5-phase engineering workflow—from structured document and table extraction to hybrid retrieval and end-to-end validation.

---

## Key Features

- **Korean RFP Document & Table Parsing:** Extracts structured text blocks and complex multi-row tables from Korean RFP documents into normalized JSONL representations (`documents.jsonl`, `tables.jsonl`, `indexable_blocks.jsonl`).
- **Hybrid Retrieval Architecture:** Combines character-level sparse/dense vector search (`char_vectors.npz`) with structured SQLite chunk storage (`chunks.sqlite3`) for high-recall Korean keyword and semantic matching.
- **Grounded Generation & Validation:** Synthesizes accurate, citation-backed responses to RFP requirements and evaluates retrieval/generation quality across test queries.

---

## Project Architecture & Phases

| Phase | Directory | Core Focus |
| :--- | :--- | :--- |
| **Phase 2** | `rfp_phase2_local_20261111/` | Document ingestion, table extraction, and block-level preprocessing |
| **Phase 3** | `rfp_phase3_local_20261111/` | Chunking strategy, SQLite metadata store, and character vector indexing |
| **Phase 4–5** | `rfp_phase5_local_20261111/` | RAG agent orchestration, prompt engineering, and final validation notebooks |

> **Note:** Large raw datasets, `.jsonl` artifacts, vector indexes (`.npz`), and SQLite databases (`.sqlite3`) are excluded from version control via `.gitignore` to keep the repository lightweight.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/TheMostEvident/kor-rfp-rag-agent.git](https://github.com/TheMostEvident/kor-rfp-rag-agent.git)
cd kor-rfp-rag-agent
```

### 2. Install dependencies
This project uses `uv` (or standard `pip`) with `pyproject.toml`:
```bash
# Using uv (recommended)
uv sync

# Or using standard venv + pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configure environment variables
Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

1. **Data Preprocessing & Indexing (Phases 2–3):** Run the ingestion and indexing scripts/notebooks to generate local chunks and vector indexes from your Korean RFP source files.
2. **RAG Evaluation & Querying (Phase 5):** Open `rfp_phase5_local_20261111/notebooks/11_younggwan_phase5_final_validation(1).ipynb` in Jupyter or VS Code to run the end-to-end RAG validation pipeline.

---

## Tech Stack

- **Language:** Python 3.11+
- **LLM & Embeddings:** OpenAI API (`python-dotenv` for environment management)
- **Storage & Search:** SQLite3, NumPy / SciPy sparse/dense vector search
- **Dependency Management:** `uv` (`pyproject.toml`, `uv.lock`)

## 팀 공유용 코드

RAG 실행에 필요한 Python 패키지와 실행 노트북 4개를 [team_handoff](team_handoff/README.md)에 정리했습니다. 원문 JSONL 입력부터 로컬 색인, 근거 답변·대화, 문서 전체 요약, 파일럿 검증까지 독립 실행할 수 있습니다. 과거 패치 노트북과 실제 데이터·색인·실행 로그는 팀 공유 폴더에 포함하지 않습니다. 팀 LangGraph/API 통합 경계와 검증 범위는 해당 README와 VALIDATION.md를 확인하세요.

