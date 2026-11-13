# 입찰메이트 RAG — Phase 3 로컬 청킹·검색

Phase 2에서 추출한 RFP 본문을 출처가 있는 청크로 나누고, 질문에 관련된 근거를 찾는 단계입니다. 기본 실행은 로컬에서 끝나며 API 키가 필요 없습니다. GitHub·Google Drive를 변경하는 동작도 없습니다.

이번 단계의 결과는 **검색 근거**입니다. 요약문이나 Q&A 답변을 생성하지 않습니다. LLM 답변 생성, 인용 검증, 답변 품질 평가는 Phase 4에서 이어갑니다.

## 실행 환경과 설치

Python 3.12와 SQLite FTS5를 사용할 수 있는 환경을 권장합니다. 아래 명령은 모두 압축을 푼 `rfp_phase3_local_20261111` 폴더에서 실행합니다.

```bash
python -m venv .venv
```

가상 환경을 활성화합니다.

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

노트북 실습용 의존성을 설치합니다.

```bash
python -m pip install -r requirements-notebook.txt
```

CLI만 사용한다면 `requirements-local.txt`만 설치해도 됩니다. 로컬 검색에 필요한 외부 라이브러리는 NumPy, SciPy, scikit-learn이며, 나머지 핵심 코드는 Python 표준 라이브러리를 사용합니다. 로컬 환경의 세 라이브러리 버전은 재현을 위해 고정했습니다.

## 노트북으로 시작하기

패키지 루트에서 Phase 3 노트북을 열고 위에서부터 실행합니다.

```bash
python -m jupyter lab notebooks/04_younggwan_phase3_retrieval.ipynb
```

실습 순서는 입력 확인 → 청킹 → 출처·커버리지 검증 → 인덱스 구축 → 검색 → 문서 선택·후속 질문 → 개발 세트 평가입니다. API 임베딩은 별도 선택 실습입니다.

같은 작업을 CLI로 실행할 수도 있습니다.

```bash
python run_phase3.py --build
python run_phase3.py --evaluate-dev
```

`--build`는 현재 설정으로 청크와 로컬 인덱스·검증 기록을 만듭니다. 평가 파일의 해시는 확인하지만 질문으로 검색하거나 정답을 채점하지 않습니다. `--evaluate-dev`는 개발 세트로 검색 성능을 측정합니다. 후보 설정을 개발 세트에서 비교하려면 `--select-on-dev`를 사용합니다.

기본 실행은 holdout을 자동 평가하지 않습니다. `--freeze-and-evaluate-holdout`은 설정 동결 후 최종 holdout 평가를 명시적으로 요청하는 옵션입니다. 이 결과로 설정을 다시 조정하면 같은 holdout을 최종 검증으로 볼 수 없습니다. 이미 저장된 동결·holdout 보고서가 있다면 실습에서는 그 결과를 읽고, 평가를 반복 실행하지 않는 것을 권장합니다.

## 데이터와 청크

- 원본 파일 ID는 100개이고, 같은 내용의 파일을 합치면 고유 문서는 99개입니다. 같은 원문을 검색 인덱스에 두 번 넣지 않으며, 100개 파일 ID는 대표 문서 ID로 해석됩니다.
- 입력은 `data/source_documents.jsonl`의 Phase 2 추출 결과입니다. 원본 바이너리를 다시 파싱하거나 Drive에서 최신본을 자동으로 가져오지 않습니다.
- 원본 SHA-256, 문서 ID, 페이지·섹션, 블록 ID와 문자 구간을 청크에 보존합니다. 문서·설정·코드 및 인덱스의 지문을 기록해 현재 자료 버전과 저장 결과를 구분합니다. 자료나 설정을 바꾸면 인덱스를 다시 구축해야 합니다.
- 기본 청크 크기는 **1,800 UTF-8 바이트**, 겹침은 **240 UTF-8 바이트**입니다. 모델 토큰 수가 아닙니다. 한글을 중간 바이트에서 자르지 않으며, 문서·PDF 페이지·HWP 섹션 경계를 넘지 않습니다.
- HWP/HWPX의 기본 본문과 확인된 전사본을 사용합니다. 호환용 중복 뷰와 검토되지 않은 OCR은 검색 본문에 넣지 않습니다. PDF의 표 후보나 추정된 표 구조는 의미 검증을 마친 표로 취급하지 않습니다.

## 기본 검색 방식

| 방식 | 역할 |
| --- | --- |
| `bm25` | SQLite FTS5로 단어·문자 조각의 일치를 검색 |
| `char_tfidf` | 문자 n-gram TF-IDF 벡터로 표현이 비슷한 청크 검색 |
| `rrf` | 두 검색기의 순위를 가중 RRF로 결합 |
| `mmr` | 결합 후보에서 관련성과 청크 간 중복을 함께 고려 |

동봉 설정의 기본 방식은 dev에서 선택한 `bm25`입니다. 사용 예의 `method`도 이 설정을 읽습니다.

TF-IDF 벡터는 로컬 어휘 벡터입니다. OpenAI의 의미 임베딩을 사용하지 않습니다. 검색 점수·코사인 거리·RRF 순위는 답변의 정확도 확률이 아닙니다.

```python
import json
from pathlib import Path
from local_rag.retrieval import LocalSearchIndex
from local_rag.session import SearchSession

config = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
index = LocalSearchIndex("index")
catalog = index.get_catalog()
session = SearchSession(index)
session.select_documents([catalog[0]["doc_id"]])  # 실제 실습에서는 원하는 문서를 선택
result = session.ask("제안서 제출 방식은?", top_k=5, method=config["search"]["method"])
result["status"], result["hits"]
```

문서 선택과 사용자 질문 이력은 세션별로 유지됩니다. 명시한 사업명이 바뀌면 검색 범위도 바꿀 수 있고, 같은 발주기관에 여러 사업이 있으면 문서 선택을 요청합니다. 연도만으로 문서 부재를 단정하지 않습니다. 후속 질문의 검색어 보완에는 현재 선택 정보와 사용자 질문을 사용하며, 이전 검색 본문이나 assistant 답변을 새 근거로 넣지 않습니다.

`status="retrieved"`는 관련 청크를 찾았다는 뜻입니다. 검색 결과가 없으면 `insufficient_evidence`, 문서 범위가 모호하면 `needs_clarification`을 반환합니다. 검색 실패로 문서에 해당 조건이 없다고 확정하지 않습니다.

전체 문서 요약을 준비할 때는 상위 5개 청크 대신 전체 문서 경로를 사용합니다.

```python
full_document = session.get_full_document_chunks(catalog[0]["doc_id"])
all_chunks = full_document["chunks"]
```

이 경로는 해당 문서에 저장된 모든 청크와 개수를 반환합니다. 구조적 청크 커버리지 100%는 인덱스의 청크를 모두 반환했다는 뜻이며, 원문 전체의 의미를 정확히 요약했다는 뜻은 아닙니다. 이 단계에서는 요약문을 생성하지 않습니다.

## 선택 기능: OpenAI 임베딩과 로컬 Chroma

기본 로컬 실습과 독립적인 `local_rag/embeddings.py` 어댑터가 있습니다. 사용할 때만 별도 의존성을 설치합니다.

```bash
python -m pip install -r requirements-openai.txt
```

기본값은 `enabled=False`입니다. 실제 호출에는 코드의 명시적 활성화, 환경 변수 `OPENAI_API_KEY`, 현재 공개 문서 ID와 원본 SHA-256 승인 목록이 필요합니다. 키를 노트북 셀·설정 파일에 저장하지 않습니다. `configs/default.json`의 값만 바꿔도 기본 빌드가 자동으로 API를 호출하는 구조는 아닙니다.

어댑터는 정확한 입력·모델·차원에 따른 캐시, 요청 전 입력·사용량 상한 검증, 선택 문서 범위 제한을 제공합니다. Chroma는 로컬 저장 폴더의 `PersistentClient`를 사용합니다. 바이트 기반 입력 상한은 보수적인 토큰 상한 관리이며, 실제 토크나이저로 측정한 토큰 수가 아닙니다.

OpenAI·Chroma SDK는 공식 문서의 현재 API를 기준으로 선택 범위를 지정했습니다. **실제 API 호출, 실제 Chroma 통합, 의미 임베딩 검색 품질·비용은 아직 측정하지 않았습니다.** 네트워크 없는 대역 테스트 통과와 실제 서비스 검증은 구분해야 합니다. 처음 실행할 때 설치된 버전과 연동을 확인하고 별도의 실험 결과를 남기세요. Chroma 로드 검사는 컬렉션 메타데이터·ID·개수 수준이며, 저장 벡터 내용의 변조·손상을 전부 감지하는 검사는 아닙니다.

참조: [OpenAI 임베딩 API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create), [Chroma 클라이언트](https://docs.trychroma.com/reference/python/client), [Chroma 컬렉션](https://docs.trychroma.com/reference/python/collection).

## 실제 측정 결과

100개 파일의 고유 원문 99개에서 18,266개 청크를 구축했습니다. 원문 비공백 문자 4,750,597자와 기존 검증 근거 684개를 모두 보존했습니다.

선택 문서가 주어진 사실·비교 질문의 BM25 근거 Recall@5 / @20은 dev 27건에서 **59.6% / 83.1%**, 설정 동결 후 holdout 18건에서 **56.5% / 74.2%**였습니다. 근거를 놓치는 사례가 있어 실무 성능 목표를 충족했다고 판정하지 않습니다. 답변 정답률도 아닙니다.

상세 비교와 한계는 [PHASE3_RESULTS.md](PHASE3_RESULTS.md), 설정 동결 기록과 평가 원자료는 `reports/`에 있습니다.

## 검증과 다음 단계

```bash
python -m unittest discover -s tests -v
```

테스트는 출처 구간 복원, 한글 바이트 경계, 중복 문서, 검색 범위 제한, 세션 격리, 평가 누출 방지, 선택 API 어댑터의 네트워크 없는 동작 등을 확인합니다. 개발 세트 평가 결과와 실행 지문은 `reports/`에서 확인합니다. 보고서에 기록된 설정·검색 방식·문서 범위·데이터 분할을 함께 읽어야 합니다.

Phase 3의 검색 지표는 생성 답변의 정답률이나 전체 에이전트 기능 완성률이 아닙니다. Phase 4에서 검색 근거를 이용한 답변 생성과 인용, 근거 부족 응답, 전체 문서 요약을 연결하고 별도의 답변 품질 평가를 진행합니다.
