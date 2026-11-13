# 입찰메이트 RAG — Phase 4 근거 기반 답변·전체 문서 요약

Phase 3의 99개 고유 문서·원문 위치·청크·검색 인덱스를 이어받아, OpenAI를 통한 근거 기반 Q&A·문서 비교·후속 질문·전체 문서 요약을 연결합니다. 기본 노트북 실행은 실제 원문과 검색 결과, 요약 계획을 점검하며 API를 호출하지 않습니다. GitHub나 Google Drive를 변경하는 작업은 없습니다.

이 환경에서는 실제 OpenAI API를 실행하지 않았습니다. 네트워크 없는 대역 테스트와 실제 모델의 답변 품질·비용 측정은 구분합니다. 라이브 실행 결과는 사용자 Mac의 `phase4_outputs/`에 새로 기록하며, 기존 Phase 3의 원문·인덱스·동결 평가 결과를 유지합니다.

실 API 옵션을 켠 실행마다 `phase4_outputs/runs/<UTC시각_UUID>/`에 새 실험 폴더를 만듭니다. `run_context.json`에는 실제 생성 설정·프롬프트 버전·검색 인덱스 지문·승인 문서 SHA·패키지 버전을 기록하고 API 키나 환경변수 전체는 저장하지 않습니다. 답변은 순서대로 JSON에 보존하고, dev 평가와 사람 검토 표는 그 실행의 `dev/`에 저장합니다. 같은 셀을 다시 실행해도 기존 기록을 덮어쓰지 않도록 새 번호를 붙입니다. 마지막 셀은 사용량·추정 비용·상태를 `run_summary.json`에 기록합니다. 기본 오프라인 실행은 새 API 실험 폴더를 만들지 않으며, 모델 캐시와 의미 검색 인덱스는 별도 공용 폴더에서 재사용합니다.

## Mac 설치: Python 경로를 직접 지정

Python 3.11 이상이 필요하며 3.12를 권장합니다. 터미널의 작업 위치를 압축을 푼 `rfp_phase4_local_20261111` 폴더로 맞춥니다.

```bash
python3 --version
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-phase4.txt
./.venv/bin/python -m ipykernel install --user --name rfp-phase4 --display-name "Python (RFP Phase 4)"
```

`.venv/bin/python`을 직접 지정하면 설치 대상이 명확해집니다. 동봉된 인덱스의 검증 버전은 `numpy==2.3.5`, `scipy==1.17.0`, `scikit-learn==1.8.0`이며 `requirements-local.txt`에서 고정합니다. 선택 OpenAI SDK 범위는 실제 서비스 연동을 검증한 고정 파일이 아닙니다. 사용자 계정의 모델 접근 권한과 설치된 SDK 연동을 첫 실습에서 확인해야 합니다.

## VS Code에서 노트북 커널 선택

1. VS Code에서 압축을 푼 Phase 4 폴더를 엽니다.
2. `notebooks/05_younggwan_phase4_grounded_answers.ipynb`를 엽니다.
3. 노트북 오른쪽 위 커널 → **Select Another Kernel → Python Environments**에서 프로젝트의 `.venv/bin/python`을 고릅니다. 등록된 커널 목록을 사용한다면 **Python (RFP Phase 4)**를 선택합니다.
4. 커널을 재시작하고 첫 셀을 실행합니다. 출력의 실행 경로와 세 라이브러리 버전 일치를 확인한 뒤 Run All을 실행합니다.

터미널에서 가상환경을 활성화해도 노트북의 커널이 자동으로 바뀌는 것은 아닙니다. `/usr/local/bin/python3`가 표시되면서 버전이 다르면 올바른 커널을 다시 선택하세요. 노트북 첫 셀에서 버전 불일치를 발견하면 검색 모듈 로딩 전에 멈추고 구체적인 해결 명령을 안내합니다. 의존성 검사를 삭제하거나 저장 인덱스를 자동으로 덮어쓰지 않습니다.

VS Code 대신 JupyterLab을 사용할 수도 있습니다.

```bash
./.venv/bin/python -m jupyter lab notebooks/05_younggwan_phase4_grounded_answers.ipynb
```

## 기본 Run All에서 확인하는 것

- Phase 3의 원문·청크·검색 인덱스 버전과 무결성
- 실제 dev 질문과 사용자 선택 범위로 검색한 원문 근거
- 원문에 없는 인용문을 거절하는 동작
- 전체 문서의 모든 청크를 빠짐없이 배치에 배정한 요약 계획
- API 미실행 상태와 사용량 기록

기본 실행은 모의 모델의 사실 문장을 실제 답변처럼 보여주지 않습니다. 코드 경로가 통과했다고 생성 답변이나 전체 요약 품질이 검증됐다고 표시하지 않습니다.

## 실제 OpenAI 실습 켜기

노트북 첫 셀의 옵션을 작업별로 선택합니다.

| 옵션 | 기본값 | 실행 범위 |
| --- | --- | --- |
| `ENABLE_OPENAI` | `False` | 외부 OpenAI API 사용 허용 |
| `RUN_LIVE_QA` | `False` | 사실 Q&A·문서 비교·후속 질문·근거 부족 예제 |
| `RUN_LIVE_SUMMARY` | `False` | 선택한 문서 전체의 배치 요약 |
| `RUN_DEV_EVALUATION` | `False` | 선택 dev 사례의 생성 결과와 검토 기록 |
| `RUN_SEMANTIC_COMPARISON` | `False` | 선택 공개 문서의 OpenAI 임베딩 검색과 BM25 비교 |

처음에는 `ENABLE_OPENAI=True`, `RUN_LIVE_QA=True`만 켜고 원문 근거와 생성 결과를 확인하세요. 키가 환경변수 `OPENAI_API_KEY`에 없으면 로컬 `getpass` 입력창에서 입력합니다. 키를 노트북 소스나 출력 파일에 넣지 않습니다.

API 사용을 허용한 `PUBLIC_DOC_IDS`와 문서별 원본 SHA가 표시됩니다. 실제 전송 범위는 승인한 문서의 검색 근거와 사용자 질문입니다. 다른 문서를 선택하면 승인 목록도 명시적으로 맞춰야 합니다. 기본 생성 모델은 설정 파일의 `gpt-5.4-mini`이며, 계정 접근 권한에 맞춰 사용자가 설정을 변경할 수 있습니다. 모델을 자동으로 대체하지 않습니다.

Responses API의 구조화 출력으로 주장을 받고 각 주장에 인용문을 연결합니다. `store=False` 옵션을 사용하지만 이것만으로 모든 운영 로그의 보관이 사라진다고 보장하지 않습니다. 외부 전송 문서는 조직의 사용 정책에 맞춰 선택하세요.

## 출력·검증·사용량

원문에 존재하고 이번 생성 근거에 포함된 청크만 인용할 수 있습니다. 인용문은 원문 문자 구간까지 복원하고 문서 범위·원본 SHA를 확인합니다. 잘못된 인용을 포함한 생성 응답은 정상 답변으로 표시하지 않습니다.

**인용문 출처의 정확성과 주장의 의미적 정답은 다릅니다.** 출처 검사는 문장이 해당 인용문에 의해 논리적으로 뒷받침되는지까지 판정하지 않습니다. 사람 검토용 기록에서 답변의 충분성·숫자와 단위·기간 기준·부가세 조건·문서별 비교 기준을 함께 확인해야 합니다.

요청 수·요청별 입력 bytes·누적 입력 bytes·출력 토큰 상한을 `configs/phase4.json`에서 관리합니다. byte 상한은 실제 토큰 수나 청구 금액이 아닙니다. 실제 API 응답의 토큰 usage와 응답시간을 따로 기록하고, 정확히 같은 요청의 로컬 캐시 적중을 새 호출과 구분합니다. API 오류·출력 잘림·거절·잘못된 JSON·근거 검증 실패를 정답으로 처리하지 않습니다.

노트북 마지막 셀은 보고된 토큰과 `pricing`의 모델별 단가·기준일로 USD 비용을 추정합니다. 실제 청구서가 아니며 세금·추가 요금과 사용량을 알 수 없는 실패 요청은 포함하지 않습니다. 실패 요청의 사용량이 미확인 상태면 추정이 불완전하다고 표시합니다. 대역 토큰의 비용은 가정값으로 구분하며, 생성 모델과 단가 모델이 다르면 계산을 생략합니다. 답변 생성 실패 시 노트북은 비밀값을 제거한 `error_detail`을 표시해 사용량 상한·설정 오류의 원인을 확인하도록 합니다.

전체 문서 요약은 top-k가 아닌 모든 인덱싱된 청크를 배치에 배정합니다. 배치 수와 전체 입력 상한을 먼저 확인하고, 상한을 초과하면 실행 전에 멈춥니다. 일부 배치가 실패한 경우 부분 상태와 커버리지를 보고합니다. 입력 청크 커버리지 100%가 문서 의미를 완벽히 요약했다는 뜻은 아닙니다.

## dev 평가와 선택 의미 검색

새 생성 평가는 dev의 질문 원문과 실제 `context.selected_doc_ids`만 생성 경로에 전달합니다. 정답 문서 ID·필수 사실·정답 값은 생성 후 채점에만 사용합니다. 이전 holdout을 본 상태이므로 Phase 4 튜닝과 최종 검증에 다시 사용하지 않습니다. 실무 완성 판정에는 새 독립 평가 세트가 필요합니다.

자동 평가는 상태와 출처 유효성, 필수 근거 구간 회수, 참고용 숫자·문자열 단서를 구분합니다. 이 자동 지표를 의미적 답변 정답률이라고 부르지 않습니다. 사람 검토 템플릿과 실제 비용·지연시간을 함께 확인하세요.

OpenAI 임베딩/Chroma 실습을 실행하려면 별도 선택 의존성을 설치합니다.

```bash
./.venv/bin/python -m pip install -r requirements-openai.txt
```

승인한 공개 문서만 색인하고 같은 문서 범위에서 BM25와 의미 검색을 비교합니다. 일부 문서만 색인한 실험 결과는 전체 문서의 의미 검색 품질을 나타내지 않습니다. 기본 Run All에서는 이 실험을 실행하거나 성능을 측정하지 않습니다.

## 파일 위치

| 경로 | 역할 |
| --- | --- |
| `rfp_phase4/` | 답변 생성·인용 검사·전체 문서 요약·dev 평가·선택 의미 검색 |
| `configs/phase4.json` | Phase 4 검색·생성·요약·API 상한 설정 |
| `notebooks/05_younggwan_phase4_grounded_answers.ipynb` | 순차 실습 노트북 |
| `phase4_outputs/runs/` | 실행별 설정·순차 답변·dev 평가·사람 검토·사용량 기록 |
| `phase4_outputs/generation_cache/`, `phase4_outputs/semantic_indexes/` | 실행 간 재사용하는 로컬 캐시·의미 검색 인덱스 |
| `phase4_reports/` | 동봉 오프라인 검증과 구현 범위 기록 |
| `local_rag/`, `data/`, `index/`, `reports/` | 보존한 Phase 3 기반 자료와 비교 기준 |

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

테스트 대역으로 검증한 것은 API 호출 계약과 처리 동작입니다. 실제 SDK 통합·계정 접근 권한·답변 품질·검색 품질·요약 품질은 네트워크 없는 테스트만으로 측정할 수 없습니다.

공식 문서: [OpenAI 구조화 출력](https://developers.openai.com/api/docs/guides/structured-outputs), [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create), [모델](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [데이터 처리](https://developers.openai.com/api/docs/guides/your-data), [VS Code 커널 선택](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).
