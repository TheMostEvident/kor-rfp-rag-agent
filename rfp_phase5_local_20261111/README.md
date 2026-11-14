# 입찰메이트 RFP RAG — Phase 5

현재 실행 노트북은 `notebooks/12_younggwan_phase5_final_env_fixed.ipynb`입니다. 기본 대상은 표본 7건이 아니라 전체 원본입니다.
기존에 성공한 Phase 3 `.venv/bin/python` 커널과 로컬 원문·인덱스를 사용하세요.
원문·인덱스·API 응답 캐시는 Git에 포함되지 않으므로, 새 clone만으로 실제 실행 데이터를 재현할 수는 없습니다.

## 모든 HWP·HWPX·PDF 인식과 처리

### 먼저 원본 등록 판정만 빠르게 확인

```bash
git pull --ff-only origin main
python rfp_phase5_local_20261111/scripts/check_corpus_sources.py
```

이 검사는 모델·SDK·인덱스 재구축 없이 원본 파일을 대조하며 API를 호출하지 않습니다.
읽기100·등록된실제파일100·미등록0이 현재100건의 정상 결과입니다.
Mac 한글 NFC/NFD 표기가 다른 경로는 실제로 같은 파일인지 확인한 뒤 연결합니다.
`._파일명` 중 AppleDouble 헤더가 확인된 부가 파일은 별도로 집계하고 RFP 수에 넣지 않습니다.
내용 SHA가 같아도 새 이름의 복사본·하드링크는 자동으로 등록 처리하지 않습니다.
진짜 미등록 파일이 남으면 보고서에 경로와 기존 원문과 내용이 같은지도 표시합니다.
노트북은 커널 Restart 후 실행해야 수정한 등록 검사가 적용됩니다.

Phase 5 종료는 전체 입력의 연결·검색 검증, 기존 평가 표본의 요약/Q&A 내용 검토,
실패·근거 부족 처리 확인을 구분합니다. 905개 요약 배치를 모두 사전 생성하는 작업은
그 기능 검증과 별개이며, 등록 검사 오류를 해결하려고 전체 유료 생성을 반복하지 않습니다.

12번 노트북은 `rfp_phase5_corpus/`의 코드로 다음을 한 번에 연결합니다.

1. 모든 원본 파일의 SHA·magic 형식과 파서 버전을 저장된 추출 원문에 연결합니다. 확장자만 믿지 않습니다.
2. 모든 대표 문서의 본문·표·원문 위치·청크 보존과 인덱스 정합성을 확인하고 실제 본문 구간으로 검색합니다.
3. 새 폴더를 지정하면 실제 파싱·청킹·색인을 별도 workspace에 구축합니다. 원래 100건과 기존 API 기록을 덮어쓰지 않습니다.
4. 전체 문서 요약을 선택한 채 이번 실행 상한만큼 처리하고 이어서 실행합니다. 전체 잔량이 예산을 넘는다고 호출 전체를 끄지 않습니다.

```bash
# 프로젝트 최상위에서 실행: 기본은 전체 원본·검색·생성 상태 검사만, API 0회.
python rfp_phase5_local_20261111/scripts/corpus_phase5.py

# 캐시 없이 모든 고유 원본을 실제 재파싱 (수 분 이상 소요).
python rfp_phase5_local_20261111/scripts/corpus_phase5.py --reparse-raw

# 새로운 입력 폴더를 실제 파싱·색인; 원래 인덱스는 보존.
python rfp_phase5_local_20261111/scripts/corpus_phase5.py --input-directory /경로/RFP폴더

# 전체 문서 중 최대20회 신규 생성. 반복하면 완료 작업을 건너뛰고 이어갑니다.
python rfp_phase5_local_20261111/scripts/corpus_phase5.py --enable-openai --max-new-calls 20 --max-total-calls 1000 --max-total-input-bytes 40000000
```

입력·파서 위치가 다르면 `--raw-directory`, `--parser-root`를 지정합니다. 새 workspace 경로는 출력에 표시됩니다.
노트북에서는 `INPUT_DIRECTORY`로 같은 작업을 수행합니다. API 옵션은 생성 셀의 `RUN_LIVE_SUMMARIES` 하나입니다.
이전 12번의 세 파일명은 같은 전체 문서 노트북 내용으로 동기화했습니다.

실제 확인한 데이터는 원본100건(HWP47/HWPX23/PDF30), 중복 제외99건입니다.
고유99개를 캐시 없이 다시 파싱한 결과 실패0건, 기존 primary blocks·tables와99/99 일치했습니다.
본문 구간 검색에서 BM25만으로 놓치는 문서가 있어 새 검색은 RRF와 긴 연속 원문 정확 일치를 결합합니다.
비교용 BM25 누락도 보고서에 그대로 남깁니다. 일반 자연어 질문에 대한 의미적 정답률은 별도 평가입니다.

`phase5_outputs/corpus_audit/documents.csv`는 실패한 새 입력까지 모든 원본 행을 보여줍니다.
`raw_sources.json`, `source_index_retrieval.json`, `corpus_audit.json`에 단계별 근거를 저장합니다.
`corpus_execution.json`은 전체 문서별 완료·미처리·실패·제외 주장 수를 보여줍니다.
현재 905개 전체 배치 중 완료47·실패1·미처리857입니다. 전체가 생성 완료됐다고 표시하지 않습니다.

PDF 이미지·벡터 숫자·표 해석은 자동 추출 성공과 구분합니다. 현재95개 원본에는 구조·시각 검토 경고가 있습니다.
경고는 파싱 실패95건이라는 뜻이 아닙니다. 숫자·VAT·기간·표의 의미적 완전성은 사람이 확인해야 합니다.
OCR 또는 일부 페이지 추출 실패 문서는 기본 자동 생성 대상에서 제외하고, 검토 후에만
`--allow-unreviewed-sources` / 노트북 `ALLOW_UNREVIEWED_SOURCES=True`로 명시적으로 포함합니다.
그 경우에도 검토 필요 표시와 전체 상태는 지우지 않습니다.

## 기존 실행의 실패 배치 복구

업로드된 `phase5_final_source_units` 실행에는 API 48회, 완료 배치 47개, 실패 배치 1개가 있습니다.
실패 작업은 D03(HF 인사이트)의 B0002이며, 당시 응답 원문은 보존되지 않아 정확한 과거 실패 원인을 확정할 수 없습니다.

프로젝트 최상위에서 최신 변경을 받은 뒤 기존 가상환경을 사용합니다.

```bash
git pull --ff-only origin main
cd rfp_phase5_local_20261111

# 기본은 진단만 수행합니다. 추가 API 호출 0회.
python scripts/recover_phase5.py

# 기록된 실패 작업만 명시적으로 재시도합니다. 추가 호출 상한은 기본 1회.
python scripts/recover_phase5.py --retry-failed --enable-openai
```

키는 이 폴더 또는 부모 폴더의 `.env`에서 읽습니다. 다른 위치는 `--env-file /경로/.env`로 지정합니다.
가상환경에서 `python`이 없다면 동일한 인터프리터의 `python3`를 사용하세요.

복구 도구는 기존 `rfp_phase4`, `rfp_phase5`, `rfp_phase5_final`을 수정하지 않습니다.
따라서 기존 `RUN_ID`, `frozen.json`, 요청 지문과 캐시를 유지하고 완료한 47개 작업을 다시 호출하지 않습니다.
기존 사용량 48회도 그대로 이어서 계산합니다. 실패 상태가 아니거나 계획에 없는 작업은 재시도하지 않습니다.

동일한 재시도 태그는 한 번만 소비합니다. 명령을 반복해도 완료 작업을 다시 호출하지 않습니다.
다시 실패하면 `recovery_audit.json`의 상세 사유와 `diagnostic_file` 경로를 먼저 확인하세요.
의도적인 추가 재시도는 새로운 `--retry-tag`를 지정해야 합니다.
인증·잔액·연결 오류를 해결한 후에만 `--resume-after-service-error`로 서비스 차단을 해제할 수 있습니다.

## 진단과 검토

- `phase5_outputs/final_source_units/phase5_final_source_units/recovery_audit.json`: 실패 작업, 누적 사용량, 제외 주장 분류.
- 같은 실행 폴더의 `recovery_last_attempt.json`: 이번 복구의 호출 수와 실행 상태.
- 응답 캐시 폴더 아래 `recovery_diagnostics/`: 검증 실패 응답과 상세 사유. 키 값은 가리고, 파일 권한은 0600으로 저장합니다. 이 폴더는 Git 및 결과 ZIP에서 제외됩니다.
- 기존 문서별 `comparison.csv`, `rejection_review.csv`: 사람이 확인할 검토표. 기존 작성 내용은 보존합니다.

응답 상태·빈 미확인 항목·잘못된 근거 번호·SOURCE 표지 오류를 구분합니다.
검증 규칙을 완화하거나 모델의 주장·category·상태를 자동 교정하지 않습니다.
제외된 주장이 잘못된 항목에 분류됐다는 사실만으로 ‘제외 적절’이나 의미적 정답을 자동 판정하지 않습니다.

복구 성공 후에도 요약 정확성·누락·조건 보존과 제외 주장 판단은 별도로 검토해야 합니다.
검토표를 작성한 뒤 12번 노트북 마지막 셀의 `REVIEWER`에 이름을 적고 해당 셀만 실행하세요.
최종 판정은 전체 ID 검사, 실제 입력 처리, 출처 검사, 사람의 내용 검토를 모두 구분합니다.

## API 없는 테스트

```bash
PYTHONPATH=.:rfp_phase5_final python -m unittest discover -s rfp_phase5_final -p 'test_*.py'
python -m unittest rfp_phase5_recovery.test_recovery
```

테스트는 합성 문서와 가짜 클라이언트만 사용합니다. 실제 API 품질 평가를 대신하지 않습니다.

이전 단계 안내: [README_PHASE5.md](README_PHASE5.md), [README_PHASE4.md](README_PHASE4.md).

