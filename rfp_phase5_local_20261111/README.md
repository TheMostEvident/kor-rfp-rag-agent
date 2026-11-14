# 입찰메이트 RFP RAG — Phase 5

현재 실행 노트북은 `notebooks/12_younggwan_phase5_final_env_fixed.ipynb`입니다.
기존에 성공한 Phase 3 `.venv/bin/python` 커널과 로컬 원문·인덱스를 사용하세요.
원문·인덱스·API 응답 캐시는 Git에 포함되지 않으므로, 새 clone만으로 실제 실행 데이터를 재현할 수는 없습니다.

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

