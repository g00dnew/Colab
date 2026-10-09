# T12 분석 인수인계 (2026-10-09 새벽 작성) — 새 세션은 이 파일부터

## 상태
- 1·2단계 분석 완료, 결과 `results/SUMMARY.md`. 노트북 `colab/T12_laryngeal.ipynb`, 모듈 `code/t12_pipeline.py`, 정렬 `scripts/align_t12.py`(.venv-tf), 분석 venv = 상위 `.venv`(py3.14).
- GitHub g00dnew/Colab `t12/`에 공개로 올라감(10/9). 사용자: 나중에 비공개 전환 예정.
- 연구 판정·수정사항은 `../core-question-v3.md` 10/9 절. 외부 검토 2건 반영: H_sampling 기각 아님(Jossinger 2026), 제목 톤 완화, PCA 누출 지적.

## 사용자가 확정한 할 일 (0·1번만. T15 재현·VOT는 보류)
### 0a. PCA 누출 수정 (필수)
- `code/t12_pipeline.py`의 분류기 경로(balanced_pair_decode, 39-way, T3·natural-class, markedness에 PCA/스케일러가 있으면 그것도)에서 스케일러+PCA를 **CV 학습 폴드 안에서만 적합**(sklearn Pipeline). 순열 null도 같은 파이프라인.
- 재실행·전후 비교: Stage1 39-way, Stage1 pair go_all(6v/6v_sup/6v_inf/44), Stage2 T1 matched-n(pooled·test-only, DoD CI), T3 primary_0729, natural-class per-pair. 순열 200회로 줄여도 됨. BLAS 2스레드.
### 0b. 파열음 전용 1차 검정 T1'
- 후두쌍 P/B·T/D·K/G만; 위치 대조 P/T·T/K·B/D·D/G; 무기식 대조 F/V·TH/DH.
- 어두 vs 비어두(어중+어말) 주검정 + 3수준. 위치별 후두−위치 차, DoD(어두−비어두) 부트스트랩 CI + 순열 p. pooled held-out 규칙 + test-only. 탄설음(모음간 T/D) 제외 민감도. 6v, seg 창, score≥0.5, 쌍별 위치 n 맞춤, 300/class 상한.
- 출력 `results/stage2_t1_stops_primary.csv`, SUMMARY 절 "T1' 파열음 전용 1차 검정".
### 1. 자질 정보의 시간 전개 (tuningTasks 음소 세션)
- 200ms 창, 50ms 간격, go 기준 −200~+1500ms(두 세션 pooled) + 04.26은 실측 발화 시작 기준. 6v, PCA-in-CV.
- 창마다: 후두 파열음쌍 평균, 후두 마찰음쌍 평균, 위치쌍 평균, 39-way. 반복 부트스트랩 CI. 피크 시점(위치 vs 후두)과 피크 차 CI, 순열 95% 넘는 첫 창(잠복기).
- 출력 `results/stage1_timecourse.csv`, `results/figures/stage1_timecourse.png`, 노트북 셀 1개, SUMMARY 절 "자질 정보의 시간 전개".

## 완료 후
- SUMMARY.md 전후표 갱신, 노트북 재검증(nbformat, tp.* 참조), GitHub 재push(사용자 지시 있었음: 공개로 우선 올리기).
- 사용자에게 전후 수치표만 짧게 보고. 제목 표현 금지어: "뇌는 숨 터뜨림으로 p/b를 구분한다".

## 진행 상황 (02:5x 기준, 자동 실행 중)
- 0a PCA 수정 **코드 완료**(`code/t12_pipeline.py`, 백업 `t12_pipeline.py.bak_prepca`, 수정 전 결과 `results_pre_pcafix/`). 재실행 체인 `code/rerun_pcafix.sh` 백그라운드 가동, 로그 `results/rerun_pcafix.log`(STAGE1→STAGE2→T3→T3PAIRS, 각 OK/FAIL 줄, 끝에 ALL_DONE). 총 2~3시간.
- 0b 파열음 전용 T1' `code/run_t1_stops_primary.py` + 1 시간전개 `code/run_stage1_timecourse.py` → `code/run_step01.sh`로 순차 실행 중, 로그 `results/step01.log`(T1STOPS OK/FAIL, TIMECOURSE OK/FAIL, STEP01_DONE). 출력: stage2_t1_stops_primary.csv, stage2_t1_stops_dod.csv, stage2_t1_stops_3lvl_interaction.csv, stage1_timecourse*.csv, figures/stage1_timecourse.png.
- 미완: (1) 전후 비교표 작성(results_pre_pcafix vs results), (2) SUMMARY.md에 T1'·시간전개 절 추가, (3) 노트북에 셀 2개 추가(두 스크립트 호출), (4) GitHub 재push(사용자 지시: 공개 우선), (5) 위치 라벨 순열검정은 생략했음(쌍 부트스트랩 CI만) — 필요하면 추가.
- 로그에 FAIL이 있으면 해당 스크립트를 venv로 직접 재실행해 traceback 확인.

## 10/9 오전: 포스터 초안 v1 (Claude Doc)
- 결과 해석·결론·포스터 초안(배경→선행연구→실험과정→결과→노벨티→한계→결론) 정본: https://claude.ai/code/artifact/2bd11949-1b5d-49bc-b768-bfb826ed53f4
- 초안 수치 중 T1 6쌍·T3·쌍 분리도는 PCA 수정 전 값. rerun_stage2.log에 STAGE2_CHAIN_DONE 뜨면 §4 표 갱신할 것.
- KASELL 겨울대회는 작년 기준 포스터 세션 없음(전부 구두 20분). 초안 구조는 그대로 발표 슬라이드 순서로 쓸 수 있음.

## 11:40 완료 표시
- 0a PCA 수정 재실행 **완료**(stage1·stage2·T3·쌍별·배열분할·onset 스캔 전부 수정판). 전후 비교표 = SUMMARY.md "PCA 폴드 내 적합 수정" 절. 질적 결론 불변.
- 0b T1′ **완료**: 파열음만 DoD 유의(0.109 [.036,.183]) 그러나 마찰음 대조군도 동일 하락 → 위치 효과 = 후두 일반, 기식 특이적 아님.
- 1 시간전개 **완료**: 파열음 후두 피크 = 조음위치 피크(onset 0ms), 마찰음 후두 +160ms.
- SUMMARY.md 갱신, 노트북 셀 9·10 추가(12셀), GitHub push 완료.
- 남은 선택지(사용자 보류): T15 재현, VOT 실측, 위치 라벨 순열검정.
- 12:3x 계획 vs 실행 분석 완료(run_t1_planning.py). 세 창 모두 교차 패턴 유지 → 계획 단계 성질. 문서·SUMMARY 반영, push 예정.
