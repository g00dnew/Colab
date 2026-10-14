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

## (완료: 10/9 저녁) 오후 계획 — 외부 검토 ③건 반영안 (아래 저녁 절에서 전부 실행됨)
현재 핵심 결과: 후두 대립 정보 어두>비어두, 조음위치는 반대(비어두↑). 파열음 3쌍 DoD .109 [.036,.183], 조음 전 300ms 창에서도 동일(.119). 아래 통제를 통과해야 보고 가능.

### 1순위 (결과를 뒤집을 수 있는 교란)
- **문장 단위 교차검증**: `balanced_pair_decode`/`_cv_accuracy`에 `groups` 추가 → `StratifiedGroupKFold(groups=trial)`. 민감도: groups=session. 순열 null도 같은 분할. 1단계(고립 음소)는 시행당 1구간이라 해당 없음.
- **단어 정체 교란**: 어말 T/D는 it/that/but vs and/had/did 등 소수 고빈도 단어. meta에 word(또는 sentence+word_idx로 단어 추출) 붙여 (a) 단어·클래스·위치별 토큰 상한 20, (b) leave-words-out CV. 이게 지금 결과의 가장 위험한 대안 설명.
- **발성 세션만 + test 전용**을 T1′ 기본 행으로(현재 pooled엔 입모양 세션 포함).
### 2순위 (통계 보강)
- 쌍 하나씩 제외(leave-one-pair-out) DoD. T/D 없이 방향 유지되는지.
- 쌍 대응 유지 통계: 쌍별 20회 독립 균형 재표집 → 쌍별 (어두−비어두) 분포 → 대립유형 간 차이를 쌍 라벨 순열로 검정.
- 구간 수준 위치 라벨 순열검정(쌍 안에서 위치 라벨 섞어 DoD 재계산, 200회) → 쌍 수 무관 p.
- 보고 3줄: 파열음 3쌍(사전 지정) / 후두 6쌍 전체 / 쌍 제외.
### 3순위 (민감도)
- 고정 길이 80ms 중심창(rnn_center)으로 T1′ 반복(길이 교란 제거).
- 세션 품질(PER) 공변량 또는 세션 그룹 CV로 흡수. 입모양 score 필터 탈락률 표기(T3는 이미 null).
### 실행 메모
- 코드 위치: `code/t12_pipeline.py`(분류기 §8, stage2_position_decode ~L1899), 스크립트 템플릿 `code/run_t1_stops_primary.py`, `code/run_t1_planning.py`. venv `../.venv`. BLAS 2스레드, 메모리 여유 확인(Chrome 닫기).
- 예상: 코딩 2~3h + 실행 2~3h. 정확도·DoD가 내려갈 것을 예상하고 결과를 그대로 보고.
- 그 뒤 보류 항목: T15 재현(10/28 이후), 레포 비공개 전환(사용자 요청 시 `gh repo edit g00dnew/Colab --visibility private`).

## 2026-10-09 저녁: 외부 검토 4건 반영 — 수용·보류 판정과 실행 결과

사전 고정 문서 `results/t1_controls_prereg.md`(실행 전 작성). 실행 `code/run_t1c_parallel.sh`(워커 5개, 특징 mmap 캐시 `results/t1c_cache/`). 보고표 `code/t1c_report.py`. 결과 정본 SUMMARY.md "T1′ 통제 분석" 절.

### 수용(구현·실행 완료)
| 검토 항목 | 구현 위치 |
|---|---|
| ① 문장 시행 그룹 CV + 블록·세션 그룹 민감도 | `t12_pipeline._cv_splits`(StratifiedGroupKFold, 반복마다 재섞기, 그룹 겹침 assert), `balanced_pair_decode(groups=)`, `stage2_position_decode(group_col=trial/block/session/word)`; `stage2_segment_features`가 `block_idx` 기록 |
| ② 대응쌍 유지 재표집·소수 쌍 의존 | `run_t1_controls.interaction()`: 쌍 단위 재표집 CI + 쌍 라벨 정확 순열; 구간 수준 위치 라벨 순열(주 검정, `posperm`); 20회 독립 재표집; 쌍 하나씩 제외(후두·위치 모두) |
| ③ 발성만·test/held_out·고정 길이 창 | 주분석 = 발성·held_out(test+07.29)·시행 CV; `test_trialcv_bal`; 80 ms 고정 중심창 `center80` |
| ④-1 대응 재표집 | 위 ② |
| ④-2 유성·무성 위치쌍 분리 | `CATS` place_vl(P/T·T/K)·place_vd(B/D·D/G), `t1c_voicing_subgroups.csv` |
| ④-3 어중·어말 구성 균형 | `select_pair_segments(sub_all=)`: 두 클래스의 어중:어말 구성을 동일하게 표집(`match_subposition_col`); 3수준 분리 실행 `main3`·`pooled3` |
| ④-4 그룹 CV·무결성·폴드별 기록 | 겹침 assert, `n_cv_groups`·`n_folds_used`·`n_test_min`·`bacc` 저장 → `t1c_cv_group_integrity.csv` |
| ④-5 발성 조건 주분석 | 위 ③. "Vocal+test 주분석" 대신 **Vocal+held_out**(test+미학습 07.29, 더 큰 n·RNN 독립)을 주분석으로, test만은 민감도 |
| ④-6B 쌍 제외 | `t1c_leave_one_pair_out.csv` |
| ④-6C 문맥 통제(부분) | 단어 상한 20 (`cap_tokens_per_word`), 단어 그룹 CV, 길이(80 ms), 세션·블록 CV |
| ④-8A TFRecord↔MAT 대응 검증 | `code/check_alignment_mat.py` → `results/alignment_mat_consistency.csv`: 48/48 세션·파티션 일치, 문장 불일치 0, 창 초과 0 |
| ④-8B pre 창 대체 버그 | `stage2_segment_features`가 `pre_valid`·`win_clipped` 플래그 기록, `run_t1_planning.py`는 pre_valid 행만·win_start≥0만 사용 |
| ④-9 파이프라인 경로·캐시·검정 범위 명시·사전 고정 | 스크립트가 `tp.__file__` 출력, GitHub `t12/t12_pipeline.py`와 `t12/code/t12_pipeline.py` 동일 파일로 push, 결과 접두사 `t1c_`·`force=True`, 순열검정은 주분석에만(나머지는 CI) 명시, `t1_controls_prereg.md` |

### 보류(이번 초록 결론에서 분리) 또는 수용 안 함
- ④-4C 블록 z-score가 평가 블록 통계를 씀 → **수용 안 함**(라벨 무관 정규화라 라벨 누출은 아님, 재구현 비용 큼). 한계로 명시.
- ④-5 입모양(nonvocal) 전용 T1′ → 보류(T3가 담당, 어중 41.6% 탈락으로 n 부족).
- ④-6C 앞·뒤 음소 통제, 통계 모형 → 보류.
- ④-7 교차 위치 해독(`run_cross_position.py`) → 통제 결과 확인 후 추가(검토도 그 순서 권장).
- ④-8B 계획 분석 재실행 → 코드는 고쳤으나 재실행 보류(탐색적으로 분리).
- ④-8C 시간전개 창별 null → 보류(탐색적).

### 결과 요약 (정본 SUMMARY.md "T1′ 통제 분석" 절)
- 이전 T1′ 재현 정확도 차 0.0000(회귀 OK). 시행 CV로만 바꾸면 D .109→.107 → CV 누출이 결과를 만든 건 아님.
- 후두쌍 Δ(비어두−어두)는 모든 통제에서 음수(−.06~−.18), 20회 재표집 D>0 20/20, 쌍 제외 7/7 양수.
- **그러나** 단어 통제 전엔 무성 위치쌍(P/T·T/K)도 후두쌍만큼 하락, 상승은 유성 위치쌍(B/D·D/G)의 어말에서만. "조음위치 안정" 문장 폐기.
- **단어 정체가 핵심 교란**(어두 T의 64% = 'to'). pooled+단어 상한 20: Δ_lar −.063, Δ_place +.022, D +.085 [+.036, +.130], 무성·유성 위치쌍 모두 D≈+.08~.09(CI 0 제외). 가장 엄격한 설정에서 원래 방향 성립, 효과는 작아짐.
- 주분석(held_out, n 작음): D +.108 [−.003, +.212]; 구간 수준 위치 라벨 순열(4표집 평균, 200회): D_obs = +.067(4표집 평균, CV 2반복), null 평균 −.003·SD .025·95% +.043, **p = .005**(200회 중 0회 ≥ 관측). vs 무성 위치쌍 D −.009, p = .60; vs 유성 위치쌍 +.143, p = .005; 마찰음 후두 vs 위치 +.054, p = .18. (단일 표집판 v1은 D +.012, p .37로 표집 잡음에 취약 → results/t1c_posperm_v1/ 보존. 관측 D의 세 추정치 .067/.087/.108은 CV 반복·표집 수 차이이며 재표집 범위 안)
- 사전 기준 판정: **지지**(구간 순열 p = .005 그리고 쌍 제외 7/7 양수). 단서: 주분석 D는 유성 위치쌍이 끌며(vs 무성 위치쌍 p = .60), 무성 위치쌍 대비 효과는 단어 통제(pooled + 상한 20, D +.092 [+.035, +.150])에서만 CI 0 제외 → 주장은 '후두 대립 정보의 비어두 약화'까지, '조음위치 안정'은 단어 통제 조건부

## 다음 세션 할 일 (2026-10-09 저녁 작성)
1. **KASELL 초록(10/16 마감)** 문장을 SUMMARY "해석(수정)" 6번 보고 문장으로 교체. 포스터 초안 Claude Doc §4 표를 t1c 수치로 갱신. 금지: "조음위치 안정·후두만 약화"를 단어 통제 없는 수치로 주장.
2. 교차 위치 해독(검토 ④-7, `code/run_cross_position.py` 신규): 어두 학습→비어두 평가와 그 반대, 같은 시행 그룹 분리·학습 폴드 PCA만 사용·n 맞춤. 단어 상한 20 적용. 예상 코딩 1h + 실행 20분(워커 병렬 틀 재사용).
3. 선택: pooled+단어 상한 20 설정에도 구간 순열 `zsh code/run_t1c_posperm.sh 100 2 pooled 20` (n 300이라 느림, ~30분).
4. 선택: 계획 분석 재실행(`run_t1_planning.py`, pre_valid 필터 반영됨, 3분) — 탐색적 절로만 유지.
5. 보류 유지: T15 재현(10/28 이후), 시간전개 창별 null, 앞뒤 음소 통제, 레포 비공개 전환(`gh repo edit g00dnew/Colab --visibility private`).
- 실행 메모: 특징 캐시 `results/t1c_cache/`(X_all.npy 등, 재추출 불필요). 워커 로그 `results/t1c_w*.log`, `t1c_pp*.log`. 분석 venv `../.venv`(nbformat 설치됨). 스모크: `T1C_SESSIONS=t12.2022.08.13,t12.2022.07.29 T1C_MIN_PER_CLASS=8 ../.venv/bin/python code/run_t1_controls.py all --perm 2 --perm-n 2 --n-resample 2`.

## 2026-10-09 밤 추가 (10/12 정리) — 교차 위치·계획 창 v2·단어 그룹 CV 확인
- **교차 위치 해독**(`code/run_cross_position.py`, results/cross_position_*.csv): 어두→비어두 전이는 후두·조음위치 모두 나쁨(전이 비율 .1~.5). 위치별 학습 ≥ 통합 학습(모든 자질). → 노벨티 문장에 "차이는 정보량, 패턴 불변성 아님" 단서 필수. 공학 함의: 위치별 모델링.
- **계획 창 v2**(`run_t1_planning.py` 개편: 발성만·시행 CV·단어 상한·구성 균형·pre_valid): pre300 D +.097 [+.042, +.153]. 탐색적 보조.
- **단어 그룹 CV 확인**(`zsh code/run_t1c_posperm.sh 120 4 pooled 0 word`, 40분): 재표집 D +.088 [+.047, +.139] 20/20, 순열 p = .008, 무성 위치쌍 대비도 p = .008. 단어 외우기 교란 배제. 초록 수치는 이 조건으로.
- 노벨티 한 줄(사용자 확정): "본 연구의 새로움은 음소 표상의 위치 효과 자체가 아니라, 단어 내 위치가 서로 다른 음성 자질의 신경표상에 차등적으로 영향을 미치는지를 직접 검증했다는 데 있다." 본문에서 '차등적' = 해독 가능한 정보량, 결과 방향(후두 감소·조음위치 유지) 명시.
- 초록 영문 초안 v2: `../abstract-kasell-draft.md` (p = .008 단어 그룹 조건으로 통일, held_out p = .005 병기 가능).

## 다음 세션 할 일 (10/12 갱신)
0. **참고문헌 정본 `../references.md`**(10/14 작성, 58건 14범주). 초록 제출 전 할 일 5개가 문서 끝에 있음 — 특히 **M절(영어 후두 음성학 5건)은 초록 첫 문장의 근거인데 현재 인용 0건**, B5 Khanna 2024·D6 Jossinger 2026은 원문 확인 필요, 한국어 선행연구 0건(KCI·RISS 손검색).
1. **KASELL 초록 10/16 마감**: `../abstract-kasell-draft.md`를 여자친구와 분량·인용 확인("sequential position" 선행 문헌 없으면 삭제) 후 제출.
2. 포스터/슬라이드 Doc §4 표를 t1c·cross_position·planning v2 수치로 교체. 금지: "조음위치 안정·후두만 약화"를 단어 통제 없는 수치로.
3. 보류: T15 재현(10/28 이후), 시간전개 창별 null, 앞뒤 음소 통제, 레포 비공개 전환.

## 2026-10-14: 한국어 초록 초안 검토
- 사용자 한국어 초안(제목 "Word-Position Effects on the Neural Decodability of Laryngeal and Place Contrasts in Attempted English Speech")의 수치 57.9→54.7 / 59.5→65.1 / p=.008 / 20회 재표집은 `t1c_resample_pairs_pooled_gword.csv`·`t1c_position_perm_pooled_gword.csv`와 일치. GitHub t12/ 결과 파일은 로컬과 해시 동일(10/12 커밋).
- 이 설정(발성 전체·단어 그룹 CV)의 세부: 후두 Δ −3.2%p [−6.3, −0.2]이나 쌍별로 P/B·K/G 감소, **T/D +5.1%p 증가**(T/D 어두 49.0% = 우연 수준). 조음위치 Δ +5.6%p, 4쌍 모두 상승. 어두 격차 +1.7%p [−1.7, +4.6], 비어두 격차 +10.4%p. D +8.8%p [+4.7, +13.9] 20/20. 순열 120회 중 0회 ≥ 관측 → p = 1/121(최솟값). 마찰 대조 62.2→57.6, D +10.2%p.
- → 결론은 "후두 저하"가 아니라 "비어두에서 조음위치 정보는 늘고 후두 정보는 늘지 않음"으로 써야 함.
- **정정**: SUMMARY.md·영문 초안의 "어두 T의 64% = 'to'"는 재현 안 됨. meta_all.pkl 재계산: 발성 전체 50%(2,532개 중), 발성 held_out 57%, 전 양식 53%.
- 수정 제안본 `../abstract-kasell-ko-v3-suggested.md`(본문 418어절, 키워드 5개, Crossref 확인 인용 5편). 원 초안은 수정하지 않음. 제출은 10/16.
