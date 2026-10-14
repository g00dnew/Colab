# 참고문헌 정본 (음성 BCI 후두 자질 연구 v4)

작성 2026-10-14. 출처: proposal-v4.md §14·§3, core-question-v3.md, trends-2026-10.md, proposal-v2.md, ideas-1to4-detailed.md, priorities.md, t12/results/SUMMARY.md를 전수 대조.

**확인 표시**
- ○ Crossref 또는 본문 직접 확인
- △ 2차 자료로만 확인 (초록 제출 전 원문 대조 필요)
- ? 프로젝트 문서에 인용은 있으나 완전한 서지가 없음 (내가 채운 서지는 대조 필요)
- ✦ 아직 목록에 없으나 현재 주장에 필요 (아래 L·M절)

**호칭 주의**: 문서 곳곳의 "Davis 2026", "Davis 랩"은 저자명이 아니라 UC Davis 연구진(= Srinivasan et al.)을 가리키는 약칭이다. 초록·발표에서는 Srinivasan으로 통일한다.

---

## A. 주 데이터셋과 그 원 논문 (우리가 직접 내려받아 쓴 것)

| # | 서지 | 식별자 | 이 연구에서의 역할 |
|---|---|---|---|
| A1 | ○ Willett FR, Kunz EM, Fan C, et al. 2023. A high-performance speech neuroprosthesis. *Nature* 620:1031. | doi:10.1038/s41586-023-06377-x · 데이터 Dryad 10.5061/dryad.x69p8czpq · 코드 github.com/fwillett/speechBCI | **주 데이터(T12)**. 24세션 9,680문장, area 6v·44. 사전학습 RNN을 CTC 강제정렬에 사용 |
| A2 | ○ Card NS, Wairagkar M, Iacobacci C, et al. 2024. An accurate and rapidly calibrating speech neuroprosthesis. *NEJM* 391:609. | Dryad 10.5061/dryad.dncjsxm85 | T15. 10/1 예비 분석(자질 전달량, 유성성 최저 .76). 양식 라벨 비공개. T15 재현은 10/28 이후 보류 |
| A3 | ○ Kunz EM, et al. 2025. Inner speech in motor cortex and implications for speech neuroprostheses. *Cell*. | doi:10.1016/j.cell.2025.06.015 · Dryad gf1vhhn1j | 내적 발화. "축소 복사본" 주장의 자질 수준 검정 대상. 재현 보류 |
| A4 | ○ He T, et al. 2025. VocalMind: A stereotactic EEG dataset for vocalized, mimed, and imagined speech in tonal language. *Sci Data*. | doi:10.1038/s41597-025-04741-2 · Zenodo 10.5281/zenodo.14696348 (CC BY-NC-ND) | 10/8 관문 실측 실패(상상=우연, 자질이 단어에서 분리 안 됨) → 주 데이터에서 제외, 보조 그림 1장 |
| A5 | ○ Srinivasan SS, et al. 2025/2026. (발화 양식·크기 디코딩, UC Davis) *bioRxiv* / *Nat Commun* 2026. | bioRxiv 10.1101/2025.05.30.657105 · Dryad 10.5061/dryad.2547d7x5w (CC0) | 무성·속삭임·보통·큰소리 4양식 94% 구별, 양식·음소 부분공간 직교. 단어 6개로 유성 최소대립쌍 없음 → 아이디어 2 상한 |

## B. 음성 BCI 디코딩 성능·음성 합성·단일뉴런

| # | 서지 | 식별자 | 비고 |
|---|---|---|---|
| B1 | ○ Wairagkar M, et al. 2025. An instantaneous voice-synthesis neuroprosthesis. *Nature*. | doi:10.1038/s41586-025-09127-3 | 언어모델 없이 직접 음성 합성 → 유성성 오류가 소리로 그대로 나옴. **우리 공학 함의의 착지점** |
| B1b | **○ Metzger SL, Littlejohn KT, Silva AB, et al. (senior: Chang EF). 2023. A high-performance neuroprosthesis for speech decoding and avatar control. *Nature* 620(7976):1037.** doi:10.1038/s41586-023-06443-4 | 2026-10-14 확인 | **A1(Willett 2023)과 같은 Nature 620권 동반 논문**. 253채널 고밀도 ECoG, 분당 78단어·WER 25%, 음성 합성+아바타. 목록에 없던 누락. 2023년 음성 BCI를 언급할 때 Willett 단독으로 쓰면 지적받는다 |
| B2 | ? Silva AB, et al. 2024. (이중언어 스페인어·영어 음절 디코딩) *Nat Biomed Eng*. | nature.com/articles/s41551-024-01207-5 | 조음 표상이 언어 간 공유 → 교차언어 아이디어 4의 전제 |
| B3 | ? Chang EF, Anumanchipalli GK. 2025. (스트리밍 brain-to-voice) *Nat Neurosci*. | nature.com/articles/s41593-025-01905-6 | |
| B4 | ? Silva AB, Chang EF. 2024. (음성 신경보철 리뷰) *Nat Rev Neurosci*. | — | "조음 표상 + 운동 계획 표상 협응"이 핵심 기제라는 정리 |
| B5 | **? Khanna AR, et al. 2024. (단일뉴런 수준 말 산출 요소) *Nature*.** | 서지 미확인 | core-question-v3.md §147에서 "운동피질이 다음 음소를 미리 담고 있다"의 근거로 인용. **계획 창(pre300) 결과를 쓸 때 반드시 원문 확인** |
| B6 | ? Stavisky SD, et al. 2019. (dorsal precentral 음소 분류) | — | 유성성 특정 분석 없음 |
| B7 | ? Wilson GH, et al. 2020. (hand knob 음소·자질 조직) | — | 유성성 특정 분석 없음 |

## C. 발화 양식 사다리 (실제 → 입모양 → 속삭임 → 상상). 경쟁 구역

| # | 서지 | 식별자 | 우리와의 차이 |
|---|---|---|---|
| C1 | ○ **Zhao Zehao**, Wang Z, Liu Y, Qian Y, Yin Y, Gao X, Yuan B, Tong SX, Tian X, Chen G, Li Y, Lu J, **Wu Jinsong**. 2026. A neural architecture for imagined and overt speech motor dynamics. *Nat Neurosci*. | doi:10.1038/s41593-026-02456-0 · 프리프린트 bioRxiv 2025.05.27.656311 (제목 *Neural hierarchy for coding articulatory dynamics in speech imagery and production*) | **2026-10-14 저자 확인: 1저자 Zehao Zhao, 책임 Jinsong Wu. "Zhou"가 아니라 Zhao다.** 음절 분류 상상 80.4%·발화 78.3%(우연 16.7%) | **최대 경쟁자**. 조음 궤적 12차원 + log F0 1차원 복원. 만다린 음절(유성 대립 없음), 후두는 F0 1차원만, **양식 간 후두 비교 없음**, 자질 비교는 조음위치만. 출간본 Methods 학교 계정 재확인 필요 |
| C2 | ○ Nie L, Lu Z. 2026. Shared and distinct temporal representations of Chinese words across imagined, silent, and overt speech. *bioRxiv*. | doi:10.64898/2026.08.25.747136 | EEG 중국어 단어 10개. 단어 기하가 입모양↔실제만 대응, 상상은 불안정. 우리 VocalMind 관문 결과와 일치. 자질 없음 |
| C3 | ○ Kwon J, Lee S, Park J, Lee S, Chung CK. 2025. Mimed speech as an intermediary state between overt and imagined speech production in an electrocorticography study. *Sci Rep*. | PMC12480242 (서울대 정천기 랩) | 입모양이 운동피질에서 실제와, 상측두회에서 상상과 유사. 자질 없음 |
| C4 | ○ Orpella J, Mantegna F, Oderbolz C, Assaneo MF, Poeppel D. 2026. Temporally structured motor and auditory representations in covert speech. *PNAS*. | PMC13578806 | MEG. 자음(조음 대비)은 운동피질, 모음(음향 대비)은 청각피질 → **자질 유형별 해리가 검출 가능함**을 보임. 우리는 자음 안에서 후두 vs 구강으로 한 단계 내려감 |
| C5 | ○ Maghsoudi A, Chillale RK, Shamma S. 2026. Relating the neural representations of vocalized, mimed, and imagined speech. | arXiv:2602.22597 | VocalMind 문장, 선형 스펙트로그램 복원이 양식 간 전이. 자질 없음 |
| C6 | ○ (저자 미확인) 2026. Mechanistic interpretability of brain-to-speech models across speech modes. | arXiv:2602.01247 | 양식 전이가 소수 뉴런 부분공간에 매개, 연속 다양체. 자질 없음 |
| C7 | ? Soroush PZ, et al. 2023. (발성 > 입모양 > 상상 중첩 위계, sEEG) *NeuroImage*. | — | 활성 수준, 자질 수준 아님. 데이터 비공개 추정 |
| C8 | ? Angrick M, et al. 2021. (속삭임·상상 발화 음성 합성, sEEG) *Comm Biol*. | — | 아이디어 2 배경 |
| C9 | ? Proix T, et al. 2022. (실제·상상 단어 ECoG) *Nat Commun*. | — | 제네바 협약 필요 → 보류 |

## D. 말 운동피질의 말소리·자질 표상 (ECoG·fMRI). 배경이자 "새롭지 않음"의 근거

| # | 서지 | 비고 |
|---|---|---|
| D1 | △ Bouchard KE, Mesgarani N, Johnson K, Chang EF. 2013. Functional organization of human sensorimotor cortex for speech articulation. *Nature* 495:327. | 조음기관별 지형, 1차 축 = 조음위치. 유성성은 약한 축. **"운동피질에서 유성성이 약하다"는 이미 알려진 사실** |
| D2 | △ Cheung C, Hamilton LS, Johnson K, Chang EF. 2016. The auditory representation of speech sounds in human motor cortex. *eLife* 5:e12577. | 같은 vSMC가 들을 때는 조음방법 중심 → 지각·산출 비대칭 |
| D3 | △ Lotte F, et al. 2015. Electrocorticographic representations of segmental features in continuous speech. *Front Hum Neurosci* 9:97. | 조음위치는 운동 네트워크 전반, 조음방법·유성성은 발화 후 청각 반응 지배 |
| D4 | ? Verwoert M, et al. 2025. (sEEG 15명 1,647접점, 조음·음향 표상 시간) *Comm Biol*. PMC11906857 | 조음 표상 피크 발화 200 ms 전, 음향 150 ms 전. 우리 pre300 창 해석의 배경 |
| D5 | ? Verwoert M, et al. 2022. (네덜란드어 단어 sEEG 데이터셋, OSF) | 공개. true-voice 언어 유성 쌍 있음 → 소리 조건 확장 후보 |
| D6 | **? Jossinger S, Arafat S, Diedrichsen J. 2026. (7T fMRI, 음절 산출 place × VOT) *bioRxiv*.** | ventral S1/M1 = 조음위치, **dorsal 감각운동피질 = VOT 타이밍**. → 우리의 "H_sampling 반대" 판정을 철회시킨 근거. T12 전극판 둘 다 ventral 6v라 dorsal 미표집. **한계 문단에 필수** |

## E. 후두 운동피질 해부·기능

| # | 서지 | 비고 |
|---|---|---|
| E1 | △ Dichter BK, Breshears JD, Leonard MK, Chang EF. 2018. The control of vocal pitch in human laryngeal motor cortex. *Cell* 174:21. | 후두 운동피질 배쪽·등쪽 두 곳, 등쪽이 음높이 |
| E2 | ? Eichert N, et al. 2020. (후두 운동피질 등쪽·배쪽 매핑) *Cereb Cortex*. | 전극 배치 함의(B편)의 해부학적 근거 |

## F. 인간 지각 혼동 행렬 (거울상 배경)

| # | 서지 | 비고 |
|---|---|---|
| F1 | △ Miller GA, Nicely PE. 1955. An analysis of perceptual confusions among some English consonants. *JASA* 27:338. | 자질별 정보 전달량의 원전. 사람 귀는 유성성·비음성이 가장 잘 들리고 조음위치가 먼저 사라짐 → 운동피질과 거울상 |
| F2 | ? Cutler A, et al. 2004. (영어 자음 혼동 행렬) | 행렬 디지털화 대상 |
| F3 | ? Phatak SA, Allen JB. 2007. (소음 속 자음 지각) | 행렬 디지털화 대상 |
| F4 | ? Tartter VC. 1989. (속삭임 음성의 지각) | 속삭임 아이디어의 지각 쪽 근거 |

## G. 음운론 이론 (예측의 출처)

| # | 서지 | 비고 |
|---|---|---|
| G1 | △ Clements GN. 1985. The geometry of phonological features. *Phonology Yearbook* 2:225. | 자질 기하. 후두 마디가 독립 가지 → 자연부류 검정의 이론 근거 |
| G2 | △ Halle M, Stevens KN. 1971. A note on laryngeal features. *MIT RLE QPR* 101. | 후두 자질 |
| G3 | △ Iverson GK, Salmons JC. 1995. Aspiration and laryngeal representation in Germanic. *Phonology* 12:369. | **laryngeal realism** 논쟁 |
| G4 | △ Honeybone P. 2005. Diachronic evidence in segmental phonology: the case of obstruent laryngeal specifications. In *The Internal Organization of Phonological Segments*. | laryngeal realism |
| G5 | △ Beckman J, Jessen M, Ringen C. 2013. Empirical evidence for laryngeal features: aspirating vs. true voice languages. *J Linguistics* 49:259. | laryngeal realism. **단 마찰음도 같이 떨어졌으므로 [voice] vs [spread glottis] 판정 근거로는 쓰지 못함(10/9 철회)** |
| G6 | △ Goldsmith J. 1976. *Autosegmental Phonology*. MIT 박사논문. | 성조가 분절과 다른 층 → 후두 마디 논리의 이론적 다리 |

## H. 심리언어학: 내적 발화의 구체성

| # | 서지 | 비고 |
|---|---|---|
| H1 | △ Oppenheim GM, Dell GS. 2008. Inner speech slips exhibit lexical bias, but not the phonemic similarity effect. *Cognition* 106:528. | 이론적 출발점 |
| H2 | △ Oppenheim GM, Dell GS. 2010. Motor movement matters: the flexible abstractness of inner speech. *Memory & Cognition* 38:1147. | "입을 움직이면 구체성이 돌아온다" |

## I. 조음 측정·조음음운론

| # | 서지 | 비고 |
|---|---|---|
| I1 | △ Narayanan S, et al. 2014. Real-time MRI and EMA databases: USC-TIMIT. *JASA* 136:1307. | Willett 2023이 음소 유사도 산출에 쓴 출처. 점진 모형 측정값의 원천 |
| I2 | ✦ Browman CP, Goldstein L. (조음음운론) | trends-2026-10.md §11에 이름만 등장, 서지 없음. 디코더 설계 원리로 채택됐다는 논의에 쓰려면 서지 확정 필요 |
| I3 | ? Lee Y, ..., Goldstein L, Narayanan S. 2025. (표면 EMG에서 조음 자질 예측) arXiv:2505.13814 | 음성학자가 자질 수준으로 EMG를 분석한 선례. A편 분석 틀의 모델 |

## J. 언어모델·뇌-언어 정렬·성조 BCI (배경, 직접 인용 가능성 낮음)

| # | 서지 | 비고 |
|---|---|---|
| J1 | ? Goldstein A, et al. 2025. (Whisper 임베딩과 일상 대화 뇌 활동) *Nat Hum Behav*. nature.com/articles/s41562-025-02105-9 | |
| J2 | ? LibriBrain / PNPL 2025. (MEG 50시간, 음소 분류 경진대회) arXiv:2506.10165 | |
| J3 | ? (저자 미확인) ICLR 2025. (sEEG 다참가자 통합 성조 디코딩) arXiv:2410.12866 | |
| J4 | ? Zhang et al. 2024. (중국어 성조 문장 brain-to-text) *Cell Reports*. | 데이터 요청 시 제공 → 보류 |

## K. 보조·후보 데이터셋 (미사용)

| # | 서지 | 비고 |
|---|---|---|
| K1 | ? Gaddy D. 2022. (표면 EMG 8채널, 소리 vs 무성 발화) Zenodo 7183877 | **말초판 동일 검정 가능**. 여자친구의 무성 발화 연구([[silent-speech-joint-research]])와 직결 |

## L. 방법론 (현재 코드가 쓰고 있으나 목록에 없음) ✦

초록·논문 Methods에 넣어야 한다. 서지는 내가 채운 것으로 대조 필요.

| # | 서지 | 어디에 쓰였나 |
|---|---|---|
| L1 | ✦ Graves A, Fernández S, Gomez F, Schmidhuber J. 2006. Connectionist temporal classification. *ICML*. | 음소 경계 = CTC 강제정렬. "CTC는 peaky하므로 경계는 ~80 ms 근사"라는 한계의 출처 |
| L2 | ✦ Ledoit O, Wolf M. 2004. A well-conditioned estimator for large-dimensional covariance matrices. *J Multivar Anal* 88:365. | 수축 LDA(`shrinkage="auto"`) |
| L3 | ✦ Pedregosa F, et al. 2011. Scikit-learn: Machine learning in Python. *JMLR* 12:2825. | LDA·PCA·StratifiedGroupKFold 전부 |
| L4 | ✦ Mantel N. 1967. The detection of disease clustering and a generalized regression approach. *Cancer Res* 27:209. | priorities.md의 "Mantel·부분상관" 계획. 현재 미실행 |

## M. 단어 위치·운율 위치의 음성학 (10/14 초록이 실제로 인용. 2026-10-14 전부 실재·내용 확인)

| # | 서지 | 확인 내용 | 역할 |
|---|---|---|---|
| M1 | ○ Mugler EM, Tate MC, Livescu K, Templer JW, Goldrick MA, Slutzky MW. 2018. Differential representation of articulatory gestures and phonemes in precentral and inferior frontal gyri. *J Neurosci* 38(46):9803. doi:10.1523/JNEUROSCI.1206-18.2018 | 원문 확인: "high gamma activity in pPCG and aPCG did vary with intraword position"(음소), 제스처는 불변. **단, 단일 CVC 단어·어두 vs 어말만(어중 없음)·참가자 2명·정확도 아닌 d′(음소 2.3±0.6 vs 제스처 0.2±0.3)** | 우리 위치 효과의 유일한 직접 선행. 초록 2문장·논의 모두에서 씀 |
| M2 | ○ Davidson L. 2016. Variability in the implementation of voicing in American English obstruents. *J Phonetics* 54:35. | 실재 확인. 어두·휴지 뒤 폐쇄음 토큰 중 파열 전 유성성이 있는 것은 1/4 미만 | VOT·폐쇄 유성성·선행 모음 길이·탄설음화를 한 건으로 묶음. 내가 앞서 제안한 M1~M5(Lisker & Abramson 등 5건)를 **이 한 건이 대체**. 더 최신·미국영어·연속발화 |
| M3 | ○ Fougeron C, Keating PA. 1997. Articulatory strengthening at edges of prosodic domains. *JASA* 101(6):3728. | 원문 확인: 영역 초 자음·영역 말 모음이 덜 축소된 설-구개 접촉 | 영역 초 강화. 두 대립 유형 모두 어두 우위를 예측 → 우리 반대 방향 결과의 **반증 대상** |
| M4 | ○ Cho T, Keating PA. 2009. Effects of initial position versus prominence in English. *J Phonetics* 37:466. | 실재·권·면 확인(Taehong Cho, 한양대 / Keating, UCLA) | 위와 같음. **1저자가 한국 소속 → KASELL 청중에 유리** |

**철회**: 내가 10/14 오전에 제안한 Lisker & Abramson 1964, Keating 1984, Westbury & Keating 1986, Docherty 1992, Zue & Laferriere 1979는 **더 이상 필요하지 않다**. Davidson 2016이 같은 내용을 더 낫게 담는다. 단 탄설음화를 따로 논증해야 할 때만 Zue & Laferriere 1979(*JASA* 66:1039)를 되살린다.

## N. 참고문헌이 아닌 것 (소프트웨어 의존성)

`t12/code/speechBCI/LanguageModelDecoder/docs/papers.md` 등에 있는 arXiv 2012.05481, 2102.01547, 2106.05642, 1408.2873은 WeNet 언어모델 디코더의 자체 문서다. Willett 2023 코드를 그대로 받아 쓴 것이므로 우리 참고문헌에 넣지 않는다. 인용은 A1(코드 출처)로 갈음한다.

---

## 초록 제출 전 할 일 (10/16 마감) — 10/14 갱신

**해결됨**: M절 공백(초록 첫 문장 근거)은 Mugler 2018·Davidson 2016·Fougeron & Keating 1997·Cho & Keating 2009 네 건으로 채워졌고 2026-10-14에 전부 실재·내용을 확인했다. "sequential position 선행 문헌 없음" 문제도 Mugler 2018로 해결.

**남은 것**
1. **Bouchard et al. 2013(D1)을 초록 1문장에 추가.** 운동피질이 조음 중심으로 조직된다는 가장 표준적인 인용인데 현재 초록은 Willett 2023만 든다. 이 분야 심사자가 가장 먼저 찾는 자리다.
2. **Mugler 2018의 본래 주장과 우리 전이 결과의 긴장을 논의에 넣기.** 그들의 결론은 "제스처는 위치 불변, 음소는 위치 의존 → 운동피질 단위는 제스처"다. 우리 교차 위치 전이는 **두 대립 유형 모두 전이가 나빴으므로** 단순한 제스처 불변성과 충돌한다. 한 문장으로 선점해야 한다.
3. **한계 문단이 없다.** 참가자 1명, 전극이 ventral 6v에 한정(→ D6 Jossinger 2026, E1 Dichter 2018로 dorsal 후두 운동피질 미표집 명시), 효과 작음.
4. **B5 Khanna 2024** 원문 확인(계획 창 해석에 쓸 때만 필요. 10/14 초록은 계획 창을 쓰지 않으므로 보류 가능).
5. **한국어 선행연구 0건.** KCI·RISS 손검색(내적 발화, 무성 발화, 유성성, 후두 자질, 자질 기하, 음성 BCI). Cho & Keating 2009의 1저자가 한양대 소속인 점은 KASELL에서 쓸 수 있다.
6. 선택: **Metzger 2023(B1b)**과 **Zhao 2026(C1)** 인용 여부 결정. 전자는 2023년 Nature 동반 논문, 후자는 최근접 경쟁자이고 자질 비교를 조음위치에만 했다는 점이 우리 공백 논증에 쓸 수 있다.

## 10/14 수치 재확인 기록 (단어 그룹 CV·재표집 추정량)

초록의 네 수치는 `t12/results/t1c_resample_pairs_pooled_gword.csv`와 정확히 일치한다: 후두 57.9% → 54.7%, 조음위치 59.5% → 65.1%, 차이 8.8pp, 20/20 양수, 순열 p = .008(120회).

**단, 두 가지를 바로잡아야 한다.**

- **"resampling range 4.7–13.9"는 범위가 아니라 2.5~97.5 백분위다.** 실제 최소~최대는 3.9~15.5. 표현을 "95% range of 20 resamplings" 또는 백분위로 고칠 것.
- **🔴 /t–d/가 효과를 끌지 않는다.** 이 조건의 쌍별 변화량(비어두−어두, pp)은 P/B −7.7, K/G −7.0, **T/D +5.1**(상승), F/V 0.0, TH/DH −9.2, B/D +10.6, D/G +6.0, P/T +3.5, T/K +2.3. 즉 후두 하락은 **P/B·K/G가 끌고 T/D는 반대 방향**이다. T/D를 빼면 D가 8.8 → **12.9pp**로 커진다(쌍 제외 8종 모두 양수, 6.5~12.9pp). → 내가 10/12 초록 초안과 NEXT_STEPS·메모리에 적은 "/t–d/가 후두 하락의 대부분을 기여"는 **held_out 조건의 이야기이고 이 조건에서는 틀렸다.** 탄설음화(Davidson 2016 목록 중 하나)도 효과를 끄는 쌍에는 적용되지 않는다. 한계 문단에 T/D를 쓰지 말 것.
- **마찰음 대조군을 되살릴 수 있다.** 같은 조건에서 F/V·TH/DH는 62.2% → 57.6%, 조음위치 대비 D = +10.2pp [5.2, 15.8], 20/20 양수. 초록이 파열음 3쌍만 쓰면서 기식 특이성을 배제하지 못하는데, 이 한 줄을 넣으면 "결국 기식 아니냐"는 질문을 막는다.
