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
| B2 | ? Silva AB, et al. 2024. (이중언어 스페인어·영어 음절 디코딩) *Nat Biomed Eng*. | nature.com/articles/s41551-024-01207-5 | 조음 표상이 언어 간 공유 → 교차언어 아이디어 4의 전제 |
| B3 | ? Chang EF, Anumanchipalli GK. 2025. (스트리밍 brain-to-voice) *Nat Neurosci*. | nature.com/articles/s41593-025-01905-6 | |
| B4 | ? Silva AB, Chang EF. 2024. (음성 신경보철 리뷰) *Nat Rev Neurosci*. | — | "조음 표상 + 운동 계획 표상 협응"이 핵심 기제라는 정리 |
| B5 | **? Khanna AR, et al. 2024. (단일뉴런 수준 말 산출 요소) *Nature*.** | 서지 미확인 | core-question-v3.md §147에서 "운동피질이 다음 음소를 미리 담고 있다"의 근거로 인용. **계획 창(pre300) 결과를 쓸 때 반드시 원문 확인** |
| B6 | ? Stavisky SD, et al. 2019. (dorsal precentral 음소 분류) | — | 유성성 특정 분석 없음 |
| B7 | ? Wilson GH, et al. 2020. (hand knob 음소·자질 조직) | — | 유성성 특정 분석 없음 |

## C. 발화 양식 사다리 (실제 → 입모양 → 속삭임 → 상상). 경쟁 구역

| # | 서지 | 식별자 | 우리와의 차이 |
|---|---|---|---|
| C1 | ○ Zhao Z, et al. 2026. A neural architecture for imagined and overt speech motor dynamics. *Nat Neurosci*. | doi:10.1038/s41593-026-02456-0 · 프리프린트 bioRxiv 2025.05.27.656311 | **최대 경쟁자**. 조음 궤적 12차원 + log F0 1차원 복원. 만다린 음절(유성 대립 없음), 후두는 F0 1차원만, **양식 간 후두 비교 없음**, 자질 비교는 조음위치만. 출간본 Methods 학교 계정 재확인 필요 |
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

## M. 초록의 첫 문장이 의존하는데 목록에 없는 것 (영어 후두 음성학) ✦

초록 도입부 "영어 자음은 어디서나 조음위치를 유지하지만 후두 대립은 그렇지 않다"는 주장의 근거다. 지금 인용이 **하나도 없다**. 심사자가 가장 먼저 물을 자리이므로 여자친구와 확정해야 한다.

| # | 서지(제안, 대조 필요) | 무엇을 뒷받침하나 |
|---|---|---|
| M1 | ✦ Lisker L, Abramson AS. 1964. A cross-language study of voicing in initial stops. *Word* 20:384. | 어두 기식(VOT)이 영어 후두 대립의 주 단서 |
| M2 | ✦ Keating PA. 1984. Phonetic and phonological representation of stop consonant voicing. *Language* 60:286. | 음성 실현과 음운 표상의 분리 |
| M3 | ✦ Westbury JR, Keating PA. 1986. On the naturalness of stop consonant voicing. *J Linguistics* 22:145. | 어말 무성화의 공기역학적 설명 ↔ 계획 설명. **우리 pre300 결과의 긴장 상대** |
| M4 | ✦ Docherty GJ. 1992. *The Timing of Voicing in British English Obstruents*. | 어말 부분 무성화 실측 |
| M5 | ✦ Zue VW, Laferriere M. 1979. Acoustic study of medial /t,d/ in American English. *JASA* 66:1039. | 탄설음화. **T/D가 어중에서 최저(.530)인 결과의 직접 근거** |

## N. 참고문헌이 아닌 것 (소프트웨어 의존성)

`t12/code/speechBCI/LanguageModelDecoder/docs/papers.md` 등에 있는 arXiv 2012.05481, 2102.01547, 2106.05642, 1408.2873은 WeNet 언어모델 디코더의 자체 문서다. Willett 2023 코드를 그대로 받아 쓴 것이므로 우리 참고문헌에 넣지 않는다. 인용은 A1(코드 출처)로 갈음한다.

---

## 초록 제출 전 할 일 (10/16 마감)

1. **M절 5건을 여자친구와 확정.** 초록 첫 문장의 근거다. 영어 음성학은 그쪽 영역이므로 다른 정본이 있으면 교체한다.
2. **B5(Khanna 2024)와 D6(Jossinger 2026) 원문 확인.** 전자는 계획 창 해석, 후자는 한계 문단에 쓰므로 서지가 틀리면 안 된다.
3. **△ 15건 중 초록에 실제로 들어갈 것만 원문 대조.** 현재 초록 본문이 인용하는 것은 A1뿐이고, 나머지는 선행연구 문단용이다.
4. **"sequential position" 선행 문헌 결정.** 초록 2문장에 "음소 표상이 순차 위치에 따라 변한다"는 선행 언급이 있는데 해당 문헌을 찾지 못했다. 없으면 그 구절을 삭제한다.
5. KCI·RISS 손검색(검색어: 내적 발화, 무성 발화, 유성성, 후두 자질, 자질 기하, 음성 BCI). 한국어 선행연구가 목록에 0건이다.
