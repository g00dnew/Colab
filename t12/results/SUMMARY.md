# T12 후두 자질 분석 요약

생성 시각: 2026-10-09 11:33

경로: /Users/hojaechoi/Documents/research/speech-bci-confusion/t12/results

## 반드시 함께 읽을 유보 조항

1. 음성 자질표는 `verified=False`. 사전등록 전 확정 금지.
   `spread_glottis` 열은 표준 기술이 아니라 laryngeal realism 쪽 가설 라벨이다.
2. 강제 정렬은 CTC 기반이고 CTC는 peaky하므로 음소 경계는 ~80 ms 수준의
   근사다. 창 선택(`seg` 160 ms vs `rnn_center` 80 ms)에 따라 수치가 바뀐다.
3. 사전학습 RNN은 competitionData의 train 파티션으로 학습됐다. 그 세션의
   정렬 PER은 0.1% 수준이고 test는 15~34% 수준이다. 주 보고값은 test 파티션.
4. `input_layer_from != session` 인 세션은 입력층을 차용했으므로 정렬이 근사다.
   T3(양식)는 양쪽의 정렬 품질을 맞춰야 하며, 각 행에 `mean_per`을 붙였다.
5. 분류기 전처리의 PCA·스케일러는 (2026-10-09 수정 후) 교차검증 학습 폴드 안에서만 적합한다. 이전 버전의 전역 PCA 수치는 results_pre_pcafix/ 에 보존.
   (순열검정의 귀무가설은 라벨 교환이므로 유효하지만, 정확도 추정에는
   약한 transduction이 들어 있다).
6. tuningTasks 고립 음소는 모두 어두 'C + AA' 발성 조건이므로 T1(위치)과
   T3(양식)에 대해서는 말할 수 없다.

## tuningTasks 인벤토리

```
                       file         task modality  n_trials  n_blocks  n_cues  go_bins_mean
   t12.2022.04.21_orofacial    orofacial  unknown       680        10      34          49.5
    t12.2022.04.21_phonemes     phonemes  unknown       640         8      40          85.1
    t12.2022.04.26_phonemes     phonemes  unknown       800        10      40          85.0
t12.2022.05.03_fiftyWordSet fiftyWordSet  unknown      1020        10      51          99.4
```

## competitionData 인벤토리

```
         partition modality  n_sessions  n_trials  total_sec
competitionHoldOut    vocal          15      1200     7160.7
              test nonvocal           4       160      881.0
              test    vocal          20       720     4408.4
             train nonvocal           4      1960    10665.2
             train    vocal          20      6840    44710.2
```

## Stage 1 — 39음소 분류 (전건전성)

```
array    acc  chance  n_per_class
   6v 0.6451  0.0256           36
   44 0.1019  0.0256           36
  all 0.4910  0.0256           36
```

## Stage 1 — 최소대립쌍 분리도 (go 구간 전체)

어레이 x 대립 유형 평균 정확도:

```
array  laryngeal  place
   44     0.5648 0.5461
   6v     0.7645 0.9065
  all     0.6692 0.8048
```

쌍별:

```
array  pair  contrast    manner  n_per_class    acc  dprime  p_perm
   6v   P/B laryngeal      stop           36 0.7483  1.3370  0.0050
   6v   T/D laryngeal      stop           36 0.7458  1.3231  0.0100
   6v   K/G laryngeal      stop           36 0.6110  0.5628  0.0746
   6v   F/V laryngeal fricative           36 0.7758  1.5946  0.0050
   6v   S/Z laryngeal fricative           36 0.7610  1.4263  0.0050
   6v SH/ZH laryngeal fricative           36 0.8683  2.2355  0.0050
   6v CH/JH laryngeal affricate           36 0.7963  1.6755  0.0050
   6v TH/DH laryngeal fricative           36 0.8093  1.7502  0.0050
   6v   P/T     place      stop           36 0.9223  2.8669  0.0050
   6v   T/K     place      stop           36 0.7652  1.4420  0.0050
   6v   B/D     place      stop           36 0.9427  3.3092  0.0050
   6v   D/G     place      stop           36 0.8352  1.9609  0.0050
   6v   F/S     place fricative           36 0.9598  3.4663  0.0050
   6v  F/TH     place fricative           36 0.9873  4.9301  0.0050
   6v  S/SH     place fricative           36 0.8350  1.9393  0.0050
   6v   V/Z     place fricative           36 0.9462  3.2480  0.0050
   6v   M/N     place     nasal           36 0.9644  3.5977  0.0050
   44   P/B laryngeal      stop           36 0.6051  0.5342  0.0846
   44   T/D laryngeal      stop           36 0.5414  0.2089  0.1095
   44   K/G laryngeal      stop           36 0.4003 -0.5129  0.9353
   44   F/V laryngeal fricative           36 0.5732  0.3753  0.2438
   44   S/Z laryngeal fricative           36 0.5110  0.0559  0.2836
   44 SH/ZH laryngeal fricative           36 0.7248  1.1835  0.0050
   44 CH/JH laryngeal affricate           36 0.5672  0.3475  0.0846
   44 TH/DH laryngeal fricative           36 0.5951  0.4771  0.0597
   44   P/T     place      stop           36 0.6229  0.6289  0.0398
   44   T/K     place      stop           36 0.4713 -0.1400  0.6070
   44   B/D     place      stop           36 0.6149  0.5856  0.0597
   44   D/G     place      stop           36 0.4590 -0.2019  0.9204
   44   F/S     place fricative           36 0.5204  0.1042  0.5224
   44  F/TH     place fricative           36 0.5599  0.3001  0.1791
   44  S/SH     place fricative           36 0.5533  0.2647  0.1741
   44   V/Z     place fricative           36 0.5374  0.1905  0.5721
   44   M/N     place     nasal           36 0.5763  0.3872  0.4378
  all   P/B laryngeal      stop           36 0.6881  0.9844  0.0149
  all   T/D laryngeal      stop           36 0.6596  0.8213  0.0199
  all   K/G laryngeal      stop           36 0.4738 -0.1322  0.9154
  all   F/V laryngeal fricative           36 0.6230  0.6421  0.0448
  all   S/Z laryngeal fricative           36 0.6181  0.6039  0.0100
  all SH/ZH laryngeal fricative           36 0.7956  1.6445  0.0050
  all CH/JH laryngeal affricate           36 0.7326  1.2897  0.0050
  all TH/DH laryngeal fricative           36 0.7626  1.4273  0.0050
  all   P/T     place      stop           36 0.8588  2.2503  0.0050
  all   T/K     place      stop           36 0.6211  0.6217  0.0299
  all   B/D     place      stop           36 0.8858  2.4598  0.0050
  all   D/G     place      stop           36 0.6890  0.9974  0.0100
  all   F/S     place fricative           36 0.8572  2.1266  0.0050
  all  F/TH     place fricative           36 0.9497  3.3977  0.0050
  all  S/SH     place fricative           36 0.6431  0.7312  0.0299
  all   V/Z     place fricative           36 0.7865  1.5928  0.0050
  all   M/N     place     nasal           36 0.9518  3.3073  0.0050
```

## Stage 1 — 창 비교 (area 6v, 지속시간 교란 점검)

`onset_*` 창은 audioEnvelope로 추정한 발화 onset 기준 고정 길이이므로
모든 쌍에 같은 길이가 적용된다(조음방법별 지속시간 차이 통제).

```
session_set        window  laryngeal  place
 audio_0426    go_0_500ms     0.7150 0.8989
 audio_0426        go_all     0.8000 0.9114
 audio_0426 onset_0_300ms     0.7625 0.8728
 audio_0426 onset_0_500ms     0.7769 0.9000
     pooled    go_0_500ms     0.7114 0.9068
     pooled        go_all     0.7645 0.9065
     pooled onset_0_300ms     0.6925 0.8516
     pooled onset_0_500ms     0.7188 0.8801
```

## Stage 1 — 후두쌍 조음방법 집계 (area 6v)

```
       window array session_set        manner   mean     sd  n_pairs
   go_0_500ms    6v  audio_0426     affricate 0.8125    NaN        1
   go_0_500ms    6v  audio_0426     fricative 0.6781 0.0502        4
   go_0_500ms    6v  audio_0426          stop 0.7317 0.1401        3
   go_0_500ms    6v      pooled     affricate 0.7461    NaN        1
   go_0_500ms    6v      pooled     fricative 0.7067 0.0704        4
   go_0_500ms    6v      pooled          stop 0.7061 0.0703        3
       go_all    6v  audio_0426     affricate 0.8375    NaN        1
       go_all    6v  audio_0426     fricative 0.8275 0.1240        4
       go_all    6v  audio_0426          stop 0.7508 0.1067        3
       go_all    6v      pooled     affricate 0.7963    NaN        1
       go_all    6v      pooled     fricative 0.8036 0.0476        4
       go_all    6v      pooled          stop 0.7017 0.0786        3
onset_0_300ms    6v  audio_0426     affricate 0.8325    NaN        1
onset_0_300ms    6v  audio_0426     fricative 0.7694 0.1445        4
onset_0_300ms    6v  audio_0426          stop 0.7300 0.1054        3
onset_0_300ms    6v      pooled     affricate 0.6462    NaN        1
onset_0_300ms    6v      pooled     fricative 0.7242 0.1056        4
onset_0_300ms    6v      pooled          stop 0.6658 0.0756        3
onset_0_500ms    6v  audio_0426     affricate 0.8525    NaN        1
onset_0_500ms    6v  audio_0426     fricative 0.7612 0.1342        4
onset_0_500ms    6v  audio_0426          stop 0.7725 0.0738        3
onset_0_500ms    6v      pooled     affricate 0.6874    NaN        1
onset_0_500ms    6v      pooled     fricative 0.7505 0.0625        4
onset_0_500ms    6v      pooled          stop 0.6870 0.0650        3
   go_0_500ms    6v  audio_0426 ALL_laryngeal 0.7150 0.0946        8
   go_0_500ms    6v      pooled ALL_laryngeal 0.7114 0.0611        8
       go_all    6v  audio_0426 ALL_laryngeal 0.8000 0.1073        8
       go_all    6v      pooled ALL_laryngeal 0.7645 0.0738        8
onset_0_300ms    6v  audio_0426 ALL_laryngeal 0.7625 0.1153        8
onset_0_300ms    6v      pooled ALL_laryngeal 0.6925 0.0872        8
onset_0_500ms    6v  audio_0426 ALL_laryngeal 0.7769 0.1012        8
onset_0_500ms    6v      pooled ALL_laryngeal 0.7188 0.0635        8
```

## Stage 1 — T2 유표성

```
array  n_pairs                                 pairs  mean_diff_voiced_minus_voiceless  n_voiced_farther  wilcoxon_stat  wilcoxon_p             verdict
   6v        8 P/B;T/D;K/G;F/V;S/Z;SH/ZH;CH/JH;TH/DH                           0.00109                 5            9.0     0.25000 판정 불가(유성-무성 차이 비유의)
   44        8 P/B;T/D;K/G;F/V;S/Z;SH/ZH;CH/JH;TH/DH                           0.00004                 5           17.0     0.94531 판정 불가(유성-무성 차이 비유의)
  all        8 P/B;T/D;K/G;F/V;S/Z;SH/ZH;CH/JH;TH/DH                           0.00057                 4           14.0     0.64062 판정 불가(유성-무성 차이 비유의)
```

## Stage 2 — 자음 x 위치 구간 수

S/Z, SH/ZH의 어두 칸이 거의 비어 있다. 영어에서 Z, ZH는 어두에
거의 나타나지 않으므로 이 두 쌍은 **구조적으로** T1 어두 조건에
기여할 수 없다. T1은 P/B, T/D, K/G, F/V, TH/DH, CH/JH 6쌍에 의존한다.

```
phoneme  final  initial  medial
      B     78     2144    1090
     CH    354      246     253
      D   3874     2260    1437
     DH    273     5687     292
      F    533     1605     673
      G    139     1517     523
     HH      0     2840      61
     JH    171      510     417
      K   1578     2228    2308
      L   2386     1573    3838
      M   1679     2227    1652
      N   4378     1524    6723
     NG   1720        0     586
      P    514     1705    1538
      R   2523     1216    3845
      S   3060     2635    2846
     SH     63      424     498
      T   8157     3036    4391
     TH    160      715     293
      V   2360      417    1385
      W      0     4059     651
      Y      0     1819     405
      Z   4335        9     607
     ZH     20        4     649
```

## Stage 2 — T1 위치 (n 맞춤, pooled)

```
array position  laryngeal  place
   6v    final     0.6292 0.7037
   6v  initial     0.6892 0.6823
   6v   medial     0.6187 0.6523
```

## Stage 2 — T1 위치 (n 맞춤, test 전용)

```
array position  laryngeal  place
   6v    final     0.5409 0.7166
   6v  initial     0.6268 0.6789
   6v   medial     0.5686 0.5910
```

## Stage 2 — T1 위치 (설탄음화 통제: 모음간 T/D 제외)

```
array position  laryngeal  place
   6v    final     0.6292 0.7037
   6v  initial     0.6892 0.6823
   6v   medial     0.6278 0.6598
```

## Stage 2 — T1 위치 (6v_sup vs 6v_inf)

```
 array position  laryngeal  place
6v_inf    final     0.6309 0.6858
6v_inf  initial     0.6764 0.6482
6v_inf   medial     0.6100 0.6378
6v_sup    final     0.5778 0.6005
6v_sup  initial     0.6321 0.6480
6v_sup   medial     0.5726 0.6137
```

## Stage 2 — T1 위치 x 대립유형 상호작용 (부트스트랩 95% CI)

```
array             position  n_laryngeal_pairs  n_place_pairs  laryngeal_acc  place_acc  gap_place_minus_laryngeal  gap_ci_lo  gap_ci_hi excluded_pairs                run excluded
   6v              initial                  6              8         0.6892     0.6823                    -0.0069    -0.0806     0.0612            NaN   all_pairs_pooled     none
   6v               medial                  6              8         0.6187     0.6523                     0.0335    -0.0523     0.1035            NaN   all_pairs_pooled     none
   6v                final                  6              8         0.6292     0.7037                     0.0745     0.0201     0.1260            NaN   all_pairs_pooled     none
   6v DoD(final - initial)                  6              8            NaN        NaN                     0.0813    -0.0052     0.1706            NaN   all_pairs_pooled     none
   6v              initial                  5              8         0.6928     0.6823                    -0.0104    -0.0932     0.0676            T/D   all_pairs_pooled      T/D
   6v               medial                  5              8         0.6335     0.6523                     0.0188    -0.0786     0.0919            T/D   all_pairs_pooled      T/D
   6v                final                  5              8         0.6460     0.7037                     0.0577     0.0092     0.1029            T/D   all_pairs_pooled      T/D
   6v DoD(final - initial)                  5              8            NaN        NaN                     0.0681    -0.0213     0.1594            T/D   all_pairs_pooled      T/D
   6v              initial                  2              4         0.6268     0.6789                     0.0520    -0.0874     0.1871            NaN all_pairs_testonly     none
   6v               medial                  2              4         0.5686     0.5910                     0.0224    -0.0040     0.0485            NaN all_pairs_testonly     none
   6v                final                  2              4         0.5409     0.7166                     0.1757     0.1007     0.2506            NaN all_pairs_testonly     none
   6v DoD(final - initial)                  2              4            NaN        NaN                     0.1236    -0.0304     0.2851            NaN all_pairs_testonly     none
   6v              initial                  1              4         0.5460     0.6789                     0.1329     0.0315     0.2029            T/D all_pairs_testonly      T/D
   6v               medial                  1              4         0.5833     0.5910                     0.0077    -0.0093     0.0247            T/D all_pairs_testonly      T/D
   6v                final                  1              4         0.5927     0.7166                     0.1239     0.0914     0.1572            T/D all_pairs_testonly      T/D
   6v DoD(final - initial)                  1              4            NaN        NaN                    -0.0089    -0.0894     0.0929            T/D all_pairs_testonly      T/D
   6v              initial                  6              8         0.6892     0.6823                    -0.0069    -0.0806     0.0612            NaN    flapctrl_pooled     none
   6v               medial                  6              8         0.6278     0.6598                     0.0320    -0.0493     0.0936            NaN    flapctrl_pooled     none
   6v                final                  6              8         0.6292     0.7037                     0.0745     0.0201     0.1260            NaN    flapctrl_pooled     none
   6v DoD(final - initial)                  6              8            NaN        NaN                     0.0813    -0.0052     0.1706            NaN    flapctrl_pooled     none
   6v              initial                  5              8         0.6928     0.6823                    -0.0104    -0.0932     0.0676            T/D    flapctrl_pooled      T/D
   6v               medial                  5              8         0.6335     0.6598                     0.0263    -0.0690     0.0955            T/D    flapctrl_pooled      T/D
   6v                final                  5              8         0.6460     0.7037                     0.0577     0.0092     0.1029            T/D    flapctrl_pooled      T/D
   6v DoD(final - initial)                  5              8            NaN        NaN                     0.0681    -0.0213     0.1594            T/D    flapctrl_pooled      T/D
```

## Stage 2 — T3 양식 (비교군별, 정렬 품질·날짜 맞춤)

```
          comparator  array modality       acc         mean_per       
                                     laryngeal  place laryngeal  place
        primary_0729     6v nonvocal    0.6047 0.6168    0.2636 0.2639
        primary_0729     6v    vocal    0.6481 0.7168    0.1858 0.1858
        primary_0729 6v_inf nonvocal    0.5451 0.6354    0.2629 0.2646
        primary_0729 6v_inf    vocal    0.6530 0.6755    0.1860 0.1858
        primary_0729 6v_sup nonvocal    0.5557 0.6008    0.2631 0.2656
        primary_0729 6v_sup    vocal    0.6011 0.6722    0.1859 0.1859
primary_0729_initial     6v nonvocal    0.7000 0.7465    0.2656 0.2681
primary_0729_initial     6v    vocal    0.7877 0.7240    0.1861 0.1862
primary_0729_initial 6v_inf nonvocal    0.6490 0.6784    0.2665 0.2660
primary_0729_initial 6v_inf    vocal    0.7240 0.6800    0.1860 0.1859
primary_0729_initial 6v_sup nonvocal    0.6585 0.6608    0.2650 0.2637
primary_0729_initial 6v_sup    vocal    0.7114 0.7052    0.1851 0.1861
 primary_0729_no0825     6v nonvocal    0.6140 0.6650    0.2530 0.2534
 primary_0729_no0825     6v    vocal    0.6481 0.7168    0.1858 0.1858
 primary_0729_no0825 6v_inf nonvocal    0.5665 0.6292    0.2523 0.2539
 primary_0729_no0825 6v_inf    vocal    0.6530 0.6755    0.1860 0.1858
 primary_0729_no0825 6v_sup nonvocal    0.5494 0.5865    0.2529 0.2542
 primary_0729_no0825 6v_sup    vocal    0.6011 0.6722    0.1859 0.1859
 secondary_datematch     6v nonvocal    0.5715 0.6397    0.2663 0.2667
 secondary_datematch     6v    vocal    0.6179 0.6476    0.1758 0.1746
 secondary_datematch 6v_inf nonvocal    0.6084 0.5922    0.2660 0.2637
 secondary_datematch 6v_inf    vocal    0.5601 0.6421    0.1745 0.1748
 secondary_datematch 6v_sup nonvocal    0.5284 0.5938    0.2640 0.2657
 secondary_datematch 6v_sup    vocal    0.5731 0.6106    0.1746 0.1745
   tertiary_allvocal     6v nonvocal    0.6478 0.6948    0.2641 0.2644
   tertiary_allvocal     6v    vocal    0.6243 0.6951    0.1873 0.1865
   tertiary_allvocal 6v_inf nonvocal    0.6210 0.6628    0.2636 0.2642
   tertiary_allvocal 6v_inf    vocal    0.6202 0.6751    0.1877 0.1872
   tertiary_allvocal 6v_sup nonvocal    0.5923 0.6254    0.2639 0.2639
   tertiary_allvocal 6v_sup    vocal    0.5976 0.6328    0.1874 0.1875
```

## Stage 2 — T3 격차와 DoD (부트스트랩 95% CI)

```
 array              modality  n_laryngeal_pairs  n_place_pairs  laryngeal_acc  place_acc  gap_place_minus_laryngeal  gap_ci_lo  gap_ci_hi  excluded_pairs           comparator positions
    6v                 vocal                  7              9         0.6481     0.7168                     0.0687    -0.0135     0.1418             NaN         primary_0729       all
    6v              nonvocal                  7              9         0.6047     0.6168                     0.0121    -0.0783     0.0921             NaN         primary_0729       all
    6v DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0566    -0.1763     0.0583             NaN         primary_0729       all
6v_sup                 vocal                  7              9         0.6011     0.6722                     0.0711     0.0011     0.1353             NaN         primary_0729       all
6v_sup              nonvocal                  7              9         0.5557     0.6008                     0.0452    -0.0171     0.1110             NaN         primary_0729       all
6v_sup DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0260    -0.1159     0.0690             NaN         primary_0729       all
6v_inf                 vocal                  7              9         0.6530     0.6755                     0.0225    -0.0517     0.0909             NaN         primary_0729       all
6v_inf              nonvocal                  7              9         0.5451     0.6354                     0.0903    -0.0007     0.1747             NaN         primary_0729       all
6v_inf DoD(nonvocal - vocal)                  7              9            NaN        NaN                     0.0678    -0.0468     0.1792             NaN         primary_0729       all
    6v                 vocal                  7              9         0.6481     0.7168                     0.0687    -0.0135     0.1418             NaN  primary_0729_no0825       all
    6v              nonvocal                  7              9         0.6140     0.6650                     0.0510    -0.0062     0.1169             NaN  primary_0729_no0825       all
    6v DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0177    -0.1137     0.0844             NaN  primary_0729_no0825       all
6v_sup                 vocal                  7              9         0.6011     0.6722                     0.0711     0.0011     0.1353             NaN  primary_0729_no0825       all
6v_sup              nonvocal                  7              9         0.5494     0.5865                     0.0372    -0.0286     0.1051             NaN  primary_0729_no0825       all
6v_sup DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0340    -0.1261     0.0647             NaN  primary_0729_no0825       all
6v_inf                 vocal                  7              9         0.6530     0.6755                     0.0225    -0.0517     0.0909             NaN  primary_0729_no0825       all
6v_inf              nonvocal                  7              9         0.5665     0.6292                     0.0626     0.0114     0.1166             NaN  primary_0729_no0825       all
6v_inf DoD(nonvocal - vocal)                  7              9            NaN        NaN                     0.0401    -0.0455     0.1298             NaN  primary_0729_no0825       all
    6v                 vocal                  6              8         0.6179     0.6476                     0.0296    -0.0149     0.0735             NaN  secondary_datematch       all
    6v              nonvocal                  6              8         0.5715     0.6397                     0.0682     0.0226     0.1168             NaN  secondary_datematch       all
    6v DoD(nonvocal - vocal)                  6              8            NaN        NaN                     0.0386    -0.0251     0.1044             NaN  secondary_datematch       all
6v_sup                 vocal                  6              8         0.5731     0.6106                     0.0374    -0.0547     0.1261             NaN  secondary_datematch       all
6v_sup              nonvocal                  6              8         0.5284     0.5938                     0.0655     0.0001     0.1295             NaN  secondary_datematch       all
6v_sup DoD(nonvocal - vocal)                  6              8            NaN        NaN                     0.0281    -0.0833     0.1393             NaN  secondary_datematch       all
6v_inf                 vocal                  6              8         0.5601     0.6421                     0.0819     0.0146     0.1444             NaN  secondary_datematch       all
6v_inf              nonvocal                  6              8         0.6084     0.5922                    -0.0162    -0.1415     0.0950             NaN  secondary_datematch       all
6v_inf DoD(nonvocal - vocal)                  6              8            NaN        NaN                    -0.0981    -0.2411     0.0310             NaN  secondary_datematch       all
    6v                 vocal                  8              9         0.6243     0.6951                     0.0708     0.0241     0.1175             NaN    tertiary_allvocal       all
    6v              nonvocal                  8              9         0.6478     0.6948                     0.0470    -0.0253     0.1071             NaN    tertiary_allvocal       all
    6v DoD(nonvocal - vocal)                  8              9            NaN        NaN                    -0.0237    -0.1084     0.0546             NaN    tertiary_allvocal       all
6v_sup                 vocal                  8              9         0.5976     0.6328                     0.0352    -0.0245     0.0922             NaN    tertiary_allvocal       all
6v_sup              nonvocal                  8              9         0.5923     0.6254                     0.0331    -0.0283     0.0869             NaN    tertiary_allvocal       all
6v_sup DoD(nonvocal - vocal)                  8              9            NaN        NaN                    -0.0021    -0.0862     0.0797             NaN    tertiary_allvocal       all
6v_inf                 vocal                  8              9         0.6202     0.6751                     0.0549     0.0013     0.1034             NaN    tertiary_allvocal       all
6v_inf              nonvocal                  8              9         0.6210     0.6628                     0.0418    -0.0184     0.1003             NaN    tertiary_allvocal       all
6v_inf DoD(nonvocal - vocal)                  8              9            NaN        NaN                    -0.0131    -0.0913     0.0663             NaN    tertiary_allvocal       all
    6v                 vocal                  4              7         0.7877     0.7240                    -0.0637    -0.1151    -0.0113             NaN primary_0729_initial   initial
    6v              nonvocal                  4              7         0.7000     0.7465                     0.0466    -0.1164     0.1732             NaN primary_0729_initial   initial
    6v DoD(nonvocal - vocal)                  4              7            NaN        NaN                     0.1102    -0.0583     0.2515             NaN primary_0729_initial   initial
6v_sup                 vocal                  4              7         0.7114     0.7052                    -0.0062    -0.0732     0.0674             NaN primary_0729_initial   initial
6v_sup              nonvocal                  4              7         0.6585     0.6608                     0.0024    -0.0902     0.0953             NaN primary_0729_initial   initial
6v_sup DoD(nonvocal - vocal)                  4              7            NaN        NaN                     0.0086    -0.1120     0.1238             NaN primary_0729_initial   initial
6v_inf                 vocal                  4              7         0.7240     0.6800                    -0.0441    -0.1461     0.0582             NaN primary_0729_initial   initial
6v_inf              nonvocal                  4              7         0.6490     0.6784                     0.0294    -0.0353     0.0898             NaN primary_0729_initial   initial
6v_inf DoD(nonvocal - vocal)                  4              7            NaN        NaN                     0.0734    -0.0471     0.1934             NaN primary_0729_initial   initial
```

## Stage 2 — T3 쌍별 낙폭 (발성 - 무성, 재표집 95% CI)

```
array  pair  contrast    manner  n_per_class  acc_vocal  acc_nonvocal    drop  drop_ci_lo  drop_ci_hi          comparator
   6v   P/B laryngeal      stop           65     0.6756        0.6004  0.0752     -0.0119      0.1787        primary_0729
   6v   T/D laryngeal      stop          140     0.6210        0.5583  0.0627      0.0098      0.1468        primary_0729
   6v   K/G laryngeal      stop           35     0.6371        0.5754  0.0618     -0.1521      0.2164        primary_0729
   6v   F/V laryngeal fricative           69     0.5863        0.5916 -0.0052     -0.0928      0.0848        primary_0729
   6v   S/Z laryngeal fricative           86     0.6153        0.5892  0.0261     -0.0460      0.1164        primary_0729
   6v SH/ZH laryngeal fricative           12     0.5662        0.5432  0.0230     -0.2455      0.2205        primary_0729
   6v TH/DH laryngeal fricative           28     0.8027        0.7568  0.0459     -0.1112      0.1364        primary_0729
   6v   P/T     place      stop           74     0.7999        0.6952  0.1048      0.0227      0.1858        primary_0729
   6v   T/K     place      stop          132     0.7048        0.6430  0.0619     -0.0049      0.1017        primary_0729
   6v   B/D     place      stop           65     0.8023        0.7048  0.0975     -0.0315      0.1712        primary_0729
   6v   D/G     place      stop           35     0.6768        0.6079  0.0689     -0.0755      0.2627        primary_0729
   6v   F/S     place fricative           69     0.6326        0.6010  0.0316     -0.0597      0.1254        primary_0729
   6v  F/TH     place fricative           28     0.7886        0.7040  0.0846     -0.0092      0.2475        primary_0729
   6v  S/SH     place fricative           12     0.6158        0.5570  0.0588     -0.1832      0.2752        primary_0729
   6v   V/Z     place fricative           77     0.6881        0.6467  0.0415     -0.0114      0.1386        primary_0729
   6v   M/N     place     nasal          131     0.7565        0.7005  0.0560     -0.0010      0.1139        primary_0729
   6v   P/B laryngeal      stop           45     0.6708        0.5697  0.1011     -0.0169      0.2046 secondary_datematch
   6v   T/D laryngeal      stop           91     0.5513        0.5469  0.0043     -0.0931      0.0771 secondary_datematch
   6v   K/G laryngeal      stop           29     0.5026        0.5297 -0.0270     -0.1436      0.0685 secondary_datematch
   6v   F/V laryngeal fricative           48     0.5925        0.5608  0.0317     -0.0559      0.1072 secondary_datematch
   6v   S/Z laryngeal fricative           39     0.6478        0.5700  0.0778     -0.0242      0.2005 secondary_datematch
   6v TH/DH laryngeal fricative           15     0.6192        0.6892 -0.0700     -0.2929      0.1850 secondary_datematch
   6v   P/T     place      stop           45     0.6997        0.6725  0.0272     -0.1061      0.1672 secondary_datematch
   6v   T/K     place      stop           86     0.6314        0.6033  0.0280     -0.0789      0.0977 secondary_datematch
   6v   B/D     place      stop           46     0.7132        0.6923  0.0209     -0.0896      0.1676 secondary_datematch
   6v   D/G     place      stop           29     0.6148        0.5774  0.0374     -0.1637      0.1843 secondary_datematch
   6v   F/S     place fricative           48     0.5927        0.5829  0.0098     -0.1283      0.1293 secondary_datematch
   6v  F/TH     place fricative           15     0.6800        0.6917 -0.0117     -0.2708      0.2517 secondary_datematch
   6v   V/Z     place fricative           39     0.7206        0.6288  0.0918      0.0096      0.1828 secondary_datematch
   6v   M/N     place     nasal          100     0.6754        0.6764 -0.0010     -0.1019      0.0726 secondary_datematch
   6v   P/B laryngeal      stop          259     0.6441        0.6380  0.0061     -0.0374      0.0411   tertiary_allvocal
   6v   T/D laryngeal      stop          300     0.5631        0.5600  0.0030     -0.0689      0.0468   tertiary_allvocal
   6v   K/G laryngeal      stop          167     0.6029        0.6338 -0.0309     -0.1320      0.0300   tertiary_allvocal
   6v   F/V laryngeal fricative          256     0.6042        0.6344 -0.0302     -0.0577      0.0177   tertiary_allvocal
   6v   S/Z laryngeal fricative          300     0.6255        0.5996  0.0259     -0.0107      0.0785   tertiary_allvocal
   6v SH/ZH laryngeal fricative           41     0.5440        0.5622 -0.0182     -0.1331      0.1011   tertiary_allvocal
   6v CH/JH laryngeal affricate           47     0.7100        0.7118 -0.0018     -0.0718      0.0623   tertiary_allvocal
   6v TH/DH laryngeal fricative           93     0.7291        0.7830 -0.0540     -0.1010      0.0376   tertiary_allvocal
   6v   P/T     place      stop          300     0.7751        0.7462  0.0289      0.0049      0.0643   tertiary_allvocal
   6v   T/K     place      stop          300     0.6664        0.6642  0.0022     -0.0236      0.0304   tertiary_allvocal
   6v   B/D     place      stop          259     0.7581        0.7508  0.0074     -0.0272      0.0404   tertiary_allvocal
   6v   D/G     place      stop          167     0.6790        0.6642  0.0148     -0.0457      0.0830   tertiary_allvocal
   6v   F/S     place fricative          256     0.5986        0.6223 -0.0238     -0.0576      0.0177   tertiary_allvocal
   6v  F/TH     place fricative           93     0.7376        0.7652 -0.0276     -0.1038      0.0271   tertiary_allvocal
   6v  S/SH     place fricative           57     0.6656        0.6081  0.0575     -0.0334      0.1547   tertiary_allvocal
   6v   V/Z     place fricative          300     0.7128        0.6994  0.0134     -0.0184      0.0424   tertiary_allvocal
   6v   M/N     place     nasal          300     0.6950        0.7138 -0.0188     -0.0588      0.0217   tertiary_allvocal
```

## Stage 2 — T3 자연부류 검정 (후두 마디가 한 부류로 약해지는가)

```
                                           group  n_pairs  mean_drop  sd_drop  uniformity_p          comparator
                                       laryngeal        7     0.0413   0.0282           NaN        primary_0729
                               place (non-nasal)        8     0.0687   0.0258           NaN        primary_0729
                                           nasal        1     0.0560      NaN           NaN        primary_0729
                                  laryngeal:stop        3     0.0666   0.0075           NaN        primary_0729
                             laryngeal:fricative        4     0.0224   0.0210           NaN        primary_0729
UNIFORMITY TEST (laryngeal SD vs random 7 of 16)        7     0.0282   0.0277        0.5467        primary_0729
                                       laryngeal        6     0.0197   0.0642           NaN secondary_datematch
                               place (non-nasal)        7     0.0291   0.0319           NaN secondary_datematch
                                           nasal        1    -0.0010      NaN           NaN secondary_datematch
                                  laryngeal:stop        3     0.0261   0.0668           NaN secondary_datematch
                             laryngeal:fricative        3     0.0132   0.0756           NaN secondary_datematch
UNIFORMITY TEST (laryngeal SD vs random 6 of 14)        6     0.0642   0.0445        0.9646 secondary_datematch
                                       laryngeal        8    -0.0125   0.0256           NaN   tertiary_allvocal
                               place (non-nasal)        8     0.0091   0.0274           NaN   tertiary_allvocal
                                           nasal        1    -0.0188      NaN           NaN   tertiary_allvocal
                                  laryngeal:stop        3    -0.0073   0.0205           NaN   tertiary_allvocal
                             laryngeal:fricative        4    -0.0191   0.0335           NaN   tertiary_allvocal
                             laryngeal:affricate        1    -0.0018      NaN           NaN   tertiary_allvocal
UNIFORMITY TEST (laryngeal SD vs random 8 of 17)        8     0.0256   0.0269        0.3885   tertiary_allvocal
```

균일성 검정: 후두쌍 낙폭의 표준편차가 전체 쌍에서 같은 수를 무작위로 뽑았을 때의 표준편차보다 작은가(= 하나의 자연부류처럼 균일하게 약해지는가). p는 무작위 SD가 관측 SD 이하일 비율이므로 작을수록 '균일하다'는 증거다.

## Stage 2 — T2 위치별 유표성

```
 array position  voiced  voiceless
    6v    final 0.00444    0.00281
    6v  initial 0.00345    0.00365
    6v   medial 0.00241    0.00209
6v_inf    final 0.00466    0.00393
6v_inf  initial 0.00315    0.00433
6v_inf   medial 0.00250    0.00264
6v_sup    final 0.00423    0.00169
6v_sup  initial 0.00374    0.00297
6v_sup   medial 0.00231    0.00153
```

## Stage 2 — Miller-Nicely 자질 전달량

```
feature     6v  6v_inf  6v_sup
 manner 0.0207  0.0161  0.0091
  nasal 0.0123  0.0157  0.0019
  place 0.0405  0.0292  0.0220
voicing 0.0130  0.0096  0.0042
```

## Stage 2 — score 필터 탈락 집계

```
modality position_in_word  n_total  n_kept  n_dropped  frac_dropped  score_min
nonvocal            final    11837    8880       2957        0.2498        0.5
nonvocal          initial    11837    8255       3582        0.3026        0.5
nonvocal           medial    17650   10301       7349        0.4164        0.5
nonvocal              sil    12666   12530        136        0.0107        0.5
nonvocal           single      829     757         72        0.0869        0.5
   vocal            final    45619   44613       1006        0.0221        0.5
   vocal          initial    45619   44480       1139        0.0250        0.5
   vocal           medial    74154   70879       3275        0.0442        0.5
   vocal              sil    48491   48409         82        0.0017        0.5
   vocal           single     2834    2822         12        0.0042        0.5
```

## 정렬 품질 (세션별 PER, 음소 수 가중)

```
       session partition modality    per  n_trials input_layer_from  borrowed
t12.2022.06.23      test nonvocal 0.2831        40   t12.2022.06.21      True
t12.2022.06.23     train nonvocal 0.2394       480   t12.2022.06.21      True
t12.2022.08.18      test nonvocal 0.2615        40   t12.2022.08.13      True
t12.2022.08.18     train nonvocal 0.2450       440   t12.2022.08.13      True
t12.2022.08.23      test nonvocal 0.2326        40   t12.2022.08.13      True
t12.2022.08.23     train nonvocal 0.2771       520   t12.2022.08.13      True
t12.2022.08.25      test nonvocal 0.3354        40   t12.2022.08.13      True
t12.2022.08.25     train nonvocal 0.2972       520   t12.2022.08.13      True
t12.2022.04.28      test    vocal 0.3266        20   t12.2022.04.28     False
t12.2022.04.28     train    vocal 0.0136       280   t12.2022.04.28     False
t12.2022.05.05      test    vocal 0.2736        20   t12.2022.05.05     False
t12.2022.05.05     train    vocal 0.0251       360   t12.2022.05.05     False
t12.2022.05.17      test    vocal 0.2316        20   t12.2022.05.17     False
t12.2022.05.17     train    vocal 0.0074       420   t12.2022.05.17     False
t12.2022.05.19      test    vocal 0.3218        20   t12.2022.05.19     False
t12.2022.05.19     train    vocal 0.0005       180   t12.2022.05.19     False
t12.2022.05.24      test    vocal 0.1426        40   t12.2022.05.24     False
t12.2022.05.24     train    vocal 0.0008       360   t12.2022.05.24     False
t12.2022.05.26      test    vocal 0.1412        40   t12.2022.05.26     False
t12.2022.05.26     train    vocal 0.0010       360   t12.2022.05.26     False
t12.2022.06.02      test    vocal 0.1495        40   t12.2022.06.02     False
t12.2022.06.02     train    vocal 0.0017       400   t12.2022.06.02     False
t12.2022.06.07      test    vocal 0.2005        40   t12.2022.06.07     False
t12.2022.06.07     train    vocal 0.0013       360   t12.2022.06.07     False
t12.2022.06.14      test    vocal 0.1673        40   t12.2022.06.14     False
t12.2022.06.14     train    vocal 0.0006       320   t12.2022.06.14     False
t12.2022.06.16      test    vocal 0.2075        40   t12.2022.06.16     False
t12.2022.06.16     train    vocal 0.0005       320   t12.2022.06.16     False
t12.2022.06.21      test    vocal 0.1890        40   t12.2022.06.21     False
t12.2022.06.21     train    vocal 0.0012       320   t12.2022.06.21     False
t12.2022.06.28      test    vocal 0.1924        40   t12.2022.06.28     False
t12.2022.06.28     train    vocal 0.0025       360   t12.2022.06.28     False
t12.2022.07.05      test    vocal 0.1597        40   t12.2022.07.05     False
t12.2022.07.05     train    vocal 0.0004       360   t12.2022.07.05     False
t12.2022.07.14      test    vocal 0.1624        40   t12.2022.07.14     False
t12.2022.07.14     train    vocal 0.0016       400   t12.2022.07.14     False
t12.2022.07.21      test    vocal 0.1688        40   t12.2022.07.21     False
t12.2022.07.21     train    vocal 0.0010       400   t12.2022.07.21     False
t12.2022.07.27      test    vocal 0.1566        40   t12.2022.07.27     False
t12.2022.07.27     train    vocal 0.0008       400   t12.2022.07.27     False
t12.2022.07.29      test    vocal 0.2053        40   t12.2022.07.27      True
t12.2022.07.29     train    vocal 0.1825       200   t12.2022.07.27      True
t12.2022.08.02      test    vocal 0.1696        40   t12.2022.08.02     False
t12.2022.08.02     train    vocal 0.0015       400   t12.2022.08.02     False
t12.2022.08.11      test    vocal 0.1668        40   t12.2022.08.11     False
t12.2022.08.11     train    vocal 0.0011       320   t12.2022.08.11     False
t12.2022.08.13      test    vocal 0.1487        40   t12.2022.08.13     False
t12.2022.08.13     train    vocal 0.0005       320   t12.2022.08.13     False
```

## Stage 2 실행 정보

```json
{
  "window_mode": "seg",
  "score_min": 0.5,
  "n_perm_primary": 100,
  "sessions_included": [
    "t12.2022.04.28",
    "t12.2022.05.05",
    "t12.2022.05.17",
    "t12.2022.05.19",
    "t12.2022.05.24",
    "t12.2022.05.26",
    "t12.2022.06.02",
    "t12.2022.06.07",
    "t12.2022.06.14",
    "t12.2022.06.16",
    "t12.2022.06.21",
    "t12.2022.06.23",
    "t12.2022.06.28",
    "t12.2022.07.05",
    "t12.2022.07.14",
    "t12.2022.07.21",
    "t12.2022.07.27",
    "t12.2022.07.29",
    "t12.2022.08.02",
    "t12.2022.08.11",
    "t12.2022.08.13",
    "t12.2022.08.18",
    "t12.2022.08.23",
    "t12.2022.08.25"
  ],
  "n_segments": 251926,
  "modality_counts": {
    "vocal": 211203,
    "nonvocal": 40723
  },
  "partition_counts": {
    "train": 232583,
    "test": 19343
  },
  "position_counts": {
    "medial": 81180,
    "sil": 60939,
    "final": 53493,
    "initial": 52735,
    "single": 3579
  },
  "arrays": [
    "6v",
    "6v_sup",
    "6v_inf"
  ],
  "max_per_class": 300,
  "structural_exclusion": "S/Z, SH/ZH는 영어에서 어두에 거의 나타나지 않아 T1 어두 조건에 기여할 수 없다",
  "caveat": "CTC는 peaky하므로 경계는 ~80 ms 근사. train 파티션은 RNN 학습에 쓰였으므로 디코더와 독립이 아니다. 모음간 T/D는 설탄음화로 통제."
}
```

## 그림

- `stage1_markedness.png`
- `stage1_pair_decoding.png`
- `stage1_timecourse.png`
- `stage2_feature_transmission.png`
- `stage2_markedness.png`
- `stage2_position.png`
- `stage2_position_synthcheck.png`

## 2026-10-09 추가 (외부 검토 반영)

### PCA 폴드 내 적합 수정 — 전후 비교 (수정 전 = results_pre_pcafix/)
| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 39음소 6v / all | 0.651 / 0.507 | 0.645 / 0.491 |
| 6v 후두쌍 / 위치쌍 (go_all) | 0.739 / 0.880 | 0.764 / 0.906 |
| 6v 유의 쌍 (p<.05) | 16/17 | 16/17 |
| 6v_sup 후두/위치/39 | 0.609/0.791/0.379 | 0.621/0.821/0.377 |
| 6v_inf 후두/위치/39 | 0.765/0.869/0.607 | 0.781/0.898/0.608 |
| T1 전체쌍 DoD(어말−어두) pooled | 0.060 [−0.029, 0.144] | 0.081 [−0.005, 0.171] |
| T1 전체쌍 test 전용 DoD | 0.108 [−0.021, 0.245] | 0.124 [−0.030, 0.285] |
| T3 primary DoD(입모양−발성) | −0.016 [−0.151, 0.120] | −0.057 [−0.176, 0.058] |
| T3 다른 대조군 DoD | 모두 CI 0 포함 | 모두 CI 0 포함 |
| 자연부류 균일성 p (primary) | 0.164 | 0.547 |
질적 결론 변화 없음. 수치는 ±0.03 안에서 움직였다.

### T1′ 파열음 전용 1차 검정 (stage2_t1_stops_primary.csv, stage2_t1_stops_dod.csv)
후두쌍 P/B·T/D·K/G, 위치 대조 P/T·T/K·B/D·D/G, 6v, 쌍별 n 300, 어두 vs 비어두.
| 실행 | 후두 어두→비어두 | 위치 어두→비어두 | 격차 어두 | 격차 비어두 | DoD |
|---|---|---|---|---|---|
| pooled | 0.664→0.599 | 0.680→0.725 | 0.016 [−.03,.06] | 0.125 [.07,.18] | **0.109 [0.036, 0.183]** |
| test 전용 | 0.690→0.581 | 0.737→0.738 | 0.047 [−.03,.12] | 0.157 [.08,.22] | **0.110 [0.004, 0.214]** |
| 탄설음 제외 | 0.664→0.620 | 0.680→0.731 | 0.016 | 0.111 | **0.094 [0.035, 0.151]** |
| 3수준 어말−어두 | | | | | **0.099 [0.016, 0.178]** |
| **마찰음 대조 F/V·TH/DH** | 0.741→0.639 | (같은 위치쌍) | −0.061 | 0.086 | **0.147 [0.036, 0.257]** |
해석: 파열음만 두면 상호작용이 유의하다. 그러나 기식 대립이 없는 마찰음 후두쌍도 같은 크기로 떨어지므로 위치 효과는 **기식([spread glottis]) 특이적이 아니라 후두 대립 일반**이다. 조음위치 정보는 어말로 가며 오히려 증가(0.68→0.73), 후두 정보는 감소. 보고 문장: "조음위치 정보는 단어 끝까지 안정적이나 후두 대립 정보는 단어 끝에서 체계적으로 약해진다." realism 판정 근거로 쓰지 않는다. 쌍 3개 부트스트랩 CI는 거칠고 위치 라벨 순열검정은 미실시.

### 자질 정보의 시간 전개 (stage1_timecourse*.csv, figures/stage1_timecourse.png)
200 ms 창·40 ms 간격, 6v. 지연 과제라 go 전부터 정보가 있어 잠복기는 해석하지 않는다.
- 발화 onset 기준(04.26): 조음위치 피크 0 ms(0.89), 파열음 후두 −40~0 ms(0.78), **마찰음 후두 +160 ms(0.79)**.
- go 기준(pooled): 조음위치 +440 ms(0.87), 파열음 후두 +240 ms(0.68), 마찰음 후두 +800 ms(0.72).
- 파열음 유성 정보는 조음위치와 같은 시점에 피크(별도의 초기 창 없음). 마찰음 유성 정보는 약 160 ms 늦다(마찰 중 지속 발성과 일치).

### 계획 vs 실행 (stage2_t1_planning.csv, stage2_t1_planning_dod.csv)
T1′(파열음 후두쌍 vs 파열음 위치쌍, 어두 vs 비어두)을 조음 구간(seg), 직전 100 ms, 전 300~100 ms 세 창에서 반복. 세 창 모두 DoD CI가 0을 제외(.109 / .123 / .119). 마찰음 대조군도 동일 방향. 위치×자질 비대칭은 조음 시작 300 ms 전 신호에 이미 존재 → 실행 산물이 아니라 계획 표상의 성질로 읽힘. 주의: 전 창은 앞 음소 실행과 겹침; 파열음 효과는 계획 창에서 T/D가 주도; n=1.

### T1′ 통제 분석 (2026-10-09 저녁, 외부 검토 ①②③④ 반영) — t1c_*.csv, 사전 고정 t1_controls_prereg.md
실행 `code/run_t1c_parallel.sh`(특징 mmap 캐시 + 워커 5개, 16분) + `code/run_t1c_posperm.sh`. 표 생성 `code/t1c_report.py`. 이전 T1′은 새 코드로 정확도 차 0.0000 재현(회귀 확인).

**무엇을 바꿨나**: 문장 시행 그룹 CV(StratifiedGroupKFold, 블록·세션·단어 그룹 민감도) / 발성 세션·held_out(test+미학습 07.29) 주분석 / 비어두의 어중:어말 구성을 두 클래스에서 동일하게 표집 / 단어 상한 20·단어 그룹 CV / 80 ms 고정 중심창 / 유성·무성 위치쌍 분리 / 쌍 대응 유지 재표집·쌍 라벨 정확 순열·구간 수준 위치 라벨 순열·20회 독립 재표집·쌍 하나씩 제외. 효과량 D = Δ_place − Δ_lar, Δ = acc(비어두) − acc(어두).

**주분석 쌍별** (발성·held_out·시행 CV·구성 균형, n/class = 쌍별 위치 간 맞춤)

| 대립 | 쌍 | 어두 | 비어두 | Δ | n | 비어두 구성(어말:어중) |
|---|---|---|---|---|---|---|
| 후두 | P/B | .708 | .635 | −.072 | 80 | 3:77 |
| 후두 | T/D | .753 | .542 | −.211 | 182 | 146:35 |
| 후두 | K/G | .673 | .513 | −.160 | 37 | 8:29 |
| 후두(마찰) | F/V | .647 | .423 | −.223 | 15 | 7:7 |
| 후두(마찰) | TH/DH | .737 | .604 | −.133 | 30 | 11:19 |
| 위치(무성) | P/T | .788 | .676 | −.112 | 129 | 34:94 |
| 위치(무성) | T/K | .766 | .631 | −.134 | 186 | 87:98 |
| 위치(유성) | B/D | .634 | .663 | +.028 | 80 | 3:77 |
| 위치(유성) | D/G | .644 | .702 | +.058 | 37 | 8:29 |

범주 Δ: 후두 파열 −.148, 후두 마찰 −.178, **무성 위치 −.123**, **유성 위치 +.043**, 위치 전체 −.040.

**상호작용 D — 실행별** (stops_lar vs place_all; [쌍 대응 유지 부트스트랩 95% CI], p = 쌍 라벨 정확 순열, 최소 .029)

| 실행 | Δ_lar | Δ_place | D [95% CI] | p_label |
|---|---|---|---|---|
| **주분석** 발성·held_out·시행CV·구성균형 | −.148 | −.040 | **+.108** [−.003, +.212] | .114 |
| test 파티션만 | −.088 | −.025 | +.063 [−.075, +.206] | .257 |
| 발성 전체(pooled) | −.115 | −.029 | +.086 [+.016, +.159] | .114 |
| held_out, 구성 비균형 | −.125 | +.010 | +.135 [+.012, +.255] | .143 |
| 블록 그룹 CV | −.154 | −.011 | +.144 [+.036, +.255] | .057 |
| 세션 그룹 CV | −.147 | −.019 | +.129 [+.016, +.253] | .086 |
| 단어 상한 20 (held_out) | −.145 | −.005 | +.139 [−.009, +.241] | .114 |
| **pooled + 단어 상한 20** | −.063 | +.022 | **+.085** [+.036, +.130] | .029 |
| 단어 그룹 CV (held_out) | −.134 | +.050 | +.184 [+.052, +.347] | .057 |
| pooled + 단어 그룹 CV | −.084 | +.065 | +.149 [+.057, +.225] | .029 |
| 80 ms 고정 중심창 | −.102 | −.001 | +.101 [−.019, +.243] | .171 |
| 후두 6쌍 vs 위치 9쌍 (held_out / pooled) | −.148 / −.115 | −.026 / −.049 | +.122 [+.017, +.214] / +.066 [+.006, +.128] | .057 / .143 |
| 이전 설정 + 시행 CV만 | −.062 | +.045 | +.107 [+.031, +.185] | .057 |
| 이전 T1′ 재현(무작위 CV, 모든 양식) | −.064 | +.045 | +.109 [+.025, +.190] | .057 |

**유성·무성 위치쌍 분리** (stops_lar 대비 D)

| 실행 | vs 무성 위치 P/T·T/K | vs 유성 위치 B/D·D/G |
|---|---|---|
| 주분석 | +.025 [−.039, +.088] | +.191 [+.115, +.254] |
| 발성 pooled | +.021 [−.014, +.074] | +.150 [+.114, +.204] |
| **pooled + 단어 상한 20** | **+.092** [+.035, +.150] | **+.077** [+.025, +.130] |
| pooled + 단어 그룹 CV | +.126 [+.031, +.187] | +.172 [+.076, +.264] |
| 단어 상한 20 (held_out) | +.130 [−.009, +.222] | +.149 [−.003, +.273] |
| 80 ms 중심창 | +.007 [−.050, +.099] | +.195 [+.052, +.338] |

**3수준 분리** (pooled, 시행 CV; 어두/어중/어말): 어두→어말 D = +.135 [+.030, +.239] (vs 무성 위치 +.050 [−.014, +.119], vs 유성 위치 +.220 [+.156, +.289]); 어두→어중 D = +.046 [−.062, +.172]. 유성 위치쌍의 상승은 **어말 특이적**(B/D 어두 .572 → 어중 .673 → 어말 .745; D/G .624 → .625 → .762). 후두쌍은 어중부터 떨어짐(T/D .705 → .530 → .572).

**주분석 통계 보강**
- 20회 독립 균형 재표집: D = +.087, 반복 2.5–97.5% [+.034, +.159], **20회 모두 D > 0**. vs 무성 위치 +.010(55%만 양수), vs 유성 위치 +.164(100%).
- 쌍 하나씩 제외(7쌍 모두): D = +.075(D/G 제외) ~ +.146(P/B 제외), 전부 양수. T/D·K/G·B/D·D/G 제외 시 CI가 0 포함.
- 구간 수준 위치 라벨 순열(4표집 평균 통계량, 200회, 같은 시행 그룹 CV): D_obs = +.067(4표집 평균, CV 2반복), null 평균 −.003·SD .025·95% +.043, **p = .005**(200회 중 0회 ≥ 관측). vs 무성 위치쌍 D −.009, p = .60; vs 유성 위치쌍 +.143, p = .005; 마찰음 후두 vs 위치 +.054, p = .18. (단일 표집판 v1은 D +.012, p .37로 표집 잡음에 취약 → results/t1c_posperm_v1/ 보존. 관측 D의 세 추정치 .067/.087/.108은 CV 반복·표집 수 차이이며 재표집 범위 안)
- 그룹 CV 무결성: 모든 실행 50/50 폴드 사용, 학습·시험 그룹 겹침 0(assert). 최소 시험 폴드 n은 held_out 3~5(세션 CV)로 작다.
- 순열검정(p_perm)은 주분석에만 적용했고 나머지 실행은 CI만 보고한다.

**해석(수정)**
1. 후두 대립 정보의 어두→비어두 감소는 **모든 통제에서 방향 유지**(Δ_lar −.06~−.18, 재표집 20/20). 시행·블록·세션 그룹 CV로 바꿔도 이전 T1′의 D(.109)는 거의 그대로(.107) → 교차검증 누출이 결과를 만든 것은 아니다.
2. 그러나 "조음위치 정보는 단어 끝까지 안정적"은 **그대로 쓸 수 없다**. 단어 통제 전에는 무성 위치쌍(P/T·T/K)이 후두쌍과 같은 폭으로 떨어지고(주분석 −.123 vs −.148, D ≈ 0), 상승은 유성 위치쌍(B/D·D/G)의 어말에서만 나온다(검토 ④-2·④-3 지적 적중).
3. **단어 정체 통제가 핵심 교란**이었다. 어두 T의 64%가 'to'라 어두 T 포함 쌍(T/D·P/T·T/K)의 어두 정확도가 부풀려져 있었다. (음소·위치·단어)별 상한 20을 걸면(pooled) 무성 위치쌍의 하락이 사라지고(Δ_vl +.029) 후두쌍은 여전히 떨어져(Δ_lar −.063), **무성·유성 위치쌍 모두에 대해 D ≈ +.08~+.09(CI 0 제외, p_label .029=최소값)**. 단어 그룹 CV도 같은 방향. → 가장 엄격한 설정에서 원래 주장(후두 선택적 약화)이 유성성과 무관하게 성립하되, 효과는 이전 보고(.109)보다 작다(.085).
4. 주분석(held_out)은 n이 작아(K/G 37, F/V 15) 쌍 대응 CI가 0을 스친다(+.108 [−.003, +.212]). 발성 pooled는 CI 0 제외(+.086 [+.016, +.159]).
5. 사전 고정 판정 기준(구간 순열 p<.05 그리고 쌍 제외 전부 D>0): **지지**(구간 순열 p = .005 그리고 쌍 제외 7/7 양수). 단서: 주분석 D는 유성 위치쌍이 끌며(vs 무성 위치쌍 p = .60), 무성 위치쌍 대비 효과는 단어 통제(pooled + 상한 20, D +.092 [+.035, +.150])에서만 CI 0 제외 → 주장은 '후두 대립 정보의 비어두 약화'까지, '조음위치 안정'은 단어 통제 조건부
6. 보고 문장(안): "후두 대립 정보는 비어두에서 체계적으로 약해진다. 조음위치 대립은 단어 정체를 통제하면 유지된다. 효과량은 작고(D ≈ 0.09) 쌍 수가 적어 통계적 확정은 제한적이며, 선택적 후두 약화인지 무성 파열음의 비어두 축약(기식 상실·미파열)인지는 현 자료로 구분하지 못한다." 금지: "조음위치는 안정, 후두만 약화"를 단어 통제 없는 수치로 주장.
7. 남은 한계: 블록 z-score가 평가 블록 통계를 씀(라벨 무관), 앞·뒤 음소 통제 없음, 입모양 조건 T1′ 미실시, 계획 창·시간전개는 탐색적.
