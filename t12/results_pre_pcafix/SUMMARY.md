# T12 후두 자질 분석 요약

생성 시각: 2026-10-09 02:41

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
5. 분류기 전처리의 PCA는 라벨과 무관하게 전체 시행에 한 번 적합한다
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
   6v 0.6513  0.0256           36
   44 0.1009  0.0256           36
  all 0.5067  0.0256           36
```

## Stage 1 — 최소대립쌍 분리도 (go 구간 전체)

어레이 x 대립 유형 평균 정확도:

```
array  laryngeal  place
   44     0.5495 0.5396
   6v     0.7388 0.8802
  all     0.6639 0.8130
```

쌍별:

```
array  pair  contrast    manner  n_per_class    acc  dprime  p_perm
   6v   P/B laryngeal      stop           36 0.7582  1.3993  0.0020
   6v   T/D laryngeal      stop           36 0.6954  1.0171  0.0180
   6v   K/G laryngeal      stop           36 0.5998  0.5055  0.1277
   6v   F/V laryngeal fricative           36 0.7387  1.2878  0.0020
   6v   S/Z laryngeal fricative           36 0.7405  1.2929  0.0020
   6v SH/ZH laryngeal fricative           36 0.8500  2.0669  0.0020
   6v CH/JH laryngeal affricate           36 0.7468  1.3222  0.0020
   6v TH/DH laryngeal fricative           36 0.7811  1.5527  0.0020
   6v   P/T     place      stop           36 0.8976  2.5442  0.0020
   6v   T/K     place      stop           36 0.7126  1.1191  0.0020
   6v   B/D     place      stop           36 0.8879  2.4684  0.0020
   6v   D/G     place      stop           36 0.8506  2.0669  0.0020
   6v   F/S     place fricative           36 0.9127  2.7283  0.0020
   6v  F/TH     place fricative           36 0.9805  4.1369  0.0020
   6v  S/SH     place fricative           36 0.8110  1.7585  0.0020
   6v   V/Z     place fricative           36 0.9074  2.6317  0.0020
   6v   M/N     place     nasal           36 0.9614  3.4975  0.0020
   44   P/B laryngeal      stop           36 0.5398  0.2018  0.1717
   44   T/D laryngeal      stop           36 0.5459  0.2300  0.2236
   44   K/G laryngeal      stop           36 0.3868 -0.5752  0.9162
   44   F/V laryngeal fricative           36 0.5376  0.1905  0.4731
   44   S/Z laryngeal fricative           36 0.5343  0.1762  0.3812
   44 SH/ZH laryngeal fricative           36 0.6938  1.0066  0.0040
   44 CH/JH laryngeal affricate           36 0.5586  0.2985  0.2236
   44 TH/DH laryngeal fricative           36 0.5992  0.5012  0.0918
   44   P/T     place      stop           36 0.5918  0.4624  0.1038
   44   T/K     place      stop           36 0.4901 -0.0486  0.4471
   44   B/D     place      stop           36 0.6052  0.5354  0.1218
   44   D/G     place      stop           36 0.5314  0.1600  0.3114
   44   F/S     place fricative           36 0.5121  0.0627  0.5489
   44  F/TH     place fricative           36 0.5906  0.4597  0.0778
   44  S/SH     place fricative           36 0.5002  0.0000  0.6168
   44   V/Z     place fricative           36 0.4710 -0.1396  0.7565
   44   M/N     place     nasal           36 0.5637  0.3228  0.4251
  all   P/B laryngeal      stop           36 0.6944  1.0144  0.0020
  all   T/D laryngeal      stop           36 0.6763  0.9141  0.0040
  all   K/G laryngeal      stop           36 0.4993  0.0000  0.5808
  all   F/V laryngeal fricative           36 0.5982  0.4993  0.0479
  all   S/Z laryngeal fricative           36 0.6724  0.8916  0.0020
  all SH/ZH laryngeal fricative           36 0.7384  1.2678  0.0060
  all CH/JH laryngeal affricate           36 0.6990  1.0897  0.0060
  all TH/DH laryngeal fricative           36 0.7331  1.2494  0.0020
  all   P/T     place      stop           36 0.8119  1.7879  0.0020
  all   T/K     place      stop           36 0.6622  0.8427  0.0299
  all   B/D     place      stop           36 0.8927  2.5193  0.0020
  all   D/G     place      stop           36 0.7085  1.1067  0.0060
  all   F/S     place fricative           36 0.8502  2.0717  0.0020
  all  F/TH     place fricative           36 0.9511  3.2927  0.0020
  all  S/SH     place fricative           36 0.7149  1.1370  0.0020
  all   V/Z     place fricative           36 0.7754  1.5074  0.0020
  all   M/N     place     nasal           36 0.9505  3.2669  0.0020
```

## Stage 1 — 창 비교 (area 6v, 지속시간 교란 점검)

`onset_*` 창은 audioEnvelope로 추정한 발화 onset 기준 고정 길이이므로
모든 쌍에 같은 길이가 적용된다(조음방법별 지속시간 차이 통제).

```
session_set        window  laryngeal  place
 audio_0426    go_0_500ms     0.6622 0.8431
 audio_0426        go_all     0.7294 0.8475
 audio_0426 onset_0_300ms     0.6800 0.7989
 audio_0426 onset_0_500ms     0.6881 0.8261
     pooled    go_0_500ms     0.6857 0.8851
     pooled        go_all     0.7388 0.8802
     pooled onset_0_300ms     0.6641 0.8165
     pooled onset_0_500ms     0.6828 0.8442
```

## Stage 1 — 후두쌍 조음방법 집계 (area 6v)

```
    window array        manner   mean     sd  n_pairs
go_0_500ms    6v     affricate 0.7365    NaN        1
go_0_500ms    6v     fricative 0.6710 0.0606        4
go_0_500ms    6v          stop 0.6884 0.0546        3
    go_all    6v     affricate 0.7468    NaN        1
    go_all    6v     fricative 0.7776 0.0521        4
    go_all    6v          stop 0.6845 0.0798        3
go_0_500ms    6v ALL_laryngeal 0.6857 0.0540        8
    go_all    6v ALL_laryngeal 0.7388 0.0715        8
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
      B     68     1783     897
     CH    291      223     231
      D   3155     1869    1221
     DH    222     4386     243
      F    436     1357     589
      G    120     1214     444
     HH      0     2338      56
     JH    151      414     373
      K   1239     1811    1910
      L   1892     1265    3197
      M   1299     1764    1330
      N   3495     1203    5403
     NG   1368        0     460
      P    400     1427    1281
      R   2040     1005    3251
      S   2493     2224    2370
     SH     59      380     426
      T   6412     2432    3558
     TH    136      563     232
      V   1837      365    1137
      W      0     3247     567
      Y      0     1417     358
      Z   3622        8     529
     ZH     20        3     572
```

## Stage 2 — T1 위치 (n 맞춤, pooled)

```
array position  laryngeal  place
   6v    final     0.6182 0.6857
   6v  initial     0.6729 0.6805
   6v   medial     0.6118 0.6361
```

## Stage 2 — T1 위치 (n 맞춤, test 전용)

```
array position  laryngeal  place
   6v    final     0.5040 0.6753
   6v  initial     0.6108 0.6740
   6v   medial     0.5730 0.6197
```

## Stage 2 — T1 위치 (설탄음화 통제: 모음간 T/D 제외)

```
array position  laryngeal  place
   6v    final     0.6182 0.6857
   6v  initial     0.6729 0.6805
   6v   medial     0.6205 0.6337
```

## Stage 2 — T1 위치 (6v_sup vs 6v_inf)

```
 array position  laryngeal  place
6v_inf    final     0.6079 0.6619
6v_inf  initial     0.6470 0.6669
6v_inf   medial     0.5889 0.6188
6v_sup    final     0.5743 0.6139
6v_sup  initial     0.6189 0.6300
6v_sup   medial     0.5620 0.6056
```

## Stage 2 — T1 위치 x 대립유형 상호작용 (부트스트랩 95% CI)

```
array             position  n_laryngeal_pairs  n_place_pairs  laryngeal_acc  place_acc  gap_place_minus_laryngeal  gap_ci_lo  gap_ci_hi excluded_pairs                run excluded
   6v              initial                  6              8         0.6729     0.6805                     0.0076    -0.0600     0.0734            NaN   all_pairs_pooled     none
   6v               medial                  6              8         0.6118     0.6361                     0.0242    -0.0588     0.0963            NaN   all_pairs_pooled     none
   6v                final                  6              8         0.6182     0.6857                     0.0675     0.0075     0.1174            NaN   all_pairs_pooled     none
   6v DoD(final - initial)                  6              8            NaN        NaN                     0.0599    -0.0292     0.1443            NaN   all_pairs_pooled     none
   6v              initial                  5              8         0.6711     0.6805                     0.0093    -0.0686     0.0861            T/D   all_pairs_pooled      T/D
   6v               medial                  5              8         0.6205     0.6361                     0.0156    -0.0809     0.0980            T/D   all_pairs_pooled      T/D
   6v                final                  5              8         0.6263     0.6857                     0.0594     0.0012     0.1092            T/D   all_pairs_pooled      T/D
   6v DoD(final - initial)                  5              8            NaN        NaN                     0.0501    -0.0451     0.1429            T/D   all_pairs_pooled      T/D
   6v              initial                  2              4         0.6108     0.6740                     0.0632    -0.0625     0.1889            NaN all_pairs_testonly     none
   6v               medial                  2              4         0.5730     0.6197                     0.0467     0.0240     0.0691            NaN all_pairs_testonly     none
   6v                final                  2              4         0.5040     0.6753                     0.1713     0.1326     0.2051            NaN all_pairs_testonly     none
   6v DoD(final - initial)                  2              4            NaN        NaN                     0.1081    -0.0213     0.2453            NaN all_pairs_testonly     none
   6v              initial                  1              4         0.5373     0.6740                     0.1367     0.0445     0.1975            T/D all_pairs_testonly      T/D
   6v               medial                  1              4         0.5627     0.6197                     0.0570     0.0360     0.0706            T/D all_pairs_testonly      T/D
   6v                final                  1              4         0.5160     0.6753                     0.1593     0.1206     0.1841            T/D all_pairs_testonly      T/D
   6v DoD(final - initial)                  1              4            NaN        NaN                     0.0226    -0.0528     0.1195            T/D all_pairs_testonly      T/D
   6v              initial                  6              8         0.6729     0.6805                     0.0076    -0.0600     0.0734            NaN    flapctrl_pooled     none
   6v               medial                  6              8         0.6205     0.6337                     0.0132    -0.0727     0.0913            NaN    flapctrl_pooled     none
   6v                final                  6              8         0.6182     0.6857                     0.0675     0.0075     0.1174            NaN    flapctrl_pooled     none
   6v DoD(final - initial)                  6              8            NaN        NaN                     0.0599    -0.0292     0.1443            NaN    flapctrl_pooled     none
   6v              initial                  5              8         0.6711     0.6805                     0.0093    -0.0686     0.0861            T/D    flapctrl_pooled      T/D
   6v               medial                  5              8         0.6205     0.6337                     0.0133    -0.0878     0.1018            T/D    flapctrl_pooled      T/D
   6v                final                  5              8         0.6263     0.6857                     0.0594     0.0012     0.1092            T/D    flapctrl_pooled      T/D
   6v DoD(final - initial)                  5              8            NaN        NaN                     0.0501    -0.0451     0.1429            T/D    flapctrl_pooled      T/D
```

## Stage 2 — T3 양식 (비교군별, 정렬 품질·날짜 맞춤)

```
          comparator  array modality       acc         mean_per       
                                     laryngeal  place laryngeal  place
        primary_0729     6v nonvocal    0.5617 0.6199    0.2636 0.2639
        primary_0729     6v    vocal    0.6216 0.6954    0.1858 0.1858
        primary_0729 6v_inf nonvocal    0.5282 0.6363    0.2629 0.2646
        primary_0729 6v_inf    vocal    0.6108 0.6431    0.1860 0.1858
        primary_0729 6v_sup nonvocal    0.5560 0.5909    0.2631 0.2656
        primary_0729 6v_sup    vocal    0.5597 0.6658    0.1859 0.1859
primary_0729_initial     6v nonvocal    0.6325 0.6817    0.2656 0.2681
primary_0729_initial     6v    vocal    0.7097 0.7042    0.1861 0.1862
primary_0729_initial 6v_inf nonvocal    0.6061 0.6407    0.2665 0.2660
primary_0729_initial 6v_inf    vocal    0.6858 0.6358    0.1860 0.1859
primary_0729_initial 6v_sup nonvocal    0.6259 0.6231    0.2650 0.2637
primary_0729_initial 6v_sup    vocal    0.6578 0.6689    0.1851 0.1861
 primary_0729_no0825     6v nonvocal    0.5814 0.6372    0.2530 0.2534
 primary_0729_no0825     6v    vocal    0.6216 0.6954    0.1858 0.1858
 primary_0729_no0825 6v_inf nonvocal    0.5508 0.6114    0.2523 0.2539
 primary_0729_no0825 6v_inf    vocal    0.6108 0.6431    0.1860 0.1858
 primary_0729_no0825 6v_sup nonvocal    0.5393 0.5584    0.2529 0.2542
 primary_0729_no0825 6v_sup    vocal    0.5597 0.6658    0.1859 0.1859
 secondary_datematch     6v nonvocal    0.5385 0.6165    0.2663 0.2667
 secondary_datematch     6v    vocal    0.5791 0.6296    0.1758 0.1746
 secondary_datematch 6v_inf nonvocal    0.5355 0.5961    0.2660 0.2637
 secondary_datematch 6v_inf    vocal    0.5716 0.6204    0.1745 0.1748
 secondary_datematch 6v_sup nonvocal    0.5109 0.5854    0.2640 0.2657
 secondary_datematch 6v_sup    vocal    0.5426 0.5624    0.1746 0.1745
   tertiary_allvocal     6v nonvocal    0.6441 0.6964    0.2641 0.2644
   tertiary_allvocal     6v    vocal    0.6278 0.6903    0.1873 0.1865
   tertiary_allvocal 6v_inf nonvocal    0.6145 0.6641    0.2636 0.2642
   tertiary_allvocal 6v_inf    vocal    0.6164 0.6747    0.1877 0.1872
   tertiary_allvocal 6v_sup nonvocal    0.5930 0.6243    0.2639 0.2639
   tertiary_allvocal 6v_sup    vocal    0.5936 0.6287    0.1874 0.1875
```

## Stage 2 — T3 격차와 DoD (부트스트랩 95% CI)

```
 array              modality  n_laryngeal_pairs  n_place_pairs  laryngeal_acc  place_acc  gap_place_minus_laryngeal  gap_ci_lo  gap_ci_hi  excluded_pairs           comparator positions
    6v                 vocal                  7              9         0.6216     0.6954                     0.0738    -0.0323     0.1768             NaN         primary_0729       all
    6v              nonvocal                  7              9         0.5617     0.6199                     0.0582    -0.0239     0.1421             NaN         primary_0729       all
    6v DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0156    -0.1511     0.1195             NaN         primary_0729       all
6v_sup                 vocal                  7              9         0.5597     0.6658                     0.1061     0.0247     0.1999             NaN         primary_0729       all
6v_sup              nonvocal                  7              9         0.5560     0.5909                     0.0350    -0.0327     0.1003             NaN         primary_0729       all
6v_sup DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0711    -0.1845     0.0364             NaN         primary_0729       all
6v_inf                 vocal                  7              9         0.6108     0.6431                     0.0323    -0.0451     0.1090             NaN         primary_0729       all
6v_inf              nonvocal                  7              9         0.5282     0.6363                     0.1081     0.0321     0.1831             NaN         primary_0729       all
6v_inf DoD(nonvocal - vocal)                  7              9            NaN        NaN                     0.0757    -0.0338     0.1830             NaN         primary_0729       all
    6v                 vocal                  7              9         0.6216     0.6954                     0.0738    -0.0323     0.1768             NaN  primary_0729_no0825       all
    6v              nonvocal                  7              9         0.5814     0.6372                     0.0558    -0.0175     0.1322             NaN  primary_0729_no0825       all
    6v DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0180    -0.1475     0.1120             NaN  primary_0729_no0825       all
6v_sup                 vocal                  7              9         0.5597     0.6658                     0.1061     0.0247     0.1999             NaN  primary_0729_no0825       all
6v_sup              nonvocal                  7              9         0.5393     0.5584                     0.0191    -0.0587     0.0946             NaN  primary_0729_no0825       all
6v_sup DoD(nonvocal - vocal)                  7              9            NaN        NaN                    -0.0870    -0.2077     0.0270             NaN  primary_0729_no0825       all
6v_inf                 vocal                  7              9         0.6108     0.6431                     0.0323    -0.0451     0.1090             NaN  primary_0729_no0825       all
6v_inf              nonvocal                  7              9         0.5508     0.6114                     0.0606    -0.0028     0.1237             NaN  primary_0729_no0825       all
6v_inf DoD(nonvocal - vocal)                  7              9            NaN        NaN                     0.0283    -0.0732     0.1292             NaN  primary_0729_no0825       all
    6v                 vocal                  6              8         0.5791     0.6296                     0.0505    -0.0159     0.1155             NaN  secondary_datematch       all
    6v              nonvocal                  6              8         0.5385     0.6165                     0.0780     0.0255     0.1311             NaN  secondary_datematch       all
    6v DoD(nonvocal - vocal)                  6              8            NaN        NaN                     0.0275    -0.0541     0.1130             NaN  secondary_datematch       all
6v_sup                 vocal                  6              8         0.5426     0.5624                     0.0198    -0.0822     0.1122             NaN  secondary_datematch       all
6v_sup              nonvocal                  6              8         0.5109     0.5854                     0.0745     0.0179     0.1284             NaN  secondary_datematch       all
6v_sup DoD(nonvocal - vocal)                  6              8            NaN        NaN                     0.0547    -0.0523     0.1691             NaN  secondary_datematch       all
6v_inf                 vocal                  6              8         0.5716     0.6204                     0.0488     0.0065     0.0883             NaN  secondary_datematch       all
6v_inf              nonvocal                  6              8         0.5355     0.5961                     0.0607    -0.0175     0.1458             NaN  secondary_datematch       all
6v_inf DoD(nonvocal - vocal)                  6              8            NaN        NaN                     0.0118    -0.0767     0.1072             NaN  secondary_datematch       all
    6v                 vocal                  8              9         0.6278     0.6903                     0.0625     0.0195     0.1071             NaN    tertiary_allvocal       all
    6v              nonvocal                  8              9         0.6441     0.6964                     0.0523    -0.0184     0.1134             NaN    tertiary_allvocal       all
    6v DoD(nonvocal - vocal)                  8              9            NaN        NaN                    -0.0102    -0.0920     0.0664             NaN    tertiary_allvocal       all
6v_sup                 vocal                  8              9         0.5936     0.6287                     0.0351    -0.0237     0.0912             NaN    tertiary_allvocal       all
6v_sup              nonvocal                  8              9         0.5930     0.6243                     0.0313    -0.0298     0.0838             NaN    tertiary_allvocal       all
6v_sup DoD(nonvocal - vocal)                  8              9            NaN        NaN                    -0.0038    -0.0870     0.0771             NaN    tertiary_allvocal       all
6v_inf                 vocal                  8              9         0.6164     0.6747                     0.0584     0.0074     0.1044             NaN    tertiary_allvocal       all
6v_inf              nonvocal                  8              9         0.6145     0.6641                     0.0496    -0.0058     0.1035             NaN    tertiary_allvocal       all
6v_inf DoD(nonvocal - vocal)                  8              9            NaN        NaN                    -0.0088    -0.0823     0.0657             NaN    tertiary_allvocal       all
    6v                 vocal                  4              7         0.7097     0.7042                    -0.0055    -0.0494     0.0398             NaN primary_0729_initial   initial
    6v              nonvocal                  4              7         0.6325     0.6817                     0.0492    -0.0409     0.1292             NaN primary_0729_initial   initial
    6v DoD(nonvocal - vocal)                  4              7            NaN        NaN                     0.0547    -0.0473     0.1492             NaN primary_0729_initial   initial
6v_sup                 vocal                  4              7         0.6578     0.6689                     0.0111    -0.0493     0.0729             NaN primary_0729_initial   initial
6v_sup              nonvocal                  4              7         0.6259     0.6231                    -0.0028    -0.1028     0.1009             NaN primary_0729_initial   initial
6v_sup DoD(nonvocal - vocal)                  4              7            NaN        NaN                    -0.0140    -0.1314     0.1034             NaN primary_0729_initial   initial
6v_inf                 vocal                  4              7         0.6858     0.6358                    -0.0500    -0.1407     0.0355             NaN primary_0729_initial   initial
6v_inf              nonvocal                  4              7         0.6061     0.6407                     0.0345    -0.0312     0.0935             NaN primary_0729_initial   initial
6v_inf DoD(nonvocal - vocal)                  4              7            NaN        NaN                     0.0846    -0.0209     0.1938             NaN primary_0729_initial   initial
```

## Stage 2 — T3 쌍별 낙폭 (발성 - 무성, 재표집 95% CI)

```
array  pair  contrast    manner  n_per_class  acc_vocal  acc_nonvocal    drop  drop_ci_lo  drop_ci_hi          comparator
   6v   P/B laryngeal      stop           65     0.6633        0.5981  0.0652     -0.0222      0.1482        primary_0729
   6v   T/D laryngeal      stop          140     0.6270        0.5609  0.0661      0.0014      0.1304        primary_0729
   6v   K/G laryngeal      stop           35     0.6607        0.5682  0.0925     -0.0725      0.2402        primary_0729
   6v   F/V laryngeal fricative           69     0.6015        0.5799  0.0216     -0.1008      0.1281        primary_0729
   6v   S/Z laryngeal fricative           86     0.6110        0.5892  0.0218     -0.0471      0.1155        primary_0729
   6v SH/ZH laryngeal fricative           12     0.4800        0.4630  0.0170     -0.1815      0.2131        primary_0729
   6v TH/DH laryngeal fricative           28     0.7225        0.6889  0.0337     -0.0934      0.1548        primary_0729
   6v   P/T     place      stop           74     0.7937        0.6882  0.1055      0.0106      0.2097        primary_0729
   6v   T/K     place      stop          132     0.6992        0.6387  0.0605     -0.0218      0.1147        primary_0729
   6v   B/D     place      stop           65     0.8033        0.6904  0.1129     -0.0068      0.1751        primary_0729
   6v   D/G     place      stop           35     0.6704        0.5900  0.0804     -0.0429      0.2562        primary_0729
   6v   F/S     place fricative           69     0.6129        0.5998  0.0131     -0.1056      0.1109        primary_0729
   6v  F/TH     place fricative           28     0.7344        0.6575  0.0769     -0.0641      0.2011        primary_0729
   6v  S/SH     place fricative           12     0.4652        0.4690 -0.0037     -0.2969      0.1694        primary_0729
   6v   V/Z     place fricative           77     0.6792        0.6561  0.0231     -0.0546      0.1047        primary_0729
   6v   M/N     place     nasal          131     0.7522        0.7049  0.0473     -0.0320      0.1282        primary_0729
   6v   P/B laryngeal      stop           45     0.6558        0.5456  0.1103     -0.0008      0.2596 secondary_datematch
   6v   T/D laryngeal      stop           91     0.5612        0.5519  0.0093     -0.0975      0.1106 secondary_datematch
   6v   K/G laryngeal      stop           29     0.5196        0.5458 -0.0262     -0.1344      0.0712 secondary_datematch
   6v   F/V laryngeal fricative           48     0.5757        0.5540  0.0218     -0.0758      0.1125 secondary_datematch
   6v   S/Z laryngeal fricative           39     0.6231        0.5681  0.0550     -0.0379      0.2091 secondary_datematch
   6v TH/DH laryngeal fricative           15     0.4892        0.5058 -0.0167     -0.2525      0.2604 secondary_datematch
   6v   P/T     place      stop           45     0.6750        0.6594  0.0156     -0.1085      0.1281 secondary_datematch
   6v   T/K     place      stop           86     0.6260        0.6072  0.0188     -0.0588      0.0860 secondary_datematch
   6v   B/D     place      stop           46     0.7282        0.6794  0.0488     -0.0854      0.1756 secondary_datematch
   6v   D/G     place      stop           29     0.5732        0.5749 -0.0017     -0.1688      0.1651 secondary_datematch
   6v   F/S     place fricative           48     0.6011        0.5806  0.0205     -0.1110      0.1681 secondary_datematch
   6v  F/TH     place fricative           15     0.5025        0.5158 -0.0133     -0.2508      0.1587 secondary_datematch
   6v   V/Z     place fricative           39     0.6776        0.6201  0.0575     -0.0339      0.1907 secondary_datematch
   6v   M/N     place     nasal          100     0.6654        0.6805 -0.0151     -0.0846      0.0638 secondary_datematch
   6v   P/B laryngeal      stop          259     0.6488        0.6403  0.0085     -0.0228      0.0391   tertiary_allvocal
   6v   T/D laryngeal      stop          300     0.5660        0.5610  0.0050     -0.0672      0.0457   tertiary_allvocal
   6v   K/G laryngeal      stop          167     0.6040        0.6342 -0.0301     -0.1333      0.0385   tertiary_allvocal
   6v   F/V laryngeal fricative          256     0.6037        0.6312 -0.0275     -0.0616      0.0131   tertiary_allvocal
   6v   S/Z laryngeal fricative          300     0.6314        0.6039  0.0275     -0.0164      0.0750   tertiary_allvocal
   6v SH/ZH laryngeal fricative           41     0.5612        0.5713 -0.0102     -0.1203      0.0980   tertiary_allvocal
   6v CH/JH laryngeal affricate           47     0.6745        0.6908 -0.0162     -0.0988      0.1046   tertiary_allvocal
   6v TH/DH laryngeal fricative           93     0.7331        0.7713 -0.0382     -0.1146      0.0402   tertiary_allvocal
   6v   P/T     place      stop          300     0.7740        0.7448  0.0292     -0.0005      0.0604   tertiary_allvocal
   6v   T/K     place      stop          300     0.6681        0.6650  0.0031     -0.0348      0.0457   tertiary_allvocal
   6v   B/D     place      stop          259     0.7574        0.7528  0.0046     -0.0286      0.0351   tertiary_allvocal
   6v   D/G     place      stop          167     0.6894        0.6723  0.0171     -0.0263      0.0815   tertiary_allvocal
   6v   F/S     place fricative          256     0.5990        0.6238 -0.0248     -0.0614      0.0344   tertiary_allvocal
   6v  F/TH     place fricative           93     0.7250        0.7551 -0.0301     -0.0943      0.0328   tertiary_allvocal
   6v  S/SH     place fricative           57     0.6508        0.6046  0.0463     -0.0680      0.1255   tertiary_allvocal
   6v   V/Z     place fricative          300     0.7159        0.7038  0.0121     -0.0111      0.0431   tertiary_allvocal
   6v   M/N     place     nasal          300     0.6946        0.7149 -0.0203     -0.0571      0.0196   tertiary_allvocal
```

## Stage 2 — T3 자연부류 검정 (후두 마디가 한 부류로 약해지는가)

```
                                           group  n_pairs  mean_drop  sd_drop  uniformity_p          comparator
                                       laryngeal        7     0.0454   0.0292           NaN        primary_0729
                               place (non-nasal)        8     0.0586   0.0434           NaN        primary_0729
                                           nasal        1     0.0473      NaN           NaN        primary_0729
                                  laryngeal:stop        3     0.0746   0.0155           NaN        primary_0729
                             laryngeal:fricative        4     0.0235   0.0071           NaN        primary_0729
UNIFORMITY TEST (laryngeal SD vs random 7 of 16)        7     0.0292   0.0351        0.1640        primary_0729
                                       laryngeal        6     0.0256   0.0506           NaN secondary_datematch
                               place (non-nasal)        7     0.0209   0.0253           NaN secondary_datematch
                                           nasal        1    -0.0151      NaN           NaN secondary_datematch
                                  laryngeal:stop        3     0.0311   0.0708           NaN secondary_datematch
                             laryngeal:fricative        3     0.0200   0.0359           NaN secondary_datematch
UNIFORMITY TEST (laryngeal SD vs random 6 of 14)        6     0.0506   0.0357        0.9380 secondary_datematch
                                       laryngeal        8    -0.0102   0.0224           NaN   tertiary_allvocal
                               place (non-nasal)        8     0.0072   0.0256           NaN   tertiary_allvocal
                                           nasal        1    -0.0203      NaN           NaN   tertiary_allvocal
                                  laryngeal:stop        3    -0.0055   0.0214           NaN   tertiary_allvocal
                             laryngeal:fricative        4    -0.0121   0.0288           NaN   tertiary_allvocal
                             laryngeal:affricate        1    -0.0162      NaN           NaN   tertiary_allvocal
UNIFORMITY TEST (laryngeal SD vs random 8 of 17)        8     0.0224   0.0243        0.3234   tertiary_allvocal
```

균일성 검정: 후두쌍 낙폭의 표준편차가 전체 쌍에서 같은 수를 무작위로 뽑았을 때의 표준편차보다 작은가(= 하나의 자연부류처럼 균일하게 약해지는가). p는 무작위 SD가 관측 SD 이하일 비율이므로 작을수록 '균일하다'는 증거다.

## Stage 2 — T2 위치별 유표성

```
 array position  voiced  voiceless
    6v    final 0.00441    0.00254
    6v  initial 0.00370    0.00350
    6v   medial 0.00237    0.00202
6v_inf    final 0.00444    0.00348
6v_inf  initial 0.00364    0.00413
6v_inf   medial 0.00244    0.00257
6v_sup    final 0.00438    0.00159
6v_sup  initial 0.00375    0.00286
6v_sup   medial 0.00230    0.00146
```

## Stage 2 — Miller-Nicely 자질 전달량

```
feature     6v  6v_inf  6v_sup
 manner 0.0233  0.0190  0.0126
  nasal 0.0148  0.0123  0.0088
  place 0.0483  0.0328  0.0266
voicing 0.0128  0.0118  0.0033
```

## Stage 2 — score 필터 탈락 집계

```
modality position_in_word  n_total  n_kept  n_dropped  frac_dropped  score_min
nonvocal            final     2928    2310        618        0.2111        0.5
nonvocal          initial     2928    2146        782        0.2671        0.5
nonvocal           medial     4155    2612       1543        0.3714        0.5
nonvocal              sil     3142    3135          7        0.0022        0.5
nonvocal           single      214     199         15        0.0701        0.5
   vocal            final    41551   40591        960        0.0231        0.5
   vocal          initial    41551   40463       1088        0.0262        0.5
   vocal           medial    67806   64661       3145        0.0464        0.5
   vocal              sil    44161   44080         81        0.0018        0.5
   vocal           single     2573    2561         12        0.0047        0.5
```

## 정렬 품질 (세션별 PER, 음소 수 가중)

```
       session partition modality    per  n_trials input_layer_from  borrowed
t12.2022.08.18      test nonvocal 0.2615        40   t12.2022.08.13      True
t12.2022.08.18     train nonvocal 0.2450       440   t12.2022.08.13      True
t12.2022.08.23      test nonvocal 0.2326        40   t12.2022.08.13      True
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
    "t12.2022.08.23"
  ],
  "n_segments": 202758,
  "modality_counts": {
    "vocal": 192356,
    "nonvocal": 10402
  },
  "partition_counts": {
    "train": 185738,
    "test": 17020
  },
  "position_counts": {
    "medial": 67273,
    "sil": 47215,
    "final": 42901,
    "initial": 42609,
    "single": 2760
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

## 추가 메모

```json
{
  "stage1": {
    "sessions": [
      "t12.2022.04.21_phonemes",
      "t12.2022.04.26_phonemes"
    ],
    "n_trials": 1440,
    "trials_per_phoneme": 36,
    "n_perm": 500,
    "arrays": [
      "6v",
      "6v_sup",
      "6v_inf",
      "44",
      "all"
    ]
  },
  "stage2": {
    "n_sessions": 24,
    "n_segments": 251926,
    "window": "seg",
    "score_min": 0.5,
    "max_per_class": 300,
    "arrays": [
      "6v",
      "6v_sup",
      "6v_inf"
    ],
    "held_out_rule": "(partition=='test') or (input_layer_from != session)"
  },
  "verdicts": {
    "T1": "어두에서 후두 분리도 최고, 어말에서 격차 최대 -> [spread glottis] 방향. DoD CI는 0 포함(시사적, 비유의).",
    "T2": "유성-무성 거리 차이 미미, 모든 어레이/위치에서 비유의 -> 판정 불가.",
    "T3": "무성 발화에서 후두와 조음위치가 함께 떨어짐. 격차의 차이 CI 모두 0 포함.",
    "T3_natural_class": "후두쌍 낙폭이 조음위치쌍보다 크지도, 더 균일하지도 않다(균일성 p = 0.16 / 0.94 / 0.32). 자연부류 모형 미지지.",
    "array": "6v_inf > 6v_sup, area 44는 거의 우연수준 -> H_sampling(dorsal 집중) 미지지."
  },
  "caveat": "음성 자질표 verified=False. 사전등록 전 확정 금지."
}
```

## 그림

- `stage1_markedness.png`
- `stage1_pair_decoding.png`
- `stage2_feature_transmission.png`
- `stage2_markedness.png`
- `stage2_position.png`
- `stage2_position_synthcheck.png`
