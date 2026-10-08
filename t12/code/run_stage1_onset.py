"""Stage 1 추가 분석: 음성 onset 기준 고정길이 창 + 조음방법별 집계.

질문: 후두 대립의 분리도가 파열음에서 가장 낮은 것이 '파열음이 짧아서'인가?
모든 쌍에 같은 길이의 창(onset 기준 0~300 ms, 0~500 ms)을 쓰고 비교한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True)
ARRAYS = ("6v", "6v_sup", "6v_inf", "44", "all")
WINDOWS = ["go_all", "go_0_500ms", "onset_0_300ms", "onset_0_500ms"]

# 04.26만 audioEnvelope가 있다. onset 기준 창의 1차 분석은 04.26 단독으로,
# 2차로 두 세션 합본(04.21은 go 기준으로 되돌아감)을 본다.
SESSION_SETS = [
    ("audio_0426", ["t12.2022.04.26_phonemes"]),
    ("pooled", None),
]

feats = tp.load_stage1_features(paths, force=True)
print("시행", feats["go_all"].shape[0], "창", feats["windows"], flush=True)
if "onset_bins" in feats:
    ob = feats["onset_bins"]
    print(f"onset 분포: 중앙 {int(pd.Series(ob).median())} bin, "
          f"p10 {int(pd.Series(ob).quantile(0.1))}, p90 {int(pd.Series(ob).quantile(0.9))}",
          flush=True)

ph = feats["phoneme"]
specs = ([(a, b, "laryngeal") for a, b in tp.LARYNGEAL_PAIRS]
         + [(a, b, "place") for a, b in tp.PLACE_CONTROL_PAIRS])
rows = []
for w in WINDOWS:
    if w not in feats:
        continue
    for arr in ARRAYS:
        X = feats[w][:, tp.array_columns(arr)]
        for a, b, contrast in specs:
            m = (ph == a) | (ph == b)
            r = tp.balanced_pair_decode(X[m], ph[m], n_perm=0, n_repeats=10,
                                        seed=20261009)
            rows.append({"window": w, "array": arr, "pair": f"{a}/{b}",
                         "contrast": contrast, "manner": tp.pair_manner(a, b),
                         "n_per_class": r["n_per_class"], "acc": r["acc"],
                         "acc_sd": r["acc_sd"], "dprime": r["dprime"],
                         "session_set": "pooled"})
    print(f"  {w} 완료", flush=True)

# 04.26 단독(진짜 onset 기준) 재실행
feats26 = tp.load_stage1_features(paths, sessions=["t12.2022.04.26_phonemes"],
                                  force=True)
ph26 = feats26["phoneme"]
print(f"\n04.26 단독: 시행 {feats26['go_all'].shape[0]}, onset 중앙 "
      f"{int(pd.Series(feats26['onset_bins']).median())} bin", flush=True)
for w in WINDOWS:
    if w not in feats26:
        continue
    for arr in ARRAYS:
        X = feats26[w][:, tp.array_columns(arr)]
        for a, b, contrast in specs:
            m = (ph26 == a) | (ph26 == b)
            r = tp.balanced_pair_decode(X[m], ph26[m], n_perm=0, n_repeats=10,
                                        seed=20261009, min_per_class=8)
            rows.append({"window": w, "array": arr, "pair": f"{a}/{b}",
                         "contrast": contrast, "manner": tp.pair_manner(a, b),
                         "n_per_class": r["n_per_class"], "acc": r["acc"],
                         "acc_sd": r["acc_sd"], "dprime": r["dprime"],
                         "session_set": "audio_0426"})
    print(f"  04.26 {w} 완료", flush=True)

df = pd.DataFrame(rows)
df["session_set"] = df["session_set"].fillna("pooled")
df.to_csv(paths["RESULT_DIR"] / "stage1_onset_window_scan.csv", index=False)

print("\n=== 04.26 단독 (진짜 음성 onset 기준), 후두쌍 x 조음방법, area 6v ===",
      flush=True)
s26 = df[(df["session_set"] == "audio_0426") & (df["array"] == "6v")
         & (df["contrast"] == "laryngeal")]
print(s26.pivot_table(index="window", columns="manner", values="acc").round(4).to_string())
print("\n=== 04.26 단독 쌍별 ===", flush=True)
print(s26.pivot_table(index=["manner", "pair"], columns="window",
                      values="acc").round(4).to_string())
print("\n=== 04.26 단독, 조음위치 통제쌍 ===", flush=True)
p26 = df[(df["session_set"] == "audio_0426") & (df["array"] == "6v")
         & (df["contrast"] == "place")]
print(p26.pivot_table(index="manner", columns="window", values="acc").round(4).to_string())

print("\n=== 창 x 어레이 x 대립 유형 평균 정확도 ===", flush=True)
print(df.pivot_table(index=["window", "array"], columns="contrast",
                     values="acc").round(4).to_string())

print("\n=== 후두쌍: 창 x 조음방법 (area 6v) ===", flush=True)
sub = df[(df["session_set"] == "pooled") & (df["array"] == "6v")
         & (df["contrast"] == "laryngeal")]
print(sub.pivot_table(index="window", columns="manner", values="acc").round(4).to_string())

print("\n=== 후두쌍 쌍별 (area 6v) ===", flush=True)
print(sub.pivot_table(index=["manner", "pair"], columns="window",
                      values="acc").round(4).to_string())

man = tp.manner_aggregate(df, group_cols=("window", "array"))
man.to_csv(paths["RESULT_DIR"] / "stage1_manner_aggregate.csv", index=False)
print("\n=== manner 집계 (area 6v) ===", flush=True)
print(man[man["array"] == "6v"].round(4).to_string(index=False))

print("\n=== 조음위치 통제쌍도 같은 창에서 (area 6v) ===", flush=True)
pl = df[(df["session_set"] == "pooled") & (df["array"] == "6v")
        & (df["contrast"] == "place")]
print(pl.pivot_table(index="manner", columns="window", values="acc").round(4).to_string())
print("\nONSET_DONE", flush=True)
