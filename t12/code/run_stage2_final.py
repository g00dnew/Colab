"""Stage 2 종합 실행 — 특징 추출 1회, 모든 통제를 그 위에서 돌린다.

T1 통제:
  (a) 위치 간 쌍별 n 맞춤 (match_n_across_positions)
  (b) 설탄음화: 모음간 T/D 제외 + T/D 전체 제외
  (c) 파티션: pooled / test 전용
  (d) 위치 x 대립유형 상호작용(DoD)에 부트스트랩 CI
민감도: 창(seg vs rnn_center), score 필터, 합성 특징 논리 점검
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 100
WINDOW = sys.argv[2] if len(sys.argv) > 2 else "seg"
paths = tp.get_paths(local=True)
PRIMARY_ARRAY = ("6v",)
SPLIT_ARRAYS = ("6v_sup", "6v_inf")
PRIMARY_VOCAL = ["t12.2022.07.29"]
SECONDARY_VOCAL = ["t12.2022.08.11", "t12.2022.08.13"]

per_table = tp.session_per_table(paths["ALIGN_DIR"])
per_table.to_csv(paths["RESULT_DIR"] / "alignment_per.csv", index=False)

align_raw = tp.load_alignments(paths["ALIGN_DIR"], window_mode=WINDOW)
align, score_report = tp.apply_score_filter(align_raw)
X, X_pre, meta = tp.stage2_segment_features(paths, align)
print(f"\n세션({meta['session'].nunique()}): {sorted(meta['session'].unique())}", flush=True)
print(f"양식: {meta['modality'].value_counts().to_dict()}", flush=True)
print(f"파티션: {meta['partition'].value_counts().to_dict()}", flush=True)
print(f"위치: {meta['position_in_word'].value_counts().to_dict()}", flush=True)
assert X.shape == (len(meta), 512), X.shape

# n-by-position 표 (SUMMARY용)
nbp = (meta[meta["phoneme"].isin(tp.CONSONANTS)]
       .groupby(["phoneme", "position_in_word"]).size().unstack(fill_value=0))
nbp.to_csv(paths["RESULT_DIR"] / "stage2_n_by_position.csv")
print("\n자음 x 위치 구간 수:")
print(nbp.to_string(), flush=True)

T1_RUNS = [
    ("all_pairs_pooled", dict(partitions=None, drop_intervocalic_flap=False), N_PERM),
    ("all_pairs_testonly", dict(partitions="test", drop_intervocalic_flap=False), 0),
    ("flapctrl_pooled", dict(partitions=None, drop_intervocalic_flap=True), 0),
]
t1_frames, inter_frames = [], []
for label, kw, nperm in T1_RUNS:
    print("\n" + "=" * 72)
    print(f"T1 [{label}] n 맞춤, 순열 {nperm}회")
    print("=" * 72, flush=True)
    d = tp.stage2_position_decode(
        paths, X, meta, arrays=PRIMARY_ARRAY, n_perm=nperm, perm_repeats=2,
        n_repeats=10, tag=f"_{label}", force=True, max_per_class=300,
        match_n_across_positions=True, **kw)
    if not len(d):
        print("결과 없음", flush=True)
        continue
    d["run"] = label
    t1_frames.append(d)
    print("\n위치 x 대립 유형:")
    print(d.pivot_table(index="position", columns="contrast",
                        values="acc").round(4).to_string())
    print("\n쌍별 (같은 쌍은 세 위치 모두 같은 n):")
    print(d.pivot_table(index=["contrast", "manner", "pair"], columns="position",
                        values="acc").round(4).to_string())
    print("\n쌍별 n per class:")
    print(d.pivot_table(index="pair", columns="position",
                        values="n_per_class").astype("Int64").to_string())
    if nperm:
        print("\n쌍별 순열 p:")
        print(d.pivot_table(index="pair", columns="position",
                            values="p_perm").round(4).to_string())
    for excl, ename in [((), "none"), (("T/D",), "T/D")]:
        it = tp.position_contrast_interaction(d, array="6v", n_boot=10000,
                                              exclude_pairs=excl)
        it["run"] = label
        it["excluded"] = ename
        inter_frames.append(it)
        print(f"\n위치별 격차 + DoD (제외: {ename}):")
        print(it.to_string(index=False), flush=True)

if t1_frames:
    pd.concat(t1_frames, ignore_index=True).to_csv(
        paths["RESULT_DIR"] / "stage2_T1_controls.csv", index=False)
if inter_frames:
    inter = pd.concat(inter_frames, ignore_index=True)
    inter.to_csv(paths["RESULT_DIR"] / "stage2_T1_interaction.csv", index=False)

# 어레이 분할 (순열 없이)
print("\n" + "=" * 72)
print("T1 어레이 분할 (6v_sup vs 6v_inf, n 맞춤, 순열 없음)")
print("=" * 72, flush=True)
dsp = tp.stage2_position_decode(paths, X, meta, arrays=SPLIT_ARRAYS, n_perm=0,
                                n_repeats=10, tag="_arraysplit", force=True,
                                max_per_class=300, match_n_across_positions=True)
if len(dsp):
    print(dsp.pivot_table(index=["array", "position"], columns="contrast",
                          values="acc").round(4).to_string())
    for arr in SPLIT_ARRAYS:
        it = tp.position_contrast_interaction(dsp, array=arr, n_boot=5000)
        print(f"\n{arr} 격차 + DoD:")
        print(it.to_string(index=False), flush=True)

# T2 / Miller-Nicely
print("\n" + "=" * 72)
print("T2 위치별 유표성")
print("=" * 72, flush=True)
m = tp.stage2_markedness_by_position(paths, X, meta,
                                     arrays=PRIMARY_ARRAY + SPLIT_ARRAYS,
                                     n_reps=400, tag="", force=True)
if len(m):
    piv = m.pivot_table(index=["array", "position"], columns="series", values="d2_cv")
    piv["voiced_minus_voiceless"] = piv["voiced"] - piv["voiceless"]
    print(piv.round(5).to_string(), flush=True)

print("\n" + "=" * 72)
print("Miller-Nicely 자질 전달량")
print("=" * 72, flush=True)
tr = tp.stage2_feature_transmission(paths, X, meta,
                                    arrays=PRIMARY_ARRAY + SPLIT_ARRAYS,
                                    positions=("all", "initial", "medial", "final"),
                                    tag="", force=True, max_per_class=300)
if len(tr):
    print(tr.pivot_table(index=["position", "feature"], columns="array",
                         values="T_rel").round(4).to_string(), flush=True)

# T3
print("\n" + "=" * 72)
print("T3 양식 (정렬 품질 맞춤)")
print("=" * 72, flush=True)
avail_vocal = sorted(set(meta.loc[meta["modality"] == "vocal", "session"]))
avail_non = sorted(set(meta.loc[meta["modality"] == "nonvocal", "session"]))
print("vocal:", avail_vocal, "\nnonvocal:", avail_non, flush=True)
t3_frames = []
if avail_non:
    for label, want, t3tag in [("primary(07.29)", PRIMARY_VOCAL, "_t3primary"),
                               ("secondary(Aug)", SECONDARY_VOCAL, "_t3secondary"),
                               ("all vocal", None, "_t3all")]:
        sel = avail_vocal if want is None else [s for s in want if s in avail_vocal]
        if not sel:
            print(f"[{label}] 해당 vocal 세션 정렬 없음 -> 생략", flush=True)
            continue
        d = tp.stage2_modality_decode(
            paths, X, meta, arrays=PRIMARY_ARRAY + SPLIT_ARRAYS, n_perm=0,
            n_repeats=10, tag=t3tag, force=True, per_table=per_table,
            max_per_class=300, session_filter={"vocal": sel, "nonvocal": None})
        if len(d):
            d["comparator"] = label
            t3_frames.append(d)
            print(f"\n[{label}] vocal={sel}")
            print(d.pivot_table(index=["array", "modality"], columns="contrast",
                                values="acc").round(4).to_string())
            print(d.groupby("modality")[["mean_per", "n_matched_per_class"]]
                  .agg({"mean_per": "mean", "n_matched_per_class": "median"})
                  .round(4).to_string(), flush=True)
    if t3_frames:
        pd.concat(t3_frames, ignore_index=True).to_csv(
            paths["RESULT_DIR"] / "stage2_t3_comparators.csv", index=False)
else:
    print("nonvocal 정렬이 아직 없어 T3 생략", flush=True)

# 그림 + 매니페스트
primary = t1_frames[0] if t1_frames else None
if primary is not None and len(primary):
    tp.fig_stage2_position(primary, paths["FIG_DIR"] / "stage2_position.png", array="6v")
if len(m):
    tp.fig_markedness(m, paths["FIG_DIR"] / "stage2_markedness.png", array="6v")
if len(tr):
    tp.fig_feature_transmission(tr, paths["FIG_DIR"] / "stage2_feature_transmission.png")
score_report.to_csv(paths["RESULT_DIR"] / "stage2_score_filter.csv", index=False)
import json  # noqa: E402

(paths["RESULT_DIR"] / "stage2_manifest.json").write_text(json.dumps({
    "window_mode": WINDOW, "score_min": tp.SCORE_MIN, "n_perm_primary": N_PERM,
    "sessions_included": sorted(meta["session"].unique().tolist()),
    "n_segments": int(len(meta)),
    "modality_counts": {k: int(v) for k, v in meta["modality"].value_counts().items()},
    "partition_counts": {k: int(v) for k, v in meta["partition"].value_counts().items()},
    "position_counts": {k: int(v) for k, v in
                        meta["position_in_word"].value_counts().items()},
    "arrays": list(PRIMARY_ARRAY + SPLIT_ARRAYS),
    "max_per_class": 300,
    "structural_exclusion": ("S/Z, SH/ZH는 영어에서 어두에 거의 나타나지 않아 "
                             "T1 어두 조건에 기여할 수 없다"),
    "caveat": ("CTC는 peaky하므로 경계는 ~80 ms 근사. train 파티션은 RNN 학습에 "
               "쓰였으므로 디코더와 독립이 아니다. 모음간 T/D는 설탄음화로 통제."),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nFINAL_DONE", flush=True)
