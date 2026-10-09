"""T1' 파열음 전용 1차 검정 (사전등록형).

후두쌍 = P/B, T/D, K/G만. 위치 대조 = 파열음 위치쌍(P/T, T/K, B/D, D/G).
무기식 대조 = 마찰음 후두쌍(F/V, TH/DH). 주검정 = 어두 vs 비어두(어중+어말).
6v, seg 창, score>=0.5, 쌍별 위치 n 맞춤, 300/class 상한. PCA는 폴드 내 적합.
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 100
paths = tp.get_paths(local=True)
R = paths["RESULT_DIR"]

def _sel(pairs, names):
    out = [p for p in pairs if f"{p[0]}/{p[1]}" in names]
    assert len(out) == len(names), (out, names)
    return out

ALL_LAR = list(tp.LARYNGEAL_PAIRS)
ALL_PLACE = list(tp.PLACE_CONTROL_PAIRS)
STOP_LAR = _sel(ALL_LAR, ["P/B", "T/D", "K/G"])
STOP_PLACE = _sel(ALL_PLACE, ["P/T", "T/K", "B/D", "D/G"])
FRIC_LAR = _sel(ALL_LAR, ["F/V", "TH/DH"])

align_raw = tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg")
align, _ = tp.apply_score_filter(align_raw)
X, X_pre, meta = tp.stage2_segment_features(paths, align)
assert X.shape == (len(meta), 512)

meta2 = meta.copy()
meta2["position_in_word"] = meta2["position_in_word"].replace(
    {"medial": "noninitial", "final": "noninitial"})

def run(label, lar, pla, positions, m, partitions, flap, nperm):
    tp.LARYNGEAL_PAIRS[:] = lar
    tp.PLACE_CONTROL_PAIRS[:] = pla
    print("\n" + "=" * 72 + f"\n{label}  (순열 {nperm})\n" + "=" * 72, flush=True)
    d = tp.stage2_position_decode(
        paths, X, m, arrays=("6v",), positions=list(positions), n_perm=nperm,
        perm_repeats=2, n_repeats=10, tag=f"_stops_{label}", force=True,
        max_per_class=300, match_n_across_positions=True, partitions=partitions,
        drop_intervocalic_flap=flap)
    d["run"] = label
    if len(d):
        print(d.pivot_table(index=["contrast", "pair"], columns="position",
                            values="acc").round(4).to_string(), flush=True)
        print(d.pivot_table(index="pair", columns="position",
                            values="n_per_class").astype("Int64").to_string(), flush=True)
    return d

def dod_boot(d, pos_a="initial", pos_b="noninitial", n_boot=10000, seed=1):
    """쌍 수준 부트스트랩: 위치별 (place - laryngeal) 격차와 DoD(pos_b - pos_a)."""
    rng = np.random.default_rng(seed)
    g = d[d["acc"].notna()]
    def grp(contrast, pos):
        return g[(g["contrast"] == contrast) & (g["position"] == pos)]["acc"].to_numpy()
    li, lb = grp("laryngeal", pos_a), grp("laryngeal", pos_b)
    pi, pb = grp("place", pos_a), grp("place", pos_b)
    if min(li.size, lb.size, pi.size, pb.size) == 0:
        return None
    def stat(li, lb, pi, pb):
        ga, gb = pi.mean() - li.mean(), pb.mean() - lb.mean()
        return ga, gb, gb - ga
    obs = stat(li, lb, pi, pb)
    bs = np.array([stat(rng.choice(li, li.size), rng.choice(lb, lb.size),
                        rng.choice(pi, pi.size), rng.choice(pb, pb.size))
                   for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"lar_a": li.mean(), "lar_b": lb.mean(), "place_a": pi.mean(), "place_b": pb.mean(),
            "gap_a": obs[0], "gap_a_lo": lo[0], "gap_a_hi": hi[0],
            "gap_b": obs[1], "gap_b_lo": lo[1], "gap_b_hi": hi[1],
            "dod_b_minus_a": obs[2], "dod_lo": lo[2], "dod_hi": hi[2],
            "n_lar_pairs": int(li.size), "n_place_pairs": int(pi.size),
            "pos_a": pos_a, "pos_b": pos_b}

frames, dods = [], []
RUNS = [
    ("pooled_2lvl", STOP_LAR, STOP_PLACE, ("initial", "noninitial"), meta2, None, False, N_PERM),
    ("testonly_2lvl", STOP_LAR, STOP_PLACE, ("initial", "noninitial"), meta2, "test", False, 0),
    ("pooled_2lvl_noflap", STOP_LAR, STOP_PLACE, ("initial", "noninitial"), meta2, None, True, 0),
    ("pooled_3lvl", STOP_LAR, STOP_PLACE, tp.POSITIONS, meta, None, False, 0),
    ("fric_pooled_2lvl", FRIC_LAR, STOP_PLACE, ("initial", "noninitial"), meta2, None, False, 0),
]
for label, lar, pla, pos, m, part, flap, nperm in RUNS:
    d = run(label, lar, pla, pos, m, part, flap, nperm)
    frames.append(d)
    if "noninitial" in pos:
        r = dod_boot(d)
        if r:
            r["run"] = label
            dods.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    else:
        it = tp.position_contrast_interaction(d, array="6v", n_boot=10000)
        it["run"] = label
        it.to_csv(R / "stage2_t1_stops_3lvl_interaction.csv", index=False)
        print(it.to_string(index=False), flush=True)

tp.LARYNGEAL_PAIRS[:] = ALL_LAR
tp.PLACE_CONTROL_PAIRS[:] = ALL_PLACE
pd.concat(frames, ignore_index=True).to_csv(R / "stage2_t1_stops_primary.csv", index=False)
pd.DataFrame(dods).to_csv(R / "stage2_t1_stops_dod.csv", index=False)
print("\nT1_STOPS_DONE", flush=True)
