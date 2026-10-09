"""계획 vs 실행: 같은 T1′(파열음 전용, 어두 vs 비어두)을 세 시간창에서 비교.

  seg      : 음소 조음 구간(정렬 seg 창)            -> 실행
  pre100   : 구간 시작 직전 100 ms (X_pre)             -> 계획+앞 음소 실행 혼재
  pre300   : 구간 시작 전 300~100 ms                   -> 계획 쪽
후두쌍 P/B·T/D·K/G, 위치쌍 P/T·T/K·B/D·D/G, 마찰음 대조 F/V·TH/DH. 6v, n 300, PCA 폴드 내.
"""
import os, sys
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True); R = paths["RESULT_DIR"]
def _sel(pairs, names):
    out = [p for p in pairs if f"{p[0]}/{p[1]}" in names]; assert len(out) == len(names); return out
ALL_LAR, ALL_PLACE = list(tp.LARYNGEAL_PAIRS), list(tp.PLACE_CONTROL_PAIRS)
STOP_LAR = _sel(ALL_LAR, ["P/B", "T/D", "K/G"]); STOP_PLACE = _sel(ALL_PLACE, ["P/T", "T/K", "B/D", "D/G"])
FRIC_LAR = _sel(ALL_LAR, ["F/V", "TH/DH"])

align_raw = tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg")
align, _ = tp.apply_score_filter(align_raw)
print("특징 1/2: seg + pre100", flush=True)
X_seg, X_pre100, meta = tp.stage2_segment_features(paths, align)
# pre300: 구간 시작 전 15~5 bin
align_sh = align.copy()
align_sh["win_end"] = align["win_start"] - 5
align_sh["win_start"] = align["win_start"] - 15
align_sh = align_sh[align_sh["win_end"] > 0]
print("특징 2/2: pre300", flush=True)
X_pre300, _, meta300 = tp.stage2_segment_features(paths, align_sh)
# meta300은 align_sh의 부분집합. 공통 키로 맞춘다
key = ["session", "partition", "trial_idx", "phone_idx"]
m300 = meta300[key].copy(); m300["_i300"] = np.arange(len(m300))
mm = meta[key].copy(); mm["_i"] = np.arange(len(mm))
j = mm.merge(m300, on=key, how="inner")
X_seg, X_pre100, meta = X_seg[j["_i"].to_numpy()], X_pre100[j["_i"].to_numpy()], meta.iloc[j["_i"].to_numpy()].reset_index(drop=True)
X_pre300 = X_pre300[j["_i300"].to_numpy()]
assert X_seg.shape == X_pre100.shape == X_pre300.shape, (X_seg.shape, X_pre300.shape)
print(f"공통 구간 {len(meta)}개", flush=True)

meta2 = meta.copy()
meta2["position_in_word"] = meta2["position_in_word"].replace({"medial": "noninitial", "final": "noninitial"})

def dod_boot(d, pos_a="initial", pos_b="noninitial", n_boot=10000, seed=1):
    rng = np.random.default_rng(seed); g = d[d["acc"].notna()]
    grp = lambda c, p: g[(g["contrast"] == c) & (g["position"] == p)]["acc"].to_numpy()
    li, lb, pi, pb = grp("laryngeal", pos_a), grp("laryngeal", pos_b), grp("place", pos_a), grp("place", pos_b)
    if min(li.size, lb.size, pi.size, pb.size) == 0: return None
    st = lambda li, lb, pi, pb: (pi.mean() - li.mean(), pb.mean() - lb.mean(), (pb.mean() - lb.mean()) - (pi.mean() - li.mean()))
    obs = st(li, lb, pi, pb)
    bs = np.array([st(rng.choice(li, li.size), rng.choice(lb, lb.size), rng.choice(pi, pi.size), rng.choice(pb, pb.size)) for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"lar_initial": li.mean(), "lar_noninitial": lb.mean(), "place_initial": pi.mean(), "place_noninitial": pb.mean(),
            "gap_initial": obs[0], "gap_i_lo": lo[0], "gap_i_hi": hi[0], "gap_noninitial": obs[1], "gap_n_lo": lo[1], "gap_n_hi": hi[1],
            "dod": obs[2], "dod_lo": lo[2], "dod_hi": hi[2], "n_lar_pairs": int(li.size), "n_place_pairs": int(pi.size)}

frames, dods = [], []
for wname, X in (("seg", X_seg), ("pre100", X_pre100), ("pre300", X_pre300)):
    for gname, lar in (("stops", STOP_LAR), ("fric", FRIC_LAR)):
        tp.LARYNGEAL_PAIRS[:] = lar; tp.PLACE_CONTROL_PAIRS[:] = STOP_PLACE
        label = f"{wname}_{gname}"
        print("\n" + "=" * 60 + f"\n{label}\n" + "=" * 60, flush=True)
        d = tp.stage2_position_decode(paths, X, meta2, arrays=("6v",), positions=["initial", "noninitial"],
                                      n_perm=0, n_repeats=10, tag=f"_plan_{label}", force=True, max_per_class=300,
                                      match_n_across_positions=True, partitions=None, drop_intervocalic_flap=False)
        d["window"], d["group"] = wname, gname; frames.append(d)
        print(d.pivot_table(index=["contrast", "pair"], columns="position", values="acc").round(3).to_string(), flush=True)
        r = dod_boot(d)
        if r: r.update({"window": wname, "group": gname}); dods.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
tp.LARYNGEAL_PAIRS[:] = ALL_LAR; tp.PLACE_CONTROL_PAIRS[:] = ALL_PLACE
pd.concat(frames, ignore_index=True).to_csv(R / "stage2_t1_planning.csv", index=False)
pd.DataFrame(dods).to_csv(R / "stage2_t1_planning_dod.csv", index=False)
print("\nPLANNING_DONE", flush=True)
