"""계획 vs 실행 v2 (2026-10-09 저녁): 같은 T1′을 세 시간창에서 비교, 통제 분석과 같은 설정.

  seg      : 음소 조음 구간(정렬 seg 창)          -> 실행
  pre100   : 구간 시작 직전 100 ms (X_pre)           -> 계획+앞 음소 실행 혼재
  pre300   : 구간 시작 전 300~100 ms                 -> 계획 쪽
v2 변경: 발성 세션만, 문장 시행 그룹 CV, (음소·위치·단어) 상한 20, 비어두 어중:어말 구성 균형,
pre100이 온전한 행(pre_valid)만, pre300은 win_start>=0인 행만, 쌍 대응 유지 부트스트랩 + 쌍 라벨 정확 순열.
후두쌍 P/B·T/D·K/G(+마찰 F/V·TH/DH), 위치쌍 P/T·T/K·B/D·D/G. 6v, n 300, PCA 폴드 내.
"""
import itertools
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True)
R = paths["RESULT_DIR"]
print("Using pipeline:", tp.__file__, flush=True)


def _sel(pairs, names):
    out = [p for p in pairs if f"{p[0]}/{p[1]}" in names]
    assert len(out) == len(names)
    return out


ALL_LAR, ALL_PLACE = list(tp.LARYNGEAL_PAIRS), list(tp.PLACE_CONTROL_PAIRS)
STOP_LAR = _sel(ALL_LAR, ["P/B", "T/D", "K/G"])
FRIC_LAR = _sel(ALL_LAR, ["F/V", "TH/DH"])
STOP_PLACE = _sel(ALL_PLACE, ["P/T", "T/K", "B/D", "D/G"])
CATS = {"stops_lar": ["P/B", "T/D", "K/G"], "fric_lar": ["F/V", "TH/DH"],
        "place_vl": ["P/T", "T/K"], "place_vd": ["B/D", "D/G"], "place_all": ["P/T", "T/K", "B/D", "D/G"]}

align_raw = tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg")
align, _ = tp.apply_score_filter(align_raw)
align = align[align["modality"] == "vocal"].reset_index(drop=True)
print(f"발성 구간 {len(align)}", flush=True)
print("특징 1/2: seg + pre100", flush=True)
X_seg, X_pre100, meta = tp.stage2_segment_features(paths, align)
align_sh = align.copy()
align_sh["win_end"] = align["win_start"] - 5
align_sh["win_start"] = align["win_start"] - 15
align_sh = align_sh[align_sh["win_start"] >= 0]  # 시행 시작 이전으로 넘어가는 창 제외
print("특징 2/2: pre300", flush=True)
X_pre300, _, meta300 = tp.stage2_segment_features(paths, align_sh)
key = ["session", "partition", "trial_idx", "phone_idx"]
m300 = meta300[key].copy(); m300["_i300"] = np.arange(len(m300))
mm = meta[key].copy(); mm["_i"] = np.arange(len(mm))
j = mm.merge(m300, on=key, how="inner")
X_seg, X_pre100 = X_seg[j["_i"].to_numpy()], X_pre100[j["_i"].to_numpy()]
meta = meta.iloc[j["_i"].to_numpy()].reset_index(drop=True)
X_pre300 = X_pre300[j["_i300"].to_numpy()]
pv = meta["pre_valid"].to_numpy()
print(f"pre100 창 온전 {int(pv.sum())}/{len(meta)} (불완전 행 제외)", flush=True)
X_seg, X_pre100, X_pre300, meta = X_seg[pv], X_pre100[pv], X_pre300[pv], meta[pv].reset_index(drop=True)
assert X_seg.shape == X_pre100.shape == X_pre300.shape
print(f"공통 구간 {len(meta)}개", flush=True)

meta2 = meta.copy()
meta2["subpos"] = meta2["position_in_word"]
meta2["position_in_word"] = meta2["position_in_word"].replace({"medial": "noninitial", "final": "noninitial"})


def pair_delta(d):
    g = d[d["acc"].notna()].pivot_table(index=["pair", "contrast"], columns="position", values="acc")
    g = g.dropna(subset=["initial", "noninitial"]).copy()
    g["delta"] = g["noninitial"] - g["initial"]
    return g.reset_index()


def interaction(pdl, lar, pla, n_boot=10000, seed=1):
    dl = pdl[pdl["pair"].isin(lar)]["delta"].to_numpy()
    dp = pdl[pdl["pair"].isin(pla)]["delta"].to_numpy()
    if dl.size == 0 or dp.size == 0:
        return None
    D = float(dp.mean() - dl.mean())
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(dp, dp.size).mean() - rng.choice(dl, dl.size).mean() for _ in range(n_boot)])
    allv = np.concatenate([dl, dp]); n, k = allv.size, dl.size
    perms = np.array([allv[[i for i in range(n) if i not in c]].mean() - allv[list(c)].mean()
                      for c in itertools.combinations(range(n), k)])
    return dict(delta_lar=float(dl.mean()), delta_place=float(dp.mean()), D=D,
                D_lo=float(np.percentile(bs, 2.5)), D_hi=float(np.percentile(bs, 97.5)),
                p_label_one=float((perms >= D - 1e-12).mean()), n_lar=int(dl.size), n_place=int(dp.size))


frames, stats = [], []
tp.LARYNGEAL_PAIRS[:] = STOP_LAR + FRIC_LAR
tp.PLACE_CONTROL_PAIRS[:] = STOP_PLACE
for wname, X in (("seg", X_seg), ("pre100", X_pre100), ("pre300", X_pre300)):
    print("\n" + "=" * 60 + f"\n{wname}\n" + "=" * 60, flush=True)
    d = tp.stage2_position_decode(paths, X, meta2, arrays=("6v",), positions=["initial", "noninitial"],
                                  n_perm=0, n_repeats=10, tag=f"_plan2_{wname}", force=True, max_per_class=300,
                                  match_n_across_positions=True, partitions=None, group_col="trial",
                                  max_per_word=20, match_subposition_col="subpos")
    d["window"] = wname
    frames.append(d)
    print(d.pivot_table(index=["contrast", "pair"], columns="position", values="acc").round(3).to_string(), flush=True)
    pdl = pair_delta(d)
    for lc, pc in [("stops_lar", "place_all"), ("stops_lar", "place_vl"), ("stops_lar", "place_vd"), ("fric_lar", "place_all")]:
        r = interaction(pdl, CATS[lc], CATS[pc])
        if r:
            r.update(window=wname, lar_cat=lc, place_cat=pc)
            stats.append(r)
            print(f"  D[{lc} vs {pc}] = {r['D']:+.3f} [{r['D_lo']:+.3f}, {r['D_hi']:+.3f}] p_label={r['p_label_one']:.3f}", flush=True)
tp.LARYNGEAL_PAIRS[:] = ALL_LAR
tp.PLACE_CONTROL_PAIRS[:] = ALL_PLACE
pd.concat(frames, ignore_index=True).to_csv(R / "stage2_t1_planning_v2.csv", index=False)
pd.DataFrame(stats).to_csv(R / "stage2_t1_planning_v2_dod.csv", index=False)
print("\nPLANNING_V2_DONE", flush=True)
