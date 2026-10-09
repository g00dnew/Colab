"""자질 정보의 시간 전개 (tuningTasks 음소 세션, 6v, PCA 폴드 내 적합).

200 ms 창(10 bin), 40 ms 간격(2 bin). go 기준(두 세션 pooled) + 04.26 발화 onset 기준.
창마다: 후두 파열음쌍 평균, 후두 마찰음쌍 평균, 위치쌍 평균, 39-way.
null: 기준창(-200ms)에서 라벨 순열 100회의 95백분위.
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

paths = tp.get_paths(local=True)
R = paths["RESULT_DIR"]
WIDTH = 10          # bins (200 ms)
STEP = 2            # bins (40 ms)
cols6v = tp.array_columns("6v")

def names(pairs, want):
    return [p for p in pairs if f"{p[0]}/{p[1]}" in want]
STOP_LAR = names(tp.LARYNGEAL_PAIRS, ["P/B", "T/D", "K/G"])
FRIC_LAR = names(tp.LARYNGEAL_PAIRS, ["F/V", "TH/DH", "S/Z", "SH/ZH"])
PLACE = list(tp.PLACE_CONTROL_PAIRS)

def windows(start, stop):
    return {f"w{s:+d}": (s, s + WIDTH) for s in range(start, stop - WIDTH + 1, STEP)}

def group_acc(X, phon, pairs, seed, n_perm=0):
    accs, nulls = [], []
    for a, b in pairs:
        m = (phon == a) | (phon == b)
        r = tp.balanced_pair_decode(X[m], phon[m], n_pca=30, n_perm=n_perm,
                                    perm_repeats=1, n_repeats=5, seed=seed)
        accs.append(r["acc"])
        if n_perm:
            nulls.append(r["acc_perm_p95"])
    return float(np.nanmean(accs)), (float(np.nanmean(nulls)) if nulls else np.nan)

def run_anchor(anchor_name, feats_by_window, phon, keep):
    rows = []
    first = True
    for wname, X in feats_by_window.items():
        s = int(wname[1:])
        Xw = X[keep][:, cols6v]
        p = phon[keep]
        nperm = 100 if first else 0
        out = {"anchor": anchor_name, "start_bin": s, "t_center_ms": (s + WIDTH / 2) * 20}
        for gname, pairs in (("lar_stop", STOP_LAR), ("lar_fric", FRIC_LAR), ("place", PLACE)):
            acc, null95 = group_acc(Xw, p, pairs, seed=20261009 + s, n_perm=nperm)
            out[f"{gname}_acc"] = acc
            if nperm:
                out[f"{gname}_null95"] = null95
        m39 = tp.multiclass_decode(Xw, p, n_pca=100, n_repeats=2, seed=20261009 + s)
        out["p39_acc"] = m39["acc"]
        rows.append(out)
        print(f"  [{anchor_name}] {out['t_center_ms']:+.0f} ms  stop {out['lar_stop_acc']:.3f}  "
              f"fric {out['lar_fric_acc']:.3f}  place {out['place_acc']:.3f}  39way {out['p39_acc']:.3f}",
              flush=True)
        first = False
    df = pd.DataFrame(rows)
    for g in ("lar_stop", "lar_fric", "place"):
        df[f"{g}_null95"] = df[f"{g}_null95"].ffill()
    return df

frames = []
# --- go 기준, 두 세션 pooled ---
GO_W = windows(-15, 75)
parts, phons = [], []
for name in tp.TUNING_PHONEME_SESSIONS:
    f = paths["TUNING_DIR"] / f"{name}.mat"
    sess = tp.load_tuning_session(f, variable_names=tp._TUNING_VARS + ["audioEnvelope"])
    ft = tp.stage1_trial_features(sess, windows=GO_W, zscore=True)
    parts.append(ft); phons.append(ft["phoneme"])
    if name.endswith("04.26_phonemes"):
        onset, flags = tp.estimate_audio_onsets(sess)
        ON_W = windows(-15, 60)
        ft_on = tp.stage1_trial_features(sess, windows=ON_W, zscore=True, anchor_bins=onset)
        keep_on = (ft_on["phoneme"] != "NOTHING") & (~flags)
        print(f"onset 기준 04.26: 시행 {int(keep_on.sum())}개", flush=True)
        frames.append(run_anchor("onset_0426", {w: ft_on[w] for w in ON_W}, ft_on["phoneme"], keep_on))
    del sess
phon = np.concatenate(phons)
feats_go = {w: np.concatenate([p[w] for p in parts], axis=0) for w in GO_W}
keep_go = phon != "NOTHING"
print(f"go 기준 pooled: 시행 {int(keep_go.sum())}개", flush=True)
frames.append(run_anchor("go_pooled", feats_go, phon, keep_go))

df = pd.concat(frames, ignore_index=True)
df.to_csv(R / "stage1_timecourse.csv", index=False)

# 요약: 피크·잠복기
summ = []
for anchor, g in df.groupby("anchor"):
    for grp in ("lar_stop", "lar_fric", "place"):
        acc = g[f"{grp}_acc"].to_numpy(); t = g["t_center_ms"].to_numpy()
        thr = g[f"{grp}_null95"].to_numpy()
        above = np.flatnonzero(acc > thr)
        summ.append({"anchor": anchor, "group": grp, "peak_ms": float(t[np.nanargmax(acc)]),
                     "peak_acc": float(np.nanmax(acc)),
                     "latency_ms": float(t[above[0]]) if above.size else np.nan,
                     "null95": float(np.nanmean(thr))})
summ = pd.DataFrame(summ)
summ.to_csv(R / "stage1_timecourse_summary.csv", index=False)
print(summ.to_string(index=False), flush=True)

# 그림
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, anchor, title in zip(axes, ("go_pooled", "onset_0426"),
                             ("go 기준 (두 세션)", "발화 onset 기준 (04.26)")):
    g = df[df["anchor"] == anchor]
    if not len(g):
        continue
    for grp, lab in (("place", "조음위치 쌍"), ("lar_stop", "후두 파열음 쌍"), ("lar_fric", "후두 마찰음 쌍")):
        ax.plot(g["t_center_ms"], g[f"{grp}_acc"], marker="o", ms=3, label=lab)
    ax.axhline(0.5, color="gray", lw=0.8); ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title(title); ax.set_xlabel("ms"); ax.set_ylabel("정확도 (우연 0.5)")
axes[0].legend(fontsize=8)
plt.tight_layout()
(R / "figures").mkdir(exist_ok=True)
plt.savefig(R / "figures" / "stage1_timecourse.png", dpi=160)
print("TIMECOURSE_DONE", flush=True)
