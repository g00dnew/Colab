"""T1′ 통제 분석 (2026-10-09 외부 검토 ①②③④ 반영). 사전 고정: results/t1_controls_prereg.md

단계형·병렬 실행. 드라이버: code/run_t1c_parallel.sh
  python run_t1_controls.py features                     특징 추출 -> results/t1c_cache/ (mmap 공유)
  python run_t1_controls.py runs --only a,b [--perm N]   실행 라벨 목록
  python run_t1_controls.py resample [--n-resample 20]   주분석 20회 독립 재표집
  python run_t1_controls.py posperm --perm-start S --perm-n N   구간 수준 위치 라벨 순열(분할)
  python run_t1_controls.py merge                        통계·표 생성
환경: T1C_SESSIONS(연기 테스트), T1C_MIN_PER_CLASS(기본 12).

주분석 = 발성 세션 · held_out(test + 미학습 07.29) · 문장 시행 그룹 CV · 비어두 어중:어말 구성 균형.
출력(results/): t1c_pair_accuracy.csv, t1c_interaction.csv, t1c_voicing_subgroups.csv,
  t1c_leave_one_pair_out.csv, t1c_resample_pairs.csv, t1c_resample_summary.csv,
  t1c_position_perm.csv, t1c_position_perm_summary.txt, t1c_cv_group_integrity.csv,
  t1c_alignment_sensitivity.csv
"""
import argparse
import itertools
import os
import pickle
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["features", "runs", "resample", "posperm", "merge", "all"])
ap.add_argument("--only", default="")
ap.add_argument("--perm", type=int, default=200)
ap.add_argument("--n-resample", type=int, default=20)
ap.add_argument("--perm-start", type=int, default=0)
ap.add_argument("--perm-n", type=int, default=200)
ap.add_argument("--n-draws", type=int, default=4, help="posperm: 표집 평균에 쓸 독립 균형 표집 수")
ap.add_argument("--part", default="held_out", help="posperm/resample 파티션: held_out | pooled | test")
ap.add_argument("--wordcap", type=int, default=0, help="posperm/resample 단어 상한(0=없음)")
ap.add_argument("--perm-rep", type=int, default=2, help="posperm 분류기 CV 반복 수")
A = ap.parse_args()

print("Using pipeline:", tp.__file__, flush=True)
SESSIONS = os.environ.get("T1C_SESSIONS")
SESSIONS = SESSIONS.split(",") if SESSIONS else None
MIN_PER_CLASS = int(os.environ.get("T1C_MIN_PER_CLASS", 12))
MAX_PER_CLASS = 300
paths = tp.get_paths(local=True)
R = paths["RESULT_DIR"]
CACHE = R / ("t1c_cache_smoke" if SESSIONS else "t1c_cache")
CACHE.mkdir(exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.1f}m] {msg}", flush=True)


def _sel(pairs, names):
    out = [p for p in pairs if f"{p[0]}/{p[1]}" in names]
    assert len(out) == len(names), (out, names)
    return out


ALL_LAR, ALL_PLACE = list(tp.LARYNGEAL_PAIRS), list(tp.PLACE_CONTROL_PAIRS)
STOP_LAR = _sel(ALL_LAR, ["P/B", "T/D", "K/G"])
FRIC_LAR = _sel(ALL_LAR, ["F/V", "TH/DH"])
STOP_PLACE = _sel(ALL_PLACE, ["P/T", "T/K", "B/D", "D/G"])
LAR6 = _sel(ALL_LAR, ["P/B", "T/D", "K/G", "F/V", "CH/JH", "TH/DH"])
SF = STOP_LAR + FRIC_LAR
CATS = {
    "stops_lar": ["P/B", "T/D", "K/G"], "fric_lar": ["F/V", "TH/DH"],
    "place_vl": ["P/T", "T/K"], "place_vd": ["B/D", "D/G"],
    "place_all": ["P/T", "T/K", "B/D", "D/G"],
    "lar6": [f"{a}/{b}" for a, b in LAR6], "place9": [f"{a}/{b}" for a, b in ALL_PLACE],
}
CONTRASTS = [("stops_lar", "place_all"), ("stops_lar", "place_vl"), ("stops_lar", "place_vd"),
             ("fric_lar", "place_all"), ("lar6", "place9")]
POS2, POS3 = ["initial", "noninitial"], list(tp.POSITIONS)

# label: (data, lar, place, positions, partitions, group_col, wordcap, subpos)
RUNS = {
    "main_heldout_trialcv_bal":        ("V", SF, STOP_PLACE, POS2, "held_out", "trial", None, True),
    "main3_heldout_trialcv":           ("V", SF, STOP_PLACE, POS3, "held_out", "trial", None, False),
    "heldout_trialcv_nobal":           ("V", SF, STOP_PLACE, POS2, "held_out", "trial", None, False),
    "test_trialcv_bal":                ("V", SF, STOP_PLACE, POS2, "test", "trial", None, True),
    "pooled_trialcv_bal":              ("V", SF, STOP_PLACE, POS2, None, "trial", None, True),
    "pooled3_trialcv":                 ("V", SF, STOP_PLACE, POS3, None, "trial", None, False),
    "heldout_blockcv_bal":             ("V", SF, STOP_PLACE, POS2, "held_out", "block", None, True),
    "heldout_sessioncv_bal":           ("V", SF, STOP_PLACE, POS2, "held_out", "session", None, True),
    "heldout_trialcv_bal_wordcap20":   ("V", SF, STOP_PLACE, POS2, "held_out", "trial", 20, True),
    "pooled_trialcv_bal_wordcap20":    ("V", SF, STOP_PLACE, POS2, None, "trial", 20, True),
    "heldout_wordcv_bal":              ("V", SF, STOP_PLACE, POS2, "held_out", "word", None, True),
    "pooled_wordcv_bal":               ("V", SF, STOP_PLACE, POS2, None, "word", None, True),
    "heldout_trialcv_bal_center80":    ("C", SF, STOP_PLACE, POS2, "held_out", "trial", None, True),
    "lar6_heldout_trialcv_bal":        ("V", LAR6, ALL_PLACE, POS2, "held_out", "trial", None, True),
    "lar6_pooled_trialcv_bal":         ("V", LAR6, ALL_PLACE, POS2, None, "trial", None, True),
    "ref_allmod_pooled_randomcv_nobal": ("A", SF, STOP_PLACE, POS2, None, None, None, False),
    "ref_allmod_pooled_trialcv_nobal":  ("A", SF, STOP_PLACE, POS2, None, "trial", None, False),
}
MAIN = "main_heldout_trialcv_bal"


# ------------------------------------------------------------------ 특징 캐시
def phase_features():
    log("정렬 로드")
    align_raw = tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg", sessions=SESSIONS)
    align, _ = tp.apply_score_filter(align_raw)
    log("특징 1/2: seg 창")
    X, _, meta = tp.stage2_segment_features(paths, align)
    assert X.shape == (len(meta), 512)
    np.save(CACHE / "X_all.npy", X.astype(np.float32))
    meta.to_pickle(CACHE / "meta_all.pkl")
    del X
    mid = (align["win_start"] + align["win_end"]) // 2
    align_c = align.assign(win_start=mid - 2, win_end=mid + 2)
    align_c = align_c[align_c["win_start"] >= 0].reset_index(drop=True)
    log("특징 2/2: 80 ms 고정 중심창")
    X_c, _, meta_c = tp.stage2_segment_features(paths, align_c)
    np.save(CACHE / "X_c.npy", X_c.astype(np.float32))
    meta_c.to_pickle(CACHE / "meta_c.pkl")
    (CACHE / "FEATURES_DONE").write_text(time.strftime("%F %T"))
    log("FEATURES_DONE")


def load_data(kind):
    """'V' 발성·2수준, 'V3' 발성·3수준, 'C' 중심창·발성·2수준, 'A' 전체 양식·2수준. mmap 공유."""
    if kind == "C":
        X = np.load(CACHE / "X_c.npy", mmap_mode="r")
        meta = pd.read_pickle(CACHE / "meta_c.pkl")
    else:
        X = np.load(CACHE / "X_all.npy", mmap_mode="r")
        meta = pd.read_pickle(CACHE / "meta_all.pkl")
    m = meta.copy()
    m["subpos"] = m["position_in_word"]
    if kind != "V3":
        m["position_in_word"] = m["position_in_word"].replace(
            {"medial": "noninitial", "final": "noninitial"})
    keep = np.ones(len(m), dtype=bool) if kind == "A" else (m["modality"] == "vocal").to_numpy()
    idx = np.flatnonzero(keep)
    return np.asarray(X[idx]), m.iloc[idx].reset_index(drop=True)


# ------------------------------------------------------------------ 실행
def phase_runs(labels, nperm):
    cache = {}
    for label in labels:
        kind, lar, pla, pos, part, grp, wcap, subpos = RUNS[label]
        dk = "V3" if (kind == "V" and pos == POS3) else kind
        if dk not in cache:
            cache[dk] = load_data(dk)
        X_, m = cache[dk]
        tp.LARYNGEAL_PAIRS[:] = lar
        tp.PLACE_CONTROL_PAIRS[:] = pla
        np_ = nperm if label == MAIN else 0
        log(f"=== {label} (perm {np_}, group={grp}, part={part}, subpos={subpos}, wordcap={wcap}) "
            f"n={len(m)}")
        d = tp.stage2_position_decode(
            paths, X_, m, arrays=("6v",), positions=list(pos), n_perm=np_,
            perm_repeats=2, n_repeats=10, tag=f"_t1c_{label}", force=True,
            max_per_class=MAX_PER_CLASS, match_n_across_positions=True, partitions=part,
            min_per_class=MIN_PER_CLASS, group_col=grp, max_per_word=wcap,
            match_subposition_col=("subpos" if subpos else None))
        if len(d):
            print(d.pivot_table(index=["contrast", "pair"], columns="position",
                                values="acc").round(3).to_string(), flush=True)
            print(d.pivot_table(index="pair", columns="position",
                                values="n_per_class").astype("Int64").to_string(), flush=True)
        (R / f"t1c_done_{label}").write_text(time.strftime("%F %T"))
    tp.LARYNGEAL_PAIRS[:] = ALL_LAR
    tp.PLACE_CONTROL_PAIRS[:] = ALL_PLACE
    log("RUNS_DONE " + ",".join(labels))


# ------------------------------------------------------------------ 주분석 보강 공통
def main_setup():
    XV2, MV2 = load_data("V")
    specs = [(a, b, "laryngeal") for a, b in SF] + [(a, b, "place") for a, b in STOP_PLACE]
    pmask, _ = tp._partition_mask(MV2, None if A.part == "pooled" else A.part)
    if A.wordcap:
        n0 = int(pmask.sum())
        pmask = pmask & tp.cap_tokens_per_word(MV2, pmask, A.wordcap, np.random.default_rng(20261009 + 7))
        log(f"  단어 상한 {A.wordcap}: {n0} -> {int(pmask.sum())}")
    phon = np.where(pmask, MV2["phoneme"].to_numpy(), "__OUT__")
    pos_all = MV2["position_in_word"].to_numpy()
    sub_all = MV2["subpos"].astype(str).to_numpy()
    groups = tp.segment_group_ids(MV2, "trial")
    Xa = np.ascontiguousarray(XV2[:, tp.array_columns("6v")])
    caps = {}
    for a, b, _ in specs:
        n_av = min(tp.select_pair_segments(phon, pos_all, a, b, p_, None, sub_all)[2] for p_ in POS2)
        caps[(a, b)] = min(n_av, MAX_PER_CLASS)
    return specs, phon, pos_all, sub_all, groups, Xa, caps


def decode_sel(Xa, phon, groups, idx, n_rep, seed, cap):
    r = tp.balanced_pair_decode(Xa[idx], phon[idx], n_pca=30, clf="lda", n_repeats=n_rep,
                                n_perm=0, seed=seed, min_per_class=MIN_PER_CLASS,
                                max_per_class=cap, groups=groups[idx])
    return r["acc"], r["n_per_class"]


def phase_resample(n_resample):
    specs, phon, pos_all, sub_all, groups, Xa, caps = main_setup()
    rows = []
    for js, (a, b, contrast) in enumerate(specs):
        cap = caps[(a, b)]
        if cap < MIN_PER_CLASS:
            continue
        for rep in range(n_resample):
            for ip, p_ in enumerate(POS2):
                rng = np.random.default_rng([4242, rep, js, ip])
                idx, mix, _ = tp.select_pair_segments(phon, pos_all, a, b, p_, cap, sub_all, rng)
                acc, n = decode_sel(Xa, phon, groups, idx, 5, 5000 + 97 * rep + js, cap)
                rows.append(dict(pair=f"{a}/{b}", contrast=contrast, rep=rep, position=p_,
                                 acc=acc, n_per_class=n, subpos_mix=mix))
        log(f"  재표집 {a}/{b} 완료 (cap {cap})")
    pd.DataFrame(rows).to_csv(R / "t1c_resample_pairs.csv", index=False)
    (R / "t1c_done_resample").write_text(time.strftime("%F %T"))
    log("RESAMPLE_DONE")


def phase_posperm(start, n):
    """구간 수준 위치 라벨 순열. 통계량 = R개 독립 균형 표집에서 구한 D의 평균(재표집 추정량과 일치).
    null: 각 표집 안에서 클래스별로 어두/비어두 라벨을 섞고 같은 그룹 CV로 D를 재계산, R개 평균."""
    specs, phon, pos_all, sub_all, groups, Xa, caps = main_setup()
    R_DRAWS, PERM_REP = A.n_draws, A.perm_rep
    sels = []
    for r in range(R_DRAWS):
        sel = {}
        for js, (a, b, contrast) in enumerate(specs):
            cap = caps[(a, b)]
            if cap < MIN_PER_CLASS:
                continue
            parts = {}
            for ip, p_ in enumerate(POS2):
                rng = np.random.default_rng([9090, r, js, ip])
                parts[p_] = tp.select_pair_segments(phon, pos_all, a, b, p_, cap, sub_all, rng)[0]
            sel[(a, b, contrast)] = parts
        sels.append(sel)

    def perm_D(assign, seed):
        deltas = {}
        for (a, b, _c), parts in assign.items():
            acc = {p_: decode_sel(Xa, phon, groups, parts[p_], PERM_REP, seed, caps[(a, b)])[0]
                   for p_ in POS2}
            deltas[f"{a}/{b}"] = acc["noninitial"] - acc["initial"]
        out = {}
        for lc, pc in CONTRASTS[:4]:
            dl = [deltas[p] for p in CATS[lc] if p in deltas]
            dp = [deltas[p] for p in CATS[pc] if p in deltas]
            out[f"{lc}|{pc}"] = (np.mean(dp) - np.mean(dl)) if dl and dp else np.nan
        return out, deltas

    def avg_over_draws(assigns, seed0):
        Ds, ds = [], []
        for r, assign in enumerate(assigns):
            D_r, d_r = perm_D(assign, seed0 + 13 * r)
            Ds.append(D_r)
            ds.append(d_r)
        D = {k: float(np.nanmean([x[k] for x in Ds])) for k in Ds[0]}
        d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
        return D, d

    rows = []
    if start == 0:
        obs_D, obs_delta = avg_over_draws(sels, 777)
        log(f"  관측 D({R_DRAWS}표집 평균, rep {PERM_REP}): " + ", ".join(f"{k}={v:+.3f}" for k, v in obs_D.items()))
        rows.append(dict(perm=-1, **obs_D, **{f"delta_{k}": v for k, v in obs_delta.items()}))
    for k in range(start, start + n):
        assigns = []
        for r, sel in enumerate(sels):
            prng = np.random.default_rng([31337, k, r])
            assign = {}
            for key, parts in sel.items():
                a, b, _ = key
                all_idx = np.concatenate([parts[p_] for p_ in POS2])
                lab = np.concatenate([np.full(parts[p_].size, p_) for p_ in POS2])
                new_lab = lab.copy()
                for c in (a, b):  # 클래스 안에서 위치 라벨만 섞는다(클래스×위치 n 보존)
                    ii = np.flatnonzero(phon[all_idx] == c)
                    new_lab[ii] = prng.permutation(lab[ii])
                assign[key] = {p_: np.sort(all_idx[new_lab == p_]) for p_ in POS2}
            assigns.append(assign)
        Dk, dk = avg_over_draws(assigns, 10_000 + 101 * k)
        rows.append(dict(perm=k, **Dk, **{f"delta_{kk}": v for kk, v in dk.items()}))
        if (k + 1 - start) % 5 == 0:
            log(f"  순열 {k + 1 - start}/{n}")
    tag = "" if (A.part == "held_out" and not A.wordcap) else f"_{A.part}_wc{A.wordcap}"
    pd.DataFrame(rows).to_csv(R / f"t1c_position_perm{tag}_part{start}.csv", index=False)
    (R / f"t1c_done_posperm{tag}{start}").write_text(time.strftime("%F %T"))
    log(f"POSPERM_DONE {start}+{n}")


# ------------------------------------------------------------------ 통계·병합
def pair_delta(d, pos_a, pos_b):
    g = d[d["acc"].notna()]
    if g.empty:
        return pd.DataFrame()
    p = g.pivot_table(index=["pair", "contrast"], columns="position", values="acc")
    if pos_a not in p.columns or pos_b not in p.columns:
        return pd.DataFrame()
    p = p.dropna(subset=[pos_a, pos_b]).copy()
    p["delta"] = p[pos_b] - p[pos_a]
    return p.reset_index()


def interaction(pdelta, lar_names, place_names, n_boot=10000, seed=1):
    """D = mean Δ_place - mean Δ_lar. 쌍 대응 유지 부트스트랩 CI + 쌍 라벨 정확 순열 p."""
    dl = pdelta[pdelta["pair"].isin(lar_names)]["delta"].to_numpy()
    dp = pdelta[pdelta["pair"].isin(place_names)]["delta"].to_numpy()
    if dl.size == 0 or dp.size == 0:
        return None
    D = float(dp.mean() - dl.mean())
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(dp, dp.size).mean() - rng.choice(dl, dl.size).mean()
                   for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    allv = np.concatenate([dl, dp])
    n, k = allv.size, dl.size
    perms = []
    for c in itertools.combinations(range(n), k):
        rest = [i for i in range(n) if i not in c]
        perms.append(allv[rest].mean() - allv[list(c)].mean())
    perms = np.asarray(perms)
    return dict(D=D, D_lo=float(lo), D_hi=float(hi),
                delta_lar=float(dl.mean()), delta_place=float(dp.mean()),
                n_lar_pairs=int(dl.size), n_place_pairs=int(dp.size),
                n_label_perm=int(perms.size),
                p_label_one=float((perms >= D - 1e-12).mean()),
                p_label_two=float((np.abs(perms) >= abs(D) - 1e-12).mean()))


def phase_merge():
    frames = []
    for label in RUNS:
        f = R / f"stage2_position_decoding_t1c_{label}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["run"] = label
            frames.append(d)
        else:
            log(f"  없음: {label}")
    allruns = pd.concat(frames, ignore_index=True)
    allruns.to_csv(R / "t1c_pair_accuracy.csv", index=False)
    inter_rows, loo_rows, cat_rows = [], [], []
    for label, d in allruns.groupby("run", sort=False):
        pos = RUNS[label][3]
        comps = [("initial", "noninitial")] if "noninitial" in pos else \
            [("initial", "medial"), ("initial", "final"), ("medial", "final")]
        for pa, pb in comps:
            pdl = pair_delta(d, pa, pb)
            if pdl.empty:
                continue
            for cat, names in CATS.items():
                s = pdl[pdl["pair"].isin(names)]
                if s.empty:
                    continue
                cat_rows.append(dict(run=label, pos_a=pa, pos_b=pb, category=cat, n_pairs=len(s),
                                     acc_a=float(s[pa].mean()), acc_b=float(s[pb].mean()),
                                     delta=float(s["delta"].mean()),
                                     delta_min=float(s["delta"].min()), delta_max=float(s["delta"].max()),
                                     pairs=";".join(f"{r.pair}:{r.delta:+.3f}" for r in s.itertuples())))
            for lc, pc in CONTRASTS:
                r = interaction(pdl, CATS[lc], CATS[pc])
                if r:
                    r.update(run=label, pos_a=pa, pos_b=pb, lar_cat=lc, place_cat=pc)
                    inter_rows.append(r)
                if lc == "stops_lar" and pc == "place_all":
                    for drop in CATS[lc] + CATS[pc]:
                        sub = pdl[pdl["pair"] != drop]
                        rr = interaction(sub, [p for p in CATS[lc] if p != drop],
                                         [p for p in CATS[pc] if p != drop], n_boot=2000)
                        if rr:
                            loo_rows.append(dict(run=label, pos_a=pa, pos_b=pb, dropped=drop, D=rr["D"],
                                                 D_lo=rr["D_lo"], D_hi=rr["D_hi"],
                                                 n_lar=rr["n_lar_pairs"], n_place=rr["n_place_pairs"]))
    it = pd.DataFrame(inter_rows)
    it.to_csv(R / "t1c_interaction.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(R / "t1c_voicing_subgroups.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(R / "t1c_leave_one_pair_out.csv", index=False)
    integ_cols = ["run", "pair", "position", "partitions", "cv_group_col", "n_cv_groups",
                  "n_folds_used", "n_test_min", "n_per_class", "subpos_mix", "max_per_word", "p_perm"]
    allruns[[c for c in integ_cols if c in allruns.columns]].to_csv(
        R / "t1c_cv_group_integrity.csv", index=False)
    sens = it[(it["lar_cat"] == "stops_lar") & (it["place_cat"] == "place_all")]
    sens.to_csv(R / "t1c_alignment_sensitivity.csv", index=False)
    print(sens[["run", "pos_a", "pos_b", "delta_lar", "delta_place", "D", "D_lo", "D_hi",
                "p_label_one"]].round(3).to_string(index=False), flush=True)

    # 회귀 확인: 이전 T1′(모든 양식·무작위 CV·비균형) 재현
    try:
        old = pd.concat([pd.read_csv(R / "stage2_position_decoding_stops_pooled_2lvl.csv"),
                         pd.read_csv(R / "stage2_position_decoding_stops_fric_pooled_2lvl.csv")])
        new = allruns[allruns["run"] == "ref_allmod_pooled_randomcv_nobal"]
        j = old.drop_duplicates(["pair", "position"]).merge(new, on=["pair", "position"],
                                                            suffixes=("_old", "_new"))
        diff = (j["acc_old"] - j["acc_new"]).abs().max()
        log(f"회귀 확인(이전 T1′ 재현): 쌍·위치 {len(j)}개, 정확도 최대 차 {diff:.4f}"
            + ("  OK" if diff < 1e-6 else "  !! 불일치"))
    except Exception as e:  # noqa: BLE001
        log(f"회귀 확인 생략: {e}")

    # 재표집 요약
    f = R / "t1c_resample_pairs.csv"
    summ = []
    if f.exists():
        rs = pd.read_csv(f)
        w = rs.pivot_table(index=["pair", "contrast", "rep"], columns="position", values="acc").reset_index()
        w["delta"] = w["noninitial"] - w["initial"]
        per_pair = w.groupby(["pair", "contrast"])["delta"].agg(["mean", "std", "count"]).reset_index()
        print(per_pair.round(4).to_string(index=False), flush=True)
        for lc, pc in CONTRASTS[:4]:
            d_rep = []
            for rep, g in w.groupby("rep"):
                dl = g[g["pair"].isin(CATS[lc])]["delta"]
                dp = g[g["pair"].isin(CATS[pc])]["delta"]
                if len(dl) and len(dp):
                    d_rep.append(dp.mean() - dl.mean())
            d_rep = np.asarray(d_rep)
            ex = interaction(per_pair.rename(columns={"mean": "delta"}), CATS[lc], CATS[pc])
            if ex is None or d_rep.size == 0:
                continue
            row = dict(lar_cat=lc, place_cat=pc, D_mean_over_reps=float(d_rep.mean()),
                       D_rep_p2_5=float(np.percentile(d_rep, 2.5)),
                       D_rep_p97_5=float(np.percentile(d_rep, 97.5)),
                       D_rep_sd=float(d_rep.std()), frac_reps_D_pos=float((d_rep > 0).mean()),
                       n_reps=int(d_rep.size), D_from_pair_means=ex["D"],
                       p_label_one=ex["p_label_one"], p_label_two=ex["p_label_two"],
                       n_label_perm=ex["n_label_perm"],
                       pairs=";".join(f"{r.pair}:{r['mean']:+.3f}±{r['std']:.3f}"
                                      for _, r in per_pair.iterrows() if r.pair in CATS[lc] + CATS[pc]))
            summ.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                   if k != "pairs"}, flush=True)
    pd.DataFrame(summ).to_csv(R / "t1c_resample_summary.csv", index=False)

    # 위치 라벨 순열 병합
    lines = []
    tags = sorted({p.name[len("t1c_position_perm"):p.name.index("_part")] for p in R.glob("t1c_position_perm*_part*.csv")})
    for tag in tags:
        parts = sorted(R.glob(f"t1c_position_perm{tag}_part*.csv"))
        pp = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        pp = pp.drop_duplicates("perm").sort_values("perm")
        pp.to_csv(R / f"t1c_position_perm{tag}.csv", index=False)
        obs = pp[pp["perm"] == -1]
        null = pp[pp["perm"] >= 0]
        lines.append(f"[{tag or 'main: held_out, no wordcap'}]")
        for col in [c for c in pp.columns if "|" in c]:
            if obs.empty:
                continue
            Do = float(obs[col].iloc[0])
            nv = null[col].dropna().to_numpy()
            if nv.size == 0 or not np.isfinite(Do):
                continue
            p1 = (1 + int((nv >= Do).sum())) / (nv.size + 1)
            p2 = (1 + int((np.abs(nv) >= abs(Do)).sum())) / (nv.size + 1)
            lines.append(f"{col}: D_obs={Do:+.3f} null_mean={nv.mean():+.3f} null_sd={nv.std():.3f} "
                         f"null_p95={np.percentile(nv, 95):+.3f} p_one={p1:.3f} p_two={p2:.3f} (n_perm={nv.size})")
    for ln in lines:
        print("  " + ln, flush=True)
    (R / "t1c_position_perm_summary.txt").write_text("\n".join(lines) + "\n")
    log("MERGE_DONE")


if A.phase == "features":
    phase_features()
elif A.phase == "runs":
    labels = [x for x in A.only.split(",") if x] or list(RUNS)
    phase_runs(labels, A.perm)
elif A.phase == "resample":
    phase_resample(A.n_resample)
elif A.phase == "posperm":
    phase_posperm(A.perm_start, A.perm_n)
elif A.phase == "merge":
    phase_merge()
else:
    phase_features()
    phase_runs(list(RUNS), A.perm)
    phase_resample(A.n_resample)
    phase_posperm(0, A.perm_n)
    phase_merge()
