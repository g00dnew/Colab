"""교차 위치 해독 + 위치 조건부 vs 통합 분류 (공학 검정). 2026-10-09 저녁.

질문 1(공학): 비어두 토큰을 맞힐 때, 비어두만으로 학습한 분류기(pos-specific)가
  어두를 섞어 학습한 분류기(pooled)보다 나은가? (데이터 적은 쪽이 같거나 이기면 자리별 모델링 근거)
질문 2(표상): 어두에서 학습한 후두 분류기가 비어두로 전이되는가? (전이 < 자기 위치 → 부호가 다름)

설정: 발성 세션 전체(pooled), 6v, seg 창, score>=0.5, 단어 상한 20(음소·위치·단어), 비어두는
어중:어말 구성 균형, 쌍별 클래스당 상한 300. 시행 그룹 5폴드 x 4반복(시험 폴드의 시행은 어떤
학습 조건에도 들어가지 않음). PCA(30)·스케일러는 학습 폴드에서만 적합.

조건(시험 = 위치 P의 시험 폴드 토큰):
  own       P 토큰만 학습 (클래스당 n_own)
  pooled    P 학습 토큰 + 다른 위치 Q 토큰 전부 (더 많은 데이터)
  matched   P 토큰 n_own/2 + Q 토큰 n_own/2 (총 n 동일)
  cross     Q 토큰만 학습 (클래스당 n_own)  -> 전이
P in {initial, noninitial}, Q = 반대 위치.
출력 results/cross_position_decoding.csv (쌍·조건·폴드 평균), results/cross_position_summary.csv
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
import t12_pipeline as tp  # noqa: E402

print("Using pipeline:", tp.__file__, flush=True)
paths = tp.get_paths(local=True)
R = paths["RESULT_DIR"]
CACHE = R / "t1c_cache"
N_SPLITS, N_REPEATS, N_PCA, CAP, WORDCAP, MIN_N = 5, 4, 30, 300, 20, 12
ONLY = os.environ.get("XP_PAIRS")  # 연기 테스트용 "P/B,T/D"
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.1f}m] {m}", flush=True)


def _sel(pairs, names):
    out = [p for p in pairs if f"{p[0]}/{p[1]}" in names]
    assert len(out) == len(names)
    return out


STOP_LAR = _sel(tp.LARYNGEAL_PAIRS, ["P/B", "T/D", "K/G"])
FRIC_LAR = _sel(tp.LARYNGEAL_PAIRS, ["F/V", "TH/DH"])
STOP_PLACE = _sel(tp.PLACE_CONTROL_PAIRS, ["P/T", "T/K", "B/D", "D/G"])
SPECS = ([(a, b, "laryngeal", "stops_lar") for a, b in STOP_LAR]
         + [(a, b, "laryngeal", "fric_lar") for a, b in FRIC_LAR]
         + [(a, b, "place", "place_vl" if a in ("P", "T", "K") else "place_vd") for a, b in STOP_PLACE])
if ONLY:
    keep = set(ONLY.split(","))
    SPECS = [s for s in SPECS if f"{s[0]}/{s[1]}" in keep]
POS = ["initial", "noninitial"]

# ---- 데이터 (t1c 캐시) ----
X = np.load(CACHE / "X_all.npy", mmap_mode="r")
meta = pd.read_pickle(CACHE / "meta_all.pkl")
m = meta.copy()
m["subpos"] = m["position_in_word"]
m["position_in_word"] = m["position_in_word"].replace({"medial": "noninitial", "final": "noninitial"})
vm = (m["modality"] == "vocal").to_numpy()
idx = np.flatnonzero(vm)
Xv = np.ascontiguousarray(np.asarray(X[idx])[:, tp.array_columns("6v")])
mv = m.iloc[idx].reset_index(drop=True)
pmask = np.ones(len(mv), dtype=bool)
n0 = int(pmask.sum())
pmask &= tp.cap_tokens_per_word(mv, pmask, WORDCAP, np.random.default_rng(20261009 + 7))
log(f"발성 구간 {n0} -> 단어 상한 {WORDCAP} 후 {int(pmask.sum())}")
phon = np.where(pmask, mv["phoneme"].to_numpy(), "__OUT__")
pos_all = mv["position_in_word"].to_numpy()
sub_all = mv["subpos"].astype(str).to_numpy()
groups = tp.segment_group_ids(mv, "trial")


def fit_predict(tr_idx, te_idx, seed):
    """학습 폴드에서만 PCA·스케일러·LDA 적합. 클래스 균형은 호출자가 맞춘다."""
    clf = tp._fold_model("lda", N_PCA, tr_idx.size, Xv.shape[1])
    clf.fit(Xv[tr_idx], phon[tr_idx])
    pred = clf.predict(Xv[te_idx])
    y = phon[te_idx]
    classes = np.unique(y)
    acc = float((pred == y).mean())
    bacc = float(np.mean([(pred[y == c] == c).mean() for c in classes])) if classes.size == 2 else acc
    return acc, bacc


def balanced_take(cands, a, b, n, rng):
    ia, ib = cands[phon[cands] == a], cands[phon[cands] == b]
    n = int(min(n, ia.size, ib.size))
    if n <= 0:
        return np.array([], dtype=int)
    return np.sort(np.concatenate([rng.choice(ia, n, replace=False), rng.choice(ib, n, replace=False)]))


rows = []
for js, (a, b, contrast, cat) in enumerate(SPECS):
    pair = f"{a}/{b}"
    # 위치별 토큰 집합(비어두는 구성 균형, 어두는 전체) + 쌍별 n 맞춤
    sel = {}
    for ip, p_ in enumerate(POS):
        rng = np.random.default_rng([7070, js, ip])
        idx_p, mix, n_av = tp.select_pair_segments(phon, pos_all, a, b, p_, CAP, sub_all, rng)
        sel[p_] = idx_p
    n_pair = min(min(int((phon[sel[p_]] == c).sum()) for c in (a, b)) for p_ in POS)
    if n_pair < MIN_N:
        log(f"  {pair}: n 부족({n_pair}) 건너뜀")
        continue
    # 두 위치 모두 클래스당 n_pair로 맞춤
    for ip, p_ in enumerate(POS):
        sel[p_] = balanced_take(sel[p_], a, b, n_pair, np.random.default_rng([7171, js, ip]))
    all_idx = np.concatenate([sel[p_] for p_ in POS])
    pos_lab = np.concatenate([np.full(sel[p_].size, p_) for p_ in POS])
    y_all = phon[all_idx]
    g_all = groups[all_idx]
    strat = np.array([f"{y}|{p}" for y, p in zip(y_all, pos_lab)])  # 클래스x위치로 층화
    res = {}
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=100 + rep)
        for fold, (tr, te) in enumerate(cv.split(all_idx, strat, g_all)):
            assert not (set(g_all[tr]) & set(g_all[te]))
            rng = np.random.default_rng([rep, fold, js])
            for P in POS:
                Q = POS[1] if P == POS[0] else POS[0]
                te_P = all_idx[te][pos_lab[te] == P]
                if np.unique(phon[te_P]).size < 2 or te_P.size < 4:
                    continue
                trP = all_idx[tr][pos_lab[tr] == P]
                trQ = all_idx[tr][pos_lab[tr] == Q]
                n_own = min(int((phon[trP] == c).sum()) for c in (a, b))
                if n_own < 6:
                    continue
                conds = {
                    "own": balanced_take(trP, a, b, n_own, rng),
                    "pooled": np.sort(np.concatenate([balanced_take(trP, a, b, n_own, rng),
                                                      balanced_take(trQ, a, b, n_own, rng)])),
                    "matched": np.sort(np.concatenate([balanced_take(trP, a, b, n_own // 2, rng),
                                                       balanced_take(trQ, a, b, n_own - n_own // 2, rng)])),
                    "cross": balanced_take(trQ, a, b, n_own, rng),
                }
                for cond, tr_idx in conds.items():
                    if tr_idx.size < 8:
                        continue
                    acc, bacc = fit_predict(tr_idx, te_P, rep)
                    res.setdefault((P, cond), []).append((acc, bacc, te_P.size, tr_idx.size))
    for (P, cond), v in res.items():
        v = np.asarray(v)
        rows.append(dict(pair=pair, contrast=contrast, category=cat, test_position=P, condition=cond,
                         n_per_class=n_pair, n_folds=len(v), acc=v[:, 0].mean(), acc_sd=v[:, 0].std(),
                         bacc=v[:, 1].mean(), n_test_mean=v[:, 2].mean(), n_train_mean=v[:, 3].mean()))
    d = pd.DataFrame([r for r in rows if r["pair"] == pair])
    log(f"  {pair} n={n_pair}: " + "; ".join(
        f"{r.test_position[:3]}/{r.condition} {r.acc:.3f}" for r in d.itertuples()))
    pd.DataFrame(rows).to_csv(R / "cross_position_decoding.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(R / "cross_position_decoding.csv", index=False)

# ---- 요약: 범주별 평균 + 쌍 대응 유지 부트스트랩 CI (조건 차이) ----
summ = []
rng = np.random.default_rng(1)
for cat in ["stops_lar", "fric_lar", "place_vl", "place_vd", "laryngeal", "place"]:
    s = df[df["category"] == cat] if cat in ("stops_lar", "fric_lar", "place_vl", "place_vd") else df[df["contrast"] == cat]
    if s.empty:
        continue
    for P in POS:
        w = s[s["test_position"] == P].pivot_table(index="pair", columns="condition", values="acc")
        if w.empty or "own" not in w:
            continue
        row = dict(category=cat, test_position=P, n_pairs=len(w))
        for cond in ["own", "pooled", "matched", "cross"]:
            if cond in w:
                row[f"acc_{cond}"] = float(w[cond].mean())
        for cond, name in [("pooled", "pooled_minus_own"), ("matched", "matched_minus_own"), ("cross", "cross_minus_own")]:
            if cond in w:
                diff = (w[cond] - w["own"]).to_numpy()
                bs = np.array([rng.choice(diff, diff.size).mean() for _ in range(5000)])
                row[name] = float(diff.mean())
                row[f"{name}_lo"], row[f"{name}_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                row[f"{name}_pairs_neg"] = int((diff < 0).sum())
        # 전이 비율: (cross - .5) / (own - .5)
        if "cross" in w:
            tr = ((w["cross"] - 0.5) / (w["own"] - 0.5)).replace([np.inf, -np.inf], np.nan)
            row["transfer_ratio"] = float(tr.mean())
        summ.append(row)
sm = pd.DataFrame(summ)
sm.to_csv(R / "cross_position_summary.csv", index=False)
print(sm.round(3).to_string(index=False), flush=True)
log("CROSS_POSITION_DONE")
