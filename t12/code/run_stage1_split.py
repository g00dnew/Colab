"""6v/44 상·하 배열 분할 스캔 재생성 (PCA 폴드 내 적합판). go_all 창, 순열 없음."""
import os, sys
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True)
feats = tp.load_stage1_features(paths)
phon = feats["phoneme"]; keep = phon != "NOTHING"
X_all = feats["go_all"]
rows = []
for arr in ("6v_sup", "6v_inf", "6v", "44_sup", "44_inf"):
    X = X_all[:, tp.array_columns(arr)]
    specs = ([(a, b, "laryngeal") for a, b in tp.LARYNGEAL_PAIRS]
             + [(a, b, "place") for a, b in tp.PLACE_CONTROL_PAIRS])
    for a, b, contrast in specs:
        m = (phon == a) | (phon == b)
        r = tp.balanced_pair_decode(X[m], phon[m], n_pca=30, n_perm=0, n_repeats=10, seed=20261009)
        rows.append({"array": arr, "pair": f"{a}/{b}", "contrast": contrast,
                     "n": r["n_per_class"], "acc": r["acc"], "dprime": r["dprime"]})
    m39 = tp.multiclass_decode(X[keep], phon[keep], n_pca=100, n_repeats=3, seed=20261009)
    rows.append({"array": arr, "pair": "39way", "contrast": "39way", "n": m39["n_per_class"],
                 "acc": m39["acc"], "dprime": float("nan")})
    print(arr, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(paths["RESULT_DIR"] / "stage1_6v_split_scan.csv", index=False)
print(df.groupby(["array", "contrast"])["acc"].mean().unstack().round(3).to_string())
print("SPLIT_DONE", flush=True)
