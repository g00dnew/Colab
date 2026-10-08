"""Stage 1 전체 실행 스크립트(순열검정 포함). 백그라운드 실행용."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True)
feats = tp.load_stage1_features(paths)

print("\n### 39음소 분류 (전건전성) ###", flush=True)
tp.stage1_phoneme39(paths, feats=feats, window="go_all", n_repeats=3, force=True)

print("\n### 창별/어레이별 무순열 스캔 ###", flush=True)
scan = []
for w in ["go_0_500ms", "go_500_1000ms", "go_all"]:
    for arr in ["6v", "44", "all"]:
        X = feats[w][:, tp.array_columns(arr)]
        specs = ([(a, b, "laryngeal") for a, b in tp.LARYNGEAL_PAIRS]
                 + [(a, b, "place") for a, b in tp.PLACE_CONTROL_PAIRS])
        for a, b, contrast in specs:
            m = (feats["phoneme"] == a) | (feats["phoneme"] == b)
            r = tp.balanced_pair_decode(X[m], feats["phoneme"][m], n_perm=0,
                                        n_repeats=10, seed=20261009)
            scan.append({"window": w, "array": arr, "pair": f"{a}/{b}",
                         "contrast": contrast, "n_per_class": r["n_per_class"],
                         "acc": r["acc"], "acc_sd": r["acc_sd"], "dprime": r["dprime"]})
pd.DataFrame(scan).to_csv(paths["RESULT_DIR"] / "stage1_window_scan.csv", index=False)
print("저장: stage1_window_scan.csv", flush=True)

print("\n### 유표성 (T2) ###", flush=True)
mark = tp.stage1_markedness(paths, feats=feats, window="go_all", n_reps=400, force=True)
tests = [tp.markedness_contrast_test(mark, array=a) for a in ("6v", "44", "all")]
pd.DataFrame(tests).to_csv(paths["RESULT_DIR"] / "stage1_markedness_test.csv", index=False)
for t in tests:
    print("  ", t, flush=True)

print("\n### 최소대립쌍 순열검정 (500회) ###", flush=True)
tp.stage1_run(paths, feats=feats, windows=("go_all", "go_0_500ms"),
              arrays=("6v", "44", "all"), n_perm=500, perm_repeats=2,
              n_repeats=10, n_pca=30, force=True)
print("\nSTAGE1_DONE", flush=True)
