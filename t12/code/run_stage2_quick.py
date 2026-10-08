"""Stage 2 빠른 중간 점검: 순열검정 없이 T1/T2 헤드라인 수치만."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True)
ARRAYS = ("6v", "6v_sup", "6v_inf")

align = tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg")
X, X_pre, meta = tp.stage2_segment_features(paths, *tp.apply_score_filter(align)[:1],
                                            verbose=True)
print(f"\n세션 {sorted(meta['session'].unique())}", flush=True)
print(f"파티션 {meta['partition'].value_counts().to_dict()}", flush=True)
print(f"위치 {meta['position_in_word'].value_counts().to_dict()}", flush=True)

for scope, parts in [("pooled(train+test)", None), ("test-only", "test")]:
    print("\n" + "=" * 72)
    print(f"T1 — {scope}")
    print("=" * 72, flush=True)
    d = tp.stage2_position_decode(paths, X, meta, arrays=ARRAYS, n_perm=0,
                                  n_repeats=10, tag=f"_quick_{scope.split('(')[0]}",
                                  force=True, partitions=parts, max_per_class=300)
    if not len(d):
        print("결과 없음")
        continue
    print(d.pivot_table(index=["array", "position"], columns="contrast",
                        values="acc").round(4).to_string())
    print("\n쌍당 클래스 시행 수(중앙) / 사용 가능 최소:")
    print(d.groupby(["position", "contrast"])[["n_per_class", "n_available_min_class"]]
          .median().round(0).to_string())
    print("\n충분한 시행이 있던 쌍 (area 6v):")
    s = d[d["array"] == "6v"]
    print(s.pivot_table(index=["contrast", "manner", "pair"], columns="position",
                        values="acc").round(4).to_string())
    print("\nn per class (area 6v):")
    print(s.pivot_table(index="pair", columns="position",
                        values="n_available_min_class").astype("Int64").to_string())

print("\n" + "=" * 72)
print("T2 — 위치별 유표성 (pooled)")
print("=" * 72, flush=True)
m = tp.stage2_markedness_by_position(paths, X, meta, arrays=ARRAYS, n_reps=300,
                                     tag="_quick", force=True)
if len(m):
    print(m.pivot_table(index=["array", "position"], columns="series",
                        values="d2_cv").round(5).to_string())
    piv = m.pivot_table(index=["array", "position"], columns="series", values="d2_cv")
    piv["voiced_minus_voiceless"] = piv["voiced"] - piv["voiceless"]
    print("\n유성 - 무성:")
    print(piv[["voiced_minus_voiceless"]].round(5).to_string())
    print("\n음소별 (area 6v):")
    print(m[m["array"] == "6v"].pivot_table(index=["series", "phoneme"],
                                            columns="position",
                                            values="d2_cv").round(5).to_string())
print("\nQUICK_DONE", flush=True)
