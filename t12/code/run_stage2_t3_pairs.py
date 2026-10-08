"""T3 쌍별 낙폭 + 자연부류 검정 (proposal-v4.md 2.3 검정 1).

후두 마디가 하나의 부류로 약해지는가(파열/마찰/파찰에 균일한 낙폭, 조음위치·비음은
낙폭 없음), 아니면 조음 차이에 비례해 약해지는가.
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

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
paths = tp.get_paths(local=True)
NONVOCAL = sorted(tp.NONVOCAL_SESSIONS)
VOCAL_PRIMARY = ["t12.2022.07.29"]
VOCAL_DATEMATCH = ["t12.2022.06.21", "t12.2022.06.28",
                   "t12.2022.08.11", "t12.2022.08.13"]

align, _ = tp.apply_score_filter(
    tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg"))
X, _, meta = tp.stage2_segment_features(paths, align)
avail = set(meta["session"])
print(f"\n세션 {meta['session'].nunique()} | held_out {int(meta['held_out'].sum())}"
      f"/{len(meta)}", flush=True)

ARMS = [("primary_0729", VOCAL_PRIMARY),
        ("secondary_datematch", VOCAL_DATEMATCH),
        ("tertiary_allvocal", None)]

frames, summaries = [], []
for label, vocal in ARMS:
    vsel = sorted(avail & set(vocal)) if vocal else None
    nsel = sorted(avail & set(NONVOCAL))
    print("\n" + "=" * 78)
    print(f"T3 쌍별 낙폭 [{label}]  vocal={vsel or 'all held-out'}  nonvocal={nsel}")
    print("=" * 78, flush=True)
    d = tp.t3_pair_drops(X, meta, array="6v", n_rep=N_REP, max_per_class=300,
                         partitions="held_out",
                         session_filter={"vocal": vsel, "nonvocal": nsel})
    if not len(d):
        print("쌍 없음", flush=True)
        continue
    d["comparator"] = label
    frames.append(d)

    print("\n쌍별 표 (area 6v):")
    show = d[["contrast", "manner", "pair", "n_per_class", "acc_vocal",
              "acc_nonvocal", "drop", "drop_ci_lo", "drop_ci_hi"]]
    print(show.sort_values(["contrast", "manner", "pair"]).round(4)
          .to_string(index=False), flush=True)

    summ = tp.t3_natural_class_test(d)
    summ["comparator"] = label
    summaries.append(summ)
    print("\n요약 (a) 평균 낙폭 / (b) 균일성 검정:")
    print(summ.round(4).to_string(index=False), flush=True)
    if "p_uniform" in summ.attrs:
        print(f"  -> 후두쌍 낙폭 SD {summ.attrs['obs_sd']:.4f} vs 무작위 SD 평균 "
              f"{summ.attrs['null_sd_mean']:.4f}, p(더 균일) = "
              f"{summ.attrs['p_uniform']:.4f}", flush=True)

if frames:
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(paths["RESULT_DIR"] / "stage2_t3_pairs.csv", index=False)
    print(f"\n저장: stage2_t3_pairs.csv ({len(allf)}행)")
if summaries:
    alls = pd.concat(summaries, ignore_index=True)
    alls.to_csv(paths["RESULT_DIR"] / "stage2_t3_natural_class.csv", index=False)
    print(f"저장: stage2_t3_natural_class.csv ({len(alls)}행)")
print("\nT3PAIRS_DONE", flush=True)
