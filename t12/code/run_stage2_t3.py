"""T3 양식(발성 vs 무성 발화) — 정렬 품질과 날짜를 맞춘 비교군 설계.

held_out = (partition=='test') or (input_layer_from != session)
  미학습 세션(06.23, 07.29, 08.18, 08.23, 08.25)은 두 파티션 모두 held out.
비교군:
  primary   : nonvocal 4일 vs vocal 07.29 (둘 다 미학습 + 입력층 차용 = 품질 일치)
  secondary : nonvocal vs 날짜 인접 vocal (08.11, 08.13, 06.21, 06.28)
  tertiary  : nonvocal vs 전체 vocal held-out
민감도: 08.25(최악 PER) 제외
각 비교군마다 후두쌍 정확도, 조음위치쌍 정확도, (조음위치 - 후두) 격차와 부트스트랩 CI.
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

paths = tp.get_paths(local=True)
ARRAYS = ("6v", "6v_sup", "6v_inf")
NONVOCAL_ALL = sorted(tp.NONVOCAL_SESSIONS)
NONVOCAL_NO0825 = [s for s in NONVOCAL_ALL if s != "t12.2022.08.25"]
VOCAL_PRIMARY = ["t12.2022.07.29"]
VOCAL_DATEMATCH = ["t12.2022.08.11", "t12.2022.08.13",
                   "t12.2022.06.21", "t12.2022.06.28"]

per_table = tp.session_per_table(paths["ALIGN_DIR"])
align_raw = tp.load_alignments(paths["ALIGN_DIR"], window_mode="seg")
align, _ = tp.apply_score_filter(align_raw)
X, _, meta = tp.stage2_segment_features(paths, align)
print(f"\n세션({meta['session'].nunique()}) 양식 {meta['modality'].value_counts().to_dict()}",
      flush=True)
print("held_out 구간:", int(meta["held_out"].sum()), "/", len(meta), flush=True)
print("\nhold-out PER 요약:")
ho = per_table.copy()
ho["held_out"] = (ho["partition"] == "test") | ho["borrowed"]
print(ho[ho["held_out"]].groupby("modality")[["per", "n_trials"]]
      .agg({"per": "mean", "n_trials": "sum"}).round(4).to_string(), flush=True)

ARMS = [
    ("primary_0729", VOCAL_PRIMARY, NONVOCAL_ALL, None),
    ("primary_0729_no0825", VOCAL_PRIMARY, NONVOCAL_NO0825, None),
    ("secondary_datematch", VOCAL_DATEMATCH, NONVOCAL_ALL, None),
    ("tertiary_allvocal", None, NONVOCAL_ALL, None),
    ("primary_0729_initial", VOCAL_PRIMARY, NONVOCAL_ALL, "initial"),
]

frames, gap_frames = [], []
for label, vocal, nonvocal, positions in ARMS:
    avail = set(meta["session"])
    vsel = sorted(avail & set(vocal)) if vocal else None
    nsel = sorted(avail & set(nonvocal))
    print("\n" + "=" * 72)
    print(f"T3 [{label}]  vocal={vsel or 'all held-out'}  nonvocal={nsel}"
          f"  positions={positions or 'all'}")
    print("=" * 72, flush=True)
    if not nsel or (vocal and not vsel):
        print("해당 세션 정렬 없음 -> 생략", flush=True)
        continue
    d = tp.stage2_modality_decode(
        paths, X, meta, arrays=ARRAYS, n_perm=0, n_repeats=10, tag=f"_{label}",
        force=True, per_table=per_table, max_per_class=300, partitions="held_out",
        positions=positions, session_filter={"vocal": vsel, "nonvocal": nsel})
    if not len(d):
        print("시행 수를 맞출 수 있는 쌍이 없음", flush=True)
        continue
    d["comparator"] = label
    frames.append(d)
    print("\n양식 x 대립 유형 (정확도):")
    print(d.pivot_table(index=["array", "modality"], columns="contrast",
                        values="acc").round(4).to_string())
    print("\n쌍별 (area 6v):")
    g = d[d["array"] == "6v"]
    print(g.pivot_table(index=["contrast", "manner", "pair"], columns="modality",
                        values="acc").round(4).to_string())
    print("\n맞춘 n per class / 세션 PER:")
    print(g.groupby("modality")[["n_matched_per_class", "mean_per"]]
          .agg({"n_matched_per_class": "median", "mean_per": "mean"})
          .round(4).to_string())
    for arr in ARRAYS:
        gap = tp.contrast_gap_ci(d, group_col="modality",
                                 levels=["vocal", "nonvocal"], array=arr,
                                 n_boot=10000, dod_pair=("vocal", "nonvocal"))
        if len(gap):
            gap["comparator"] = label
            gap["positions"] = positions or "all"
            gap_frames.append(gap)
            if arr == "6v":
                print(f"\n격차 + DoD (area {arr}):")
                print(gap.to_string(index=False), flush=True)

if frames:
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(paths["RESULT_DIR"] / "stage2_t3_comparators.csv", index=False)
    print("\n" + "=" * 72)
    print("T3 요약: area 6v, 비교군 x 양식")
    print("=" * 72)
    print(allf[allf["array"] == "6v"]
          .pivot_table(index=["comparator", "modality"], columns="contrast",
                       values="acc").round(4).to_string())
if gap_frames:
    gaps = pd.concat(gap_frames, ignore_index=True)
    gaps.to_csv(paths["RESULT_DIR"] / "stage2_t3_gaps.csv", index=False)
    print("\nT3 격차 (area 6v):")
    print(gaps[gaps["array"] == "6v"].to_string(index=False))
print("\nT3_DONE", flush=True)
