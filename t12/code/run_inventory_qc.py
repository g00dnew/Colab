"""인벤토리(셀 3) + QC(셀 4)를 실제 데이터로 실행한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t12_pipeline as tp  # noqa: E402

paths = tp.get_paths(local=True)

print("### 음성 자질표 ###", flush=True)
tp.build_phonetic_feature_table(paths, force=True)

print("\n### tuningTasks 인벤토리 ###", flush=True)
ti = tp.tuning_inventory(paths, force=True)
print(ti[["file", "task", "modality", "n_trials", "n_blocks", "n_cues",
          "trials_per_cue_min", "go_bins_mean", "total_sec",
          "trial_states"]].to_string(index=False), flush=True)

print("\n### tuningTasks cue별 시행 수 ###", flush=True)
cc = tp.tuning_cue_counts(paths, force=True)
print(cc.head(8).to_string(index=False), flush=True)

print("\n### tuningTasks QC ###", flush=True)
tq = tp.tuning_qc(paths, force=True)
print(tq.to_string(index=False), flush=True)

print("\n### competitionData 인벤토리 ###", flush=True)
ci = tp.competition_inventory(paths, force=True)
print(ci.drop(columns=["blocks"]).to_string(index=False), flush=True)
print("\n파티션 x 양식:", flush=True)
print(ci.groupby(["partition", "modality"]).agg(
    n_sessions=("session", "nunique"), n_trials=("n_trials", "sum"),
    total_sec=("total_sec", "sum")).to_string(), flush=True)
nonvocal = sorted(ci.loc[ci["modality"] == "nonvocal", "session"].unique())
print("nonvocal 세션:", nonvocal, flush=True)
assert set(nonvocal) <= tp.NONVOCAL_SESSIONS, "양식 라벨 불일치"

print("\n### competitionData QC ###", flush=True)
cq = tp.competition_qc(paths, force=True)
print(cq.head(16).to_string(index=False), flush=True)
print("\nNaN 총합:", int(cq["n_nan"].sum()), "| Inf 총합:", int(cq["n_inf"].sum()),
      "| 죽은 채널 최대:", int(cq["n_dead_chan"].max()), flush=True)
print("z 후 채널 평균 절대최대:", round(float(cq["z_chan_mean_absmax"].max()), 6),
      "| z 후 채널 sd 중앙:", round(float(cq["z_chan_std_med"].median()), 4), flush=True)
print("\n특징별 원 채널 통계 요약:", flush=True)
print(cq.groupby("feature")[["raw_chan_mean_med", "raw_chan_std_med",
                             "raw_chan_std_min", "raw_chan_std_max",
                             "n_dead_chan"]].describe().T.round(4).to_string(), flush=True)
print("\nINVQC_DONE", flush=True)
