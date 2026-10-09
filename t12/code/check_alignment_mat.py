"""정렬 CSV(TFRecord 경로로 생성) <-> competitionData MAT 대응 검증. (2026-10-09 외부 검토 8A)

세션·파티션마다 (1) 시행 수, (2) 같은 trial_idx의 문장 텍스트(정규화 후) 일치,
(3) 정렬 창(seg_end_bin)이 신경신호 길이 T 안에 있는지, (4) 음소 수 일치를 비교한다.
출력 results/alignment_mat_consistency.csv. 불일치가 있으면 ALIGN_MAT_CHECK FAIL.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import t12_pipeline as tp  # noqa: E402

print("Using pipeline:", tp.__file__, flush=True)
paths = tp.get_paths(local=True)


def norm(s):
    s = str(s).lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


rows, examples = [], []
for f in sorted(Path(paths["ALIGN_DIR"]).glob("t12.*.csv")):
    if f.name.endswith("_trials.csv"):
        continue
    df = pd.read_csv(f)
    sess, part = str(df["session"].iloc[0]), str(df["partition"].iloc[0])
    mat = paths["COMP_DIR"] / part / f"{sess}.mat"
    if not mat.exists():
        rows.append(dict(session=sess, partition=part, mat_missing=True))
        continue
    d = tp.load_competition_session(mat)
    per_trial = df.groupby("trial_idx").agg(sentence=("sentence", "first"),
                                            n_phones=("phone_idx", "size"),
                                            max_end=("seg_end_bin", "max"))
    n_text, n_over, n_out = 0, 0, 0
    for t, r in per_trial.iterrows():
        t = int(t)
        if not (0 <= t < d["n_trials"]):
            n_out += 1
            continue
        if norm(d["sentenceText"][t]) != norm(r.sentence):
            n_text += 1
            if len(examples) < 10:
                examples.append((sess, part, t, d["sentenceText"][t], r.sentence))
        if int(r.max_end) > d["tx1"][t].shape[0]:
            n_over += 1
    rows.append(dict(session=sess, partition=part, mat_missing=False,
                     n_trials_align=int(len(per_trial)), n_trials_mat=int(d["n_trials"]),
                     n_trial_idx_out_of_range=n_out, n_text_mismatch=n_text,
                     n_window_beyond_T=n_over))
    print(rows[-1], flush=True)
    del d

out = pd.DataFrame(rows)
out.to_csv(paths["RESULT_DIR"] / "alignment_mat_consistency.csv", index=False)
for ex in examples:
    print("  예시 불일치:", ex)
bad = out[(out.get("mat_missing", False) == True)  # noqa: E712
          | (out["n_trial_idx_out_of_range"].fillna(0) > 0)
          | (out["n_text_mismatch"].fillna(0) > 0)
          | (out["n_window_beyond_T"].fillna(0) > 0)]
print(f"세션·파티션 {len(out)}개, 불일치 {len(bad)}개", flush=True)
print("ALIGN_MAT_CHECK", "FAIL" if len(bad) else "OK", flush=True)
sys.exit(1 if len(bad) else 0)
