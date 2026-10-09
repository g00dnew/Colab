"""t1c_*.csv -> SUMMARY.md용 마크다운 표. 사용: python code/t1c_report.py > /tmp/t1c_report.md"""
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
MAIN = "main_heldout_trialcv_bal"
it = pd.read_csv(R / "t1c_interaction.csv")
pa = pd.read_csv(R / "t1c_pair_accuracy.csv")
sub = pd.read_csv(R / "t1c_voicing_subgroups.csv")
loo = pd.read_csv(R / "t1c_leave_one_pair_out.csv")
rs = pd.read_csv(R / "t1c_resample_summary.csv")
ORDER = ["main_heldout_trialcv_bal", "test_trialcv_bal", "pooled_trialcv_bal", "heldout_trialcv_nobal",
         "heldout_blockcv_bal", "heldout_sessioncv_bal", "heldout_trialcv_bal_wordcap20",
         "pooled_trialcv_bal_wordcap20", "heldout_wordcv_bal", "pooled_wordcv_bal",
         "heldout_trialcv_bal_center80", "lar6_heldout_trialcv_bal", "lar6_pooled_trialcv_bal",
         "ref_allmod_pooled_trialcv_nobal", "ref_allmod_pooled_randomcv_nobal"]
DESC = {"main_heldout_trialcv_bal": "**주분석** 발성·held_out·시행CV·구성균형",
        "test_trialcv_bal": "test 파티션만", "pooled_trialcv_bal": "발성 전체(pooled)",
        "heldout_trialcv_nobal": "held_out, 구성 비균형", "heldout_blockcv_bal": "블록 그룹 CV",
        "heldout_sessioncv_bal": "세션 그룹 CV", "heldout_trialcv_bal_wordcap20": "단어 상한 20",
        "pooled_trialcv_bal_wordcap20": "pooled + 단어 상한 20", "heldout_wordcv_bal": "단어 그룹 CV",
        "pooled_wordcv_bal": "pooled + 단어 그룹 CV", "heldout_trialcv_bal_center80": "80 ms 고정 중심창",
        "lar6_heldout_trialcv_bal": "후두 6쌍 vs 위치 9쌍 (held_out)", "lar6_pooled_trialcv_bal": "후두 6쌍 vs 위치 9쌍 (pooled)",
        "ref_allmod_pooled_trialcv_nobal": "이전 설정 + 시행 CV만", "ref_allmod_pooled_randomcv_nobal": "이전 T1′ 재현(무작위 CV)"}


def f3(x):
    return "—" if pd.isna(x) else f"{x:+.3f}"


def table_inter(lc, pc, pos_a="initial", pos_b="noninitial"):
    s = it[(it.lar_cat == lc) & (it.place_cat == pc) & (it.pos_a == pos_a) & (it.pos_b == pos_b)]
    s = s.set_index("run").reindex([r for r in ORDER if r in set(s.run)])
    print(f"\n**{lc} vs {pc}** ({pos_a}→{pos_b}). Δ = acc({pos_b}) − acc({pos_a}), D = Δ_place − Δ_lar, [쌍 대응 유지 부트스트랩 95% CI], p = 쌍 라벨 정확 순열(단측)\n")
    print("| 실행 | Δ_lar | Δ_place | D [95% CI] | p_label | 쌍 수 |")
    print("|---|---|---|---|---|---|")
    for run, r in s.iterrows():
        print(f"| {DESC.get(run, run)} | {f3(r.delta_lar)} | {f3(r.delta_place)} | **{f3(r.D)}** [{f3(r.D_lo)}, {f3(r.D_hi)}] | {r.p_label_one:.3f} | {int(r.n_lar_pairs)}+{int(r.n_place_pairs)} |")


print("### 상호작용 D — 실행별")
for lc, pc in [("stops_lar", "place_all"), ("stops_lar", "place_vl"), ("stops_lar", "place_vd"), ("fric_lar", "place_all"), ("lar6", "place9")]:
    table_inter(lc, pc)

print("\n### 주분석 쌍별 정확도 (n = 클래스당, 비어두 = 어중:어말 구성 균형)\n")
m = pa[pa.run == MAIN]
pv = m.pivot_table(index=["contrast", "pair"], columns="position", values=["acc", "n_per_class", "p_perm"])
print("| 대립 | 쌍 | 어두 | 비어두 | Δ | n | p_perm(어두/비어두) | 비어두 구성 |")
print("|---|---|---|---|---|---|---|---|")
mix = m[m.position == "noninitial"].set_index("pair")["subpos_mix"]
for (c, p), r in pv.iterrows():
    print(f"| {c} | {p} | {r[('acc','initial')]:.3f} | {r[('acc','noninitial')]:.3f} | {r[('acc','noninitial')]-r[('acc','initial')]:+.3f} | {int(r[('n_per_class','initial')])} | {r[('p_perm','initial')]:.3f}/{r[('p_perm','noninitial')]:.3f} | {mix.get(p,'')} |")

print("\n### 범주별 (주분석)\n")
s = sub[(sub.run == MAIN)]
print("| 범주 | 쌍 수 | 어두 | 비어두 | Δ | 쌍별 Δ |")
print("|---|---|---|---|---|---|")
for _, r in s.iterrows():
    print(f"| {r.category} | {r.n_pairs} | {r.acc_a:.3f} | {r.acc_b:.3f} | {r.delta:+.3f} | {r.pairs} |")

print("\n### 쌍 하나씩 제외 (주분석, stops_lar vs place_all)\n")
l = loo[loo.run == MAIN]
print("| 제외 | D [95% CI] |"); print("|---|---|")
for _, r in l.iterrows():
    print(f"| {r.dropped} | {r.D:+.3f} [{r.D_lo:+.3f}, {r.D_hi:+.3f}] |")

print("\n### 20회 독립 재표집 (주분석)\n")
print("| 대조 | D 평균 | 반복 2.5–97.5% | D>0 비율 | p_label(쌍 평균) | 쌍별 Δ 평균±SD |"); print("|---|---|---|---|---|---|")
for _, r in rs.iterrows():
    print(f"| {r.lar_cat} vs {r.place_cat} | {r.D_mean_over_reps:+.3f} | [{r.D_rep_p2_5:+.3f}, {r.D_rep_p97_5:+.3f}] | {r.frac_reps_D_pos:.2f} | {r.p_label_one:.3f} | {r.pairs} |")

print("\n### 구간 수준 위치 라벨 순열 (주분석, 주 검정)\n```")
print((R / "t1c_position_perm_summary.txt").read_text().strip())
print("```")

print("\n### 3수준 분리 (어두·어중·어말 각각, 구성 문제 없음)\n")
for run in ["main3_heldout_trialcv", "pooled3_trialcv"]:
    d = pa[pa.run == run]
    if d.empty:
        continue
    pv = d.pivot_table(index=["contrast", "pair"], columns="position", values="acc")
    n = d.pivot_table(index=["contrast", "pair"], columns="position", values="n_per_class")
    print(f"\n**{run}**\n\n| 대립 | 쌍 | 어두 | 어중 | 어말 | n |\n|---|---|---|---|---|---|")
    for (c, p), r in pv.iterrows():
        print(f"| {c} | {p} | {r.get('initial', float('nan')):.3f} | {r.get('medial', float('nan')):.3f} | {r.get('final', float('nan')):.3f} | {int(n.loc[(c,p)].min())} |")
    for pa_, pb_ in [("initial", "medial"), ("initial", "final")]:
        s = sub[(sub.run == run) & (sub.pos_a == pa_) & (sub.pos_b == pb_)]
        if s.empty:
            continue
        print(f"\n{pa_}→{pb_} 범주별 Δ: " + "; ".join(f"{r.category} {r.delta:+.3f} ({r.n_pairs}쌍)" for _, r in s.iterrows()))
        ii = it[(it.run == run) & (it.pos_a == pa_) & (it.pos_b == pb_)]
        print("  D: " + "; ".join(f"{r.lar_cat} vs {r.place_cat} {r.D:+.3f} [{r.D_lo:+.3f},{r.D_hi:+.3f}]" for _, r in ii.iterrows()))
