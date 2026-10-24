"""05 - Individual differences (Study 3): do impulsivity and anhedonia predict how strongly the inter-trial interval (ITI) affects switching?

Preregistered (Study 3): impulsivity (BIS-11, Barratt Impulsiveness Scale, 8-item subset) is POSITIVELY and
anhedonia (SHAPS, Snaith-Hamilton Pleasure Scale) NEGATIVELY associated with the participant-level ITI
effect, p(switch | prior ITI 4 s) - p(switch | prior ITI 1 s) over all trials; one-sided Pearson
correlations (directions in analysis_common/config.py, prereg_direction).
Exploratory: impulsivity against the post-win ITI effect (Pearson and Spearman), and against each
participant's ITI slope from the mixed-effects model with the linear trial term (the posterior mean of the
fixed + random prev_iti effect, written by 02_iti_switching.py).

Inputs: data/participants.csv (questionnaire totals); results/02_switching/per_subject_switching.csv and
per_subject_iti_slope_trial_adjusted.csv (run 02 first).
Outputs (results/05_individual_differences/): correlations.csv (one row per questionnaire x index: Pearson r
with df and two- and one-sided p, Spearman rho with p) and per_subject_effects_with_questionnaires.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
import pandas as pd
from analysis_common.runner import Runner, fmt_p
from analysis_common.behavior import correlate

R = Runner(__file__)
R.banner("05 individual differences")
out = R.out("05_individual_differences")
trials, P = R.load()
qs = R.study["questionnaires"]
# Participant indices from 02, joined with the questionnaire totals.
sw = R.results_dir / "02_switching"
eff = pd.read_csv(sw / "per_subject_switching.csv")
adj = pd.read_csv(sw / "per_subject_iti_slope_trial_adjusted.csv")
for d in (eff, adj, P):
    d["subject_id"] = d["subject_id"].astype(str)
df = eff.merge(adj, on="subject_id", how="left").merge(P[["subject_id"] + [q["col"] for q in qs.values()]], on="subject_id", how="left")
df.to_csv(out / "per_subject_effects_with_questionnaires.csv", index=False)

# Correlations: the preregistered tests (each questionnaire x the ITI effect over all trials, one-sided in the
# predicted direction) and the exploratory tests of impulsivity (post-win ITI effect; trial-adjusted slope).
tests = [(qk, "iti_effect_all", True) for qk in qs] + [("bis", "iti_effect_win", False), ("bis", "iti_slope_trial_adjusted", False)]
rows = []
for qk, idx, prereg in tests:
    q = qs[qk]
    c = correlate(df[q["col"]], df[idx], direction=q["prereg_direction"] if prereg else None)
    c.update(questionnaire=q["label"], questionnaire_key=qk, index=idx, preregistered=prereg, df=c["n"] - 2)
    rows.append(c)
    one = f", one-sided p = {c['pearson_p_one_sided']:.3f}" if "pearson_p_one_sided" in c else ""
    print(f"  {q['label']:28s} x {idx:26s}: r({c['n'] - 2}) = {c['pearson_r']:+.3f}, {fmt_p(c['pearson_p_two_sided'])}{one}; "
          f"rho = {c['spearman_rho']:+.3f}, {fmt_p(c['spearman_p_two_sided'])} [{'preregistered' if prereg else 'exploratory'}]")
pd.DataFrame(rows).to_csv(out / "correlations.csv", index=False)
print("wrote", out)
