"""02 - Preregistered main effect: a longer prior inter-trial interval (ITI) increases switching away from the previous choice.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ analyses study N (analysis_common/runner.py reads the folder name).

Question: is p(switch) on trial t higher when the ITI just before the choice (prev_iti, the ITI that
followed trial t-1) was 4 s rather than 1 s? A "switch" means choosing a different option from the one
chosen on the previous trial. Only pairs of consecutive responded trials are used (no missed trial
between them), so prev_iti is the actual wait between the two choices.

Analyses:
  - paired t test of each participant's p(switch) after a 1 s vs a 4 s ITI, over all trials, after a win
    and after a loss, and a paired t test of the post-win against the post-loss effect;
  - the preregistered mixed model with a linear trend over the session, fitted in PyMC.

Mixed models (generalized linear mixed models, GLMMs; analysis_common/bayes_glmm.py). Bernoulli-logit,
sampled with the No-U-Turn sampler (NUTS) via nutpie, 4 chains x 1000 draws after 1000 warm-up draws.
Priors: N(0, 1.5) on the intercept, N(0, 1) on the slopes, HalfNormal(1) on the random-effect standard
deviations. Participant random effects are independent of each other. prev_iti is in seconds (1 or 4).
trial_c = (trial - 50.5) / 50 is the linear trend; it enters every model as a fixed effect and as a
participant random slope:
  preregistered   switch ~ prev_iti + trial_c + (1 + prev_iti + trial_c | subject_id)
  by outcome      the same model fitted to the trials after a win and after a loss separately
  interaction     switch ~ prev_iti * prev_win + trial_c + (1 + prev_iti + trial_c | subject_id)
Reported per term: posterior mean and 95% credible interval, and the odds ratio for 4 s vs 1 s for prev_iti
terms. The participant-level ITI slopes (posterior means of the fixed + random prev_iti effect) are written
for 05_individual_differences.py.

Inputs: data/trials.csv, main sample.
Outputs (results/02_switching/): main_effect.json, switching_by_outcome.json, per_subject_switching.csv,
glmm.json (keys trial_controlled, trial_effect, by_outcome_trial_controlled, interaction_trial_controlled),
bglmm_*_fixef.csv / _subject.csv / .json, per_subject_iti_slope_trial_adjusted.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
from scipy import stats
from analysis_common.runner import Runner, fmt_p
from analysis_common.behavior import switch_table, paired_iti_test, subject_iti_effects
from analysis_common.bayes_glmm import fit_bglmm, summary_row

R = Runner(__file__)
R.banner("02 ITI -> switching (preregistered main effect)")
out = R.out("02_switching")
# One row per responded trial with lagged predictors (prev_iti, prev_win, ...); keep only rows whose
# previous trial was also responded and directly preceding (consec1).
trials, P = R.load()
tab = switch_table(trials)
tab = tab[tab["consec1"]].copy()
print(f"analysed: {tab.subject_id.nunique()} participants")

# ---- paired t tests: p(switch) after 1 s vs 4 s prior ITI -----------------------------------
main, _ = paired_iti_test(tab, outcome="switch")
print(f"all trials: p(switch) 1 s = {main['mean_short']:.3f}, 4 s = {main['mean_long']:.3f}, t({main['df']}) = {main['t']:.2f}, "
      f"{fmt_p(main['p_two_sided'])}")
R.save_json(out, "main_effect", dict(paired_test=main))

# By the previous outcome (prev_win = 1 after a win, 0 after a loss), and the post-win against the
# post-loss effect (per-participant difference of differences, paired t test).
by_out = {}
by_out["switch_after_win"], _ = paired_iti_test(tab[tab["prev_win"] == 1], outcome="switch")
by_out["switch_after_loss"], _ = paired_iti_test(tab[tab["prev_win"] == 0], outcome="switch")
for k in ("switch_after_win", "switch_after_loss"):
    r = by_out[k]
    print(f"{k}: 1 s = {r['mean_short']:.3f}, 4 s = {r['mean_long']:.3f}, t({r['df']}) = {r['t']:.2f}, {fmt_p(r['p_two_sided'])}")
eff = subject_iti_effects(tab)
both = eff.dropna(subset=["iti_effect_win", "iti_effect_loss"])
t_int, p_int = stats.ttest_rel(both["iti_effect_win"], both["iti_effect_loss"])
by_out["win_vs_loss_effect_difference"] = dict(n=len(both), df=len(both) - 1, mean_win=both["iti_effect_win"].mean(),
                                               mean_loss=both["iti_effect_loss"].mean(), t=float(t_int), p_two_sided=float(p_int))
print(f"post-win vs post-loss effect: t({len(both) - 1}) = {t_int:.2f}, {fmt_p(p_int)}")
R.save_json(out, "switching_by_outcome", by_out)
# Per participant: p(switch) after 1 s and 4 s ITIs and their difference, overall, after wins and after
# losses (read by 05_individual_differences.py).
eff.to_csv(out / "per_subject_switching.csv", index=False)


# ---- preregistered GLMM with a linear trend over the session (PyMC) ---------------------------
def report(lab, r):
    """Print one fixed effect: posterior mean, 95% credible interval and, for prev_iti terms, the odds ratio
    for 4 s vs 1 s."""
    print(f"    {lab}: b = {r['estimate']:+.4f}, CI = [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
          f"OR{' (4 vs 1 s)' if r['or_scale'] == 3 else ''} = {r['odds_ratio']:.3f}, CI = [{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}]; "
          f"R-hat {r['rhat_max']:.3f}")


# trial_c centres trial number on the middle of the 100 trials (50.5), one unit = 50 trials.
# summary_row(..., scale=3) gives the odds ratio for prev_iti = 4 s vs 1 s.
g = tab[["subject_id", "switch", "prev_iti", "prev_win", "trial"]].copy()
g["subject_id"] = g["subject_id"].astype(str)
g["trial_c"] = (g["trial"] - 50.5) / 50.0
F_MAIN = "switch ~ prev_iti + trial_c + (1 + prev_iti + trial_c | subject_id)"
res_tc = fit_bglmm(g, F_MAIN, out_prefix=out / "bglmm_main")
row_tc = summary_row(res_tc, "prev_iti", scale=3); row_trial = summary_row(res_tc, "trial_c")
report("prev_iti", row_tc); report("trial_c (per 50 trials)", row_trial)
# Interaction model: does the ITI effect differ after wins vs losses (prev_iti:prev_win)?
F_INT = "switch ~ prev_iti * prev_win + trial_c + (1 + prev_iti + trial_c | subject_id)"
res_int_tc = fit_bglmm(g, F_INT, out_prefix=out / "bglmm_iti_by_outcome")
row_int_tc = summary_row(res_int_tc, "prev_iti:prev_win", scale=3)
report("prev_iti:prev_win", row_int_tc)
# The preregistered model fitted separately to trials after a win and after a loss.
by_out_tc = {}
for w, lab in [(1, "after_win"), (0, "after_loss")]:
    sub = g[g["prev_win"] == w][["subject_id", "switch", "prev_iti", "trial_c"]]
    r = fit_bglmm(sub, F_MAIN, out_prefix=out / f"bglmm_{lab}")
    by_out_tc[lab] = summary_row(r, "prev_iti", scale=3)
    report(f"{lab}, prev_iti", by_out_tc[lab])
# Participant-level ITI slopes (posterior means of fixed + random effect) for 05_individual_differences.py.
res_tc["subject"][["subject_id", "prev_iti"]].rename(columns={"prev_iti": "iti_slope_trial_adjusted"}).to_csv(
    out / "per_subject_iti_slope_trial_adjusted.csv", index=False)
R.save_json(out, "glmm", dict(trial_controlled=row_tc, trial_effect=row_trial, interaction_trial_controlled=row_int_tc,
                              by_outcome_trial_controlled=by_out_tc))
print("wrote", out)
