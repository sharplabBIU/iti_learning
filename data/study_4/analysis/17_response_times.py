"""17 - Response times (RTs) after short and long inter-trial intervals (ITIs): the model-free response-time test.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ analyses study N (analysis_common/runner.py reads the folder name).

Prediction (Methods, Response times): if the ITI competes with value, a long ITI slows staying relative to switching after a
win and not after a loss; an outcome-blind cost or a fading trace of the last choice predicts the shift after both outcomes.
No model was fitted to response times.

Trials: every scored choice (a valid choice with a valid, consecutive previous choice) with RT >= 150 ms. Log RT (s) is
regressed by a Bayesian linear mixed model in PyMC (analysis_common.bayes_glmm, Gaussian family; No-U-Turn sampler,
4 chains x 1000 draws; independent random effects) with the last ITI (iti_c: -0.5 after 1 s, +0.5 after 4 s), the last
outcome (win_c: -0.5 loss, +0.5 win), stay (stay_c: -0.5 switch, +0.5 stay) and their interactions, with trial number
(centred, per 50 trials) as a fixed effect and a participant random intercept and trial slope:
  full:                                   log_rt ~ iti_c * win_c * stay_c + trial_c + (1 + trial_c | subject_id)
  per outcome (after wins; after losses): log_rt ~ iti_c * stay_c + trial_c + (1 + trial_c | subject_id)
Reported: posterior mean and 95% credible interval of the three-way interaction (full model) and of the ITI x stay
interaction after wins and after losses.

Inputs: data/trials.csv, main sample.
Outputs (results/17_response_times/): mixed_model_coefficients.csv (every fixed effect), response_times.json (the three
reported terms)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
import numpy as np
import pandas as pd
from analysis_common.runner import Runner
from analysis_common.behavior import switch_table
from analysis_common.bayes_glmm import fit_bglmm, summary_row

R = Runner(__file__)
R.banner("17 response times after short and long ITIs (model-free)")
out = R.out("17_response_times")
trials, P = R.load()
# Scored choices with RT >= 150 ms, and the effect-coded predictors (each -0.5 / +0.5, so main effects are
# averaged over the other factors).
tab = switch_table(trials)
tab = tab[tab.consec1 & (tab.rt >= 150)].copy()
d = pd.DataFrame(dict(subject_id=tab.subject_id, trial=tab.trial, log_rt=np.log(tab.rt / 1000.0), trial_c=(tab.trial - 50.5) / 50,
                      prev_iti=tab.prev_iti, prev_win=tab.prev_win.astype(int), stay=1 - tab.switch.astype(int)))
d["iti_c"] = np.where(d.prev_iti == 4, 0.5, -0.5); d["win_c"] = d.prev_win - 0.5; d["stay_c"] = d.stay - 0.5
print(f"{d.subject_id.nunique()} participants")

RE = " + trial_c + (1 + trial_c | subject_id)"   # time-on-task control shared by every model
specs = [("all", "log_rt ~ iti_c * win_c * stay_c" + RE), ("after_win", "log_rt ~ iti_c * stay_c" + RE),
         ("after_loss", "log_rt ~ iti_c * stay_c" + RE)]
rows, fits = [], {}
for subset, f in specs:
    dd = d if subset == "all" else d[d.prev_win == (1 if subset == "after_win" else 0)]
    r = fit_bglmm(dd, f, family="gaussian"); fits[subset] = r
    for term in r["fixed_terms"]:
        rows.append(dict(subset=subset, formula=f, **{k: v for k, v in summary_row(r, term).items() if k not in ("formula",)}))
    print(f"  {subset}: {f}: max R-hat {r['rhat_max']:.3f}", flush=True)
coef = pd.DataFrame(rows)
coef.to_csv(out / "mixed_model_coefficients.csv", index=False)
print(coef[["subset", "term", "estimate", "ci_low", "ci_high"]].round(4).to_string(index=False))

pick = lambda r: {k: r[k] for k in ("estimate", "sd", "ci_low", "ci_high", "rhat_max")}
R.save_json(out, "response_times", dict(
    n_participants=int(d.subject_id.nunique()),
    three_way=pick(summary_row(fits["all"], "iti_c:win_c:stay_c")),
    after_win_iti_x_stay=pick(summary_row(fits["after_win"], "iti_c:stay_c")),
    after_loss_iti_x_stay=pick(summary_row(fits["after_loss"], "iti_c:stay_c"))))
print("wrote", out)
