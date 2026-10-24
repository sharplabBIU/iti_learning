"""The mixed model of switching on the jittered inter-trial interval (ITI) in the Tomov et al. (2020) fMRI bandit (Fig. 6a).

Question: in a public fMRI two-armed bandit in which the ITI before each choice was jittered for imaging, is a switch
away from the previous choice more likely after a longer ITI? (Results; Methods, External dataset.)

Trial table (from data/tomov2020/tidy.csv, written by prepare_tomov2020.py). A trial enters if a valid choice was
registered on it and on the immediately preceding trial of the same ten-trial run (one pair of arms); missed responses
(choice "0") are dropped first, so a trial after a miss does not enter. No participant is excluded. Columns:
  is_shift   1 if the arm chosen differs from the previous trial's, 0 if it is the same;
  iti_pre    the scheduled ITI before the choice, in seconds;
  iti_c      iti_pre minus the participant's median iti_pre (within-participant variation);
  t          trial within the scanning session, (block - 1) * 40 + (run - 1) * 10 + trial (session 2 numbers its runs
             5-8 in the raw files, so its t runs 41-200);
  trial_c    the trial within the current pair of arms, (t - 1) mod 10 + 1, centred on its midpoint and scaled per ten
             trials ((trial - 5.5) / 10), so the trend starts afresh with every new pair of arms;
  pid        participant.

Model (analysis_common/bayes_glmm.py, as for our own studies):
    is_shift ~ iti_c + trial_c + (1 + iti_c + trial_c | pid)
Bernoulli-logit; priors N(0, 1.5) on the intercept, N(0, 1) on the slopes, HalfNormal(1) on the participant SDs,
independent participant effects; NUTS via nutpie, 4 chains x 1000 draws after 1000 warm-up, target acceptance 0.9.
Reported: the posterior mean of the ITI slope (log-odds per second) and its 95% credible interval, and the odds ratio
for a 3 s difference.

Inputs: data/tomov2020/tidy.csv. Outputs: data/tomov2020/model_table.csv (the trial table; not shared, as the data are
not redistributed), results/tomov2020_iti_glmm_fixef.csv / _subject.csv / .json (the fit), and
results/tomov2020_iti_glmm_summary.json / .txt (the reported numbers).
Run: python3 tomov_iti_glmm.py (after prepare_tomov2020.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent))                     # the repository root
from analysis_common.bayes_glmm import fit_bglmm, summary_row   # noqa: E402

D = ROOT / "data" / "tomov2020"; RES = ROOT / "results"; RES.mkdir(exist_ok=True)
FORMULA = "is_shift ~ iti_c + trial_c + (1 + iti_c + trial_c | pid)"

# ---- the trial table -------------------------------------------------------------------------------------------------
d = pd.read_csv(D / "tidy.csv", dtype={"choice": str})
d["subject"] = d["subject"].astype(str); d["participant"] = d["participant"].astype(str)
for c in ("trial", "iti", "reward"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["choice"] = d["choice"].astype(str).str.strip().str.lower()
d = d[~d["choice"].isin(["0", "nan", "none", "", "-1", "na"])]          # missed responses
# The first trial of each run has no preceding ITI and is dropped here, before the previous-trial values are taken, so
# trial 2 fails the consecutive-trial check below and the usable trials of each run start at trial 3.
d = d.dropna(subset=["trial", "iti", "reward"])
d = d.sort_values(["subject", "trial"]).reset_index(drop=True)
g = d.groupby("subject", sort=False)                                     # subject = one ten-trial run of one session
d["prev_choice"] = g["choice"].shift(1); d["prev_reward"] = g["reward"].shift(1); d["prev_trial"] = g["trial"].shift(1)
d["iti_pre"] = d["iti"]                                                  # tidy.csv holds the ITI before each choice
d = d[d["trial"] - d["prev_trial"] == 1]
d["is_shift"] = (d["choice"] != d["prev_choice"]).astype(float)
d = d.dropna(subset=["is_shift", "iti_pre", "prev_reward"])
bl = d["subject"].str.extract(r"_b(\d+)_r(\d+)$").astype(int)
d["t"] = (bl[0] - 1) * 40 + (bl[1] - 1) * 10 + d["trial"]
d["iti_c"] = d["iti_pre"] - d.groupby("participant")["iti_pre"].transform("median")
d["pid"] = "tomov2020_" + d["participant"]
d["t_drift"] = (d["t"] - 1) % 10 + 1                                     # trial within the pair of arms
d["trial_c"] = (d["t_drift"] - (d["t_drift"].max() + 1) / 2) / 10.0
T = d[["pid", "is_shift", "iti_pre", "iti_c", "t", "trial_c", "prev_reward"]].reset_index(drop=True)
T.to_csv(D / "model_table.csv", index=False)

# ---- the fit ---------------------------------------------------------------------------------------------------------
lines = []
def out(s=""):
    """Print a line and keep it for the text report."""
    print(s, flush=True); lines.append(s)


out(f"Tomov et al. (2020): {T.pid.nunique()} participants, {len(T)} trials, switch rate {T.is_shift.mean():.3f}, "
    f"ITI {T.iti_pre.min():.2f}-{T.iti_pre.max():.2f} s")
out(FORMULA)
r = fit_bglmm(T, FORMULA, out_prefix=RES / "tomov2020_iti_glmm")
out(f"R-hat max {r['rhat_max']:.4f}, ESS min {r['ess_min']:.0f}, divergences {r['n_divergences']}, {r['seconds'] / 60:.1f} min")
out(r["fixef"].round(4).to_string(index=False))
iti3, tr = summary_row(r, "iti_c", scale=3), summary_row(r, "trial_c")
tau = r["_tau"]
sds = {c: dict(mean=float(tau[:, k].mean()), ci_low=float(np.quantile(tau[:, k], .025)), ci_high=float(np.quantile(tau[:, k], .975)))
       for k, c in enumerate(r["random_terms"])}
out(f"ITI slope: b = {iti3['estimate']:+.4f} log-odds per second, CI = [{iti3['ci_low']:+.4f}, {iti3['ci_high']:+.4f}]; odds ratio for "
    f"3 s {iti3['or_median']:.3f} [{iti3['or_ci_low']:.3f}, {iti3['or_ci_high']:.3f}]")
out(f"linear trend per ten trials: {tr['estimate']:+.4f} [{tr['ci_low']:+.4f}, {tr['ci_high']:+.4f}]")
out("participant SDs: " + ", ".join(f"{c} {v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]" for c, v in sds.items()))
summary = dict(formula=FORMULA, n_participants=int(T.pid.nunique()), n_trials=int(len(T)), switch_rate=float(T.is_shift.mean()),
               iti_c=iti3, trial_c=tr, participant_sd=sds,
               fit=dict(seconds=r["seconds"], rhat_max=r["rhat_max"], ess_min=r["ess_min"], n_divergences=r["n_divergences"]))
json.dump(summary, open(RES / "tomov2020_iti_glmm_summary.json", "w"), indent=1, default=float)
(RES / "tomov2020_iti_glmm_summary.txt").write_text("\n".join(lines) + "\n")
print("wrote", RES / "tomov2020_iti_glmm_summary.json")
