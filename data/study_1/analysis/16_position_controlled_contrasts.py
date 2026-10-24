"""16 - The inter-trial interval (ITI) that followed an earlier choice, with the decline in switching over the session controlled.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ analyses study N (analysis_common/runner.py reads the folder name).

Question: does the ITI that followed the choice two, three or four trials back still affect switching on
trial t, once the ITI just elapsed is in the model (Supplementary Figs. 1 and 2; Results, carry-over)? The ITI
sequence is identical for every participant of a study, so a trial whose earlier ITI was 4 s comes from a
different part of the session than one whose earlier ITI was 1 s, while switching falls over the session.
Each contrast is therefore reported in two ways:
  raw   (descriptive, uncontrolled) each participant's p(switch) for the two trial types (ITI after the choice k back 1 s or
        4 s), their mean +/- s.e.m. across participants, the paired difference (4 s minus 1 s) and the mean trial number of
        each trial type (the position confound);
  GLMM  the preregistered mixed-effects logistic regression with a linear trend over the session: trial_c = (trial - 50.5) / 50
        entered as a fixed effect and as a participant random slope. Fitted in PyMC (analysis_common.bayes_glmm.fit_bglmm:
        No-U-Turn sampler via nutpie, 4 chains x 1000 draws, independent random effects). Reported per term: posterior mean and
        95% credible interval, and the odds ratio exp(b) with its 95% credible interval.
The predictor is prevK_long = 1 if the ITI that followed the choice K trials back was 4 s, 0 if 1 s, so exp(b) is the odds
ratio for 4 s against 1 s. The ITI just elapsed (prev_iti, s) enters, centred within each model's data, as prev_iti_c.

Models (key in summary.json "glmm"; files glmm_<key>_fixef.csv / _subject.csv / .json):
    twoback_same, twoback_different, threeback_different, fourback_different
        switch ~ prevK_long + prev_iti_c + trial_c + (1 + prevK_long + trial_c | subject_id)
        same: the option chosen two back is the option chosen last, a(t-2) = a(t-1); different: the option chosen K back
        differs from the last one, a(t-K) != a(t-1); the K preceding trials responded and consecutive.

Inputs: data/trials.csv, main sample.
Outputs (results/16_position_controlled/): raw_contrasts.csv, glmm_<key>_fixef.csv / _subject.csv / .json, summary.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
import pandas as pd
from analysis_common.runner import Runner
from analysis_common.behavior import switch_table, paired_iti_test
from analysis_common.bayes_glmm import fit_bglmm, summary_row

LAG_WORD = {2: "twoback", 3: "threeback", 4: "fourback"}
FORMULA = "switch ~ prev{k}_long + prev_iti_c + trial_c + (1 + prev{k}_long + trial_c | subject_id)"


def model_frame(d: pd.DataFrame, k: int) -> pd.DataFrame:
    """The columns a model needs: switch, prevK_long (4 s = 1, 1 s = 0), prev_iti_c (centred within these data), trial_c.
    Rows with a missing value are dropped; rows are sorted by participant and trial."""
    dd = d[["subject_id", "trial", "switch", "prev_iti", f"prev{k}_iti"]].dropna().copy()
    dd[f"prev{k}_long"] = (dd[f"prev{k}_iti"] == 4).astype(float)
    dd["prev_iti_c"] = dd["prev_iti"] - dd["prev_iti"].mean()
    dd["trial_c"] = (dd["trial"] - 50.5) / 50.0
    return dd.sort_values(["subject_id", "trial"], kind="stable").reset_index(drop=True)


def raw_contrast(dd: pd.DataFrame, k: int) -> dict:
    """Descriptive: participant means of p(switch) by the ITI after the choice k back (participants with both trial types),
    the paired difference 4 s minus 1 s, and the mean trial number of each trial type over trials (the position confound)."""
    s, _ = paired_iti_test(dd, "switch", cond=f"prev{k}_iti", levels=(1, 4))
    mt = dd.groupby(f"prev{k}_iti")["trial"].mean()
    return dict(n=s["n"], mean_trial_1s=float(mt.loc[1.0]), mean_trial_4s=float(mt.loc[4.0]),
                mean_1s=s["mean_short"], sem_1s=s["sem_short"], mean_4s=s["mean_long"], sem_4s=s["sem_long"],
                diff=s["diff_mean"], diff_sem=s["diff_sem"], t=s["t"], df=s["df"], p_two_sided=s["p_two_sided"])


R = Runner(__file__)
R.banner("16 ITI after an earlier choice: raw contrasts and mixed models with a linear trend over the session")
out = R.out("16_position_controlled")
trials, P = R.load()
tab = switch_table(trials)
two = tab[tab.consec2]                                          # the two preceding trials responded and consecutive


def lag_different(k):
    """a(t-k) != a(t-1), the k preceding trials responded and consecutive."""
    return tab[tab[f"consec{k}"] & (tab[f"prev{k}_choice"] != tab["prev1_choice"])]


# One entry per model: (data, lag k).
SPECS = {"twoback_same": (two[two.same_prev2], 2)}
for k in (2, 3, 4):
    SPECS[f"{LAG_WORD[k]}_different"] = (lag_different(k), k)

rows, gl = [], {}
for key, (d, k) in SPECS.items():
    dd = model_frame(d, k)
    rows.append(dict(subset=key, lag=k, **raw_contrast(dd, k)))
    formula = FORMULA.format(k=k)
    res = fit_bglmm(dd, formula, out_prefix=out / f"glmm_{key}")
    rec = summary_row(res, f"prev{k}_long", scale=1.0)
    rec["last_iti_per_s"] = summary_row(res, "prev_iti_c", scale=3.0)   # the ITI just elapsed, in the same model
    gl[key] = rec
    print(f"[{key}] {formula}: b = {rec['estimate']:+.4f}, CI = [{rec['ci_low']:+.4f}, {rec['ci_high']:+.4f}]; "
          f"OR {rec['odds_ratio']:.3f}, CI = [{rec['or_ci_low']:.3f}, {rec['or_ci_high']:.3f}]; R-hat {rec['rhat_max']:.3f}", flush=True)
Cr = pd.DataFrame(rows); Cr.to_csv(out / "raw_contrasts.csv", index=False)
print(Cr.round(4).to_string(index=False))
R.save_json(out, "summary", dict(raw_contrasts=Cr.to_dict(orient="records"), glmm=gl))
print("wrote", out)
