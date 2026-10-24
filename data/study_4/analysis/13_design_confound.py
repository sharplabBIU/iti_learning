"""13 - The fixed inter-trial interval (ITI) sequence and the decline in switching over the session.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ analyses study N (analysis_common/runner.py reads the folder name).

The ITI sequence is a fixed sequence shared by all participants (front-loaded with long ITIs in Studies 1-3, balanced in
Study 4), while switching declines over the session. This script documents both (Results and Methods):
  - the share of long (4 s) ITIs in successive blocks of twenty trials, and the correlation between a long ITI and trial
    number in the first (trials 1-50) and in the second half (trials 51-100) of the session;
  - p(switch) in successive blocks of twenty trials (participant means, then averaged), and each participant's
    least-squares slope of switching on trial number (per 100 trials), tested against zero (one-sample t test).
The ITI effect itself is estimated with trial number in every mixed model (02_iti_switching.py).

Inputs: data/trials.csv, main sample.
Outputs (results/13_design_confound/): design_confound.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
import numpy as np
import pandas as pd
from scipy import stats
from analysis_common.runner import Runner, fmt_p
from analysis_common.behavior import switch_table

R = Runner(__file__)
R.banner("13 fixed ITI sequence and the decline in switching")
out = R.out("13_design_confound")
trials, P = R.load()

# The ITI that followed each trial position (the same for every participant), by 20-trial block, and its
# correlation with trial number within each half of the session.
seq = trials.groupby("trial")["iti"].agg(lambda x: x.mode().iloc[0])
long_ = (seq == 4).astype(float)
by_block = long_.groupby(pd.cut(long_.index, [0, 20, 40, 60, 80, 100]), observed=True).mean()
halves = {}
for lab, (a, b) in (("trials_1_50", (1, 50)), ("trials_51_100", (51, 100))):
    x = long_[(long_.index >= a) & (long_.index <= b)]
    r, p = stats.pearsonr(x.index, x.values)
    halves[lab] = dict(r=float(r), p=float(p))
# Switching by 20-trial block (participant means, then averaged), and each participant's linear slope of
# switching (0/1) on trial / 100, tested against 0.
tab = switch_table(trials); tab = tab[tab["consec1"]].copy()
sw_by_block = tab.assign(block=pd.cut(tab["trial"], [1, 20, 40, 60, 80, 100])).groupby(["subject_id", "block"], observed=True)["switch"].mean().unstack().mean()
slopes = tab.groupby("subject_id").apply(lambda g: np.polyfit(g["trial"] / 100.0, g["switch"], 1)[0] if g["trial"].nunique() > 5 else np.nan).dropna()
t_sl, p_sl = stats.ttest_1samp(slopes, 0)
R.save_json(out, "design_confound", dict(
    long_iti_share_by_20trial_block={str(k): float(v) for k, v in by_block.items()},
    corr_long_iti_with_trial_by_half=halves,
    p_switch_by_20trial_block={str(k): float(v) for k, v in sw_by_block.items()},
    within_participant_switch_slope_per_100_trials=dict(mean=float(slopes.mean()), t=float(t_sl), df=int(len(slopes) - 1), p=float(p_sl))))
print("share of 4 s ITIs by 20-trial block: " + ", ".join(f"{v:.2f}" for v in by_block))
print("correlation of a 4 s ITI with trial number: " + ", ".join(f"{k} r = {v['r']:+.2f}" for k, v in halves.items()))
print("p(switch) by 20-trial block: " + ", ".join(f"{v:.3f}" for v in sw_by_block)
      + f"; within-participant slope {slopes.mean():+.3f} per 100 trials, t({len(slopes) - 1}) = {t_sl:.1f}, {fmt_p(p_sl)}")
print("wrote", out)
