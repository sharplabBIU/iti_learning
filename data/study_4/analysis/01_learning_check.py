"""01 - Did participants learn? The chose-best rate against chance, and the participant flow.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ analyses study N (analysis_common/runner.py reads the folder name).

Question: do participants choose the option with the highest current reward probability ("chose best")
more often than chance (1/3 with three options)? Each participant's chose-best rate over responded trials
is tested against 1/3 with a one-sample t test (Results: "participants learned overall").

Sample: participants whose session reached trial 90 of 100 (the main sample, config.EXCLUSION_MAIN).

Inputs: data/participants.csv and data/exclusion_flow.csv (written by 00_prepare_data.py).
Outputs (results/01_learning/): learning_stats.json (n, mean chose-best rate, t, df, p) and exclusion_flow.csv
(how many participants pass each inclusion stage; the inclusion numbers of the Methods).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
import pandas as pd
from scipy import stats
from analysis_common.runner import Runner, fmt_p

R = Runner(__file__)
R.banner("01 learning check")
out = R.out("01_learning")
trials, P = R.load()

# Per-participant chose-best rate (responded trials) against chance, one-sample t test.
cb = P.set_index("subject_id")["chose_best_rate"].dropna()
t, p = stats.ttest_1samp(cb, 1 / 3)
R.save_json(out, "learning_stats", dict(n=int(len(cb)), chose_best_mean=float(cb.mean()), chose_best_sd=float(cb.std()),
                                        t_vs_chance=float(t), df=int(len(cb) - 1), p_vs_chance=float(p)))
pd.read_csv(R.paths["flow_csv"]).to_csv(out / "exclusion_flow.csv", index=False)
print(f"chose-best rate (n = {len(cb)}): M = {cb.mean():.3f} against 1/3, t({len(cb) - 1}) = {t:.2f}, {fmt_p(p)}")
print("wrote", out)
