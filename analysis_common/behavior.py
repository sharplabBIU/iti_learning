"""Model-agnostic behavioural analyses shared by all studies.

These functions measure the inter-trial interval (ITI) effect directly from choices, without a
computational model: does a participant switch to another option more often after a long (4 s) than
after a short (1 s) ITI? Nothing is read from or written to disk here; the study scripts call these
functions and save the results.

All functions take the tidy trial table from data.py (or the switch table built here):
  * switch_table: one row per responded choice with lagged trials, ITIs, choices and outcomes
  * paired_iti_test: p(switch) per participant after a 1 s vs a 4 s prior ITI, paired t test
  * subject_iti_effects: participant-level ITI effects (overall, after wins, after losses)
  * correlate: Pearson and Spearman correlations, with one-sided p in a preregistered direction

"PRIOR ITI" is the ITI just before the current choice, i.e. after the previous trial's outcome.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# trial-level switch table
# ---------------------------------------------------------------------------
def switch_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Responded trials with lagged predictors. Row t describes the choice on trial t.

    prevK_trial, prevK_iti, prevK_choice, prevK_win (K = 1-4) = trial number, ITI that followed,
                choice and outcome of the K-th previous responded trial; prev_iti (= prev1_iti) is
                the ITI that preceded the current choice (iti stored on row t-1)
    consecK   = the K preceding responded trials are consecutive in trial number
                (no missed trial in between), so the lagged ITIs are the actual gaps
    switch    = 1 if the choice differs from the previous responded choice (stay = 1 - switch)
    prev_win  = 1 if the previous responded trial was won
    same_prev2 = the two previous choices were the same option
    half      = "early" (trials 1-50) or "late" (51-100)
    The first responded trial of each participant has no previous choice and is dropped.
    """
    v = trials[trials["valid"]].sort_values(["subject_id", "trial"]).copy()
    g = v.groupby("subject_id")
    # Lags 1-4 of trial number, ITI, choice and outcome, over each participant's responded trials
    for k in (1, 2, 3, 4):
        v[f"prev{k}_trial"] = g["trial"].shift(k)
        v[f"prev{k}_iti"] = g["iti"].shift(k)
        v[f"prev{k}_choice"] = g["choice"].shift(k)
        v[f"prev{k}_win"] = g["win"].shift(k)
    v["consec1"] = (v["trial"] - v["prev1_trial"]) == 1
    v["consec2"] = v["consec1"] & ((v["prev1_trial"] - v["prev2_trial"]) == 1)
    v["consec3"] = v["consec2"] & ((v["prev2_trial"] - v["prev3_trial"]) == 1)
    v["consec4"] = v["consec3"] & ((v["prev3_trial"] - v["prev4_trial"]) == 1)
    # Outcome variables and short aliases for the lag-1 columns
    v = v[v["prev1_choice"].notna()].copy()
    v["switch"] = (v["choice"] != v["prev1_choice"]).astype(int)
    v["stay"] = 1 - v["switch"]
    v["prev_iti"] = v["prev1_iti"]
    v["prev_win"] = v["prev1_win"]
    v["prev_choice"] = v["prev1_choice"]
    v["same_prev2"] = (v["prev2_choice"] == v["prev1_choice"]) & v["prev2_choice"].notna()
    v["half"] = np.where(v["trial"] <= 50, "early", "late")
    return v


# ---------------------------------------------------------------------------
# paired comparisons
# ---------------------------------------------------------------------------
def paired_iti_test(tab: pd.DataFrame, outcome: str = "switch", cond: str = "prev_iti",
                    levels=(1, 4), min_per_cell: int = 1):
    """Per-participant mean of `outcome` at cond==levels[0] ('short') vs levels[1] ('long').

    Returns (summary dict, per-participant DataFrame). Effect = long - short.
    Only participants with at least `min_per_cell` trials at both levels enter. With fewer than 3
    such participants the statistics are returned as NaN. The summary holds the means and standard
    errors of the two levels and of the difference, the paired t test (two-sided p, and one-sided p
    for long > short), Cohen's dz (mean difference / SD of the differences) and the share of
    participants with a positive difference.
    """
    # Trial counts and means per participant (rows) and level (columns); a missing level gets count 0
    cnt = tab.pivot_table(index="subject_id", columns=cond, values=outcome, aggfunc="count")
    mean = tab.pivot_table(index="subject_id", columns=cond, values=outcome, aggfunc="mean")
    for lv in levels:
        if lv not in cnt.columns:
            cnt[lv] = 0; mean[lv] = np.nan
    ok = (cnt[levels[0]].fillna(0) >= min_per_cell) & (cnt[levels[1]].fillna(0) >= min_per_cell)
    m = mean.loc[ok, list(levels)].dropna()
    m.columns = ["short", "long"]
    m["diff"] = m["long"] - m["short"]
    n = len(m)
    out = dict(n=n, n_trials=int(cnt.loc[m.index, list(levels)].sum().sum()) if n else 0)
    if n < 3:
        out.update(dict(mean_short=np.nan, mean_long=np.nan, diff_mean=np.nan, t=np.nan, p_two_sided=np.nan,
                        p_one_sided_greater=np.nan, dz=np.nan))
        return out, m
    # Paired t test (two-sided and one-sided long > short) and effect sizes
    diff = m["diff"]
    t, p2 = stats.ttest_rel(m["long"], m["short"])
    p1 = stats.ttest_rel(m["long"], m["short"], alternative="greater").pvalue
    out.update(dict(
        mean_short=m["short"].mean(), mean_long=m["long"].mean(),
        sem_short=m["short"].sem(), sem_long=m["long"].sem(),
        diff_mean=diff.mean(), diff_sd=diff.std(ddof=1), diff_sem=diff.sem(),
        t=float(t), df=n - 1, p_two_sided=float(p2), p_one_sided_greater=float(p1),
        dz=diff.mean() / diff.std(ddof=1), pct_positive=float((diff > 0).mean()),
    ))
    return out, m


# ---------------------------------------------------------------------------
# participant-level indices for individual differences
# ---------------------------------------------------------------------------
def subject_iti_effects(tab: pd.DataFrame) -> pd.DataFrame:
    """Per participant: p(switch | prior ITI 1 s / 4 s) overall, after wins and after losses, and the
    difference scores (4 s - 1 s) used as the participant-level ITI effect.

    Returns one row per participant with columns p_switch_1s_<set>, p_switch_4s_<set>,
    iti_effect_<set> (set = all, win, loss) and n_trials.
    """
    out = {}
    for label, sub in [("all", tab), ("win", tab[tab["prev_win"] == 1]), ("loss", tab[tab["prev_win"] == 0])]:
        _, m = paired_iti_test(sub, outcome="switch")
        out[f"p_switch_1s_{label}"] = m["short"]
        out[f"p_switch_4s_{label}"] = m["long"]
        out[f"iti_effect_{label}"] = m["diff"]
    df = pd.DataFrame(out)
    df["n_trials"] = tab.groupby("subject_id").size()
    return df.reset_index()


def correlate(x: pd.Series, y: pd.Series, direction=None) -> dict:
    """Pearson and Spearman correlations; one-tailed p in `direction` (+1/-1).

    x and y are aligned on their index (participant); pairs with a missing value are dropped. With
    fewer than 5 pairs only n is returned. `direction` is the preregistered sign (config.STUDIES
    questionnaires); with None, only two-sided p values are given.
    """
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 5:
        return dict(n=n)
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    rho, ps = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    out = dict(n=n, pearson_r=float(r), pearson_p_two_sided=float(p), spearman_rho=float(rho), spearman_p_two_sided=float(ps))
    if direction in (1, -1):
        # One-sided p: half the two-sided p if the sign matches the prediction, else 1 minus that
        out["pearson_p_one_sided"] = float(p / 2 if np.sign(r) == direction else 1 - p / 2)
        out["spearman_p_one_sided"] = float(ps / 2 if np.sign(rho) == direction else 1 - ps / 2)
        out["predicted_direction"] = "+" if direction == 1 else "-"
    return out
