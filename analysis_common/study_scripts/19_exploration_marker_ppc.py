"""19 - A model-agnostic marker of exploratory state and its posterior predictive check (Fig. 4f; Results; Methods,
Exploratory state). Run from the Study 3 folder; it analyses Studies 3 and 4.

ITI = inter-trial interval, 1 s or 4 s. Question: does the effect of the ITI after a win (more switching after a 4 s than
after a 1 s ITI) grow with how rarely the option just chosen had been chosen lately? The script answers it in the data and
in data simulated from each fitted model (a posterior predictive check, PPC).

Marker. u = 1 / (1 + n) for the option chosen on the trial before the ITI, read before that choice. n counts the option's
earlier choices: it rises by 1 each time the option is chosen and loses a fraction lambda of its size on every trial the
option is not chosen, with lambda = 0.85. The ITI-competes-with-value model (m12) uses the limit lambda = 1, in which n is
the run of consecutive choices of the option. u near 1: the option had not been chosen in the last trial or two (an
exploratory choice); u small: the option had been chosen repeatedly (exploitation). Quintiles of u are cut within
participant and outcome (ties in random order, fixed seed).

Measure: the ITI effect on switching (p(switch) after 4 s minus after 1 s) per quintile, after a win and after a loss,
averaged across participants. For the data the script also computes
  - each participant's least-squares slope of the effect over the quintiles, tested against 0 (one-sample t test);
  - the same trend for lambda = 0.5, 0.6, 0.7, 0.8 and 0.9 (Methods: "the pattern reported was the same");
  - a Bayesian mixed-effects logistic regression (PyMC, analysis_common/bayes_glmm.py) on post-win and on post-loss
    trials, switch ~ ITI x quintile + trial, with participant random intercepts and random slopes on ITI and trial; the
    reported term is the ITI x quintile interaction (log odds per quintile).

Check. Every participant is simulated 20 times from their posterior means in their own task environment (reward walks,
ITI sequence, missed trials), for the model without an ITI term (m08) and the three ITI models (m10, m11, m12). The marker
is computed on every simulated data set exactly as on the data. Reported: the mean and 95% range over the 20 data sets,
and the share of data sets whose post-win slope over the quintiles is positive.

Inputs: each study's trial table (main sample) and the participant-level fits results/06_model_fits/<model>_subjects.csv
(run 06 for Studies 3 and 4 first).
Usage: python3 19_exploration_marker_ppc.py [--studies 3 4] [--sims 20]
Output: results/19_exploration_marker_ppc/ in each study's folder: marker.csv, ppc.csv, lambda_robustness.csv, stats.json
"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                     # repository root, so analysis_common can be imported
import numpy as np
import pandas as pd
from scipy import stats
from analysis_common.data import load_study, subjects_for_fitting
from analysis_common.behavior import switch_table
from analysis_common.simulate import simulate_dataset
from analysis_common.bayes_glmm import fit_bglmm, contrast_row
from analysis_common import likelihoods as L
from analysis_common import config as C

ROOT = HERE.parents[1]                                       # the data/ folder that holds study_1 ... study_4
LAM = 0.85                                                   # leak of the choice count behind u
LAM_CHECK = (0.5, 0.6, 0.7, 0.8, 0.9)
MODELS = ["m08_trend_beta", "m10_iti_bias", "m11_iti_target", "m12_iti_compete"]


def arg_list(k, default):
    """All values after the command-line option k up to the next option, or the default if k is absent."""
    if k not in sys.argv:
        return default
    i = sys.argv.index(k) + 1; v = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        v.append(sys.argv[i]); i += 1
    return v


STUDIES = [int(x) for x in arg_list("--studies", [3, 4])]
SIMS = int(arg_list("--sims", [20])[0])
L.register_models()                                          # register every model so its parameters and likelihood are known


def u_table(t, lam=LAM):
    """The marker u = 1 / (1 + n) of the chosen option on every responded trial, read before that choice.

    After each choice, every count keeps a share 1 - lam of its size and the chosen option's count becomes its old
    value + 1. The trial column is named prev1_trial so that it can be merged onto the trial before the ITI in the
    switch table.
    """
    rows = []
    for sid, g in t[t["valid"]].sort_values(["subject_id", "trial"]).groupby("subject_id"):
        n = np.zeros(3)                                      # leaky choice counts of the three options
        for tr, c in zip(g["trial"].to_numpy(), g["choice"].to_numpy().astype(int) - 1):
            rows.append((sid, tr, 1.0 / (1.0 + n[c]))); nc = n[c] + 1.0; n *= (1.0 - lam); n[c] = nc
    return pd.DataFrame(rows, columns=["subject_id", "prev1_trial", "u"])


def marker(t, rng, keep=False, lam=LAM):
    """ITI effect on switching by quintile of u, after a win and after a loss, for one data set (real or simulated).

    Returns, per outcome, the mean effect and s.e.m. per quintile, the mean u per quintile, and the mean per-participant
    slope over the quintiles with its one-sample t test. With keep=True it also returns the trial tables with their
    quintiles (for the mixed model).
    """
    tab = switch_table(t); tab = tab[tab["consec1"]].merge(u_table(t, lam), on=["subject_id", "prev1_trial"])
    res, xs = {}, {}
    for w, lab in ((1, "win"), (0, "loss")):
        x = tab[tab["prev_win"] == w].copy(); x["jit"] = rng.random(len(x))   # jit: random tie-breaker
        # quintile (1-5) of u within participant: sort by u, ties in random order, cut the sorted positions into 5 groups
        x["q"] = x.groupby("subject_id", group_keys=False).apply(
            lambda g: pd.Series(pd.qcut(g.sort_values(["u", "jit"]).reset_index().index, 5, labels=False) + 1, index=g.sort_values(["u", "jit"]).index))
        # e: participants x quintiles, the ITI effect p(switch | 4 s) - p(switch | 1 s)
        e = x.groupby(["subject_id", "q", "prev_iti"])["switch"].mean().unstack("prev_iti").pipe(lambda c: c[4] - c[1]).unstack("q")
        # each participant's linear slope over the quintiles (needs at least 3 quintiles), then a one-sample t test
        sl = e.apply(lambda v: np.polyfit(v.dropna().index.to_numpy(float), v.dropna().to_numpy(), 1)[0] if v.notna().sum() >= 3 else np.nan, axis=1).dropna()
        t_, p_ = stats.ttest_1samp(sl, 0.0)
        res[lab] = dict(effect=[float(e[q].mean()) for q in range(1, 6)], sem=[float(e[q].sem()) for q in range(1, 6)],
                        mean_u=[float(x[x.q == q]["u"].mean()) for q in range(1, 6)], slope=float(sl.mean()), slope_sem=float(sl.sem()),
                        t=float(t_), p=float(p_), df=int(len(sl) - 1))
        xs[lab] = x
    return (res, xs) if keep else res


for s in STUDIES:
    out = ROOT / f"study_{s}/analysis/results/19_exploration_marker_ppc"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1900 + s)                    # one generator per study, for tie-breaking and simulation
    t = load_study(f"study_{s}", C.EXCLUSION_MAIN)[0]
    # The marker in the data.
    D, X = marker(t, rng, keep=True); st = dict(study=s, leak=LAM, sims=SIMS, data=D)
    for lab in ("win", "loss"):
        print(f"Study {s} data, after a {lab}: " + " ".join(f"{v:+.3f}" for v in D[lab]["effect"]) +
              f"; trend {D[lab]['slope']:+.4f}, t({D[lab]['df']}) = {D[lab]['t']:.2f}, p = {D[lab]['p']:.4f}", flush=True)
        # Bayesian mixed logistic regression on the data: iti4 = 1 after 4 s; qc = quintile centred (-2 to 2).
        x = X[lab].copy(); x["iti4"] = (x["prev_iti"] == 4).astype(float); x["trial_c"] = (x["trial"] - 50.5) / 50; x["qc"] = x["q"] - 3
        f_ = fit_bglmm(x, "switch ~ iti4 * qc + trial_c + (1 + iti4 + trial_c | subject_id)", draws=1000, tune=1000, chains=4, seed=19)
        st[f"glmm_{lab}"] = g = contrast_row(f_, {"iti4:qc": 1.0}, f"ITI x u quintile after a {lab}")
        print(f"   mixed model, ITI x quintile (log odds): {g['estimate']:+.3f}, CI = [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}], "
              f"R-hat {g['rhat_max']:.3f}", flush=True)
    # The same data trend for other leaks of the count.
    lam_rows = []
    for lam in LAM_CHECK:
        M = marker(t, np.random.default_rng(1900 + s), lam=lam)
        lam_rows += [dict(lam=lam, outcome=lab, slope=M[lab]["slope"], t=M[lab]["t"], df=M[lab]["df"], p=M[lab]["p"]) for lab in ("win", "loss")]
    lam_tab = pd.DataFrame(lam_rows); lam_tab.to_csv(out / "lambda_robustness.csv", index=False)
    print(lam_tab.round(4).to_string(index=False))
    # Posterior predictive check: simulate each model SIMS times from the participants' posterior means and recompute
    # the marker.
    subs = subjects_for_fitting(t); rows = []
    for m in MODELS:
        fit = pd.read_csv(ROOT / f"study_{s}/analysis/results/06_model_fits/{m}_subjects.csv").set_index("subject_id")
        ss = [x for x in subs if x["subject_id"] in fit.index]
        # parameter matrix (participants x parameters, posterior means) in the order the model expects
        th = fit.loc[[x["subject_id"] for x in ss], [f"{p}_mean" for p in L.PARAMS[m]]].to_numpy()
        for r in range(SIMS):
            M = marker(simulate_dataset(m, th, ss, rng, rep=r), rng)
            for lab in ("win", "loss"):
                rows.append(dict(model=m, rep=r, outcome=lab, slope=M[lab]["slope"], **{f"q{q + 1}": M[lab]["effect"][q] for q in range(5)}))
        Pm = pd.DataFrame([r_ for r_ in rows if r_["model"] == m])
        for lab in ("win", "loss"):
            p_ = Pm[Pm.outcome == lab]
            print(f"   {m:<24} after a {lab}: slope {p_['slope'].mean():+.4f} [{p_['slope'].quantile(.025):+.4f}, "
                  f"{p_['slope'].quantile(.975):+.4f}], positive in {(p_['slope'] > 0).mean():.0%}", flush=True)
    pd.DataFrame(rows).to_csv(out / "ppc.csv", index=False)
    pd.DataFrame([dict(outcome=lab, quintile=q + 1, effect=D[lab]["effect"][q], sem=D[lab]["sem"][q], mean_u=D[lab]["mean_u"][q])
                  for lab in ("win", "loss") for q in range(5)]).to_csv(out / "marker.csv", index=False)
    json.dump(st, open(out / "stats.json", "w"), indent=1, default=float)
    print("wrote", out, flush=True)
