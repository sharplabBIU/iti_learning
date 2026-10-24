"""10 - Posterior predictive checks (PPCs) of the model sequence: do data simulated from each fitted model reproduce the behavioural signatures of the real data?

This script runs for any study whose models are fitted (Studies 3 and 4 in the paper; Fig. 4b,c). The
study is chosen by the folder the script is run from: a copy in data/study_N/analysis/ analyses study N
(analysis_common/runner.py reads the folder name).

Method (Methods, Model fitting and comparison). Every participant is simulated from each fitted model's
posterior means (from 06_fit_models_nuts.py) in their own task environment (reward walks, inter-trial interval
(ITI) sequence, missed-trial structure), 20 times by default. The same behavioural signatures are computed on
the simulated and on the real data. Each stage of the sequence has the signatures that justify it:
  A. value learning  the learning curve: p(choose the best option) in 10-trial windows
  B. choice bias     p(switch) after a stay and after a switch on the previous trial, and among switches that
                     follow a switch, the share that return to the option chosen two trials back (B2, novelty)
  C. trend           p(switch) in 10-trial windows over the session (the decline)
  D. ITI             the preregistered mixed model with a linear trend over the session, fitted with the
                     No-U-Turn sampler (NUTS; analysis_common.bayes_glmm) to the real data and to every
                     simulated data set of the ITI-relevant models (--glmm-models):
                       switch ~ prev_iti + trial_c + (1 + prev_iti + trial_c | subject)            -> glmm_iti (log-odds per s)
                       switch ~ prev_iti * prev_win + trial_c + (1 + prev_iti + trial_c | subject) -> glmm_iti_win, glmm_iti_loss
                     (posterior means; 2 chains x 500 draws per replicate, the model compiled once)

Inputs: data/trials.csv, main sample; results/06_model_fits/<model>_subjects.csv.
Usage: python3 10_posterior_predictive_checks.py [--reps 20] [--models ...]   (default: every fitted model of the sequence)
Other options: --seed, --glmm-models (default m08_trend_beta and the three ITI models m10_iti_bias,
m11_iti_target, m12_iti_compete).
Outputs (results/10_ppc/): ppc_signatures_all_reps.csv (every replicate, and the data), ppc_signatures.csv (mean and 95%
range per model), ppc_curves.csv (the two curves per replicate). Fig. 4b,c is drawn from them by
figures/make_manuscript_figures.py.
"""
import argparse
import sys
from pathlib import Path
# Put the repository root (the first parent folder that contains analysis_common/) on the import path.
_here = Path(__file__).resolve()
for _cand in _here.parents:
    if (_cand / "analysis_common").is_dir():
        sys.path.insert(0, str(_cand)); break
import numpy as np
import pandas as pd

# Command-line options (see the module docstring).
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--glmm-models", nargs="*", default=["m08_trend_beta", "m10_iti_bias", "m11_iti_target", "m12_iti_compete"],
                help="models whose simulated data also get the mixed-model interval signatures (the interval stages and their parent)")
args = ap.parse_args()

from analysis_common.runner import Runner
import analysis_common.likelihoods as L
from analysis_common.simulate import simulate_dataset
from analysis_common.behavior import switch_table

R = Runner(__file__)
R.banner(f"10 posterior predictive checks of the model sequence ({args.reps} simulated datasets per model)")
out = R.out("10_ppc")
fit_dir = R.results_dir / "06_model_fits"
# Models to check: --models, or every registered model with a saved fit in 06_model_fits.
stages = L.register_models()
MODELS = args.models or [m for _, ms in stages for m in ms if (fit_dir / f"{m}_subjects.csv").exists()]
subs, P = R.subjects()
sid_index = {s["subject_id"]: i for i, s in enumerate(subs)}
W = 10   # window width (trials) of the learning and switching curves


def signatures(trials):
    """The choice-bias signatures of one data set (real or simulated), as a dict of scalars.

    Rates are averaged within each participant and then across participants. Uses runs of three consecutive
    responded trials (consec2)."""
    tab = switch_table(trials); t2 = tab[tab["consec2"]].copy()
    t2["prev_switched"] = (t2["prev1_choice"] != t2["prev2_choice"]).astype(int)   # 1 if trial t-1 was a switch
    per = lambda d, col: d.groupby("subject_id")[col].mean().mean()
    sw2 = t2[(t2["switch"] == 1) & (t2["prev_switched"] == 1)]   # switches after a switch (only then can t return to t-2)
    ret = (sw2["choice"] == sw2["prev2_choice"]).groupby(sw2["subject_id"]).mean().mean()
    return dict(switch_after_stay=float(per(t2[t2.prev_switched == 0], "switch")), switch_after_switch=float(per(t2[t2.prev_switched == 1], "switch")),
                return_two_back=float(ret))


# The two mixed-model formulas of stage D: the ITI slope, and the ITI slope by previous outcome.
GLMM_F = ("switch ~ prev_iti + trial_c + (1 + prev_iti + trial_c | subject_id)",
          "switch ~ prev_iti * prev_win + trial_c + (1 + prev_iti + trial_c | subject_id)")


def glmm_table(trials):
    """Trial table for the mixed models: consecutive responded pairs, trial_c = (trial - 50.5) / 50, sorted
    by participant so that real and simulated tables have the same shape."""
    t1 = switch_table(trials); t1 = t1[t1["consec1"]].copy()
    t1["trial_c"] = (t1["trial"] - 50.5) / 50.0; t1["prev_win"] = t1["prev_win"].astype(float)
    return t1[["subject_id", "switch", "prev_iti", "prev_win", "trial_c"]].sort_values(["subject_id"], kind="stable").reset_index(drop=True)


def glmm_signatures(tbl, fitters, seed):
    """Posterior-mean ITI slopes: overall (glmm_iti), after a loss (prev_iti, prev_win = 0) and after a win
    (prev_iti + prev_iti:prev_win)."""
    a = fitters[0].fit(tbl, seed=seed)["mean"]; b = fitters[1].fit(tbl, seed=seed + 1)["mean"]
    return dict(glmm_iti=a["prev_iti"], glmm_iti_loss=b["prev_iti"], glmm_iti_win=b["prev_iti"] + b["prev_iti:prev_win"])


def curves(trials):
    """p(choose best) and p(switch) in 10-trial windows (mean over participants)."""
    v = trials[trials["valid"]].assign(w=lambda d: ((d["trial"] - 1) // W).astype(int))
    tab = switch_table(trials); t1 = tab[tab["consec1"]].assign(w=lambda d: ((d["trial"] - 1) // W).astype(int))
    cb = v.groupby(["w", "subject_id"])["chose_best"].mean().groupby("w").mean()
    sw = t1.groupby(["w", "subject_id"])["switch"].mean().groupby("w").mean()
    return pd.DataFrame(dict(w=cb.index, chose_best=cb.values, p_switch=sw.reindex(cb.index).values))


# Signatures of the real data. RepeatedBGLMM compiles each mixed model once on the real table and refits it
# to every simulated table of the same shape.
trials, _ = R.load()
real = signatures(trials)
from analysis_common.bayes_glmm import RepeatedBGLMM
tbl_real = glmm_table(trials)
fitters = [RepeatedBGLMM(f, tbl_real) for f in GLMM_F]
real.update(glmm_signatures(tbl_real, fitters, seed=args.seed))
print("real data, mixed-model interval slopes:", {k: round(real[k], 4) for k in ("glmm_iti", "glmm_iti_win", "glmm_iti_loss")}, flush=True)
GLMM_MODELS = [m for m in args.glmm_models if m in MODELS]
rows = [dict(model="data", rep=-1, **real)]
crv = [curves(trials).assign(model="data", rep=-1)]
# Simulate each model --reps times from its participants' posterior means and compute the signatures
# (plus the mixed-model signatures for the models in --glmm-models).
rng = np.random.default_rng(args.seed)
for m in MODELS:
    fit = pd.read_csv(fit_dir / f"{m}_subjects.csv").set_index("subject_id")
    ids = [s["subject_id"] for s in subs if s["subject_id"] in fit.index]; sub_m = [subs[sid_index[i]] for i in ids]
    theta = fit.loc[ids, [f"{p}_mean" for p in L.PARAMS[m]]].to_numpy()
    for rep in range(args.reps):
        sim = simulate_dataset(m, theta, sub_m, rng, rep=rep)
        sg = signatures(sim)
        if m in GLMM_MODELS:
            tb = glmm_table(sim)
            try:
                sg.update(glmm_signatures(tb, fitters, seed=args.seed + 17 * rep))
            except ValueError:                  # shape differs from the real data (should not happen): fit from scratch
                from analysis_common.bayes_glmm import fit_bglmm
                r1 = fit_bglmm(tb, GLMM_F[0], draws=500, tune=500, chains=2); r2 = fit_bglmm(tb, GLMM_F[1], draws=500, tune=500, chains=2)
                f1 = r1["fixef"].set_index("term")["mean"]; f2 = r2["fixef"].set_index("term")["mean"]
                sg.update(glmm_iti=f1["prev_iti"], glmm_iti_loss=f2["prev_iti"], glmm_iti_win=f2["prev_iti"] + f2["prev_iti:prev_win"])
        rows.append(dict(model=m, rep=rep, **sg)); crv.append(curves(sim).assign(model=m, rep=rep))
    print(f"  simulated {m} ({len(sub_m)} participants x {args.reps})", flush=True)
res = pd.DataFrame(rows); cur = pd.concat(crv, ignore_index=True)
res.to_csv(out / "ppc_signatures_all_reps.csv", index=False); cur.to_csv(out / "ppc_curves.csv", index=False)
# Per model: mean and 95% range (2.5th and 97.5th percentiles) of each signature across replicates.
keys = [k for k in real]
summ = res[res.model != "data"].groupby("model")[keys].agg(["mean", lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)])
summ.columns = [f"{a}_{ {'mean': 'mean', '<lambda_0>': 'lo', '<lambda_1>': 'hi'}[b] }" for a, b in summ.columns]
summ = summ.reindex([m for m in MODELS]).reset_index(); summ.to_csv(out / "ppc_signatures.csv", index=False)
print("\nreal data:", {k: round(v, 3) for k, v in real.items()})
print(summ[["model"] + [f"{k}_mean" for k in keys]].round(3).to_string(index=False))
print("wrote", out)
