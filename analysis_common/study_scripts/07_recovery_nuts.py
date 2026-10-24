"""07 - Parameter and model recovery of the ITI models, by NUTS with the priors and settings of the real fits (Fig. 4d,e).

The script is copied to every study folder and runs for the study chosen by the folder it is run from
(analysis_common/runner.py reads the folder name), but it stops unless that study is Study 3.

Question: can the fitting procedure recover known parameters, and can it tell the inter-trial interval (ITI)
models apart? The compared models are the three ITI models (--models; default m10_iti_bias, m11_iti_target,
m12_iti_compete). The code calls this list `three`. Fitting uses the No-U-Turn sampler (NUTS) with the priors
and settings of the real fits (4 chains x 1000 draws after 2000 warm-up iterations).

Parameter recovery (--part parameter --model M; the paper runs it for the winning model, m12_iti_compete): one
synthetic agent per participant, playing that participant's environment (reward trajectories, ITI sequence and
missed-trial structure), with every parameter drawn independently and uniformly over its full range
(FULL_RANGE); simulated once and refitted with the same hierarchy. Reported: Pearson and Spearman correlations
between generating and recovered participant posterior means. parameter_ranges.csv shows how much of the real
fitted participant means lies inside each range.

Model recovery (--part model --gen G): for each generating model, data sets --reps a .. b-1 (default 0 .. 4,
five data sets) of --n-subjects participants (default 50). For each data set, n real participants are drawn
without replacement, and each plays their own environment under G with parameters spread evenly over
FULL_RANGE (--theta-source grid, a Latin hypercube; --theta-source fitted uses the participants' fitted
posterior means instead). All compared models are fitted to every data set. Scoring: the winner of a data set
is the model with the highest summed elpd (expected log predictive density, from PSIS-LOO, Pareto-smoothed
importance-sampling leave-one-out cross-validation); a model is recovered on a data set if it is the generating
model and wins (1) and not otherwise (0), and the recovery rate is the mean over data sets. Reported: the
confusion matrix of elpd winners (rows = generating model; model_recovery_confusion.csv) and, secondarily, the
winners by protected exceedance probability (PXP) from random-effects Bayesian model selection (BMS) on the
participant-level elpd (model_recovery_confusion_pxp.csv) and the mean PXP.

Inputs: data/trials.csv (participant environments), results/06_model_fits/<model>_subjects.csv (fitted
participant posterior means).
Usage (the runs of the paper; reproduce.sh --with-recovery):
       python3 07_recovery_nuts.py --part parameter --model m12_iti_compete
       python3 07_recovery_nuts.py --part model --gen G          (G = each of the three ITI models)
       python3 07_recovery_nuts.py --collect-only
Other options: --n-subjects, --reps, --models, --theta-source {grid, fitted}, --tag (suffix of the
model-recovery folder, default iti3_grid; the collected tables and figure are written there), and the sampler
settings --draws, --tune, --chains, --target-accept, --seed. Existing parameter-recovery summaries and
model-recovery data sets are not refitted, so a run can be resumed (delete results/07_recovery/ to start
afresh).
Outputs (results/07_recovery/): parameter_recovery_<M>.csv / _summary.csv, parameter_recovery_fit_<M>_*
(the refit), parameter_ranges.csv, fig_parameter_recovery_<M>.png, and in model_recovery_n<n>_<tag>/ one
<G>_rep<k>.json per data set and the collected model_recovery_datasets.csv, model_recovery_confusion.csv
(elpd winners), model_recovery_confusion_pxp.csv, model_recovery_mean_pxp.csv, fig_model_recovery.png and
recovery.json.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
# Put the repository root (the first parent folder that contains analysis_common/) on the import path.
_here = Path(__file__).resolve()
for _cand in _here.parents:
    if (_cand / "analysis_common").is_dir():
        sys.path.insert(0, str(_cand)); break
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# Full range of each parameter for the parameter recovery and for --theta-source grid. alpha / alpha_r =
# learning rates after a win / a loss; decay = retention of unchosen values; alpha_c, beta_c = rate and
# weight of the choice kernel; beta = inverse temperature on values; stick = constant stay bias; c1, c2 =
# option intercepts.
FULL_RANGE = {"alpha": (0.0, 1.0), "alpha_r": (0.0, 1.0), "decay": (0.0, 1.0), "alpha_c": (0.0, 1.0),
              "beta": (0.0, 10.0), "beta_c": (0.0, 6.0), "stick": (-3.0, 3.0), "c1": (-3.0, 3.0), "c2": (-3.0, 3.0),
              # Trends: change in the log of the weight per 50 trials (+/-2: up to e^4, about 55-fold, over the session)
              "beta_slope": (-2.0, 2.0), "bias_slope": (-2.0, 2.0),
              # Novelty bonus: omega = the constant bonus on the option chosen furthest back
              "omega": (-3.0, 3.0),
              # ITI terms (ITI coded 0 after 1 s, 1 after 4 s): gamma = log gain of the long wait on the choice-bias
              # weight (m10); cost = cost of the long wait in the value target (m11); kappa = share of beta moved from
              # value to the wait just taken at full uncertainty, u = 1 (m12)
              "gamma": (-2.0, 2.0), "cost": (-1.0, 1.0), "kappa": (0.0, 1.0)}

# Everything below runs only when the script is executed directly, not when it is imported.
if __name__ == "__main__":
    from scipy import stats
    from analysis_common.runner import Runner, BLUE
    from analysis_common import config as C
    from analysis_common import pymc_models as PM
    from analysis_common.bms import bms_table
    from analysis_common.likelihoods import PARAMS, register_models
    from analysis_common.data import subjects_for_fitting
    from analysis_common.simulate import simulate_dataset

    # Command-line options (see the module docstring).
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["parameter", "model", "all"])
    ap.add_argument("--model", default=None, help="parameter recovery: one model (default: each of the compared models)")
    ap.add_argument("--gen", default=None, help="model recovery: one generating model (default: the three)")
    ap.add_argument("--reps", nargs=2, type=int, default=[0, 5], help="model recovery: data sets a .. b-1")
    ap.add_argument("--n-subjects", type=int, default=50, help="participants per synthetic data set (results in model_recovery_n<n>_<tag>/)")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--models", nargs="+", default=["m10_iti_bias", "m11_iti_target", "m12_iti_compete"], help="the compared models")
    ap.add_argument("--theta-source", default="grid", choices=["grid", "fitted"], help="model recovery: generating parameters spread evenly "
                    "over the full range of every parameter (grid: Latin hypercube over FULL_RANGE, each participant a random real task "
                    "environment) or the sampled participants' fitted posterior means (fitted)")
    ap.add_argument("--tag", default="iti3_grid", help="suffix of the model-recovery folder; the collected tables and figure are written there")
    args = ap.parse_args()
    R = Runner(__file__)
    if R.study["key"] != "study_3":
        sys.exit("07_recovery_nuts.py is run on Study 3 only")
    R.banner("07 parameter and model recovery by NUTS (the three ITI models, or --models)")
    # mr_dir holds one JSON file per model-recovery data set; cdir receives the collected tables.
    out = R.out("07_recovery"); mr_dir = out / (f"model_recovery_n{args.n_subjects}" + (f"_{args.tag}" if args.tag else "")); mr_dir.mkdir(exist_ok=True)
    cdir = mr_dir if args.tag else out
    fit_dir = R.results_dir / "06_model_fits"
    # The compared models (--models).
    register_models()
    three = args.models
    roles = {m: C.MODEL_SHORT[m] for m in three}
    print("models: " + ", ".join(f"{m} ({roles[m]})" for m in three))
    # The real participants: their environments are reused for the synthetic agents.
    subs, P = R.subjects(); idx = {s["subject_id"]: i for i, s in enumerate(subs)}
    plt = R.plt

    def fit_nuts(model, sim_subs, seed):
        """Fit one model to a list of (synthetic) participants by NUTS and return the postprocessed result
        (participant posterior means, elpd, diagnostics, fitting time)."""
        t0 = time.time()
        data = PM.pad_subjects(sim_subs)
        m = PM.build_model(model, data)
        idata, info = PM.sample_model(m, draws=args.draws, tune=args.tune, chains=args.chains, seed=seed, target_accept=args.target_accept,
                                      verbose=len(sim_subs) > 100)
        res = PM.postprocess(idata, model, data, sim_subs)
        res["summary"].update(info); res["summary"]["wall_seconds"] = time.time() - t0
        del idata, m
        return res

    # ------------------------------------------------------------------ parameter recovery, full ranges
    # Per model: record the ranges, draw uniform parameters for one agent per participant, simulate,
    # refit, and correlate generating with recovered posterior means. Models with a summary already on disk
    # are skipped unless --model names one.
    if args.part in ("parameter", "all") and not args.collect_only:
        for k, m in enumerate([args.model] if args.model else three):
            if (out / f"parameter_recovery_{m}_summary.csv").exists() and not args.model:
                print(f"parameter recovery of {m}: done already"); continue
            names = PARAMS[m]
            fit = pd.read_csv(fit_dir / f"{m}_subjects.csv")
            rr = pd.DataFrame([dict(model=m, parameter=p, range_lo=FULL_RANGE[p][0], range_hi=FULL_RANGE[p][1],
                                    fitted_min=fit[f"{p}_mean"].min(), fitted_max=fit[f"{p}_mean"].max(),
                                    share_fitted_inside=float(fit[f"{p}_mean"].between(*FULL_RANGE[p]).mean())) for p in names])
            f_rng = out / "parameter_ranges.csv"
            old = pd.read_csv(f_rng) if f_rng.exists() else pd.DataFrame()
            pd.concat([old[old.model != m] if len(old) else old, rr]).to_csv(f_rng, index=False)
            print(f"\nparameter recovery, {m}: ranges (share of fitted participant means inside)"); print(rr.round(3).to_string(index=False))
            rng = np.random.default_rng(args.seed + 100 * (k + 1))
            theta = np.column_stack([rng.uniform(*FULL_RANGE[p], len(subs)) for p in names])
            sim_subs = subjects_for_fitting(simulate_dataset(m, theta, subs, rng))
            res = fit_nuts(m, sim_subs, args.seed + 1)
            PM.save_pymc_result(res, out, f"parameter_recovery_fit_{m}")
            rec = res["subjects"].set_index("subject_id")
            rows = [dict(model=m, subject_id=s["subject_id"], parameter=p, generating=theta[i, j], recovered=rec.loc[s["subject_id"], f"{p}_mean"])
                    for i, s in enumerate(subs) if s["subject_id"] in rec.index for j, p in enumerate(names)]
            pr = pd.DataFrame(rows); pr.to_csv(out / f"parameter_recovery_{m}.csv", index=False)
            summ = pr.groupby("parameter", sort=False).apply(lambda g: pd.Series(dict(
                pearson_r=stats.pearsonr(g.generating, g.recovered)[0], spearman_rho=stats.spearmanr(g.generating, g.recovered)[0],
                range_lo=FULL_RANGE[g.name][0], range_hi=FULL_RANGE[g.name][1], n=len(g)))).reset_index()
            summ.to_csv(out / f"parameter_recovery_{m}_summary.csv", index=False)
            s = res["summary"]
            print(f"parameter recovery, {m} ({C.MODEL_SHORT[m]}; {len(subs)} agents; NUTS {s['wall_seconds'] / 60:.0f} min, divergences {s['n_divergences']}, "
                  f"max R-hat {s['rhat_max']:.3f}, min ESS {s['ess_bulk_min']:.0f}):")
            print(summ.round(3).to_string(index=False), flush=True)
            ncol = int(np.ceil(len(names) / 2))
            fig, axes = plt.subplots(2, ncol, figsize=(2.4 * ncol, 4.9))
            for ax, p in zip(axes.ravel(), names):
                g = pr[pr.parameter == p]
                ax.scatter(g.generating, g.recovered, s=4, alpha=0.4, color=BLUE); ax.plot(FULL_RANGE[p], FULL_RANGE[p], color="grey", lw=0.8, ls="--")
                ax.set_title(f"{p}  r = {summ.set_index('parameter').loc[p, 'pearson_r']:.2f}", fontsize=8)
            for ax in axes.ravel()[len(names):]:
                ax.axis("off")
            fig.suptitle(f"{R.label}: parameter recovery by NUTS, {C.MODEL_SHORT[m]} (generating uniform over full ranges)", fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"fig_parameter_recovery_{m}.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ model recovery: data sets of n subjects per generating model
    # Each data set: sample n real participants, simulate them under the generating model, fit every compared
    # model, and store elpd, participant elpd, BMS results and winners in <G>_rep<k>.json. Data sets already
    # on disk are skipped.
    if args.part in ("model", "all") and not args.collect_only:
        ids_all = np.array([s["subject_id"] for s in subs])
        for gen in ([args.gen] if args.gen else three):
            fit = pd.read_csv(fit_dir / f"{gen}_subjects.csv").set_index("subject_id")
            for rep in range(*args.reps):
                f = mr_dir / f"{gen}_rep{rep:02d}.json"
                if f.exists():
                    continue
                rng = np.random.default_rng(args.seed + 1000 * (three.index(gen) + 1) + rep)
                ids = rng.choice([i for i in ids_all if i in fit.index], size=args.n_subjects, replace=False)
                if args.theta_source == "grid":           # every parameter stratified evenly over its full range, random pairing
                    # (Latin hypercube: one draw in each of n equal bins, bins shuffled across participants)
                    n_ = args.n_subjects
                    theta = np.column_stack([FULL_RANGE[p][0] + (FULL_RANGE[p][1] - FULL_RANGE[p][0]) * (rng.permutation(n_) + rng.uniform(0, 1, n_)) / n_
                                             for p in PARAMS[gen]])
                else:
                    theta = fit.loc[ids, [f"{p}_mean" for p in PARAMS[gen]]].to_numpy()
                sim_subs = subjects_for_fitting(simulate_dataset(gen, theta, [subs[idx[i]] for i in ids], rng, rep=rep))
                t0 = time.time(); fits = {}
                for fm in three:
                    res = fit_nuts(fm, sim_subs, args.seed + 7 * rep + three.index(fm))
                    fits[fm] = dict(elpd=float(res["summary"]["elpd_loo"]), subject_elpd=res["subjects"]["elpd_loo"].tolist(),
                                    rhat_max=float(res["summary"]["rhat_max"]), n_divergences=int(res["summary"]["n_divergences"]))
                E = np.column_stack([fits[fm]["subject_elpd"] for fm in three])
                b, _ = bms_table(E, three)
                pxp = b.set_index("model")["protected_exceedance"].to_dict()
                rec = dict(generating=gen, rep=rep, subjects=list(map(str, ids)), fits=fits, pxp=pxp, theta_source=args.theta_source,
                           theta={p: theta[:, j].tolist() for j, p in enumerate(PARAMS[gen])},
                           winner_pxp=max(three, key=lambda m: pxp[m]), winner_elpd=max(three, key=lambda m: fits[m]["elpd"]),
                           expected_frequency=b.set_index("model")["expected_frequency"].to_dict(), seconds=time.time() - t0)
                json.dump(rec, open(f, "w"), indent=1)
                print(f"  {gen} data set {rep + 1:2d}: PXP winner {rec['winner_pxp']} (" + ", ".join(f"{C.MODEL_SHORT[m].split(':')[0]} {pxp[m]:.2f}" for m in three)
                      + f"); elpd winner {rec['winner_elpd']}; {rec['seconds'] / 60:.1f} min", flush=True)
                # Print the generating model's summed elpd advantage over each other model (SE over participants).
                eg = np.asarray(fits[gen]["subject_elpd"])
                for fm in three:
                    if fm != gen:
                        dd = eg - np.asarray(fits[fm]["subject_elpd"])
                        print(f"      generating minus {fm}: {dd.sum():+.1f} elpd (SE {np.sqrt(len(dd)) * dd.std(ddof=1):.1f}); R-hat max "
                              f"{fits[gen]['rhat_max']:.3f} / {fits[fm]['rhat_max']:.3f}", flush=True)

    # ------------------------------------------------------------------ collect
    # Read every saved data set and build the confusion matrices: the share of data sets of each generating
    # model (rows) won by each fitted model (columns), by summed elpd (primary) and by PXP (secondary). The
    # diagonal is the recovery rate. Then collect the parameter-recovery summaries into recovery.json.
    recs = [json.load(open(f)) for f in sorted(mr_dir.glob("*_rep*.json"))]
    info = dict(models=three, roles=roles, n_subjects=args.n_subjects, ranges=FULL_RANGE)
    if recs:
        rows = []
        for r in recs:
            row = dict(generating=r["generating"], rep=r["rep"], winner_pxp=r["winner_pxp"], winner_elpd=r["winner_elpd"])
            for m in three:
                row[f"pxp_{m}"] = r["pxp"][m]; row[f"elpd_{m}"] = r["fits"][m]["elpd"]; row[f"rhat_{m}"] = r["fits"][m]["rhat_max"]
            rows.append(row)
        D = pd.DataFrame(rows); D.to_csv(cdir / "model_recovery_datasets.csv", index=False)
        gens = [g for g in three if g in set(D.generating)]
        conf = (D.groupby(["generating", "winner_pxp"]).size().unstack(fill_value=0).reindex(index=gens, columns=three, fill_value=0))
        n_sets = D.groupby("generating").size().reindex(gens)
        conf = conf.div(n_sets, axis=0); conf.index.name = "generating"; conf.to_csv(cdir / "model_recovery_confusion_pxp.csv")
        conf_e = (D.groupby(["generating", "winner_elpd"]).size().unstack(fill_value=0).reindex(index=gens, columns=three, fill_value=0)).div(n_sets, axis=0)
        conf_e.index.name = "generating"; conf_e.to_csv(cdir / "model_recovery_confusion.csv")
        mean_pxp = D.groupby("generating")[[f"pxp_{m}" for m in three]].mean().reindex(gens); mean_pxp.columns = three
        mean_pxp.to_csv(cdir / "model_recovery_mean_pxp.csv")
        info.update(reps=int(n_sets.min()), reps_by_generating=n_sets.to_dict(), confusion_pxp=conf.round(4).to_dict(orient="index"),
                    confusion_elpd=conf_e.round(4).to_dict(orient="index"), mean_pxp=mean_pxp.round(4).to_dict(orient="index"),
                    recovery_rate_pxp={g: float(conf.loc[g, g]) for g in gens}, recovery_rate_elpd={g: float(conf_e.loc[g, g]) for g in gens},
                    scoring="elpd winner (1 if the generating model has the highest summed elpd, else 0), averaged over data sets")
        lab = lambda m: C.MODEL_SHORT[m]
        print(f"\nmodel recovery: share of data sets ({args.n_subjects} subjects each; n per generating model: {n_sets.to_dict()}) in which each fitted model had the highest summed elpd")
        print(conf_e.rename(index=lab, columns=lab).round(2).to_string())
        print("\nsecondary: share won by the highest PXP"); print(conf.rename(index=lab, columns=lab).round(2).to_string())
        print("\nmean PXP"); print(mean_pxp.rename(index=lab, columns=lab).round(2).to_string())
        # Heat map of the elpd confusion matrix.
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.imshow(conf_e.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(three))); ax.set_xticklabels([lab(m) for m in three], rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(gens))); ax.set_yticklabels([lab(m) for m in gens], fontsize=7)
        for i in range(len(gens)):
            for j in range(len(three)):
                v = conf_e.iloc[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "black", fontsize=9)
        ax.set_xlabel("fitted model with the highest summed elpd"); ax.set_ylabel("generating model")
        ax.set_title(f"{R.label}: model recovery, {int(n_sets.min())} data sets x {args.n_subjects} subjects", fontsize=9)
        fig.tight_layout(); fig.savefig(cdir / "fig_model_recovery.png", dpi=150); plt.close(fig)
    pr_files = sorted(out.glob("parameter_recovery_*_summary.csv"))
    if pr_files:
        info["parameter_recovery"] = {f.stem.replace("parameter_recovery_", "").replace("_summary", ""): pd.read_csv(f).to_dict(orient="records") for f in pr_files}
    R.save_json(cdir, "recovery", info)
    print("wrote", out)
