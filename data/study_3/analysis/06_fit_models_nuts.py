"""06 - Fit the computational model sequence as hierarchical Bayesian models and compare the models stage by stage.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ fits study N (analysis_common/runner.py reads the folder name). The paper fits
the whole sequence to Studies 3 and 4, and the model without an ITI term (m08) to Study 2 (Fig. 2e).

What it does. The reinforcement-learning models of analysis_common/likelihoods.py are fitted as
hierarchical Bayesian models with the No-U-Turn sampler (NUTS; PyMC + nutpie). They are compared by the
expected log predictive density (elpd) from Pareto-smoothed importance-sampling leave-one-out
cross-validation (PSIS-LOO), stage by stage. This is the only model fitting in the pipeline. Before any
fit, the PyTensor likelihood of each model is checked against the numpy likelihood of likelihoods.py.

The sequence. Each stage builds on the winner of the one before; the choices made are recorded in
likelihoods.CHOICE_A / CHOICE_B / CHOICE_B2 / CHOICE_C.
  A. value learning   m01 intercepts only (fixed preferences); m02 Rescorla-Wagner (RW), one learning rate;
                      m03 separate learning rates after a win and after a loss; m04 = m03 + decay of
                      unchosen values toward the neutral value
  B. choice bias      m05 constant stay bias; m06 choice kernel (one trace per option)
  B2. novelty         m07 = the stage-B winner + a constant bonus omega on the option chosen furthest back
  C. trend            a log-linear trend over the session, exp(slope * (trial - 50.5) / 50): m08 on the
                      inverse temperature (beta); m09 on the choice-bias weight
  D. inter-trial interval (ITI), coded w = 0 after 1 s and 1 after 4 s; only the wait just elapsed enters:
                      m10 ITI on choice bias: the wait scales the choice-bias weight;
                      m11 ITI as learning target: the wait enters the value target of the option it followed;
                      m12 ITI competes with value: after a long wait a share kappa * u of beta moves from the
                      values to the wait just taken, attached to the option chosen last, with
                      u = 1 / (1 + its run of consecutive choices before that choice).
Coding. Outcomes are +1 for a win and -1 for a loss; values start at 0 and unchosen values decay toward 0.
The choice-kernel counters move toward 1 for the chosen option and 0 for the others.

Stage rules (all on elpd; SE = standard error of the paired difference across observations):
  A: each addition (m02 over m01, m03 over m02, m04 over the model kept so far) is kept if it gains more
     than 2 SE;
  B: the choice-bias term with the largest elpd (among converged models); B2: the novelty bonus is kept if
     it gains more than 2 SE;
  C: the better trend if it gains more than 2 SE over the stage-B winner, otherwise no trend;
  D: every ITI model against its parent. The final comparison takes the no-ITI model (the stage-C winner)
     and the ITI models of stage D (m10, m11 and m12, as listed in
     likelihoods.register_models), with random-effects Bayesian model selection (BMS) on participant-level
     elpd among them. Every model is also compared overall, against the best converged model
     (model_comparison_loo.csv: elpd_minus_best, se_vs_best).
A model can win a stage only if its group-level R-hat (convergence diagnostic; mu and sigma of every
parameter) is at most 1.1. The script prints the choices the data support; if they differ from those
recorded in likelihoods.py it says so (the recorded choices define the models of the later stages).

Priors (pymc_models.build_model), non-centred: unit-interval parameters invlogit(mu + sigma z),
mu ~ N(0, 1.5); positive parameters exp(mu + sigma z), mu ~ N(0.5, 1); real-valued parameters
mu + sigma z, mu ~ N(0, 1); sigma ~ HalfNormal(1). Sampling: 4 chains x 1000 draws after 2000 warm-up
iterations, target acceptance 0.9. In Study 4, m08 and m12 were refitted with 4 chains x 2000 draws after
4000 warm-up iterations (--models m08_trend_beta m12_iti_compete --tune 4000 --draws 2000), because the
spread of the choice-kernel learning rate mixed slowly (Methods). A choice made right after a missed trial
is not scored.

Inputs: data/trials.csv (per-participant choices, outcomes, ITIs, trial numbers), main sample;
previously saved fits in results/06_model_fits/ for --skip-existing and --collect-only.
Usage: python3 06_fit_models_nuts.py [--models ...] [--skip-existing] [--collect-only]
  --models          fit only these models (default: every registered model, in stage order)
  --skip-existing   skip models whose <model>_summary.json already exists
  --collect-only    fit nothing; rebuild the comparison from the saved fits
  --draws, --tune, --chains, --target-accept, --seed   sampler settings (defaults 1000, 2000, 4, 0.9, 2026)
Outputs (results/06_model_fits/): <model>_{subjects,group,summary,pointwise,group_posterior_draws}, model_comparison_loo.csv,
stage_comparison.csv, final_comparison.csv, bms_final.csv, subject_elpd_loo.csv, comparison.json, fig_model_comparison.png
(also run_info.json with the arguments and the models found).
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

RHAT_GATE = 1.1   # a model can win a stage only if its largest group-level R-hat is at most this

# Everything below runs only when the script is executed directly, not when it is imported.
if __name__ == "__main__":
    from analysis_common.runner import Runner, BLUE, GREY
    from analysis_common import config as C
    from analysis_common import pymc_models as PM
    from analysis_common.bms import bms_table
    import analysis_common.likelihoods as L

    # Command-line options (see the module docstring).
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    R = Runner(__file__)
    R.banner("06 model fitting by NUTS (the model sequence), PSIS-LOO comparison by stage")
    out = R.out("06_model_fits")
    # Register the models of every stage (likelihoods.register_models): [(stage label, [model names])].
    stages = L.register_models()
    order = [m for _, ms in stages for m in ms]
    models = args.models or order
    # One dict per participant (choices, rewards, ITIs, trial numbers), padded into trial x participant
    # arrays. SC marks the scored choices (not those made right after a missed trial).
    subs, P = R.subjects()
    data = PM.pad_subjects(subs)
    print(f"{data['N']} participants, {int(data['M'].sum())} responded trials, {int(data['SC'].sum())} scored choices")
    if args.skip_existing:
        models = [m for m in models if not (out / f"{m}_summary.json").exists()]
    if not args.collect_only:
        # The PyTensor likelihood must match the numpy likelihood (8 participants, 3 random parameter draws).
        for m in models:
            err = PM.check_against_numpy(m, subs[:8], S=3, seed=args.seed)
            assert err < 1e-6, (m, err)
        # Fit each model, then save its participant and group summaries, pointwise elpd and diagnostics.
        for model in models:
            print(f"\n--- {model}: {C.MODEL_SHORT[model]} ({len(L.PARAMS[model])} parameters)", flush=True)
            t0 = time.time()
            mdl = PM.build_model(model, data)
            idata, info = PM.sample_model(mdl, draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed, target_accept=args.target_accept)
            res = PM.postprocess(idata, model, data, subs)
            res["summary"].update(info); res["summary"]["wall_seconds"] = time.time() - t0; res["summary"]["spec"] = L.parse(model)
            PM.save_pymc_result(res, out, model)
            s = res["summary"]
            print(f"    elpd_loo = {s['elpd_loo']:.1f} (SE {s['se_elpd_loo']:.1f}), p_loo = {s['p_loo']:.1f}, Pareto k > 0.7: {s['pareto_k_frac_gt_0p7']:.1%}; "
                  f"divergences = {s['n_divergences']}, max R-hat = {s['rhat_max']:.3f}, min ESS = {s['ess_bulk_min']:.0f}", flush=True)
            del idata, mdl

    # ------------------------------------------------------------------ comparison
    # Load every saved fit of a registered model (also fits from earlier runs).
    fitted = {}
    for m in order:
        f = out / f"{m}_summary.json"
        if f.exists():
            info = json.load(open(f)); info["subjects"] = pd.read_csv(out / f"{m}_subjects.csv"); info["group"] = pd.read_csv(out / f"{m}_group.csv")
            info["elpd_i"] = np.load(out / f"{m}_pointwise.npz")["elpd_i"]; fitted[m] = info
    if not fitted:
        sys.exit("no fits found")

    def dse(a, b):
        """Paired elpd difference a - b summed over observations, and its standard error sqrt(n * var(d))."""
        d = fitted[a]["elpd_i"] - fitted[b]["elpd_i"]
        return float(d.sum()), float(np.sqrt(len(d) * np.var(d)))

    def rhat_group(m):
        """Largest R-hat among the group-level parameters (mu and sigma) of model m."""
        g = fitted[m]["group"]
        return float(np.nanmax(g[["rhat_mu", "rhat_sigma"]].to_numpy()))

    ok = lambda m: m in fitted and rhat_group(m) <= RHAT_GATE   # fitted and converged
    stage_of = {m: lab for lab, ms in stages for m in ms}
    # One row per fitted model: elpd, diagnostics, the difference from its stage parent, and the group-level
    # mean of each parameter on its natural scale.
    rows = []
    for m, s in fitted.items():
        par = L.stage_parent(m)
        row = dict(stage=stage_of[m], model=m, label=C.MODEL_SHORT[m], n_params=len(s["group"]), elpd_loo=s["elpd_loo"], se_elpd_loo=s["se_elpd_loo"],
                   p_loo=s["p_loo"], pareto_k_frac_gt_0p7=s["pareto_k_frac_gt_0p7"], n_divergences=s["n_divergences"], rhat_group_max=rhat_group(m),
                   ess_bulk_min=s["ess_bulk_min"], parent=par)
        if par in fitted:
            row["elpd_minus_parent"], row["se_vs_parent"] = dse(m, par)
        for _, g in s["group"].iterrows():
            row[f"{g['parameter']}_group_mean"] = g["group_mean_constrained"]
        rows.append(row)
    comp = pd.DataFrame(rows)
    for col in ("elpd_minus_parent", "se_vs_parent"):
        if col not in comp.columns:
            comp[col] = np.nan
    # Overall comparison: every model against the converged model with the highest elpd
    # (against the best of all models if none converged).
    best_all = max([m for m in fitted if ok(m)] or list(fitted), key=lambda m: fitted[m]["elpd_loo"])
    ob = [dse(m, best_all) if m != best_all else (0.0, 0.0) for m in comp["model"]]
    comp["best_model"] = best_all; comp["elpd_minus_best"] = [d for d, _ in ob]; comp["se_vs_best"] = [se for _, se in ob]
    comp.to_csv(out / "model_comparison_loo.csv", index=False)
    comp[["stage", "model", "label", "n_params", "elpd_loo", "se_elpd_loo", "parent", "elpd_minus_parent", "se_vs_parent", "n_divergences",
          "rhat_group_max"]].to_csv(out / "stage_comparison.csv", index=False)
    print("\nPSIS-LOO by stage (elpd; higher is better; difference against the model each stage builds on):")
    print(comp[["stage", "model", "n_params", "elpd_loo", "elpd_minus_parent", "se_vs_parent", "n_divergences", "rhat_group_max"]].round(2).to_string(index=False))

    # The stage choices the data support, by the stage rules in the module docstring, compared with the
    # choices recorded in likelihoods.py.
    summary = dict(study=R.key, n_subjects=data["N"], n_scored=int(data["SC"].sum()), rhat_gate=RHAT_GATE,
                   recorded_choices=dict(A=L.CHOICE_A, B=L.CHOICE_B, B2=L.CHOICE_B2, C=L.CHOICE_C))
    supported = {}
    ST = {lab.split(".")[0]: ms for lab, ms in L.STAGES}             # stage models by stage key (A, B, B2, C, D)
    if all(m in fitted for m in ST["A"]):
        pick = "m01_intercepts"   # stage A: keep each addition in turn if it converged and gains > 2 SE
        for m in ("m02_rw_1lr", "m03_rw_2lr", "m04_rw_2lr_decay"):
            d, se = dse(m, pick)
            if ok(m) and d > 2 * se:
                pick = m
        supported["A"] = pick
    if "B" in ST and all(m in fitted for m in ST["B"]):
        cand = [m for m in ST["B"] if ok(m)] or ST["B"]   # stage B: highest elpd among converged models
        supported["B"] = L.SPECS[max(cand, key=lambda m: fitted[m]["elpd_loo"])]["bias"]
    if "B2" in ST and all(m in fitted for m in ST["B2"]):                # B2, the novelty bonus: kept if it gains more than 2 SE
        m, par = ST["B2"][0], L.stage_parent(ST["B2"][0])
        if par in fitted:
            d, se = dse(m, par); supported["B2"] = bool(ok(m) and d > 2 * se)
    if "C" in ST and all(m in fitted for m in ST["C"]):
        # Stage C: the better converged trend model, kept only if it beats the stage-B winner by > 2 SE.
        par = L.stage_parent(ST["C"][0])
        cand = [m for m in ST["C"] if ok(m)]
        best = max(cand, key=lambda m: fitted[m]["elpd_loo"]) if cand else None
        if best and par in fitted and dse(best, par)[0] > 2 * dse(best, par)[1]:
            supported["C"] = "beta_trend" if L.SPECS[best]["beta_trend"] else "bias_trend"
        elif par in fitted:
            supported["C"] = "none"
    summary["supported_choices"] = supported
    for k, v in supported.items():
        rec = summary["recorded_choices"][k]
        flag = "" if rec == v else ("  <-- NOT YET RECORDED in likelihoods.py" if rec is None else f"  <-- DIFFERS from the recorded choice {rec!r}")
        print(f"stage {k}: the data support {v!r}{flag}")

    # Final comparison: the model without an ITI term (the parent of the stage-D models)
    # and the ITI models of stage D, with BMS among the converged ones. It runs only if the no-ITI model
    # converged and at least one ITI model did. roles gives each model a plain description.
    if "D" in ST:
        base = L.stage_parent(ST["D"][0])
        roles = {base: "no interval", "m10_iti_bias": "ITI on choice bias", "m11_iti_target": "ITI as learning target",
                 "m12_iti_compete": "ITI competes with value"}
        final = [m for m in [base] + list(ST["D"]) if m in fitted]
        conv = [m for m in final if ok(m)]
        if base in conv and len(conv) > 1:
            if len(conv) < len(final):
                print("not converged, left out of the final comparison:", [m for m in final if m not in conv])
            # Differences from the best model (top) and from the no-ITI model (base), then BMS on each
            # participant's elpd: expected frequency, exceedance probability, protected exceedance probability
            # (PXP), number of participants best fitted, and Bayesian omnibus risk (BOR).
            top = max(conv, key=lambda m: fitted[m]["elpd_loo"])
            t3 = pd.DataFrame([dict(role=roles[m], model=m, label=C.MODEL_SHORT[m], elpd_loo=fitted[m]["elpd_loo"],
                                    elpd_minus_best=dse(m, top)[0] if m != top else 0.0, se_vs_best=dse(m, top)[1] if m != top else 0.0,
                                    elpd_minus_base=dse(m, base)[0] if m != base else 0.0, se_vs_base=dse(m, base)[1] if m != base else 0.0)
                               for m in conv])
            elpd3 = pd.DataFrame({m: fitted[m]["subjects"].set_index("subject_id")["elpd_loo"] for m in conv})
            b3, _ = bms_table(elpd3.to_numpy(), list(elpd3.columns))
            t3 = t3.merge(b3[["model", "expected_frequency", "exceedance_probability", "protected_exceedance", "n_subjects_best", "bor"]], on="model", how="left")
            t3.to_csv(out / "final_comparison.csv", index=False); b3.to_csv(out / "bms_final.csv", index=False)
            print(f"\nfinal comparison (elpd differences with SE; BMS among the {len(conv)}):")
            print(t3[["role", "model", "elpd_loo", "elpd_minus_best", "se_vs_best", "elpd_minus_base", "se_vs_base", "expected_frequency",
                      "protected_exceedance", "n_subjects_best"]].round(3).to_string(index=False))
            summary.update(final=conv, winner_elpd=top, winner_pxp=str(b3.sort_values("protected_exceedance", ascending=False).iloc[0]["model"]),
                           final_table=t3.to_dict(orient="records"))
    # Participant-level elpd of every fitted model (one column per model).
    elpd_all = pd.DataFrame({m: r["subjects"].set_index("subject_id")["elpd_loo"] for m, r in fitted.items()}); elpd_all.to_csv(out / "subject_elpd_loo.csv")
    summary["best_overall"] = best_all
    R.save_json(out, "comparison", summary)

    # Figure: elpd of every fitted model relative to the best model overall, with error
    # bars of 2 SE of the paired difference, on a symmetric-log x axis (linear within +/- 20).
    plt = R.plt
    c2 = comp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(c2) + 1.4))
    ax.barh(range(len(c2)), c2["elpd_minus_best"], xerr=2 * c2["se_vs_best"], color=BLUE, height=0.7)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(range(len(c2))); ax.set_yticklabels([C.MODEL_SHORT[m] for m in c2["model"]], fontsize=8)
    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlabel(f"elpd relative to the best model, {C.MODEL_SHORT[best_all]} (PSIS-LOO; error bar = 2 SE of the paired difference; symlog)")
    ax.set_title(f"{R.label}: the model sequence by NUTS (n = {data['N']})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_model_comparison.png", dpi=150); plt.close(fig)
    R.save_json(out, "run_info", dict(vars(args), n_subjects=data["N"], models_fitted=list(fitted)))
    print("wrote", out)
