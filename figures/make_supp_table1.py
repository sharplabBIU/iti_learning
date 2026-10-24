"""Supplementary Table 1: the model-building sequence of the manuscript, fitted to Studies 3 and 4.

Model names are those of the manuscript (config.MODEL_SHORT, with the en dash of Rescorla–Wagner) and the parameters are given in
the notation of the Methods. For each model: its stage, the model it builds on, its parameters, ELPD (SE), the gain over the model it
builds on and the difference from the best converged model (each with the SE of the paired difference), and convergence. Then the
final comparison with random-effects Bayesian model selection (Study 3).
Inputs: data/study_{3,4}/analysis/results/06_model_fits/ (model_comparison_loo.csv, final_comparison.csv, subject_elpd_loo.csv).
Outputs: figures/SupplementaryTable1_model_comparison.md and .csv.
Run: python3 figures/make_supp_table1.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from analysis_common import config as C              # noqa: E402
from analysis_common.likelihoods import PARAMS       # noqa: E402

OUT = HERE
RES = lambda s: REPO / f"data/study_{s}/analysis/results/06_model_fits"
STAGES = {"A": "Value learning", "B": "Choice bias", "B2": "Novelty", "C": "Trend over the session", "D": "ITI"}
SYMBOL = {"c1": "c₁", "c2": "c₂", "alpha_r": "α⁻", "beta": "β", "decay": "d", "stick": "s", "alpha_c": "α_C",
          "beta_c": "β_C", "omega": "ω", "beta_slope": "κ_β", "bias_slope": "κ_b", "gamma": "γ", "cost": "c",
          "kappa": "κ"}
LABEL = {3: "Study 3 (preregistered replication)", 4: "Study 4 (reversed ITI sequence)"}


def name(m):
    return C.MODEL_SHORT[m].replace("Rescorla-Wagner", "Rescorla–Wagner")


def params(m):
    ps = PARAMS[m]
    alpha = "α" if "alpha_r" not in ps else "α⁺"           # one learning rate, or the rate after a win
    return ", ".join(alpha if p == "alpha" else SYMBOL[p] for p in ps)


def fmt(x, signed=False, d=1):
    if pd.isna(x):
        return ""
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


md = ["## Supplementary Table 1. The model-building sequence", "",
      "Every model was fitted to the main sample (sessions that reached trial 90) as a hierarchical Bayesian model with the "
      "No-U-Turn sampler, using four chains of 1,000 draws after 2,000 warm-up iterations. In Study 4, the exploration-trend "
      "model and the ITI-competes-with-value model were refitted with four chains of 2,000 draws after 4,000 warm-up "
      "iterations. Models were compared by Pareto-smoothed importance-sampling leave-one-out cross-validation (Methods). The "
      "stages were decided on Study 3, and the same models were fitted separately to Study 4. Parameters are in the notation "
      "of the Methods (α, one learning rate; α⁺ and α⁻, learning rates after a win and after a loss; β, inverse temperature; "
      "d, value decay; s, stay bias; α_C and β_C, rate and weight of the choice kernel; ω, novelty bonus; κ_β and κ_b, trends "
      "in the inverse temperature and in the choice-kernel weight; γ, c and κ, the ITI parameters; c₁ and c₂, option "
      "intercepts). ELPD, expected log predictive density (higher is better), with its standard error (SE). Gain, ELPD minus "
      "that of the model it builds on. Vs best, ELPD minus that of the best converged model of the study. Both differences "
      "are given with the standard error of the paired difference. R̂, largest group-level split-R̂ (convergence criterion "
      "1.1). ESS, smallest bulk effective sample size. Div., divergent transitions. The final comparison gives the protected "
      "exceedance probability (PXP) from random-effects Bayesian model selection on participant-level ELPD. In Study 4 it was "
      "not run, because the model without an ITI term did not meet the convergence criterion. "
      "Source: data/study_N/analysis/results/06_model_fits/.", ""]
rows = []
for s in (3, 4):
    comp = pd.read_csv(RES(s) / "model_comparison_loo.csv")
    n = pd.read_csv(RES(s) / "subject_elpd_loo.csv").shape[0]
    md += [f"### {LABEL[s]}, n = {n}", "",
           "| Stage | Model | Builds on | Parameters | ELPD | SE | Gain | SE | Vs best | SE | R̂ | ESS | Div. |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        stage = STAGES[r["stage"].split(".")[0]]
        parent = name(r["parent"]) if isinstance(r.get("parent"), str) else ""
        md.append(f"| {stage} | {name(r['model'])} | {parent} | {params(r['model'])} | {fmt(r['elpd_loo'])} | "
                  f"{fmt(r['se_elpd_loo'])} | {fmt(r['elpd_minus_parent'], True)} | {fmt(r['se_vs_parent'])} | "
                  f"{fmt(r['elpd_minus_best'], True)} | {fmt(r['se_vs_best'])} | {r['rhat_group_max']:.2f} | "
                  f"{r['ess_bulk_min']:.0f} | {int(r['n_divergences'])} |")
        rows.append(dict(study=s, stage=stage, model=r["model"], label=name(r["model"]), builds_on=parent,
                         parameters=params(r["model"]), elpd=r["elpd_loo"], se=r["se_elpd_loo"],
                         gain=r["elpd_minus_parent"], se_gain=r["se_vs_parent"], vs_best=r["elpd_minus_best"],
                         se_vs_best=r["se_vs_best"], rhat_max=r["rhat_group_max"], ess_min=r["ess_bulk_min"],
                         divergences=r["n_divergences"]))
    fin = RES(s) / "final_comparison.csv"
    if s == 3 and fin.exists():
        f = pd.read_csv(fin)
        md += ["", "Final comparison:", "", "| Model | ELPD | Vs no ITI term | SE | PXP | Best for n |", "|---|---|---|---|---|---|"]
        for _, r in f.iterrows():
            md.append(f"| {name(r['model'])} | {fmt(r['elpd_loo'])} | {fmt(r['elpd_minus_base'], True)} | "
                      f"{fmt(r['se_vs_base'])} | {r['protected_exceedance']:.2f} | {int(r['n_subjects_best'])} |")
    md.append("")
(OUT / "SupplementaryTable1_model_comparison.md").write_text("\n".join(md) + "\n")
pd.DataFrame(rows).to_csv(OUT / "SupplementaryTable1_model_comparison.csv", index=False)
print("\n".join(md))
