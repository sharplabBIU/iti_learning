"""Compare the regenerated results with every number reported in the main text and the Supplementary Information.

Run after reproduce.sh (it is its last step): python3 check_reported_numbers.py
Writes reproduction_report.md: one row per reported number, with the value in the paper, the regenerated value at the
paper's precision, and the verdict:
  match     the regenerated value, rounded as in the paper, equals the reported value (or satisfies the reported bound,
            e.g. p < 10^-6)
  close     a value estimated by Markov chain Monte Carlo (MCMC) or by simulation that differs only in the last digits:
            within 3 units of the last reported digit or 2% of the value, whichever is larger; for a posterior mean or
            credible-interval bound, within 5% of the width of the reported 95% interval (about two Monte Carlo standard
            errors of a bound at the effective sample sizes of these fits) if that is larger. Posterior summaries and
            simulations vary slightly between runs and machines, so the paper's run and this one need not agree exactly.
  DIFFERS   anything else (listed again at the top of the report)
  (to add)  a number the paper still has as a placeholder; the regenerated value is given to fill it in
The references in brackets give the paragraph of the manuscript (Results, Methods) or the item of the supplement.
"""
from __future__ import annotations
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = lambda s: ROOT / f"data/study_{s}/analysis/results"
FIG = ROOT / "figures"


def J(path):
    return json.load(open(path))


def jr(s, rel):
    return J(RES(s) / rel)


def csv(s, rel):
    return pd.read_csv(RES(s) / rel)


def pw_diff(s, a, b):
    """Paired ELPD difference a - b over the pointwise observations, and its SE sqrt(n var(d))."""
    ea = np.load(RES(s) / f"06_model_fits/{a}_pointwise.npz")["elpd_i"]; eb = np.load(RES(s) / f"06_model_fits/{b}_pointwise.npz")["elpd_i"]
    d = ea - eb
    return float(d.sum()), float(np.sqrt(len(d) * np.var(d)))


def mu(s, model, p, support):
    """Posterior mean and 95% credible interval of the group location mu of parameter p, on the parameter's scale."""
    x = pd.read_csv(RES(s) / f"06_model_fits/{model}_group_posterior_draws.csv.gz")[f"mu_{p}"].to_numpy()
    x = 1 / (1 + np.exp(-x)) if support == "unit" else (np.exp(x) if support == "pos" else x)
    return float(x.mean()), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def comp(s):
    return csv(s, "06_model_fits/model_comparison_loo.csv").set_index("model")


def ppc(s):
    return csv(s, "10_ppc/ppc_signatures.csv").set_index("model")


def ppc_data(s):
    a = csv(s, "10_ppc/ppc_signatures_all_reps.csv")
    return a[a.model == "data"].iloc[0]


def curves(s):
    return csv(s, "10_ppc/ppc_curves.csv")


def rmse_chose_best(s, m):
    c = curves(s); obs = c[c.model == "data"].set_index("w")["chose_best"]
    sim = c[c.model == m].groupby("w")["chose_best"].mean()
    return float(np.sqrt(((sim - obs.reindex(sim.index)) ** 2).mean()))


def corr(s, q, idx):
    c = csv(s, "05_individual_differences/correlations.csv")
    return c[(c.questionnaire_key == q) & (c["index"] == idx)].iloc[0]


def glmm16(s, key):
    return jr(s, "16_position_controlled/summary.json")["glmm"][key]


def raw16(s, key):
    r = jr(s, "16_position_controlled/summary.json")["raw_contrasts"]
    return next(x for x in r if x["subset"] == key)


def fixef16(s, model, term, scale):
    f = csv(s, f"16_position_controlled/glmm_{model}_fixef.csv").set_index("term").loc[term]
    return scale * f["mean"], scale * f["ci_low"], scale * f["ci_high"]


def marker(s):
    return jr(s, "19_exploration_marker_ppc/stats.json")


def marker_ppc(s, model, outcome):
    p = csv(s, "19_exploration_marker_ppc/ppc.csv")
    return p[(p.model == model) & (p.outcome == outcome)]["slope"]


def agents():
    return pd.read_csv(ROOT / "ai_agents/results/agent_metrics.csv")


def figure_numbers():
    f = FIG / "figure_numbers.json"
    return J(f) if f.exists() else {}


def rt_floor_share():
    """Share of the Study 2 main sample's responded choices faster than 250 ms (treated as missed trials)."""
    t = pd.read_csv(ROOT / "data/study_2/data/trials.csv"); P = pd.read_csv(ROOT / "data/study_2/data/participants.csv")
    t = t[t.subject_id.isin(P.loc[P.reached_min_trials, "subject_id"]) & t.valid]
    return float((t.rt < 250).mean())


def flow(s, stage_start):
    f = pd.read_csv(ROOT / f"data/study_{s}/data/exclusion_flow.csv")
    return int(f[f.stage.str.strip().str.startswith(stage_start)].iloc[0]["n"])


def recovery_margins():
    out = []
    for f in sorted(glob.glob(str(RES(3) / "07_recovery/model_recovery_n50_iti3_grid/*_rep*.json"))):
        r = J(f); g = r["generating"]; others = [m for m in r["fits"] if m != g]
        el = {m: np.asarray(r["fits"][m]["subject_elpd"]) for m in r["fits"]}
        out.append(dict(won=r["winner_elpd"] == g, margin=float(el[g].sum() - max(el[m].sum() for m in others)),
                        rhat=max(float(v["rhat_max"]) for v in r["fits"].values())))
    return out


def recovery_in_range(p):
    d = csv(3, "07_recovery/parameter_recovery_m12_iti_compete.csv"); rr = csv(3, "07_recovery/parameter_ranges.csv")
    rr = rr[rr.model == "m12_iti_compete"].set_index("parameter")
    g = d[(d.parameter == p) & d.generating.between(rr.loc[p, "fitted_min"], rr.loc[p, "fitted_max"])]
    return float(np.corrcoef(g.generating, g.recovered)[0, 1])


# ---- the checks ---------------------------------------------------------------------------------------------------------
# (where, what, paper value, decimals, getter, kind); kind: "exact", "mcmc" (MCMC or simulation), "lt" (paper reports a
# bound: p < value), "todo" (placeholder in the paper; paper value None)
C = []
def add(where, what, paper, dec, get, kind="exact", tol=0.0):
    C.append(dict(where=where, what=what, paper=paper, dec=dec, get=get, kind=kind, tol=tol))


def add_ci(where, what, vals, dec, get3, names=("b", "CI low", "CI high")):
    """A posterior mean and its 95% credible interval (three checks); get3() returns the three regenerated values."""
    w = abs(vals[2] - vals[1])
    for j, nm in enumerate(names):
        add(where, f"{what}: {nm}", vals[j], dec, lambda j=j: get3()[j], "mcmc", tol=0.05 * w)


pt = lambda s, k: jr(s, "02_switching/switching_by_outcome.json")[k]
me = lambda s: jr(s, "02_switching/main_effect.json")["paired_test"]
ph = lambda s, k: jr(s, "02_switching/phase_by_outcome.json")[k]
gl = lambda s: jr(s, "02_switching/glmm.json")
lr = lambda s: jr(s, "01_learning/learning_stats.json")

# Sample sizes and inclusion (Results [20]; Methods [78])
for s, n, tot in ((1, 96, 101), (2, 96, 96), (3, 439, 488), (4, 117, 135)):
    add("Methods [78]", f"Study {s}: participants with trial data", tot, 0, lambda s=s: flow(s, "participants with trial data"))
    add("Results [20]; Methods [78]", f"Study {s}: main sample (session reached trial 90)", n, 0, lambda s=s: flow(s, "main sample:"))
for s in (1, 2, 3, 4):
    add("Methods [78]", f"Study {s}: largest number of missed trials of an included participant (<= 5)", 5, 0,
        lambda s=s: flow(s, "main sample, largest number"))
add("Methods [83]", "Study 3: age and birth year differ by more than two years", 5, 0, lambda: flow(3, "|age - (2026 - birth year)| > 2"))
for s in (3, 4):
    add("Methods [86]", f"Study {s}: failed the BIS attention check", 1, 0, lambda s=s: flow(s, "failed the BIS attention check"))

# Study 1 (Results [21])
add("Results [21]", "S1 p(switch) after 4 s, %", 45.1, 1, lambda: 100 * me(1)["mean_long"])
add("Results [21]", "S1 p(switch) after 1 s, %", 38.9, 1, lambda: 100 * me(1)["mean_short"])
add("Results [21]", "S1 ITI effect, t(95)", 5.54, 2, lambda: me(1)["t"])
add("Results [21]", "S1 ITI effect, p < 10^-6", 1e-6, None, lambda: me(1)["p_two_sided"], "lt")
add("Results [21]", "S1 after a win, 1 s, %", 20.5, 1, lambda: 100 * pt(1, "switch_after_win")["mean_short"])
add("Results [21]", "S1 after a win, 4 s, %", 28.2, 1, lambda: 100 * pt(1, "switch_after_win")["mean_long"])
add("Results [21]", "S1 after a win, t(95)", 5.89, 2, lambda: pt(1, "switch_after_win")["t"])
add("Results [21]", "S1 after a win, p < 10^-7", 1e-7, None, lambda: pt(1, "switch_after_win")["p_two_sided"], "lt")
add("Results [21]", "S1 after a loss, 1 s, %", 69.3, 1, lambda: 100 * pt(1, "switch_after_loss")["mean_short"])
add("Results [21]", "S1 after a loss, 4 s, %", 70.3, 1, lambda: 100 * pt(1, "switch_after_loss")["mean_long"])
add("Results [21]", "S1 after a loss, t(95)", 0.61, 2, lambda: pt(1, "switch_after_loss")["t"])
add("Results [21]", "S1 after a loss, p", 0.54, 2, lambda: pt(1, "switch_after_loss")["p_two_sided"])
for k, a, b, t, p, pk in (("early_after_win", 26.8, 32.5, 3.16, 0.002, "p"), ("late_after_win", 17.8, 23.5, 4.05, 1e-3, "lt"),
                          ("early_after_loss", 73.9, 72.6, -0.67, 0.51, "p"), ("late_after_loss", 65.2, 66.4, 0.51, 0.61, "p")):
    add("Results [21]", f"S1 {k}: 1 s, %", a, 1, lambda k=k: 100 * ph(1, k)["mean_short"])
    add("Results [21]", f"S1 {k}: 4 s, %", b, 1, lambda k=k: 100 * ph(1, k)["mean_long"])
    add("Results [21]", f"S1 {k}: t(95)", t, 2, lambda k=k: ph(1, k)["t"])
    add("Results [21]", f"S1 {k}: p" + (" < 10^-3" if pk == "lt" else ""), p, None if pk == "lt" else (3 if p < 0.01 else 2),
        lambda k=k: ph(1, k)["p_two_sided"], "lt" if pk == "lt" else "exact")
add("Results [21]", "S1 chose-best rate, %", 47, 0, lambda: 100 * lr(1)["chose_best_mean"])
add("Results [21]", "S1 chose-best vs 1/3, t(95)", 15.7, 1, lambda: lr(1)["t_vs_chance"])
add("Results [21]", "S1 chose-best vs 1/3, p < 10^-27", 1e-27, None, lambda: lr(1)["p_vs_chance"], "lt")

# Study 2 (Results [25], [26]; Methods [78])
cn = lambda k: figure_numbers()["study2_contamination"][k]
add("Results [25]", "S2 after a missed downward display at 1 s: next choices within 250 ms, %", 24, 0, lambda: 100 * cn("a_Study 2: missed downward")["share_lt_250"])
add("Results [25]", "S2 after a missed downward display at 1 s: next choices within 50 ms, %", 8, 0, lambda: 100 * cn("a_Study 2: missed downward")["share_lt_50"])
add("Results [25]; Methods [78]", "S2 after a 1 s ITI: choices within 250 ms, %", 11.3, 1,
    lambda: 100 * (cn("a_Study 2: missed downward")["share_lt_250"] * cn("a_Study 2: missed downward")["n"]
                   + cn("a_Study 2: other displays")["share_lt_250"] * cn("a_Study 2: other displays")["n"])
    / (cn("a_Study 2: missed downward")["n"] + cn("a_Study 2: other displays")["n"]))
add("Results [25]; Methods [78]", "S1 after a 1 s ITI: choices within 250 ms, %", 2.5, 1, lambda: 100 * cn("a_Study 1")["share_lt_250"])
add("Results [25]; Methods [78]", "S2 choices faster than 250 ms (treated as missed), %", 6.7, 1, lambda: 100 * rt_floor_share())
add("Results [26]", "S2 p(switch) 1 s, %", 49.9, 1, lambda: 100 * me(2)["mean_short"])
add("Results [26]", "S2 p(switch) 4 s, %", 55.9, 1, lambda: 100 * me(2)["mean_long"])
add("Results [26]", "S2 ITI effect, t(95)", 5.51, 2, lambda: me(2)["t"])
add("Results [26]", "S2 ITI effect, p < 10^-6", 1e-6, None, lambda: me(2)["p_two_sided"], "lt")
add("Results [26]", "S2 after a win, 1 s, %", 33.5, 1, lambda: 100 * pt(2, "switch_after_win")["mean_short"])
add("Results [26]", "S2 after a win, 4 s, %", 39.3, 1, lambda: 100 * pt(2, "switch_after_win")["mean_long"])
add("Results [26]", "S2 after a win, t(95)", 4.11, 2, lambda: pt(2, "switch_after_win")["t"])
add("Results [26]", "S2 after a win, p < 10^-4", 1e-4, None, lambda: pt(2, "switch_after_win")["p_two_sided"], "lt")
add("Results [26]", "S2 after a loss, 1 s, %", 75.9, 1, lambda: 100 * pt(2, "switch_after_loss")["mean_short"])
add("Results [26]", "S2 after a loss, 4 s, %", 78.1, 1, lambda: 100 * pt(2, "switch_after_loss")["mean_long"])
add("Results [26]", "S2 after a loss, t(95)", 1.54, 2, lambda: pt(2, "switch_after_loss")["t"])
add("Results [26]", "S2 after a loss, p", 0.13, 2, lambda: pt(2, "switch_after_loss")["p_two_sided"])
for k, a, b, t, p, pk in (("early_after_win", 33.5, 39.2, 3.18, 0.002, "p"), ("late_after_win", 33.6, 40.0, 3.86, 1e-3, "lt"),
                          ("early_after_loss", 77.6, 79.9, 1.14, 0.26, "p"), ("late_after_loss", 74.7, 73.4, -0.58, 0.57, "p")):
    add("Results [26]", f"S2 {k}: 1 s, %", a, 1, lambda k=k: 100 * ph(2, k)["mean_short"])
    add("Results [26]", f"S2 {k}: 4 s, %", b, 1, lambda k=k: 100 * ph(2, k)["mean_long"])
    add("Results [26]", f"S2 {k}: t(95)", t, 2, lambda k=k: ph(2, k)["t"])
    add("Results [26]", f"S2 {k}: p" + (" < 10^-3" if pk == "lt" else ""), p, None if pk == "lt" else (3 if p < 0.01 else 2),
        lambda k=k: ph(2, k)["p_two_sided"], "lt" if pk == "lt" else "exact")
rdm = lambda: jr(2, "11_rdm_task/rdm_stats.json")["accuracy_by_iti"]
add("Results [26]", "S2 dot-motion accuracy, 4 s displays, %", 77, 0, lambda: 100 * rdm()["acc_4s"])
add("Results [26]", "S2 dot-motion accuracy, 1 s displays, %", 58, 0, lambda: 100 * rdm()["acc_1s"])
add("Results [26]", "S2 dot-motion accuracy 4 s vs 1 s, t(95)", 12.1, 1, lambda: rdm()["t"])
add("Results [26]", "S2 dot-motion accuracy 4 s vs 1 s, p < 10^-20", 1e-20, None, lambda: rdm()["p"], "lt")

# Study 3 (Results [29]-[32])
add("Results [29]", "S3 ITI effect, t(438)", 12.30, 2, lambda: me(3)["t"])
add("Results [29]", "S3 ITI effect, p < 10^-29", 1e-29, None, lambda: me(3)["p_two_sided"], "lt")
add("Results [29]", "S3 after a win, 1 s, %", 29.3, 1, lambda: 100 * pt(3, "switch_after_win")["mean_short"])
add("Results [29]", "S3 after a win, 4 s, %", 36.5, 1, lambda: 100 * pt(3, "switch_after_win")["mean_long"])
add("Results [29]", "S3 after a win, t(438)", 12.30, 2, lambda: pt(3, "switch_after_win")["t"])
add("Results [29]", "S3 after a win, p < 10^-29", 1e-29, None, lambda: pt(3, "switch_after_win")["p_two_sided"], "lt")
add("Results [29]", "S3 after a loss, 1 s, %", 71.8, 1, lambda: 100 * pt(3, "switch_after_loss")["mean_short"])
add("Results [29]", "S3 after a loss, 4 s, %", 72.9, 1, lambda: 100 * pt(3, "switch_after_loss")["mean_long"])
add("Results [29]", "S3 after a loss, t(438)", 1.74, 2, lambda: pt(3, "switch_after_loss")["t"])
add("Results [29]", "S3 after a loss, p", 0.083, 3, lambda: pt(3, "switch_after_loss")["p_two_sided"])
dc = lambda s: jr(s, "13_design_confound/design_confound.json")
blk = lambda s, key, i: list(dc(s)[key].values())[i]
add("Results [30]", "S3 p(switch), first twenty trials, %", 62, 0, lambda: 100 * blk(3, "p_switch_by_20trial_block", 0))
add("Results [30]", "S3 p(switch), last twenty trials, %", 40, 0, lambda: 100 * blk(3, "p_switch_by_20trial_block", 4))
add_ci("Results [31]", "S3 ITI slope with the linear trial term", (0.040, 0.025, 0.055), 3,
       lambda: tuple(gl(3)["trial_controlled"][k] for k in ("estimate", "ci_low", "ci_high")))
for s in (1, 2):
    add("Results [31]", f"S{s} ITI slope with the linear trial term, b [CI]", None, 3,
        lambda s=s: (gl(s)["trial_controlled"]["estimate"], gl(s)["trial_controlled"]["ci_low"], gl(s)["trial_controlled"]["ci_high"]), "todo")
for i, v in enumerate((70, 60, 40, 40, 40)):
    add("Results [32]; Methods [80]", f"S1-3 share of 4 s ITIs, trials {20 * i + 1}-{20 * i + 20}, %", v, 0, lambda i=i: 100 * blk(3, "long_iti_share_by_20trial_block", i))
add("Results [32]; Methods [90]", "S3 correlation of a 4 s ITI with trial number, trials 51-100", -0.05, 2, lambda: dc(3)["corr_long_iti_with_trial_by_half"]["trials_51_100"]["r"])
add("Results [32]; Methods [90]", "S3 correlation of a 4 s ITI with trial number, trials 1-50", -0.15, 2, lambda: dc(3)["corr_long_iti_with_trial_by_half"]["trials_1_50"]["r"])
for k, a, b, t, p, pk, dp in (("late_after_win", 27.4, 32.3, 7.10, 1e-11, "lt", None), ("early_after_win", 33.4, 40.3, 8.37, 1e-15, "lt", None),
                              ("early_after_loss", 75.4, 75.4, 0.03, 0.97, "p", 2), ("late_after_loss", 69.4, 67.8, -1.80, 0.07, "p", 2)):
    add("Results [32]", f"S3 {k}: 1 s, %", a, 1, lambda k=k: 100 * ph(3, k)["mean_short"])
    add("Results [32]", f"S3 {k}: 4 s, %", b, 1, lambda k=k: 100 * ph(3, k)["mean_long"])
    add("Results [32]", f"S3 {k}: t(438)", t, 2, lambda k=k: ph(3, k)["t"])
    add("Results [32]", f"S3 {k}: p" + (f" < {p:g}" if pk == "lt" else ""), p, dp, lambda k=k: ph(3, k)["p_two_sided"], "lt" if pk == "lt" else "exact")

# The earlier ITI (Results [36]; Methods [94]; Supplementary Note 1)
add_ci("Results [36]", "S3 ITI two trials back (other option)", (0.028, -0.019, 0.073), 3,
       lambda: tuple(glmm16(3, "twoback_different")[k] for k in ("estimate", "ci_low", "ci_high")))
add("Methods [94]", "S3 mean trial of trials whose earlier ITI was 4 s", 41, 0, lambda: raw16(3, "twoback_different")["mean_trial_4s"])
add("Methods [94]", "S3 mean trial of trials whose earlier ITI was 1 s", 54, 0, lambda: raw16(3, "twoback_different")["mean_trial_1s"])
SN1 = {3: [("last ITI", "twoback_different", "prev_iti_c", 3.0, (0.130, 0.068, 0.191)), ("2 back, same option", "twoback_same", "prev2_long", 1.0, (0.009, -0.059, 0.077)),
           ("2 back, other option", "twoback_different", "prev2_long", 1.0, (0.025, -0.040, 0.090)), ("3 back", "threeback_different", "prev3_long", 1.0, (0.036, -0.022, 0.094)),
           ("4 back", "fourback_different", "prev4_long", 1.0, (-0.023, -0.083, 0.038))],
       4: [("last ITI", "twoback_different", "prev_iti_c", 3.0, (0.130, 0.015, 0.250)), ("2 back, same option", "twoback_same", "prev2_long", 1.0, (0.043, -0.091, 0.174)),
           ("2 back, other option", "twoback_different", "prev2_long", 1.0, (-0.070, -0.191, 0.056)), ("3 back", "threeback_different", "prev3_long", 1.0, (-0.104, -0.230, 0.023)),
           ("4 back", "fourback_different", "prev4_long", 1.0, (0.012, -0.109, 0.134))]}
for s, rows in SN1.items():
    for lab, model, term, sc, vals in rows:
        add_ci("Supplementary Note 1", f"S{s} {lab}", vals, 3, lambda s=s, model=model, term=term, sc=sc: fixef16(s, model, term, sc))
add("Supplementary Note 1", "largest R-hat of the earlier-ITI models (<= 1.02)", 1.02, 2,
    lambda: max(J(f)["rhat_max"] for s in (3, 4) for f in glob.glob(str(RES(s) / "16_position_controlled/glmm_*.json"))), "le")
add_ci("Results [55]", "S4 ITI two trials back (other option), odds ratio", (0.93, 0.83, 1.06), 2,
       lambda: tuple(glmm16(4, "twoback_different")[k] for k in ("odds_ratio", "or_ci_low", "or_ci_high")), ("OR", "CI low", "CI high"))

# The outcome asymmetry (Results [37])
for s, w, l, inter in ((3, (1.44, 1.33, 1.56), (0.95, 0.88, 1.02), (0.145, 0.112, 0.178)), (1, (1.42, 1.12, 1.74), (0.97, 0.81, 1.15), None),
                       (2, (1.35, 1.13, 1.59), (1.10, 0.91, 1.34), (0.053, -0.023, 0.129))):
    for lab, vals in (("after_win", w), ("after_loss", l)):
        add_ci("Results [37]", f"S{s} {lab}, odds ratio 4 s vs 1 s", vals, 2,
               lambda s=s, lab=lab: tuple(gl(s)["by_outcome_trial_controlled"][lab][k] for k in ("odds_ratio", "or_ci_low", "or_ci_high")), ("OR", "CI low", "CI high"))
    if inter:
        add_ci("Results [37]", f"S{s} ITI x outcome interaction", inter, 3,
               lambda s=s: tuple(gl(s)["interaction_trial_controlled"][k] for k in ("estimate", "ci_low", "ci_high")))

# The model sequence, Study 3 (Results [42]-[45])
gain = lambda s, m, k="elpd_minus_parent": comp(s).loc[m, k]
add("Results [42]", "S3 RW vs fixed preferences, ELPD gain", 7518, 0, lambda: gain(3, "m02_rw_1lr"), "mcmc")
add("Results [42]", "S3 RW vs fixed preferences, SE", 121, 0, lambda: gain(3, "m02_rw_1lr", "se_vs_parent"), "mcmc")
add("Results [42]", "S3 separate learning rates, gain", 346, 0, lambda: gain(3, "m03_rw_2lr"), "mcmc")
add("Results [42]", "S3 separate learning rates, SE", 35, 0, lambda: gain(3, "m03_rw_2lr", "se_vs_parent"), "mcmc")
add("Results [42]", "S3 value decay, gain", 1575, 0, lambda: gain(3, "m04_rw_2lr_decay"), "mcmc")
add("Results [42]", "S3 value decay, SE", 66, 0, lambda: gain(3, "m04_rw_2lr_decay", "se_vs_parent"), "mcmc")
cb_w = lambda: curves(3)[lambda d: d.model == "data"].set_index("w")["chose_best"]
add("Results [42]", "S3 share of choices of the best option, lowest 10-trial window, %", 31, 0, lambda: 100 * cb_w().min())
add("Results [42]", "S3 share of choices of the best option, highest 10-trial window, %", 60, 0, lambda: 100 * cb_w().max())
add("Results [42]", "S3 RMSE of the learning curve, best learning model", 0.035, 3, lambda: min(rmse_chose_best(3, m) for m in ("m02_rw_1lr", "m03_rw_2lr", "m04_rw_2lr_decay")), "mcmc")
add("Results [42]", "S3 RMSE of the learning curve, worst learning model", 0.042, 3, lambda: max(rmse_chose_best(3, m) for m in ("m02_rw_1lr", "m03_rw_2lr", "m04_rw_2lr_decay")), "mcmc")
add("Results [42]", "S3 RMSE of the learning curve, fixed preferences", 0.068, 3, lambda: rmse_chose_best(3, "m01_intercepts"), "mcmc")
add("Results [43]", "S3 choice kernel over value decay, gain", 717.3, 1, lambda: gain(3, "m06_choice_kernel"), "mcmc")
add("Results [43]", "S3 choice kernel, SE", 42.7, 1, lambda: gain(3, "m06_choice_kernel", "se_vs_parent"), "mcmc")
add("Results [43]", "S3 constant stay bias over value decay, gain", 634.1, 1, lambda: gain(3, "m05_stick"), "mcmc")
add("Results [43]", "S3 constant stay bias, SE", 33.8, 1, lambda: gain(3, "m05_stick", "se_vs_parent"), "mcmc")
add("Results [43]", "S3 choice kernel minus stay bias", 83.2, 1, lambda: pw_diff(3, "m06_choice_kernel", "m05_stick")[0], "mcmc")
add("Results [43]", "S3 choice kernel minus stay bias, SE", 48.4, 1, lambda: pw_diff(3, "m06_choice_kernel", "m05_stick")[1], "mcmc")
add("Results [43]", "S3 p(switch) after a switch (data), %", 60.9, 1, lambda: 100 * ppc_data(3)["switch_after_switch"])
add("Results [43]", "S3 p(switch) after a stay (data), %", 40.5, 1, lambda: 100 * ppc_data(3)["switch_after_stay"])
for m, a, b in (("m05_stick", 58.5, 42.2), ("m06_choice_kernel", 57.8, 39.9)):
    add("Results [43]", f"S3 {m}: simulated p(switch) after a switch, %", a, 1, lambda m=m: 100 * ppc(3).loc[m, "switch_after_switch_mean"], "mcmc")
    add("Results [43]", f"S3 {m}: simulated p(switch) after a stay, %", b, 1, lambda m=m: 100 * ppc(3).loc[m, "switch_after_stay_mean"], "mcmc")
add("Results [44]", "S3 switches that return to the option chosen two back (data), %", 38.6, 1, lambda: 100 * ppc_data(3)["return_two_back"])
add("Results [44]", "S3 the same, choice-kernel model, %", 51.7, 1, lambda: 100 * ppc(3).loc["m06_choice_kernel", "return_two_back_mean"], "mcmc")
add("Results [44]", "S3 novelty bonus, gain", 913.7, 1, lambda: gain(3, "m07_novelty"), "mcmc")
add("Results [44]", "S3 novelty bonus, SE", 44.4, 1, lambda: gain(3, "m07_novelty", "se_vs_parent"), "mcmc")
add("Results [45]", "S3 exploration trend, gain", 559.8, 1, lambda: gain(3, "m08_trend_beta"), "mcmc")
add("Results [45]", "S3 exploration trend, SE", 35.7, 1, lambda: gain(3, "m08_trend_beta", "se_vs_parent"), "mcmc")
add("Results [45]", "S3 exploration trend minus choice-bias trend", 114.7, 1, lambda: pw_diff(3, "m08_trend_beta", "m09_trend_bias")[0], "mcmc")
add("Results [45]", "S3 exploration trend minus choice-bias trend, SE", 15.7, 1, lambda: pw_diff(3, "m08_trend_beta", "m09_trend_bias")[1], "mcmc")

# The exploration marker (Results [48], [56]; Methods [95])
for s, df_, w0, w4, tw, pw, l0, l4, tl, pl in ((3, 438, 0.048, 0.083, 3.41, 0.0007, -0.036, -0.001, 2.01, 0.045),
                                              (4, 116, -0.008, 0.077, 3.49, 0.0007, None, None, 0.04, 0.967)):
    where = "Results [48]" if s == 3 else "Results [56]"
    add(where, f"S{s} ITI effect after a win, most exploitative fifth", w0, 3, lambda s=s: marker(s)["data"]["win"]["effect"][0])
    add(where, f"S{s} ITI effect after a win, most exploratory fifth", w4, 3, lambda s=s: marker(s)["data"]["win"]["effect"][4])
    add(where, f"S{s} trend across quintiles after a win, t({df_})", tw, 2, lambda s=s: marker(s)["data"]["win"]["t"])
    add(where, f"S{s} trend after a win, p", pw, 4, lambda s=s: marker(s)["data"]["win"]["p"])
    if l0 is not None:
        add(where, f"S{s} ITI effect after a loss, most exploitative fifth", l0, 3, lambda s=s: marker(s)["data"]["loss"]["effect"][0])
        add(where, f"S{s} ITI effect after a loss, most exploratory fifth", l4, 3, lambda s=s: marker(s)["data"]["loss"]["effect"][4])
    add(where, f"S{s} trend across quintiles after a loss, t({df_})", tl, 2, lambda s=s: marker(s)["data"]["loss"]["t"])
    add(where, f"S{s} trend after a loss, p", pl, 3, lambda s=s: marker(s)["data"]["loss"]["p"])
for s in (3, 4):
    add("Methods [95]", f"S{s} the post-win trend is positive and p < 0.05 for every leak 0.5-0.9", 1, 0,
        lambda s=s: float(all((r.slope > 0) and (r.p < 0.05) for r in csv(s, "19_exploration_marker_ppc/lambda_robustness.csv").itertuples() if r.outcome == "win")))
hmm = lambda s, k: jr(s, "20_exploration_proxy_hmm/hmm_correlation.json")[k]["mean_within_participant_spearman"]
for s in (3, 4):
    add("Methods [95]", f"S{s} rank correlation of u with P(explore), post-win", 0.93, 2, lambda s=s: hmm(s, "after_win"))
    add("Methods [95]", f"S{s} rank correlation of u with P(explore), post-loss", 0.87, 2, lambda s=s: hmm(s, "after_loss"))

# The ITI models, Study 3 (Results [50], [51])
fin3 = lambda: csv(3, "06_model_fits/final_comparison.csv").set_index("model")
add("Results [50]", "S3 ITI competes with value over no ITI term", 118.2, 1, lambda: fin3().loc["m12_iti_compete", "elpd_minus_base"], "mcmc")
add("Results [50]", "S3 the same, SE", 16.8, 1, lambda: fin3().loc["m12_iti_compete", "se_vs_base"], "mcmc")
add("Results [50]", "S3 ITI competes with value over ITI as learning target", 104.1, 1, lambda: pw_diff(3, "m12_iti_compete", "m11_iti_target")[0], "mcmc")
add("Results [50]", "S3 the same, SE", 16.9, 1, lambda: pw_diff(3, "m12_iti_compete", "m11_iti_target")[1], "mcmc")
add("Results [50]", "S3 ITI competes with value over ITI on choice bias", 110.9, 1, lambda: pw_diff(3, "m12_iti_compete", "m10_iti_bias")[0], "mcmc")
add("Results [50]", "S3 the same, SE", 16.5, 1, lambda: pw_diff(3, "m12_iti_compete", "m10_iti_bias")[1], "mcmc")
add("Results [50]", "S3 ITI as learning target over no ITI term", 14.1, 1, lambda: fin3().loc["m11_iti_target", "elpd_minus_base"], "mcmc")
add("Results [50]", "S3 the same, SE", 8.7, 1, lambda: fin3().loc["m11_iti_target", "se_vs_base"], "mcmc")
add("Results [50]", "S3 ITI on choice bias over no ITI term", 7.3, 1, lambda: fin3().loc["m10_iti_bias", "elpd_minus_base"], "mcmc")
add("Results [50]", "S3 the same, SE", 5.3, 1, lambda: fin3().loc["m10_iti_bias", "se_vs_base"], "mcmc")
add("Results [50]", "S3 largest group-level R-hat of the four final models", 1.08, 2,
    lambda: max(comp(3).loc[m, "rhat_group_max"] for m in ("m08_trend_beta", "m10_iti_bias", "m11_iti_target", "m12_iti_compete")), "mcmc")
add("Results [50]", "S3 participants best fitted by ITI competes with value", 433, 0, lambda: fin3().loc["m12_iti_compete", "n_subjects_best"], "mcmc")
add("Results [50]", "S3 participants best fitted by ITI as learning target", 6, 0, lambda: fin3().loc["m11_iti_target", "n_subjects_best"], "mcmc")
add("Results [50]", "S3 protected exceedance probability of ITI competes with value (> 0.99)", 0.99, None,
    lambda: fin3().loc["m12_iti_compete", "protected_exceedance"], "gt")
for lab, model, p, sup, vals in (("kappa", "m12_iti_compete", "kappa", "unit", (0.096, 0.066, 0.126)),
                                  ("gamma", "m10_iti_bias", "gamma", "real", (-0.097, -0.169, -0.030)),
                                  ("c", "m11_iti_target", "cost", "real", (0.060, 0.027, 0.094))):
    add_ci("Results [50]", f"S3 {lab} ({model})", vals, 3, lambda model=model, p=p, sup=sup: mu(3, model, p, sup), ("posterior mean", "CI low", "CI high"))
sig = lambda m, k: ppc(3).loc[m, k]
for k, lab, vals in (("glmm_iti", "all trials", (0.036, 0.025, 0.050)), ("glmm_iti_win", "after wins", (0.098, 0.080, 0.115)),
                     ("glmm_iti_loss", "after losses", (0.001, -0.027, 0.019))):
    add_ci("Results [51]", f"S3 ITI competes with value, simulated ITI slope {lab}", vals, 3,
           lambda k=k: tuple(sig("m12_iti_compete", f"{k}_{suf}") for suf in ("mean", "lo", "hi")), ("mean", "2.5%", "97.5%"))
add("Results [51]", "S3 observed ITI slope, all trials", 0.040, 3, lambda: ppc_data(3)["glmm_iti"], "mcmc")
add("Results [51]", "S3 observed ITI slope, after wins", 0.111, 3, lambda: ppc_data(3)["glmm_iti_win"], "mcmc")
add("Results [51]", "S3 observed ITI slope, after losses", -0.035, 3, lambda: ppc_data(3)["glmm_iti_loss"], "mcmc")
add("Results [51]", "S3 ITI as learning target, simulated slope after wins", 0.036, 3, lambda: sig("m11_iti_target", "glmm_iti_win_mean"), "mcmc")
add("Results [51]", "S3 ITI on choice bias, simulated slope after wins", 0.026, 3, lambda: sig("m10_iti_bias", "glmm_iti_win_mean"), "mcmc")

# Recovery (Results [50]; kept from the run of the paper unless reproduce.sh ran without --skip-recovery)
prs = lambda: csv(3, "07_recovery/parameter_recovery_m12_iti_compete_summary.csv").set_index("parameter")["pearson_r"]
add("Results [50]", "parameter recovery: largest group-level R-hat of the refit", 1.01, 2,
    lambda: jr(3, "07_recovery/parameter_recovery_fit_m12_iti_compete_summary.json")["rhat_max"], "mcmc")
add("Results [50]", "parameter recovery: r for kappa", 0.80, 2, lambda: prs()["kappa"], "mcmc")
add("Results [50]", "parameter recovery: r for kappa within the range of the fitted estimates", 0.59, 2, lambda: recovery_in_range("kappa"), "mcmc")
add("Results [50]", "parameter recovery: lowest r of the other parameters (learning rate after a win)", 0.65, 2, lambda: prs().drop("kappa").min(), "mcmc")
add("Results [50]", "parameter recovery: highest r of the other parameters (novelty bonus)", 0.92, 2, lambda: prs().drop("kappa").max(), "mcmc")
add("Results [50]", "model recovery: data sets won by the generating model (of 15)", 15, 0, lambda: sum(r["won"] for r in recovery_margins()))
add("Results [50]", "model recovery: smallest margin over the better of the other two (>= 58)", 58, 0, lambda: min(r["margin"] for r in recovery_margins()), "ge")
add("Results [50]", "model recovery: largest group-level R-hat of any fit", 1.10, 2, lambda: max(r["rhat"] for r in recovery_margins()), "mcmc")

# Response times (Results [52], [55])
rt = lambda s, k: jr(s, "17_response_times/response_times.json")[k]
for s, where, vals in ((3, "Results [52]", dict(after_win_iti_x_stay=(0.052, 0.035, 0.069), after_loss_iti_x_stay=(-0.019, -0.040, 0.001), three_way=(0.073, 0.047, 0.100))),
                       (4, "Results [55]", dict(after_win_iti_x_stay=(0.069, 0.037, 0.100), after_loss_iti_x_stay=(-0.061, -0.100, -0.021), three_way=(0.128, 0.079, 0.178)))):
    for k, v in vals.items():
        add_ci(where, f"S{s} response time, {k}", v, 3, lambda s=s, k=k: tuple(rt(s, k)[nm] for nm in ("estimate", "ci_low", "ci_high")))

# Individual differences (Results [53])
add("Results [53]", "S3 impulsivity x ITI effect, r(437)", 0.05, 2, lambda: corr(3, "bis", "iti_effect_all")["pearson_r"])
add("Results [53]", "S3 impulsivity x ITI effect, one-sided p", 0.16, 2, lambda: corr(3, "bis", "iti_effect_all")["pearson_p_one_sided"])
add("Results [53]", "S3 anhedonia x ITI effect, r(437)", 0.02, 2, lambda: corr(3, "shaps", "iti_effect_all")["pearson_r"])
add("Results [53]", "S3 anhedonia x ITI effect, one-sided p", 0.62, 2, lambda: corr(3, "shaps", "iti_effect_all")["pearson_p_one_sided"])
add("Results [53]", "S3 impulsivity x post-win ITI effect, r(437)", 0.11, 2, lambda: corr(3, "bis", "iti_effect_win")["pearson_r"])
add("Results [53]", "S3 impulsivity x post-win ITI effect, p", 0.023, 3, lambda: corr(3, "bis", "iti_effect_win")["pearson_p_two_sided"])
add("Results [53]", "S3 impulsivity x post-win ITI effect, Spearman rho(437)", 0.13, 2, lambda: corr(3, "bis", "iti_effect_win")["spearman_rho"])
add("Results [53]", "S3 impulsivity x post-win ITI effect, Spearman p", 0.008, 3, lambda: corr(3, "bis", "iti_effect_win")["spearman_p_two_sided"])
add("Results [53]", "S3 impulsivity x trial-adjusted ITI slope, r(437)", 0.08, 2, lambda: corr(3, "bis", "iti_slope_trial_adjusted")["pearson_r"], "mcmc")
add("Results [53]", "S3 impulsivity x trial-adjusted ITI slope, p", 0.11, 2, lambda: corr(3, "bis", "iti_slope_trial_adjusted")["pearson_p_two_sided"], "mcmc")
add("Results [53]; Methods [97]", "parameter-trait correlations (Studies 3 and 4): smallest FDR-corrected p", None, 3,
    lambda: float(csv(3, "09_parameter_traits/parameter_trait_correlations.csv")["p_fdr"].min()), "todo")

# Study 4 (Results [54]-[56]; Methods [84])
add("Results [54]", "S4 chose-best rate", 0.45, 2, lambda: lr(4)["chose_best_mean"])
add("Results [54]", "S4 chose-best vs 1/3, t(116)", 13.7, 1, lambda: lr(4)["t_vs_chance"])
add("Results [54]", "S4 chose-best vs 1/3, p < 10^-25", 1e-25, None, lambda: lr(4)["p_vs_chance"], "lt")
add("Results [54]", "S4 p(switch), first twenty trials, %", 63, 0, lambda: 100 * blk(4, "p_switch_by_20trial_block", 0))
add("Results [54]", "S4 p(switch), last twenty trials, %", 41, 0, lambda: 100 * blk(4, "p_switch_by_20trial_block", 4))
sl4 = lambda: dc(4)["within_participant_switch_slope_per_100_trials"]
add("Results [54]", "S4 within-participant slope of switching per 100 trials", -0.26, 2, lambda: sl4()["mean"])
add("Results [54]", "S4 slope, t(116)", -8.1, 1, lambda: sl4()["t"])
add("Results [54]", "S4 slope, p < 10^-12", 1e-12, None, lambda: sl4()["p"], "lt")
for i, v in enumerate((45, 50, 50, 50, 55)):
    add("Methods [84]; Fig. 5c", f"S4 share of 4 s ITIs, trials {20 * i + 1}-{20 * i + 20}, %", v, 0, lambda i=i: 100 * blk(4, "long_iti_share_by_20trial_block", i))
add("Results [55]", "S4 after a win, 1 s, %", 28.1, 1, lambda: 100 * pt(4, "switch_after_win")["mean_short"])
add("Results [55]", "S4 after a win, 4 s, %", 32.3, 1, lambda: 100 * pt(4, "switch_after_win")["mean_long"])
add("Results [55]", "S4 after a win, t(116)", 3.74, 2, lambda: pt(4, "switch_after_win")["t"])
add("Results [55]", "S4 after a win, p", 0.0003, 4, lambda: pt(4, "switch_after_win")["p_two_sided"])
add("Results [55]", "S4 after a loss, 1 s, %", 72.6, 1, lambda: 100 * pt(4, "switch_after_loss")["mean_short"])
add("Results [55]", "S4 after a loss, 4 s, %", 71.5, 1, lambda: 100 * pt(4, "switch_after_loss")["mean_long"])
add("Results [55]", "S4 after a loss, t(116)", -0.77, 2, lambda: pt(4, "switch_after_loss")["t"])
add("Results [55]", "S4 after a loss, p", 0.45, 2, lambda: pt(4, "switch_after_loss")["p_two_sided"])
add("Results [55]", "S4 post-win vs post-loss effect, t(116)", 2.90, 2, lambda: pt(4, "win_vs_loss_effect_difference")["t"])
add("Results [55]", "S4 post-win vs post-loss effect, p", 0.004, 3, lambda: pt(4, "win_vs_loss_effect_difference")["p_two_sided"])
for m, g, se in (("m02_rw_1lr", 2198.7, 63.7), ("m03_rw_2lr", 62.7, 17.0), ("m04_rw_2lr_decay", 343.8, 32.2), ("m06_choice_kernel", 284.2, 28.0),
                 ("m07_novelty", 201.5, 22.6), ("m08_trend_beta", 103.6, 16.5)):
    add("Results [56]", f"S4 {m}, gain over the model it builds on", g, 1, lambda m=m: gain(4, m), "mcmc")
    add("Results [56]", f"S4 {m}, SE", se, 1, lambda m=m: gain(4, m, "se_vs_parent"), "mcmc")
add("Results [56]", "S4 choice kernel minus stay bias", 172.5, 1, lambda: pw_diff(4, "m06_choice_kernel", "m05_stick")[0], "mcmc")
add("Results [56]", "S4 choice kernel minus stay bias, SE", 29.5, 1, lambda: pw_diff(4, "m06_choice_kernel", "m05_stick")[1], "mcmc")
for other, d, se in (("m08_trend_beta", 16.0, 6.5), ("m11_iti_target", 27.9, 7.1), ("m10_iti_bias", 12.6, 10.2)):
    add("Results [56]", f"S4 ITI competes with value minus {other}", d, 1, lambda other=other: pw_diff(4, "m12_iti_compete", other)[0], "mcmc")
    add("Results [56]", f"S4 ITI competes with value minus {other}, SE", se, 1, lambda other=other: pw_diff(4, "m12_iti_compete", other)[1], "mcmc")
add_ci("Results [56]", "S4 kappa", (0.087, 0.038, 0.142), 3, lambda: mu(4, "m12_iti_compete", "kappa", "unit"), ("posterior mean", "CI low", "CI high"))
add_ci("Results [56]", "S4 ITI competes with value: simulated post-win trend per quintile", (0.013, 0.003, 0.027), 3,
       lambda: (marker_ppc(4, "m12_iti_compete", "win").mean(), marker_ppc(4, "m12_iti_compete", "win").quantile(0.025),
                marker_ppc(4, "m12_iti_compete", "win").quantile(0.975)), ("mean", "2.5%", "97.5%"))
add("Results [56]", "S4 the same: positive in all 20 simulated data sets", 1, 0, lambda: float((marker_ppc(4, "m12_iti_compete", "win") > 0).all()), "mcmc")
add("Results [56]", "S4 observed post-win trend per quintile", 0.022, 3, lambda: marker(4)["data"]["win"]["slope"])
add("Results [56]", "S4 other models: largest simulated post-win trend (<= 0.001)", 0.001, 3,
    lambda: max(marker_ppc(4, m, "win").mean() for m in ("m08_trend_beta", "m10_iti_bias", "m11_iti_target")), "mcmc")
add("Results [56]", "S4 ITI competes with value: simulated post-loss trend", 0.0006, 4, lambda: marker_ppc(4, "m12_iti_compete", "loss").mean(), "mcmc")

# External dataset (Results [59])
tom = lambda: J(ROOT / "fMRI_study_generalization/results/tomov2020_iti_glmm_summary.json")
add("Results [59]", "Tomov et al. (2020): participants", 31, 0, lambda: tom()["n_participants"])
add_ci("Results [59]", "Tomov ITI slope per second", (0.130, 0.003, 0.259), 3, lambda: tuple(tom()["iti_c"][k] for k in ("estimate", "ci_low", "ci_high")))

# Language-model agents (Results [61])
A = lambda: agents()
f6 = lambda: figure_numbers()["fig6"]
add("Results [61]", "agents: number", 14, 0, lambda: len(A()))
add("Results [61]", "agents: chose-best rate", 0.56, 2, lambda: A()["chose_best"].mean())
add("Results [61]", "Study 4 people: chose-best rate", 0.45, 2, lambda: f6()["chose_best_people"]["mean"])
add("Results [61]", "agents: stay rate after a win", 0.99, 2, lambda: (A().eval("stay_after_win * (n_win_1s + n_win_4s)").sum()) / A().eval("n_win_1s + n_win_4s").sum())
add("Results [61]", "agents: switches after a win", 6, 0, lambda: A().eval("switch_win_1s * n_win_1s + switch_win_4s * n_win_4s").sum())
add("Results [61]", "agents: post-win trials", 891, 0, lambda: A().eval("n_win_1s + n_win_4s").sum())
add("Results [61]", "agents: stay rate after a loss", 0.60, 2, lambda: A()["stay_after_loss"].mean())
add("Results [61]", "agents: overall switch rate", 0.15, 2, lambda: A()["p_switch"].mean())
add("Results [61]", "agents: highest switch rate of an agent", 0.24, 2, lambda: A()["p_switch"].max())
add("Results [61]", "agents: reasoning summaries returned", 148, 0, lambda: f6()["reasoning_summaries"])
add("Results [61]", "agents: reasoning summaries that mention the wait", 0, 0, lambda: f6()["reasoning_mentioning_the_wait"])

# Supplementary Note 2 / Supplementary Fig. 3
sn2 = lambda: figure_numbers()["supp_fig3"]
add("Supplementary Note 2", "correlation of E with trial number, rho = 0.02", -0.83, 2, lambda: sn2()["correlation_with_trial"]["0.02"])
add("Supplementary Note 2", "correlation of E with trial number, rho = 0.2", -0.68, 2, lambda: sn2()["correlation_with_trial"]["0.2"])
add("Supplementary Note 2", "rho = 0.05: mean E over the first twenty trials", 0.15, 2, lambda: sn2()["mean_first20_last20"]["0.05"][0])
add("Supplementary Note 2", "rho = 0.05: mean E over the last twenty trials", -0.19, 2, lambda: sn2()["mean_first20_last20"]["0.05"][1])
add("Supplementary Note 2", "correlation at rho = 0.5", -0.42, 2, lambda: sn2()["correlation_with_trial"]["0.5"])
add("Supplementary Note 2", "correlation at rho = 1", -0.24, 2, lambda: sn2()["correlation_with_trial"]["1"])

# Supplementary Table 1: every model of the sequence, Studies 3 and 4 (study, model, ELPD, SE, gain, SE, vs best, SE, R-hat, ESS, div.)
SUPP_TABLE1 = [
    (3, 'm01_intercepts', -42143.0, 92.5, None, None, -11747.4, 132.3, 1.0, 753, 0),
    (3, 'm02_rw_1lr', -34624.7, 130.8, 7518.3, 121.1, -4229.2, 91.9, 1.01, 425, 0),
    (3, 'm03_rw_2lr', -34279.1, 130.4, 345.6, 34.8, -3883.6, 88.1, 1.04, 162, 1),
    (3, 'm04_rw_2lr_decay', -32704.5, 136.1, 1574.5, 65.7, -2309.0, 68.7, 1.07, 47, 0),
    (3, 'm05_stick', -32070.4, 137.9, 634.1, 33.8, -1674.9, 68.2, 1.01, 247, 0),
    (3, 'm06_choice_kernel', -31987.2, 137.0, 717.3, 42.7, -1591.7, 58.6, 1.05, 74, 0),
    (3, 'm07_novelty', -31073.5, 141.0, 913.7, 44.4, -678.0, 38.1, 1.03, 210, 0),
    (3, 'm08_trend_beta', -30513.7, 140.1, 559.8, 35.7, -118.2, 16.8, 1.04, 94, 0),
    (3, 'm09_trend_bias', -30958.8, 140.6, 114.7, 15.7, -563.3, 38.9, 1.08, 33, 0),
    (3, 'm10_iti_bias', -30506.4, 140.1, 7.3, 5.3, -110.9, 16.5, 1.08, 35, 0),
    (3, 'm11_iti_target', -30499.6, 140.2, 14.1, 8.7, -104.1, 16.9, 1.05, 90, 0),
    (3, 'm12_iti_compete', -30395.5, 140.5, 118.2, 16.8, 0.0, 0.0, 1.05, 82, 0),
    (4, 'm01_intercepts', -11095.3, 50.2, None, None, -3210.5, 69.0, 1.01, 265, 0),
    (4, 'm02_rw_1lr', -8896.6, 68.8, 2198.7, 63.7, -1011.8, 46.5, 1.04, 126, 0),
    (4, 'm03_rw_2lr', -8833.9, 67.6, 62.7, 17.0, -949.1, 45.2, 1.01, 537, 0),
    (4, 'm04_rw_2lr_decay', -8490.1, 70.7, 343.8, 32.2, -605.3, 36.5, 1.01, 385, 1),
    (4, 'm05_stick', -8378.4, 71.2, 111.7, 15.1, -493.6, 37.1, 1.02, 327, 3),
    (4, 'm06_choice_kernel', -8205.9, 71.7, 284.2, 28.0, -321.1, 29.1, 1.01, 337, 0),
    (4, 'm07_novelty', -8004.4, 73.9, 201.5, 22.6, -119.6, 17.4, 1.03, 280, 0),
    (4, 'm08_trend_beta', -7900.8, 73.3, 103.6, 16.5, -16.0, 6.5, 1.19, 15, 0),
    (4, 'm09_trend_bias', -7973.5, 73.9, 31.0, 8.1, -88.6, 18.6, 1.08, 35, 0),
    (4, 'm10_iti_bias', -7897.4, 73.4, 3.4, 7.9, -12.6, 10.2, 1.53, 7, 0),
    (4, 'm11_iti_target', -7912.7, 73.4, -11.9, 3.6, -27.9, 7.1, 1.4, 8, 0),
    (4, 'm12_iti_compete', -7884.8, 73.4, 16.0, 6.5, 0.0, 0.0, 1.05, 45, 0),
]
for s, m, elpd, se, g, sg, vb, svb, rh, ess, dv in SUPP_TABLE1:
    for nm, val, col in (("ELPD", elpd, "elpd_loo"), ("SE", se, "se_elpd_loo"), ("gain", g, "elpd_minus_parent"), ("SE of gain", sg, "se_vs_parent"),
                         ("vs best", vb, "elpd_minus_best"), ("SE vs best", svb, "se_vs_best")):
        if val is not None:
            add("Supplementary Table 1", f"S{s} {m}: {nm}", val, 1, lambda s=s, m=m, col=col: comp(s).loc[m, col], "mcmc")
    add("Supplementary Table 1", f"S{s} {m}: R-hat", rh, 2, lambda s=s, m=m: comp(s).loc[m, "rhat_group_max"], "mcmc")


# ---- evaluate -----------------------------------------------------------------------------------------------------------
def fmt(v, dec):
    if isinstance(v, tuple):
        return "b = " + f"{v[0]:+.{dec}f}" + ", CI = [" + ", ".join(f"{x:+.{dec}f}" for x in v[1:]) + "]"
    if dec is None:
        return f"{v:.2g}"
    return f"{v:.{dec}f}"


def verdict(c, v):
    p, dec, kind = c["paper"], c["dec"], c["kind"]
    if kind == "todo":
        return "(to add)"
    if kind == "lt":
        return "match" if v < p else "DIFFERS"
    if kind == "gt":
        return "match" if v > p else "DIFFERS"
    if kind == "le":
        return "match" if round(v, dec) <= p else "DIFFERS"
    if kind == "ge":
        return "match" if round(v, dec) >= p else "DIFFERS"
    if round(v, dec) == round(p, dec) or abs(v - p) < 0.5 * 10 ** -dec + 1e-12:
        return "match"
    if kind == "mcmc" and abs(v - p) <= max(3 * 10 ** -dec, 0.02 * abs(p), c["tol"]):
        return "close"
    return "DIFFERS"


rows = []
for c in C:
    try:
        v = c["get"]()
        v = tuple(float(x) for x in v) if isinstance(v, tuple) else float(v)
        rows.append(dict(where=c["where"], what=c["what"], paper="" if c["paper"] is None else (f"< {c['paper']:g}" if c["kind"] == "lt" else
                                                                                                f"> {c['paper']:g}" if c["kind"] == "gt" else fmt(c["paper"], c["dec"])),
                         reproduced=fmt(v, c["dec"]), verdict=verdict(c, v)))
    except Exception as e:                                       # a missing output (step not run or failed)
        rows.append(dict(where=c["where"], what=c["what"], paper="" if c["paper"] is None else str(c["paper"]), reproduced=f"missing ({type(e).__name__}: {e})",
                         verdict="MISSING"))
R = pd.DataFrame(rows)
counts = R["verdict"].value_counts().to_dict()
lines = ["# Reproduction report", "",
         "Every number reported in the main text and the Supplementary Information, compared with the results regenerated by "
         "`reproduce.sh` (written by `check_reported_numbers.py`; the verdicts are defined in its docstring). References in brackets "
         "are paragraphs of the manuscript.", "",
         "Summary: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f" (of {len(R)} numbers).", ""]
st = ROOT / "logs/status.tsv"
if st.exists():
    S = pd.read_csv(st, sep="\t", names=["step", "status", "seconds"])
    lines += [f"Steps of this run: {int((S.status == 'ok').sum())} ok, {int((S.status != 'ok').sum())} failed; computing time per step in logs/status.tsv.", ""]
bad = R[R.verdict.isin(["DIFFERS", "MISSING"])]
if len(bad):
    lines += ["## Numbers that differ from the paper", "", "| where | number | paper | reproduced |", "|---|---|---|---|"]
    lines += [f"| {r.where} | {r.what} | {r.paper} | {r.reproduced} |" for r in bad.itertuples()]
    lines.append("")
todo = R[R.verdict == "(to add)"]
if len(todo):
    lines += ["## Placeholders in the paper, with the regenerated values", "", "| where | number | reproduced |", "|---|---|---|"]
    lines += [f"| {r.where} | {r.what} | {r.reproduced} |" for r in todo.itertuples()]
    lines.append("")
lines += ["## All numbers", "", "| where | number | paper | reproduced | verdict |", "|---|---|---|---|---|"]
lines += [f"| {r.where} | {r.what} | {r.paper} | {r.reproduced} | {r.verdict} |" for r in R.itertuples()]
(ROOT / "reproduction_report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:8]))
if len(bad):
    print(bad[["where", "what", "paper", "reproduced"]].to_string(index=False))
print("wrote", ROOT / "reproduction_report.md")
