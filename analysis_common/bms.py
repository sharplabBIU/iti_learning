"""Random-effects Bayesian model selection (BMS; Stephan et al. 2009; Rigoux et al. 2014).

Random-effects BMS asks which computational model is most common in the population, allowing
different participants to be best described by different models. Each participant's model is treated
as drawn from a multinomial distribution with unknown model frequencies r, which have a
Dirichlet(alpha) distribution. A variational Bayes scheme alternates between (a) each participant's
posterior probability of each model and (b) the Dirichlet parameters alpha, until the free energy (a
lower bound on the log evidence of the whole group) stops changing.

Ported from study_3/task/analysis_noa/bms_utils.py (self-contained, no groupBMC).
Input: log-evidence matrix (N subjects x K models). Output: Dirichlet alpha, expected
frequencies, exceedance probabilities, protected exceedance probabilities (PXP),
Bayesian omnibus risk (BOR) and per-subject posterior model probabilities.

Definitions
  * expected frequency = posterior mean of r (the share of participants best described by a model).
  * exceedance probability = posterior probability that a model is more frequent than all others.
  * BOR = posterior probability that all models are equally frequent (the "null" in which
    differences in frequency are due to chance).
  * PXP = exceedance probability corrected for that null: PXP = XP (1 - BOR) + BOR / K.

The study scripts call bms_table(); nothing is read from or written to disk here.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, log

import numpy as np
from scipy import integrate
from scipy.special import digamma, gammaln, gammainc, softmax
from scipy.stats import dirichlet

_EPS = np.finfo(float).eps   # added inside logarithms so a probability of 0 does not give log(0)


@dataclass
class BMSResult:
    """All outputs of run_random_effects_bms(); arrays have one entry per model unless noted."""
    expected_frequency: np.ndarray
    exceedance_probability: np.ndarray
    protected_exceedance: np.ndarray
    dirichlet_alpha: np.ndarray
    subject_posterior: np.ndarray   # N x K: each participant's posterior probability of each model
    bor: float                      # Bayesian omnibus risk (one number for the model set)
    frequency_var: np.ndarray       # posterior variance of each model frequency


def _dirichlet_exceedance(alpha: np.ndarray) -> np.ndarray:
    """Exceedance probabilities of a Dirichlet(alpha): P(r_i > r_j for all j != i), for each model i.

    With two models this is a Beta tail probability. With more, it uses the gamma representation of
    the Dirichlet (r_i = g_i / sum g, g_i ~ Gamma(alpha_i)): P(model i largest) = integral over x of
    the density of g_i at x times the product over j != i of P(g_j < x), computed by numerical
    integration and then normalised to sum to 1.
    """
    alpha = np.asarray(alpha, dtype=float).flatten()
    k = len(alpha)
    if k == 2:
        from scipy.stats import beta as beta_dist
        p1 = 1.0 - beta_dist.cdf(0.5, alpha[0], alpha[1])
        return np.array([p1, 1.0 - p1])
    gterms = [gammaln(a) for a in alpha]

    def integrand(x: float, i: int) -> float:
        # product of Gamma CDFs (regularised lower incomplete gamma) times the Gamma(alpha_i) density
        prod = 1.0
        for j in range(k):
            if j != i:
                prod *= gammainc(alpha[j], x)
        return prod * exp((alpha[i] - 1.0) * log(x) - x - gterms[i])

    xp = np.array([integrate.quad(lambda x, i=i: integrand(x, i), 0.0, np.inf, limit=200)[0] for i in range(k)])
    return xp / xp.sum()


def _free_energy_null(lme, alpha0):
    """Free energy (log evidence) of the null hypothesis that all models are equally frequent.

    lme is the K x N log-evidence matrix; alpha0 the prior Dirichlet parameters, whose logarithms
    enter as log prior model probabilities (1/K each with the default total of 1).
    """
    w = softmax(lme, axis=0)
    a0 = alpha0.flatten()[:, None]
    return float((w * (lme + np.log(a0 + _EPS) - np.log(w + _EPS))).sum())


def _free_energy_alt(lme, alpha, z, alpha0):
    """Variational free energy of the random-effects model (model frequencies differ).

    lme: K x N log evidences; alpha: posterior Dirichlet parameters; z: K x N posterior model
    probabilities per participant; alpha0: prior Dirichlet parameters. Free energy = expected log
    joint + entropy of z + entropy of the Dirichlet posterior.
    """
    alpha = alpha.flatten(); alpha0 = alpha0.flatten()
    e_log_r = digamma(alpha) - digamma(alpha.sum())   # E[log r_k] under Dirichlet(alpha)
    e_log_joint = (z * (lme + e_log_r[:, None])).sum()
    e_log_joint += ((alpha0 - 1.0) * e_log_r).sum()
    e_log_joint += gammaln(alpha0.sum()) - gammaln(alpha0).sum()
    entropy_z = -(z * np.log(z + _EPS)).sum()
    entropy_a = gammaln(alpha).sum() - gammaln(alpha.sum()) - ((alpha - 1.0) * e_log_r).sum()
    return float(e_log_joint + entropy_z + entropy_a)


def run_random_effects_bms(log_evidence: np.ndarray, *, models_axis: int = 1, alpha0: float = 1.0,
                           max_iter: int = 256, tol: float = 1e-8) -> BMSResult:
    """Run random-effects BMS on a log-evidence matrix.

    log_evidence: N subjects x K models by default (models_axis=1); models_axis=0 for K x N. alpha0
    is the total prior Dirichlet count, spread evenly over the K models. Iterates at most max_iter
    times, stopping early once the free energy changes by less than tol.
    """
    lme = np.asarray(log_evidence, dtype=float)
    if models_axis == 1:
        lme = lme.T                       # -> K x N
    k, n = lme.shape
    if np.any(~np.isfinite(lme)):
        raise ValueError("log_evidence contains NaN or inf")
    # Start from the prior: alpha = alpha0 / K per model, equal model probabilities for everyone
    alpha0_vec = np.full(k, float(alpha0) / k)
    alpha = alpha0_vec.copy()
    z = np.tile(alpha0_vec / alpha0_vec.sum(), (n, 1)).T
    f_hist = []
    # Variational updates: model probabilities z given alpha, then alpha = prior + summed z
    for _ in range(max_iter):
        log_z = lme + digamma(alpha)[:, None] - digamma(alpha.sum())
        z = softmax(log_z, axis=0)
        alpha = alpha0_vec + z.sum(axis=1)
        f_hist.append(_free_energy_alt(lme, alpha, z, alpha0_vec))
        if len(f_hist) > 1 and abs(f_hist[-1] - f_hist[-2]) < tol:
            break
    # BOR compares the evidence of the null (equal frequencies, f0) with that of the full model (f1)
    f1 = f_hist[-1]; f0 = _free_energy_null(lme, alpha0_vec)
    bor = 1.0 / (1.0 + exp(f1 - f0))
    xp = _dirichlet_exceedance(alpha)
    pxp = xp * (1.0 - bor) + bor / k
    return BMSResult(expected_frequency=dirichlet.mean(alpha), exceedance_probability=xp,
                     protected_exceedance=pxp, dirichlet_alpha=alpha, subject_posterior=z.T,
                     bor=float(bor), frequency_var=dirichlet.var(alpha))


def bms_table(log_evidence: np.ndarray, model_names: list[str]):
    """DataFrame with one row per model (sorted by PXP) + the raw result.

    log_evidence is N subjects x K models, with columns in the order of model_names. n_subjects_best
    counts the participants whose highest posterior model probability is for that model; bms_rank is
    1 for the highest PXP.
    """
    import pandas as pd
    res = run_random_effects_bms(log_evidence, models_axis=1)
    df = pd.DataFrame({
        "model": model_names,
        "dirichlet_alpha": res.dirichlet_alpha,
        "expected_frequency": res.expected_frequency,
        "exceedance_probability": res.exceedance_probability,
        "protected_exceedance": res.protected_exceedance,
        "n_subjects_best": np.bincount(res.subject_posterior.argmax(axis=1), minlength=len(model_names)),
        "bor": res.bor,
    }).sort_values("protected_exceedance", ascending=False).reset_index(drop=True)
    df["bms_rank"] = np.arange(1, len(df) + 1)
    return df, res
