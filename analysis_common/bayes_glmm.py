"""Bayesian mixed-effects regressions in PyMC, used for every mixed model of the paper.

These generalized linear mixed models (GLMMs) test behavioural effects, such as more switching
after a long inter-trial interval (ITI), with participants as the grouping factor.

fit_bglmm(df, formula, family="bernoulli" | "gaussian") takes an lme4-style formula with one
grouping factor, e.g.
    switch ~ prev_iti + trial_c + (1 + prev_iti + trial_c | subject_id)
and fits
    y_ij ~ Bernoulli(logit^-1(eta_ij))   or   Normal(eta_ij, sigma_y)
    eta_ij = X_ij . beta + Z_ij . u_j,     u_jk ~ N(0, tau_k^2)
where i indexes observations (trials) and j groups (participants), X holds the fixed-effect
predictors and Z the random-effect predictors, beta are the fixed effects, u_j participant j's
random deviations and tau their standard deviations.

Priors: beta_intercept ~ N(0, 1.5), other beta ~ N(0, 1) (Bernoulli; for Gaussian, N(mean(y), 2 sd(y))
and N(0, sd(y))), tau ~ HalfNormal(1) (Gaussian: HalfNormal(sd(y))), sigma_y ~ HalfNormal(sd(y));
non-centred u_j (u_j = tau * z_j with z_j ~ N(0, 1)). Random effects are independent: the random ITI
slope varies little between participants, so its correlations with the other random effects are not
identified (Methods). Convergence is judged on beta and tau. Sampled by the No-U-Turn sampler (NUTS)
in nutpie, 4 chains x 1000 draws after 1000 warm-up, target acceptance 0.9, as for the computational
models.

Reported per fixed effect: posterior mean, sd, 95% credible interval, the posterior probability that
it is positive, p_direction = 2 * min(P(beta > 0), P(beta < 0)) (two-sided posterior tail
probability), R-hat (convergence diagnostic) and bulk effective sample size (ESS). Subject-level
coefficients (beta_k + u_jk, posterior means) are returned for the random terms.

Other entry points: summary_row() and contrast_row() turn a fit into the standard JSON-ready record of
one effect (with odds ratios); RepeatedBGLMM fits the same model to many data sets of the same shape
(posterior predictive checks). With out_prefix, fit_bglmm writes <prefix>_fixef.csv,
<prefix>_subject.csv and <prefix>.json.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _parse(formula: str):
    """Split an lme4-style formula into (outcome, fixed-effects part, random-effects part, grouping factor).

    Exactly one random term "(... | group)" is supported; it is removed from the right-hand side,
    and what is left (or "1" if nothing is) is the fixed-effects formula for patsy.
    """
    lhs, rhs = [x.strip() for x in formula.split("~", 1)]
    m = re.search(r"\(([^|()]+)\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", rhs)
    if not m:
        raise ValueError("formula needs one random term (... | group)")
    rand, group = m.group(1).strip(), m.group(2)
    fixed = (rhs[:m.start()] + rhs[m.end():]).strip().strip("+").strip()
    fixed = re.sub(r"\+\s*\+", "+", fixed) or "1"
    return lhs, fixed, rand, group


def fit_bglmm(df: pd.DataFrame, formula: str, family: str = "bernoulli", draws: int = 1000, tune: int = 1000, chains: int = 4,
              seed: int = 1, target_accept: float = 0.9, out_prefix: Path | None = None):
    """Fit one mixed model by NUTS and summarise it (model, priors and outputs: see the module docstring).

    Returns a dict with the fixed-effect table (fixef), subject-level coefficients (subject), sample
    sizes, convergence measures (rhat_max, ess_min, n_divergences) and run time, plus the raw draws
    used by the other functions here: _B (draws x fixed effects), _U (draws x groups x random terms)
    and _tau (draws x random-effect SDs).
    """
    import patsy
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    from .pymc_models import sample_model
    # Design matrices: X for the fixed effects, Z for the random effects (rows without missing values)
    lhs, fixed, rand, group = _parse(formula)
    d = df.dropna(subset=[lhs, group]).copy()
    X = patsy.dmatrix(fixed, d, return_type="dataframe")
    Z = patsy.dmatrix(rand, d, return_type="dataframe")
    keep = X.index.intersection(Z.index)
    d, X, Z = d.loc[keep], X.loc[keep], Z.loc[keep]
    y = d[lhs].to_numpy(float)
    gcodes, glevels = pd.factorize(d[group])          # group index of each row, and the group labels
    J, p, q = len(glevels), X.shape[1], Z.shape[1]    # number of groups, fixed effects, random effects
    xn, zn = list(X.columns), list(Z.columns)
    gauss = family == "gaussian"
    ysd = float(np.std(y)) if gauss else 1.0          # prior scale: sd(y) if Gaussian, 1 on the logit scale
    # The model: priors on beta, independent random effects, and the likelihood
    with pm.Model() as m:
        if gauss:
            b_sd = np.array([2 * ysd if c == "Intercept" else ysd for c in xn]); b_mu = np.array([float(np.mean(y)) if c == "Intercept" else 0.0 for c in xn])
        else:
            b_sd = np.array([1.5 if c == "Intercept" else 1.0 for c in xn]); b_mu = np.zeros(p)
        beta = pm.Normal("beta", b_mu, b_sd, shape=p)
        tau = pm.HalfNormal("tau", ysd, shape=q)
        zr = pm.Normal("z", 0.0, 1.0, shape=(J, q))
        u = zr * tau
        pm.Deterministic("u", u)
        # Linear predictor: X . beta + each row's random-effect predictors times its group's deviations
        eta = pt.dot(X.to_numpy(float), beta) + (Z.to_numpy(float) * u[gcodes]).sum(axis=1)
        if gauss:
            sig = pm.HalfNormal("sigma_y", ysd)
            pm.Normal("y", eta, sig, observed=y)
        else:
            pm.Bernoulli("y", logit_p=eta, observed=y)
    # Sample, then summarise each fixed effect from its pooled draws (all chains together)
    t0 = time.time()
    idata, info = sample_model(m, draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept, verbose=False)
    post = idata.posterior
    B = post["beta"].to_numpy().reshape(-1, p)
    summ = az.summary(idata, var_names=["beta"], kind="diagnostics")
    rows = []
    for k, c in enumerate(xn):
        s = B[:, k]; pp = float((s > 0).mean())
        rows.append(dict(term=c, mean=float(s.mean()), sd=float(s.std()), ci_low=float(np.quantile(s, 0.025)), ci_high=float(np.quantile(s, 0.975)),
                         p_positive=pp, p_direction=float(2 * min(pp, 1 - pp)), rhat=float(summ["r_hat"].iloc[k]), ess_bulk=float(summ["ess_bulk"].iloc[k])))
    fixef = pd.DataFrame(rows)
    # Subject-level coefficients: fixed effect + the subject's deviation, averaged over draws
    # (a random term without a matching fixed effect gets the deviation alone)
    U = post["u"].to_numpy().reshape(-1, J, q)
    subj = pd.DataFrame({group: glevels})
    for k, c in enumerate(zn):
        bk = B[:, xn.index(c)] if c in xn else 0.0
        subj[f"{c}"] = (U[:, :, k] + (bk[:, None] if np.ndim(bk) else 0.0)).mean(axis=0)
    # Convergence: worst R-hat and smallest ESS over the fixed effects and random-effect SDs
    # (and sigma_y if Gaussian)
    rh = az.rhat(idata); es = az.ess(idata)
    gate = [v for v in rh.data_vars if v in ("beta", "tau", "sigma_y")]
    rhat_max = float(max(float(np.nanmax(rh[v].to_numpy())) for v in gate))
    ess_min = float(min(float(np.nanmin(es[v].to_numpy())) for v in gate))
    res = dict(formula=formula, family=family, n_obs=int(len(y)), n_groups=int(J), fixef=fixef, subject=subj, fixed_terms=xn, random_terms=zn,
               rhat_max=rhat_max, ess_min=ess_min, n_divergences=int(info.get("n_divergences", -1)) if isinstance(info, dict) else -1,
               seconds=float(time.time() - t0))
    # sample_model's info does not report divergences, so count them from the sampler statistics
    if isinstance(info, dict) and "n_divergences" not in info:
        try:
            res["n_divergences"] = int(np.asarray(idata.sample_stats["diverging"]).sum())
        except Exception:
            pass
    # Optional files: <prefix>_fixef.csv, <prefix>_subject.csv and <prefix>.json (all but the draws)
    if out_prefix is not None:
        out_prefix = Path(out_prefix)
        fixef.to_csv(f"{out_prefix}_fixef.csv", index=False); subj.to_csv(f"{out_prefix}_subject.csv", index=False)
        json.dump({k: v for k, v in res.items() if k not in ("fixef", "subject")} | dict(fixef=fixef.to_dict(orient="records")),
                  open(f"{out_prefix}.json", "w"), indent=1)
    res["_B"] = B
    res["_U"] = U                                            # (draws, groups, random terms), for per-group posteriors
    res["_tau"] = post["tau"].to_numpy().reshape(-1, q)      # random-effect SDs
    return res


def summary_row(res, term: str, scale: float = 1.0) -> dict:
    """The standard record of one fixed effect, used by every script (JSON-serialisable). `scale` multiplies the coefficient for the
    odds ratio (e.g. scale = 3 for prev_iti in seconds gives the odds ratio for 4 s against 1 s).

    p_direction, the two-sided posterior tail probability 2 * min(P(beta > 0), P(beta < 0)), is
    floored at 1 / (number of draws), since a smaller tail probability cannot be resolved from the
    draws. odds_ratio is the posterior mean of exp(scale * beta); or_median and the odds-ratio
    credible interval are exp(scale * ...) of the median and 2.5% / 97.5% quantiles of beta."""
    B = res["_B"]; k = res["fixed_terms"].index(term); s = B[:, k]
    pp = float((s > 0).mean())
    out = dict(engine="pymc", formula=res["formula"], term=term, n_obs=res["n_obs"], n_groups=res["n_groups"],
               estimate=float(s.mean()), sd=float(s.std()), ci_low=float(np.quantile(s, 0.025)), ci_high=float(np.quantile(s, 0.975)),
               p_positive=pp, p_direction=float(max(2 * min(pp, 1 - pp), 1.0 / len(s))), or_scale=scale,
               odds_ratio=float(np.exp(scale * s).mean()), or_median=float(np.exp(scale * np.median(s))),
               or_ci_low=float(np.exp(scale * np.quantile(s, 0.025))), or_ci_high=float(np.exp(scale * np.quantile(s, 0.975))),
               rhat_max=res["rhat_max"], ess_min=res["ess_min"], n_divergences=res["n_divergences"])
    return out


def contrast_row(res, weights: dict, label: str, scale: float = 1.0) -> dict:
    """A linear combination of fixed effects (e.g. the prev_iti slope after wins = prev_iti + prev_iti:prev_win), same record.

    weights maps term names to their weights, e.g. {"prev_iti": 1, "prev_iti:prev_win": 1};
    label names the result.
    """
    B = res["_B"]; s = sum(w * B[:, res["fixed_terms"].index(t)] for t, w in weights.items())
    pp = float((s > 0).mean())
    return dict(engine="pymc", formula=res["formula"], term=label, weights=weights, n_obs=res["n_obs"], n_groups=res["n_groups"],
                estimate=float(s.mean()), sd=float(s.std()), ci_low=float(np.quantile(s, 0.025)), ci_high=float(np.quantile(s, 0.975)),
                p_positive=pp, p_direction=float(max(2 * min(pp, 1 - pp), 1.0 / len(s))), or_scale=scale,
                odds_ratio=float(np.exp(scale * s).mean()), or_median=float(np.exp(scale * np.median(s))),
                or_ci_low=float(np.exp(scale * np.quantile(s, 0.025))), or_ci_high=float(np.exp(scale * np.quantile(s, 0.975))),
                rhat_max=res["rhat_max"], ess_min=res["ess_min"], n_divergences=res["n_divergences"])


class RepeatedBGLMM:
    """The same Bernoulli mixed model fitted to many data sets of identical shape (posterior predictive checks: the real data and each
    simulated replicate), compiled once (nutpie with_data). Same priors and independent random effects as fit_bglmm; by default fewer
    draws (2 chains x 500 after 500 warm-up), since only the posterior mean of a fixed effect is
    used per replicate.

    Usage: build once with a template data set, then call fit(df) for each data set. Each df must
    give design matrices of the template's shape and contain only groups present in the template."""

    def __init__(self, formula: str, template: pd.DataFrame, draws: int = 500, tune: int = 500, chains: int = 2, target_accept: float = 0.9):
        """Build the model on `template` with the data in pm.Data containers; compile it once with nutpie."""
        import patsy
        import pymc as pm
        import pytensor.tensor as pt
        import nutpie
        self.formula, self.draws, self.tune, self.chains, self.ta = formula, draws, tune, chains, target_accept
        self.lhs, fixed, rand, self.group = _parse(formula)
        # Keep the patsy design information so later data sets are coded exactly like the template
        X = patsy.dmatrix(fixed, template, return_type="dataframe"); Z = patsy.dmatrix(rand, template, return_type="dataframe")
        self.xinfo, self.zinfo = X.design_info, Z.design_info
        self.xn, self.zn = list(X.columns), list(Z.columns)
        self.n, self.p, self.q = len(X), X.shape[1], Z.shape[1]
        self.levels = pd.Index(pd.unique(template[self.group]))
        J = len(self.levels)
        # Same Bernoulli model and priors as fit_bglmm (independent random effects); the data are
        # swappable containers
        with pm.Model() as m:
            Xd = pm.Data("X", X.to_numpy(float)); Zd = pm.Data("Z", Z.to_numpy(float))
            gd = pm.Data("g", self.levels.get_indexer(template[self.group]).astype("int64")); yd = pm.Data("yobs", template[self.lhs].to_numpy(float))
            beta = pm.Normal("beta", 0.0, np.array([1.5 if c == "Intercept" else 1.0 for c in self.xn]), shape=self.p)
            tau = pm.HalfNormal("tau", 1.0, shape=self.q); zr = pm.Normal("z", 0.0, 1.0, shape=(J, self.q))
            u = zr * tau
            pm.Bernoulli("y", logit_p=pt.dot(Xd, beta) + (Zd * u[gd]).sum(axis=1), observed=yd)
        self.compiled = nutpie.compile_pymc_model(m, backend="numba")

    def fit(self, df: pd.DataFrame, seed: int = 1) -> dict:
        """Sample the compiled model on a new data set.

        Returns the fixed-effect terms, their posterior means and the draws (_B).
        """
        import patsy
        import nutpie
        X = np.asarray(patsy.build_design_matrices([self.xinfo], df)[0], float)
        Z = np.asarray(patsy.build_design_matrices([self.zinfo], df)[0], float)
        if X.shape != (self.n, self.p) or not set(pd.unique(df[self.group])) <= set(self.levels):
            raise ValueError("data set does not match the compiled design")
        c = self.compiled.with_data(X=X, Z=Z, g=self.levels.get_indexer(df[self.group]).astype("int64"), yobs=df[self.lhs].to_numpy(float))
        idata = nutpie.sample(c, draws=self.draws, tune=self.tune, chains=self.chains, seed=seed, target_accept=self.ta, progress_bar=False)
        B = idata.posterior["beta"].to_numpy().reshape(-1, self.p)
        return dict(terms=self.xn, mean={t: float(B[:, k].mean()) for k, t in enumerate(self.xn)}, _B=B)
