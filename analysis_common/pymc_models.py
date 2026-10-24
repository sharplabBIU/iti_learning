"""Hierarchical Bayesian fits of the reinforcement-learning models (PyMC, NUTS) and PSIS-LOO comparison.

Overview
    The trial-by-trial likelihoods of analysis_common/likelihoods.py, written in PyTensor, fitted as
    hierarchical Bayesian models with the No-U-Turn sampler (NUTS, run by nutpie) and compared by
    Pareto-smoothed importance-sampling leave-one-out cross-validation (PSIS-LOO). The model
    definitions, the coding conventions (outcomes +1 / -1, ITI = inter-trial interval coded per
    model) and the model sequence are those of likelihoods.py. Callers such as
    analysis_common/study_scripts/06_fit_models_nuts.py use the functions in this order.

Main steps
    1. pad_subjects: stack all participants into (trial, participant) arrays, padded to a common
       length with a mask.
    2. make_step and _scan_loglik: the trial-by-trial likelihood is written once (make_step) and
       evaluated for all participants at once in one pytensor.scan over trials. check_against_numpy
       verifies it against the NumPy engine of likelihoods.py.
    3. build_model: the PyMC model, with the priors below and the summed log-likelihood as a
       Potential.
    4. sample_model: compile and sample with nutpie.
    5. postprocess: PSIS-LOO (trial level, computed in blocks, and subject level), convergence
       diagnostics (R-hat, effective sample size (ESS), divergences) and parameter summaries.
    6. save_pymc_result: write the results to disk.

Priors (non-centred; independent per parameter; z_i is participant i's standardised deviation)
  unit-interval parameters (learning rates, decay):  theta_i = invlogit(mu + sigma z_i),  mu ~ N(0, 1.5)
  positive (beta, beta_c):                           theta_i = exp(mu + sigma z_i),       mu ~ N(0.5, 1)
  real (intercepts, stick, slopes, cost, gamma):     theta_i = mu + sigma z_i,            mu ~ N(0, 1)
  sigma ~ HalfNormal(1);  z_i ~ N(0, 1)
  The support of each parameter comes from likelihoods.SUPPORT.

Model comparison
    Trial-level PSIS-LOO excludes each participant's first trial, whose probability is 1/3 under every
    model with learning. The per-participant summed expected log predictive density (elpd) feeds
    random-effects Bayesian model selection (BMS). A subject-level PSIS-LOO (one observation per
    participant) is computed as well.

ITI forms in the PyTensor step
    make_step has a branch for each ITI form of likelihoods.ITI_FORMS: "bias" (m10), "target" (m11)
    and "compete" (m12); check_against_numpy compares the two engines.

Outputs (save_pymc_result, in out_dir)
    {prefix}_subjects.csv, {prefix}_group.csv, {prefix}_pointwise.npz,
    {prefix}_group_posterior_draws.csv.gz and {prefix}_summary.json.
"""
from __future__ import annotations
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .likelihoods import PARAMS, SUPPORT, INIT, Q_INIT, DECAY_TARGET, parse, iti_code, iti_code_long_only, reward_code, scored_mask, beta_trend_covariate

# Silence DeprecationWarning and FutureWarning messages.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def param_support(model: str, name: str) -> str:
    """Support of a parameter ('unit', 'pos' or 'real'; likelihoods.SUPPORT). `model` is not used."""
    return SUPPORT[name]


def pad_subjects(subjects: list[dict]) -> dict:
    """Stack participants into (T, N) arrays (trial x participant), padded to the longest session.

    Each subject dict has choices (0-indexed), rewards (+1 win / -1 loss), itis (seconds, the ITI that
    followed each trial) and subject_id, and optionally trial (the trial numbers of the responded
    choices). Returns C (choices), R (outcomes +1 / -1), W (centred ITI code), WS (ITIs in seconds),
    M (1 = real trial, 0 = padding), SC (1 = scored choice), TB (trend covariate), N (participants),
    T (longest session) and subject_ids."""
    N = len(subjects); T = max(len(s["choices"]) for s in subjects)
    C = np.zeros((T, N), dtype="int64"); R = np.zeros((T, N)); W = np.zeros((T, N)); M = np.zeros((T, N)); SC = np.zeros((T, N))
    WS = np.zeros((T, N))                                       # the ITIs in seconds; coded per model in _scan_loglik
    TB = np.zeros((T, N))                                       # trend covariate from each participant's actual trial numbers
    for i, s in enumerate(subjects):
        n = len(s["choices"])
        # r = +1 / -1; w = -1 after 1 s, +1 after 4 s (centred).
        C[:n, i] = s["choices"]; R[:n, i] = reward_code(s["rewards"]); W[:n, i] = iti_code(s["itis"]); M[:n, i] = 1.0
        WS[:n, i] = np.asarray(s["itis"], dtype=float)
        SC[:n, i] = scored_mask(s["trial"]) if "trial" in s else 1.0     # not a choice made right after a missed trial
        TB[:n, i] = beta_trend_covariate(s["trial"] if "trial" in s else None, n)
    return dict(C=C, R=R, W=W, WS=WS, M=M, SC=SC, TB=TB, N=N, T=T, subject_ids=[s["subject_id"] for s in subjects])


def make_step(model: str, names: list[str], N: int):
    """One trial for all participants at once (pytensor.scan step); mirrors likelihoods._decision_values / _update.

    `names` are the parameter names in the order of the non-sequence tensors passed to scan; N is the
    number of participants."""
    import pytensor.tensor as pt
    ar = pt.arange(N)
    opts = pt.arange(3)[None, :]
    spec = parse(model)

    def step(c, r, w, mk, sc, tb, Q, CC, LT, prev_oh, prev_w, prev_r, NS, UP, *theta):
        """One trial. Inputs for this trial, each (N,): choice c, outcome r, coded ITI w that followed
        it, mask mk (1 = real trial, 0 = padding), scored flag sc, trend covariate tb. State carried
        from the trial before:
          Q (N, 3)        values
          CC (N, 3)       choice-kernel traces
          LT (N, 3)       trial number at which each option was last chosen (0 = never)
          prev_oh (N, 3)  one-hot of the last choice (all 0 before the first choice)
          prev_w, prev_r  coded ITI and outcome of the last trial
          NS (N, 3)       run counts (m12): consecutive choices of each option up to the last choice
          UP (N,)         u = 1 / (1 + n) of the option chosen last, read before it was counted (m12)
        theta: the (N,) parameter tensors in the order of `names`. Returns the updated state and the
        trial log-likelihood ll (N,). On padded trials the state is carried over unchanged."""
        p = dict(zip(names, theta))
        mk2 = mk[:, None]                                              # the mask broadcast over options
        dv = pt.zeros((N, 3))
        if spec["intercepts"]:
            dv = dv + pt.stack([p["c1"], p["c2"], pt.zeros(N)], axis=1)
        if spec["learn"]:
            beta = p["beta"] * pt.exp(p["beta_slope"] * tb) if spec["beta_trend"] else p["beta"]
            if spec["iti"] == "compete":
                # m12: after a long wait pi = kappa * u * L of beta moves from the values to the wait just
                # taken, on the option chosen last (prev_w = L, UP = u; prev_oh is 0 before the first choice).
                pi = p["kappa"] * prev_w * UP
                dv = dv + (beta * (1.0 - pi))[:, None] * Q - (beta * pi)[:, None] * prev_oh
            else:
                dv = dv + beta[:, None] * Q
        n_t = 50.0 * tb + 50.5                                        # trial number (from the trend covariate)
        if spec["novelty"]:                                            # constant bonus on the option(s) chosen furthest back
            is_c = pt.eq(LT, LT.min(axis=1)[:, None]).astype("float64")
            dv = dv + is_c * p["omega"][:, None]
        if spec["bias"] != "none":
            # Gain on the choice-bias weight: trend (m09) and ITI just elapsed (form "bias", m10).
            g = pt.ones(N)
            if spec["bias_trend"]:
                g = g * pt.exp(p["bias_slope"] * tb)
            if spec["iti"] == "bias":
                g = g * pt.exp(p["gamma"] * prev_w)
            if spec["bias"] == "choice_kernel":
                dv = dv + (g * p["beta_c"])[:, None] * CC
            else:                                                      # stay bias: prev_oh is 0 before the first choice
                dv = dv + (prev_oh * sc[:, None]) * (g * p["stick"])[:, None]
        ll = (dv[ar, c] - pt.logsumexp(dv, axis=1)) * mk * sc          # unscored: padding, and a choice right after a miss
        oh = pt.eq(opts, c[:, None]).astype("float64")                 # one-hot of this choice
        # Choice kernel: each trace moves toward 1 if its option was chosen, toward 0 if not.
        CC_out = CC + mk2 * p["alpha_c"][:, None] * (oh - CC) if spec["bias"] == "choice_kernel" else CC
        if spec["learn"]:
            y = r - p["cost"] * w if spec["iti"] == "target" else r    # m11: the wait that followed in the value target
            # Learning rate alpha after a win (or the only rate), alpha_r after a loss; then the
            # update of the chosen value.
            lr = p["alpha"] if spec["lr"] == "one" else pt.switch(r > 0.5, p["alpha"], p["alpha_r"])
            Qn = Q + oh * (lr * (y - Q[ar, c]))[:, None]
            if spec["decay"]:
                Qn = oh * Qn + (1.0 - oh) * (DECAY_TARGET + p["decay"][:, None] * (Qn - DECAY_TARGET))   # unchosen decay toward 0 (neutral)
            Q_out = mk2 * Qn + (1.0 - mk2) * Q
        else:
            Q_out = Q
        if spec["iti"] == "compete":
            # m12: u = 1 / (1 + n_c) of the option just chosen (read before it is counted); then its run
            # count goes up by 1 and every other option's count is reset to 0.
            UP_out = mk * (1.0 / (1.0 + NS[ar, c])) + (1.0 - mk) * UP
            NS_out = mk2 * (oh * (NS + 1.0)) + (1.0 - mk2) * NS
        else:
            NS_out, UP_out = NS, UP
        LT_out = mk2 * (oh * n_t[:, None] + (1.0 - oh) * LT) + (1.0 - mk2) * LT     # trial each option was last chosen
        # Remember this trial for the next decision (kept unchanged on padded trials).
        prev_oh_out = mk2 * oh + (1.0 - mk2) * prev_oh
        prev_w_out = mk * w + (1.0 - mk) * prev_w
        prev_r_out = mk * r + (1.0 - mk) * prev_r
        return Q_out, CC_out, LT_out, prev_oh_out, prev_w_out, prev_r_out, NS_out, UP_out, ll
    return step


def _scan_loglik(model, names, data, theta_list):
    """Trial-wise log-likelihood (T, N) of all participants, from one pytensor.scan over trials.

    data is a padded data dict (pad_subjects); theta_list holds one (N,) parameter tensor per name, in
    the order of `names`. Padded and unscored trials get 0."""
    import pytensor
    import pytensor.tensor as pt
    N = data["N"]
    step = make_step(model, names, N)
    SC = data["SC"] if "SC" in data else data["M"]
    spec = parse(model)                                          # the ITI code the model uses (likelihoods.coded_itis)
    if spec["iti_code"] == "centred":
        Wm = data["W"]
    elif "WS" in data:
        Wm = iti_code_long_only(data["WS"])
    else:
        raise ValueError("this data dict has no intervals in seconds (pad_subjects); needed for iti_code = 'long_only'")
    # Per-trial inputs, in the order of the step's first arguments: c, r, w, mk, sc, tb.
    seqs = [pt.as_tensor(data["C"]), pt.as_tensor(data["R"]), pt.as_tensor(Wm), pt.as_tensor(data["M"]), pt.as_tensor(SC),
            pt.as_tensor(data["TB"])]
    f64 = lambda x: pt.cast(x, "float64")
    i3 = lambda v: f64(pt.zeros((N, 3)) + v)                   # values start at Q_INIT (0), choice-kernel traces at INIT (0.5)
    # Initial state, in the order Q, CC, LT, prev_oh, prev_w, prev_r, NS, UP; the trial
    # log-likelihood (last output) is not carried over (None).
    init = [i3(Q_INIT), i3(INIT), f64(pt.zeros((N, 3))), f64(pt.zeros((N, 3))), f64(pt.zeros(N)), f64(pt.zeros(N)),
            f64(pt.zeros((N, 3))), f64(pt.zeros(N)), None]      # run counts (m12) start at 0, u at 0
    # Some PyTensor versions accept return_updates=False and return the outputs only; others raise a
    # TypeError for it and return (outputs, updates).
    try:
        outputs = pytensor.scan(step, sequences=seqs, outputs_info=init, non_sequences=theta_list, return_updates=False)
    except TypeError:
        outputs, _ = pytensor.scan(step, sequences=seqs, outputs_info=init, non_sequences=theta_list)
    return outputs[-1]   # (T, N) trial-wise log-likelihood (0 on padded and unscored trials)


def build_model(model: str, data: dict):
    """The PyMC model of `model` for a padded data dict (pad_subjects).

    Each parameter gets the non-centred hierarchical prior of the module docstring; the participants'
    constrained values are stored as Deterministic `name` (dims subject), the trial-wise
    log-likelihood as Deterministic ll_trial (subject, trial), and its sum enters as a Potential."""
    import pymc as pm
    import pytensor.tensor as pt
    names = PARAMS[model]
    coords = {"subject": data["subject_ids"], "trial": np.arange(data["T"])}
    with pm.Model(coords=coords) as m:
        theta = []
        for name in names:
            sup = param_support(model, name)
            # Group mean: N(0.5, 1) for positive, N(0, 1.5) for unit-interval, N(0, 1) for real parameters.
            mu = pm.Normal(f"mu_{name}", 0.5 if sup == "pos" else 0.0, 1.5 if sup == "unit" else 1.0)
            z = pm.Normal(f"z_{name}", 0.0, 1.0, dims="subject")
            sigma = pm.HalfNormal(f"sigma_{name}", 1.0)
            raw = mu + sigma * z
            # Map to the parameter's support: inverse logit, exponential, or identity.
            val = pm.math.invlogit(raw) if sup == "unit" else (pt.exp(raw) if sup == "pos" else raw)
            theta.append(pm.Deterministic(name, val, dims="subject"))
        ll = _scan_loglik(model, names, data, theta)
        pm.Deterministic("ll_trial", ll.T, dims=("subject", "trial"))
        pm.Potential("loglik", ll.sum())
    return m


def _draw_params(model, names, N, rng):
    """Random parameter values for N participants on each parameter's support (for
    check_against_numpy): unit ~ Uniform(0.02, 0.98), pos ~ Gamma(shape 2, scale 1.5),
    real ~ Normal(0, 0.6)."""
    th = {}
    for n in names:
        sup = param_support(model, n)
        th[n] = (rng.uniform(0.02, 0.98, N) if sup == "unit" else
                 rng.gamma(2, 1.5, N) if sup == "pos" else rng.normal(0, 0.6, N))
    return th


def check_against_numpy(model: str, subjects: list[dict], S: int = 5, seed: int = 0) -> float:
    """Max abs difference between the PyTensor and numpy log-likelihoods (per subject) for
    random parameter vectors drawn on the NUTS supports.

    S is the number of random parameter draws; each draw gives every participant its own values."""
    import pytensor
    import pytensor.tensor as pt
    from .likelihoods import loglik
    data = pad_subjects(subjects)
    names = PARAMS[model]
    # Compile the PyTensor log-likelihood as a function of the parameter vectors.
    theta_syms = [pt.dvector(n) for n in names]
    f = pytensor.function(theta_syms, _scan_loglik(model, names, data, theta_syms))
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(S):
        th = _draw_params(model, names, data["N"], rng)
        ll_pt = f(*[th[n] for n in names]).sum(axis=0)          # (N,)
        for i, s in enumerate(subjects):
            ll_np = loglik(model, np.array([[th[n][i] for n in names]]), s["choices"], s["rewards"], s["itis"], trials=s.get("trial"))[0]
            worst = max(worst, abs(ll_pt[i] - ll_np))
    return worst


def sample_model(m, draws=1000, tune=1000, chains=4, seed=0, backend="numba", target_accept=0.9, verbose=True, progress_every=120):
    """Compile and sample with nutpie. With verbose, a progress line is printed every `progress_every` seconds (the sampler
    runs in the background and is inspected: post-warm-up draws per chain so far and divergences), so a log file shows
    how far a long run has got.

    Returns (idata, info): the posterior from nutpie, and a dict with the compile and sampling times
    in seconds and the sampler settings."""
    import nutpie
    t0 = time.time()
    compiled = nutpie.compile_pymc_model(m, backend=backend)
    t_comp = time.time() - t0
    t0 = time.time()
    if not verbose or not progress_every:
        idata = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept, progress_bar=False)
    else:
        # Sample in the background (blocking=False) and check every 2 s whether it has finished.
        sampler = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept,
                                progress_bar=False, blocking=False)
        def finished():
            """True once the background sampler is done; a TimeoutError while asking counts as not done."""
            try:
                return sampler.is_finished
            except TimeoutError:
                return False
        last = time.time()
        while not finished():
            time.sleep(2.0)
            if time.time() - last >= progress_every:
                last = time.time()
                # Inspect the draws so far: 0 post-warm-up draws means the sampler is still warming up.
                try:
                    part = sampler.inspect()
                    post = part.posterior.to_dataset() if hasattr(part.posterior, "to_dataset") else part.posterior
                    n = int(post.sizes.get("draw", 0))
                    ss = part.sample_stats.to_dataset() if hasattr(part.sample_stats, "to_dataset") else part.sample_stats
                    div = int(np.asarray(ss["diverging"]).sum()) if "diverging" in ss else -1
                    stage = "warm-up" if n == 0 else f"{n}/{draws} draws per chain ({100 * n / draws:.0f}%)"
                    print(f"    [{(time.time() - t0) / 60:5.1f} min] {stage}, divergences so far {div}", flush=True)
                except Exception as e:                       # inspection is best-effort; never interrupt the sampler
                    print(f"    [{(time.time() - t0) / 60:5.1f} min] running (progress unavailable: {type(e).__name__})", flush=True)
        idata = sampler.wait()
    t_samp = time.time() - t0
    if verbose:
        print(f"    compiled in {t_comp:.0f}s ({backend}), sampled {chains}x{draws} (+{tune} tune) in {t_samp:.0f}s", flush=True)
    return idata, dict(compile_seconds=t_comp, sample_seconds=t_samp, backend=backend, draws=draws, tune=tune, chains=chains,
                       target_accept=target_accept)


LOO_BLOCK = 4000   # observations per PSIS-LOO call in postprocess (limits memory)


class _LooFields:
    """Uniform view of az.loo output: arviz >= 1 (elpd, se, p, elpd_i, pareto_k) and arviz 0.x
    (elpd_loo, se, p_loo, loo_i, pareto_k)."""
    def __init__(self, lo):
        self.raw = lo

        def g(*keys):
            """The first of `keys` found in the LOO result (KeyError if none is)."""
            # attribute access first (arviz_stats ELPDData has no __contains__, so `k in lo` falls
            # back to integer indexing and fails), then item access (arviz 0.x ELPDData is a pandas
            # Series)
            for k in keys:
                for get in (lambda o, kk: getattr(o, kk), lambda o, kk: o[kk]):
                    try:
                        v = get(lo, k)
                    except (AttributeError, KeyError, TypeError, IndexError):
                        continue
                    if v is not None:
                        return v
            raise KeyError(f"none of {keys} in the LOO result")
        self.elpd = float(g("elpd", "elpd_loo")); self.se = float(g("se")); self.p = float(g("p", "p_loo"))
        self.elpd_i = np.asarray(g("elpd_i", "loo_i")); self.pareto_k = np.asarray(g("pareto_k"))


def postprocess(idata, model: str, data: dict, subjects: list[dict]) -> dict:
    """LOO (trial-level and subject-level), convergence diagnostics, parameter summaries.

    Returns a dict: summary (model-level numbers), subjects (one row per participant: elpd, Pareto k
    and posterior summaries of each parameter), group (one row per parameter: group-level posterior
    and diagnostics), the pointwise arrays elpd_i, pareto_k, subj_of_obs and ll_obs_mean over the LOO
    observations, the raw az.loo outputs loo and loo_subject, and hyper_draws (the posterior draws of
    the group-level parameters)."""
    import arviz as az
    names = PARAMS[model]
    # arviz >= 1 stores groups as DataTree nodes (need .to_dataset()); arviz 0.x stores xarray Datasets
    _ds = lambda grp: grp.to_dataset() if hasattr(grp, "to_dataset") else grp
    _pack = (lambda d: xr.DataTree.from_dict(d)) if hasattr(idata.posterior, "to_dataset") else (lambda d: az.InferenceData(**d))
    ds = _ds(idata.posterior)
    valid = (data["SC"] if "SC" in data else data["M"]).T.astype(bool)   # (N, T) scored choices
    # The first trial of every participant has log-likelihood log(1/3) under every model with learning
    # and every parameter value (Q = 0, no previous choice), so it is uninformative and constant across
    # draws. PSIS cannot smooth a constant, so it is excluded from the pointwise LOO set (for every
    # model, including the intercepts-only m01).
    loo_mask = valid.copy(); loo_mask[:, 0] = False
    subj_of_obs = np.where(loo_mask)[0]                       # participant index of each LOO observation
    ll_obs = ds["ll_trial"].values[:, :, loo_mask]           # (chain, draw, n_obs)
    # Group-level parameters: mu and sigma of every parameter.
    hyper_names = [f"mu_{n}" for n in names] + [f"sigma_{n}" for n in names]
    small_post = ds[hyper_names]
    # Observations PSIS cannot smooth: (i) a scored choice whose log-likelihood is the same under every
    # draw (e.g. all three values equal, as after three losses on three different options in a model
    # without decay or choice bias); (ii) a tail of the importance ratios made of identical values (many
    # draws giving the same log-likelihood, e.g. a probability that rounds to 1). These are left out of
    # az.loo, and the plain importance-sampling estimate -log mean(exp(-ll)) is used instead (it equals
    # the constant in case (i)); they are counted in n_obs_plain_is and get Pareto k = 0.
    # M_tail is the standard PSIS tail length, min(0.2 S, 3 sqrt(S)) for S draws.
    S_draws, n_obs = ll_obs.shape[0] * ll_obs.shape[1], ll_obs.shape[-1]
    M_tail = int(np.ceil(min(0.2 * S_draws, 3 * np.sqrt(S_draws))))
    plain = np.zeros(n_obs, dtype=bool)
    # In blocks of 4000 observations (memory): -ll are the log importance ratios; tail = the
    # M_tail + 1 largest per observation. Plain if the tail is flat or the log-likelihood barely varies.
    for j0 in range(0, n_obs, 4000):
        blk = -ll_obs[:, :, j0:j0 + 4000].reshape(S_draws, -1)
        tail = np.partition(blk, S_draws - M_tail - 1, axis=0)[S_draws - M_tail - 1:]
        plain[j0:j0 + 4000] = (np.ptp(tail, axis=0) < 1e-12) | (blk.std(axis=0) < 1e-10)
    # PSIS-LOO in blocks of observations: PSIS is computed observation by observation, so
    # the result is the same as one call on all observations, but az.loo holds several copies of the
    # (chain, draw, obs) array, which for Study 3 (4 x 1000 draws x ~43,000 trials) peaked near 20 GB
    # per fit and, with fits running in parallel, exhausted the machine's memory.
    dt = _pack({"posterior": small_post})
    idx = np.where(~plain)[0]; e_parts, k_parts, p_tot = [], [], 0.0
    for j0 in range(0, len(idx), LOO_BLOCK):
        sl_ = idx[j0:j0 + LOO_BLOCK]
        dtb = _pack({"posterior": small_post, "log_likelihood": xr.Dataset({"ll": xr.DataArray(ll_obs[:, :, sl_], dims=("chain", "draw", "obs"))})})
        lb = _LooFields(az.loo(dtb, var_name="ll")); e_parts.append(np.asarray(lb.elpd_i)); k_parts.append(np.asarray(lb.pareto_k)); p_tot += lb.p
        del dtb
    # lo keeps the fields of the last block (lo.raw is that block's az.loo output); its p_loo becomes
    # the sum over blocks, and its elpd and SE are recomputed below over all observations.
    lo = lb; lo.p = p_tot
    elpd_i = np.empty(n_obs); k = np.zeros(n_obs)
    elpd_i[~plain] = np.concatenate(e_parts); k[~plain] = np.concatenate(k_parts)
    if plain.any():
        # Plain importance sampling: elpd_i = -log mean over draws of exp(-ll).
        from scipy.special import logsumexp as _lse
        neg = -ll_obs[:, :, plain].reshape(S_draws, -1)
        elpd_i[plain] = -(_lse(neg, axis=0) - np.log(S_draws))
    lo.elpd = float(elpd_i.sum()); lo.se = float(np.sqrt(len(elpd_i) * np.var(elpd_i)))
    # Per participant: summed pointwise elpd, worst Pareto k, and the summed log-likelihood per draw
    # for a subject-level PSIS-LOO (one observation per participant).
    N = data["N"]
    subj_elpd = np.bincount(subj_of_obs, weights=elpd_i, minlength=N)
    subj_kmax = np.array([k[subj_of_obs == i].max() for i in range(N)])
    ll_subj = np.zeros((ll_obs.shape[0], ll_obs.shape[1], N))
    for i in range(N):
        ll_subj[:, :, i] = ll_obs[:, :, subj_of_obs == i].sum(axis=2)
    dt_s = _pack({"posterior": small_post,
                  "log_likelihood": xr.Dataset({"ll": xr.DataArray(ll_subj, dims=("chain", "draw", "subject"))})})
    lo_s = _LooFields(az.loo(dt_s, var_name="ll"))
    # Convergence: divergences (-1 if not recorded), and the worst R-hat and smallest bulk ESS of each
    # group-level parameter.
    ss = _ds(idata.sample_stats)
    div = int(np.asarray(ss["diverging"]).sum()) if "diverging" in ss.data_vars else -1
    rhat_ds = az.rhat(dt, var_names=hyper_names); ess_ds = az.ess(dt, var_names=hyper_names)
    rhat = {v: float(np.asarray(rhat_ds[v]).max()) for v in hyper_names}
    ess = {v: float(np.asarray(ess_ds[v]).min()) for v in hyper_names}
    # One row per participant: LOO results and the posterior mean, SD and 5th / 95th percentiles of
    # each parameter on its natural scale.
    rows = []
    for i, s in enumerate(subjects):
        row = dict(subject_id=s["subject_id"], n_trials=len(s["choices"]), elpd_loo=subj_elpd[i], pareto_k_max=subj_kmax[i],
                   elpd_loo_subjectlevel=float(np.asarray(lo_s.elpd_i)[i]), pareto_k_subjectlevel=float(np.asarray(lo_s.pareto_k)[i]))
        for n in names:
            arr = ds[n].values[:, :, i]
            row[f"{n}_mean"] = arr.mean(); row[f"{n}_sd"] = arr.std()
            row[f"{n}_ci05"], row[f"{n}_ci95"] = np.percentile(arr, [5, 95])
        rows.append(row)
    subj = pd.DataFrame(rows)
    # One row per parameter: the posterior of mu and sigma, the group mean of the constrained values
    # (over draws and participants), the SD across participants of their posterior means, and the
    # diagnostics.
    grows = []
    for n in names:
        mu = ds[f"mu_{n}"].values.ravel(); sg = ds[f"sigma_{n}"].values.ravel()
        th = ds[n].values.reshape(-1, N)
        row = dict(parameter=n, support=param_support(model, n), mu_mean=mu.mean(), mu_sd=mu.std(),
                   mu_ci05=np.percentile(mu, 5), mu_ci95=np.percentile(mu, 95), sigma_mean=sg.mean(), sigma_sd=sg.std(),
                   group_mean_constrained=th.mean(), group_sd_constrained=th.mean(axis=0).std(),
                   rhat_mu=rhat[f"mu_{n}"], rhat_sigma=rhat[f"sigma_{n}"], ess_bulk_mu=ess[f"mu_{n}"], ess_bulk_sigma=ess[f"sigma_{n}"])
        grows.append(row)
    group = pd.DataFrame(grows)
    # Model-level summary. looic = -2 elpd; first_trial_constant_loglik_total = N log(1/3), the
    # log-likelihood of the excluded first trials; mean_total_loglik = the sum over LOO observations of
    # the posterior-mean log-likelihood.
    summary = dict(model=model, n_subjects=N, n_obs=int(loo_mask.sum()), n_obs_incl_first_trials=int(valid.sum()), n_obs_plain_is=int(plain.sum()),
                   first_trial_constant_loglik_total=float(N * np.log(1 / 3)),
                   elpd_loo=float(lo.elpd), se_elpd_loo=float(lo.se), p_loo=float(lo.p), looic=float(-2 * lo.elpd),
                   pareto_k_frac_gt_0p7=float((k > 0.7).mean()), pareto_k_max=float(k.max()),
                   elpd_loo_subjectlevel=float(lo_s.elpd), se_subjectlevel=float(lo_s.se), p_loo_subjectlevel=float(lo_s.p),
                   pareto_k_subjectlevel_frac_gt_0p7=float((np.asarray(lo_s.pareto_k) > 0.7).mean()),
                   mean_total_loglik=float(ll_obs.mean(axis=(0, 1)).sum()), n_divergences=div,
                   rhat_max=max(rhat.values()), ess_bulk_min=min(ess.values()))
    return dict(summary=summary, subjects=subj, group=group, elpd_i=elpd_i, pareto_k=k, subj_of_obs=subj_of_obs,
                ll_obs_mean=ll_obs.mean(axis=(0, 1)), loo=lo.raw, loo_subject=lo_s.raw,
                hyper_draws=small_post.to_dataframe().reset_index())


def save_pymc_result(res: dict, out_dir, prefix: str):
    """Write a postprocess result to out_dir: {prefix}_subjects.csv, {prefix}_group.csv,
    {prefix}_pointwise.npz (elpd_i, pareto_k, subj_of_obs, ll_obs_mean),
    {prefix}_group_posterior_draws.csv.gz (if the draws are present) and {prefix}_summary.json."""
    from pathlib import Path
    import json
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res["subjects"].to_csv(out_dir / f"{prefix}_subjects.csv", index=False)
    res["group"].to_csv(out_dir / f"{prefix}_group.csv", index=False)
    np.savez_compressed(out_dir / f"{prefix}_pointwise.npz", elpd_i=res["elpd_i"], pareto_k=res["pareto_k"],
                        subj_of_obs=res["subj_of_obs"], ll_obs_mean=res["ll_obs_mean"])
    if "hyper_draws" in res:
        res["hyper_draws"].to_csv(out_dir / f"{prefix}_group_posterior_draws.csv.gz", index=False, compression="gzip")
    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(res["summary"], f, indent=2, default=float)
