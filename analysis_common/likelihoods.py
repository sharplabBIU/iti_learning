"""Reinforcement-learning models of the ITI project: specifications, likelihood and simulator.

Overview
    This module defines the reinforcement-learning models fitted to the three-armed restless bandit
    task, and one engine that evaluates them. The engine works on plain NumPy arrays and is vectorised
    over S parameter samples. It gives the log-likelihood of one participant's choices (loglik) and
    plays the task with a synthetic agent (simulate_choices). analysis_common/pymc_models.py writes the
    same likelihood in PyTensor for fitting with the No-U-Turn sampler (NUTS); its check_against_numpy
    compares the two. The module holds the models of the paper's model sequence.

The question
    Does the inter-trial interval (ITI, 1 s or 4 s) act on the CHOICE BIAS (the tendency to repeat an
    action), on the VALUE the option learns, or does it COMPETE WITH VALUE for control of the next
    choice? Only the ITI just elapsed enters any model: no model-agnostic test found an effect of
    earlier ITIs on switching.

Main parts
    1. Coding helpers: iti_code, iti_code_long_only, coded_itis, reward_code, beta_trend_covariate,
       scored_mask.
    2. Model specifications. A model is a dict of switches (SPEC_DEFAULTS, checked by make_spec). Its
       parameter names come from params_for, and the support of each parameter from SUPPORT.
    3. The model sequence: the stage dicts, the recorded stage choices, register_models and
       stage_parent. register_models() runs on import and fills SPECS, PARAMS and STAGES.
    4. The engine (_init_state, _decision_values, _update) and its two entry points, loglik and
       simulate_choices.

Coding conventions
    - Outcomes r: +1 for a win, -1 for a loss (REWARD_CODING = "pm1"). A win is r > 0.5 everywhere
      in the engine.
    - Values Q start at 0 (Q_INIT). The values of unchosen options decay toward 0 (DECAY_TARGET), the
      neutral point between +1 and -1.
    - The choice-kernel traces C start at 0.5 (INIT). A trace moves toward 1 when its option is
      chosen and toward 0 when it is not.
    - ITI code w, set per model by the iti_code switch:
        "centred"    w = -1 after a 1 s ITI, +1 after a 4 s ITI (the default; models without an ITI
                     term do not use w).
        "long_only"  w = 0 after a 1 s ITI, 1 after a 4 s ITI: only the long wait carries anything.
                     Every stage-D (ITI) model uses this code.
    - The ITI stored with each trial is the one that FOLLOWED it, so w_{t-1} is the wait just
      elapsed before choice t.
    - Choices are 0-indexed (options 0, 1, 2).

The model sequence
    The models are fitted in stages on Study 3 (the preregistered online replication). Each stage
    builds on the winner of the stage before (register_models, stage_parent). The stage rules are
    given in the comments around STAGE_A. "(chosen)" marks the recorded choice.
    A. Value learning
         m01_intercepts      option intercepts only (no learning)
         m02_rw_1lr          Rescorla-Wagner, one learning rate
         m03_rw_2lr          separate learning rates after a win and after a loss
         m04_rw_2lr_decay    m03 + decay of the values of unchosen options toward the neutral
                             value (chosen)
    B. Choice bias: one choice-bias term each, on the stage-A winner
         m05_stick           a constant bias toward the last choice
         m06_choice_kernel   choice kernel: one trace per option, moved toward 1 when the option is
                             chosen and toward 0 when it is not, with its own learning rate and
                             weight (chosen)
    B2. Novelty
         m07_novelty         the stage-B winner + a constant bonus omega on the option chosen
                             furthest back (never chosen counts as chosen before trial 1). It is
                             kept for the rest of the sequence only if it improves on the stage-B
                             winner (kept).
    C. Trend, on the stage-B winner (with the novelty bonus if kept)
         m08_trend_beta      a log-linear trend in the inverse temperature on value (chosen)
         m09_trend_bias      the same trend on the weight of the choice-bias term
    D. ITI, on the stage-C winner, with the long_only ITI code (L_t = w_{t-1} = 1 after a 4 s wait)
         m10_iti_bias        ITI on choice bias: the wait just elapsed scales the weight of the
                             choice-bias term by exp(gamma L_t)
         m11_iti_target      ITI as learning target: the wait that followed a choice enters that
                             option's value target, y = r - cost w_t
         m12_iti_compete     ITI competes with value: after a long wait, a share pi_t = kappa u_t L_t
                             of the inverse temperature moves from the learned values to the wait
                             just taken, which is attached to the option chosen last; u_t =
                             1 / (1 + n_t), n_t the number of consecutive choices of that option just
                             before it was chosen (its uncertainty: u = 1 after a switch to it)

The choice rule (all models)
    P(a_t = a) = softmax over options of  beta_t Q(a) + h_t(a) + b_t 1[a = a_{t-1}] + n_t(a)
      beta_t   inverse temperature: beta, or beta exp(beta_slope * tb_t) (m08), with
               tb_t = (trial - 50.5) / 50 from the participant's actual trial number. In m12 the
               value term is beta_t (1 - pi_t) Q(a) - beta_t pi_t 1[a = a_{t-1}] (the "compete" form).
      g_t      gain on the choice-bias weight: exp(bias_slope * tb_t) (m09) times
               exp(gamma * L_t) (ITI form "bias", m10)
      h_t(a)   choice kernel: g_t * beta_c * C_t(a)
      b_t      constant stay bias on the option chosen last: g_t * stick (m05)
      n_t(a)   novelty (m07): omega for the option(s) whose last choice is furthest back, 0 for the
               others
    A choice made right after a missed trial is not scored and gets no stay bias. It still updates
    learning and the traces.

Updates after trial t (choice c, outcome r, ITI w_t that followed the trial)
    C(a) <- C(a) + alpha_c (x_a - C(a)), x_a = 1 for the chosen option, 0 for the others
    Q(c) <- Q(c) + alpha^{+/-} (y_t - Q(c)), with y_t = r, or y_t = r - cost * w_t (m11)
    Q(a) <- DECAY_TARGET + decay (Q(a) - DECAY_TARGET) for each unchosen option a
    m12 only: the run counts behind u (the chosen option's count + 1, every other option's count reset
    to 0), read for the chosen option before it is counted.

ITI forms (the iti switch): "none", "bias" (m10), "target" (m11), "compete" (m12). Each form is
    described above its branch in params_for.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

# ITI lengths in seconds.
ITI_SHORT, ITI_LONG = 1.0, 4.0
# Trend covariate: (trial - 50.5) / 50, from the participant's actual trial number.
BTREND_CENTER, BTREND_SCALE = 50.5, 50.0
# Outcome and value coding. Outcomes are +1 (win) / -1 (loss). Values start at 0 and decay toward 0,
# the neutral point between +1 and -1 and the natural point for them to regress to. The choice-kernel
# traces start at 0.5.
REWARD_CODING = "pm1"
Q_INIT = 0.0                                     # starting value of the values (the prior)
INIT = 0.5                                       # start of the choice-kernel traces
DECAY_TARGET = 0.0                               # values decay toward the neutral value


def iti_code(itis) -> np.ndarray:
    """Centred ITI code: -1 after a 1 s ITI, +1 after a 4 s ITI (linear in seconds)."""
    x = np.asarray(itis, dtype=float)
    return (x - (ITI_SHORT + ITI_LONG) / 2.0) / ((ITI_LONG - ITI_SHORT) / 2.0)


def iti_code_long_only(itis) -> np.ndarray:
    """Long-only ITI code: 0 after a 1 s ITI, 1 after a 4 s ITI (only the long wait carries a cost)."""
    x = np.asarray(itis, dtype=float)
    return (x > (ITI_SHORT + ITI_LONG) / 2.0).astype(float)


def coded_itis(spec: dict, itis) -> np.ndarray:
    """The ITI code a model uses, chosen by spec['iti_code'] ("centred" or "long_only")."""
    return iti_code(itis) if spec["iti_code"] == "centred" else iti_code_long_only(itis)


def reward_code(rewards) -> np.ndarray:
    """+1 for a win, -1 for a loss, as the data store them.

    Any reward above 0 counts as a win. A win is r > 0.5 everywhere in the engine."""
    return np.where(np.asarray(rewards, dtype=float) > 0, 1.0, -1.0)


def beta_trend_covariate(trials, T=None) -> np.ndarray:
    """Trend covariate (trial - 50.5) / 50 for each responded trial, from the actual trial numbers.

    Without trial numbers (trials=None) the count of responded trials (1, 2, ..., T) is used."""
    tr = np.asarray(trials, dtype=float) if trials is not None else np.arange(T, dtype=float) + 1.0
    return (tr - BTREND_CENTER) / BTREND_SCALE


def scored_mask(trials) -> np.ndarray:
    """Which responded choices enter the likelihood.

    Scored are the first choice and every choice whose previous trial was also responded; a choice
    made right after a missed trial is not scored. `trials` are the trial numbers of the responded
    choices."""
    tr = np.asarray(trials)
    sc = np.ones(len(tr), dtype=bool)
    sc[1:] = np.diff(tr) == 1
    return sc


# ---------------------------------------------------------------------------
# Model specifications
# A model is a dict of switches; SPEC_DEFAULTS lists them with their default values.
#   intercepts   option intercepts c1, c2 (the third option is the reference)
#   learn        value learning on (off only in the intercepts-only model m01)
#   lr           "one" learning rate, or "two": alpha after a win, alpha_r after a loss
#   decay        unchosen values decay toward the neutral value
#   bias         the choice-bias term (one of BIASES)
#   novelty      novelty bonus on the option chosen furthest back (m07)
#   beta_trend   log-linear trend in the inverse temperature (m08)
#   bias_trend   log-linear trend in the choice-bias weight (m09)
#   iti          the ITI form (one of ITI_FORMS; "none" = no ITI effect)
#   iti_code     "centred" or "long_only" ITI code (ITI_CODES)
# ---------------------------------------------------------------------------
SPEC_DEFAULTS = dict(intercepts=False, learn=True, lr="two", decay=True, bias="none", novelty=False, beta_trend=False, bias_trend=False,
                     iti="none", iti_code="centred")
# The ITI forms: none, m10 (bias), m11 (target), m12 (compete).
ITI_FORMS = ("none", "bias", "target", "compete")
ITI_CODES = ("centred", "long_only")
# Choice-bias terms: "stick" = constant stay bias (m05); "choice_kernel" = one trace per option (m06).
BIASES = ("none", "stick", "choice_kernel")


def make_spec(**kw) -> dict:
    """Build and check a model specification from keyword switches (see SPEC_DEFAULTS).

    Raises KeyError for an unknown switch and ValueError for an invalid value or combination."""
    bad = set(kw) - set(SPEC_DEFAULTS)
    if bad:
        raise KeyError(f"unknown specification switches {sorted(bad)}")
    s = dict(SPEC_DEFAULTS); s.update(kw)
    if s["lr"] not in ("one", "two"):
        raise ValueError("lr must be 'one' or 'two'")
    if s["bias"] not in BIASES:
        raise ValueError(f"bias must be one of {BIASES}")
    if s["iti"] not in ITI_FORMS:
        raise ValueError(f"iti must be one of {ITI_FORMS}")
    if s["iti_code"] not in ITI_CODES:
        raise ValueError(f"iti_code must be one of {ITI_CODES}")
    # Combinations that do not make a valid model.
    if not s["learn"] and (s["bias"] != "none" or s["beta_trend"] or s["iti"] != "none" or s["novelty"]):
        raise ValueError("the no-learning model has intercepts only")
    if (s["bias_trend"] or s["iti"] == "bias") and s["bias"] == "none":
        raise ValueError("a trend or interval effect on the choice-bias weight needs a choice-bias term")
    if s["iti"] == "compete" and s["iti_code"] != "long_only":
        raise ValueError("compete moves weight only after a long ITI (w = 1 after 4 s, 0 after 1 s): iti_code long_only")
    return s


def params_for(spec: dict) -> list[str]:
    """Parameter names of a model specification, in the column order of theta (stored as PARAMS).

    The comment above each ITI branch describes the form. In the formulas, w is the coded ITI that
    followed the trial and c a cost."""
    n = []
    if spec["intercepts"]:
        n += ["c1", "c2"]
    # Learning: alpha is the learning rate after a win (or the only rate), alpha_r the rate after a
    # loss; beta the inverse temperature; decay the retention of unchosen values.
    if spec["learn"]:
        n += ["alpha"] + (["alpha_r"] if spec["lr"] == "two" else []) + ["beta"] + (["decay"] if spec["decay"] else [])
    # Choice bias: stick (m05); alpha_c and beta_c, the choice kernel (m06).
    n += {"none": [], "stick": ["stick"], "choice_kernel": ["alpha_c", "beta_c"]}[spec["bias"]]
    # Novelty (m07): a constant bonus omega on the option chosen furthest back.
    if spec["novelty"]:
        n += ["omega"]
    if spec["beta_trend"]:
        n.append("beta_slope")
    if spec["bias_trend"]:
        n.append("bias_slope")

    # --- ITI forms (L_t = w_{t-1}: 1 if the wait just elapsed was 4 s) ---
    # "bias" (m10, ITI on choice bias): the wait scales the choice-bias weight by exp(gamma L_t),
    # outcome-blind by construction.
    if spec["iti"] == "bias":
        n += ["gamma"]
    # "target" (m11, ITI as learning target): the wait that followed a choice enters that option's
    # value target, y = r - cost w; it shifts the target after wins and after losses alike.
    elif spec["iti"] == "target":
        n += ["cost"]
    # "compete" (m12, ITI competes with value): after a long wait a share pi_t = kappa u_t L_t of beta
    # moves from the values to the wait just taken, attached to the option chosen last:
    # dv = beta (1 - pi_t) Q - beta pi_t 1[a = a_{t-1}]. u_t = 1 / (1 + n), n the number of
    # consecutive choices of a_{t-1} just before trial t-1 (u = 1 when t-1 was a switch to it).
    elif spec["iti"] == "compete":
        n += ["kappa"]
    return n


# Support of each parameter: 'unit' = (0, 1), 'pos' = (0, inf), 'real' = the real line.
# pymc_models.py maps each support to its prior and transform.
SUPPORT = {"c1": "real", "c2": "real", "alpha": "unit", "alpha_r": "unit", "beta": "pos", "decay": "unit",
           "stick": "real", "alpha_c": "unit", "beta_c": "pos", "omega": "real", "beta_slope": "real", "bias_slope": "real",
           "gamma": "real", "cost": "real", "kappa": "unit"}
# Parameters that belong to an ITI form (reported by iti_params).
ITI_PARAMETERS = ("gamma", "cost", "kappa")

# Registries filled by register(): model name -> specification, and model name -> parameter names.
SPECS: dict[str, dict] = {}
PARAMS: dict[str, list[str]] = {}


def register(name: str, **kw) -> str:
    """Register a model under `name` with the given switches (see make_spec); returns the name."""
    SPECS[name] = make_spec(**kw); PARAMS[name] = params_for(SPECS[name])
    return name


def parse(model: str) -> dict:
    """A copy of the specification of a registered model."""
    if model not in SPECS:
        raise KeyError(f"unknown model {model!r} (registered: {', '.join(SPECS)})")
    return dict(SPECS[model])


def iti_params(model: str) -> list[str]:
    """The ITI parameters of a model (for reporting)."""
    return [p for p in PARAMS[model] if p in ITI_PARAMETERS]


# ---------------------------------------------------------------------------
# The model sequence. The stage choices are made on Study 3 by
# analysis_common/study_scripts/06_fit_models_nuts.py, using the expected log predictive density
# (elpd) from Pareto-smoothed importance-sampling leave-one-out cross-validation (PSIS-LOO):
#   A: an addition is kept if it improves elpd by more than 2 SE of the paired difference;
#   B: the largest elpd;
#   B2: the novelty bonus is kept if it gains more than 2 SE;
#   C: the better trend if it gains more than 2 SE.
# 06 prints the choices the data support and warns when they differ from the recorded ones below.
# ---------------------------------------------------------------------------
# Stage A (value learning): the models, and the model each one is compared with.
STAGE_A = {"m01_intercepts": dict(intercepts=True, learn=False, decay=False),
           "m02_rw_1lr": dict(lr="one", decay=False),
           "m03_rw_2lr": dict(lr="two", decay=False),
           "m04_rw_2lr_decay": dict(lr="two", decay=True)}
STAGE_A_PARENT = {"m01_intercepts": None, "m02_rw_1lr": "m01_intercepts", "m03_rw_2lr": "m02_rw_1lr", "m04_rw_2lr_decay": "m03_rw_2lr"}
# Recorded choices, made on Study 3: the stage-A winner, the stage-B choice-bias term, whether the
# novelty bonus is kept (B2), and the stage-C trend switch.
CHOICE_A = "m04_rw_2lr_decay"
CHOICE_B = "choice_kernel"
CHOICE_B2 = True
CHOICE_C = "beta_trend"
# Stage labels and their model names, filled by register_models().
STAGES: list[tuple[str, list[str]]] = []
# Stage B (choice bias): model name -> bias switch.
STAGE_B = {"m05_stick": "stick", "m06_choice_kernel": "choice_kernel"}
# Stage D (ITI): model name -> ITI form, each with the long_only ITI code.
STAGE_D = {"m10_iti_bias": "bias", "m11_iti_target": "target", "m12_iti_compete": "compete"}


def register_models() -> list[tuple[str, list[str]]]:
    """Register the models of every stage whose parent is decided. Returns [(stage label, [model names])].

    A stage is registered only when the choice of the stage before it is recorded; a recorded choice
    of None stops the sequence there."""
    SPECS.clear(); PARAMS.clear(); STAGES.clear()
    # Stage A: value learning.
    for n, kw in STAGE_A.items():
        register(n, **kw)
    STAGES.append(("A. value learning", list(STAGE_A)))
    if CHOICE_A is None:
        return list(STAGES)
    # Stage B: each choice-bias term on the stage-A winner.
    base = dict(STAGE_A[CHOICE_A])
    for n, b in STAGE_B.items():
        register(n, **base, bias=b)
    STAGES.append(("B. choice bias", list(STAGE_B)))
    if CHOICE_B is None:
        return list(STAGES)
    # Stage B2: the novelty bonus on the stage-B winner.
    base_b = dict(base, bias=CHOICE_B)
    register("m07_novelty", **base_b, novelty=True)
    STAGES.append(("B2. novelty", ["m07_novelty"]))
    if CHOICE_B2 is None:
        return list(STAGES)
    # Stage C: the two trends, on the stage-B winner with the novelty bonus if it was kept.
    base_b = dict(base_b, novelty=bool(CHOICE_B2))
    register("m08_trend_beta", **base_b, beta_trend=True); register("m09_trend_bias", **base_b, bias_trend=True)
    STAGES.append(("C. trend", ["m08_trend_beta", "m09_trend_bias"]))
    if CHOICE_C is None:
        return list(STAGES)
    # Stage D (ITI): each ITI model on the stage-C winner (the chosen trend switched on).
    base_c = dict(base_b, **({CHOICE_C: True} if CHOICE_C != "none" else {}))
    for n, f in STAGE_D.items():
        register(n, **base_c, iti=f, iti_code="long_only")
    STAGES.append(("D. interval", list(STAGE_D)))
    return list(STAGES)


def stage_parent(model: str) -> str | None:
    """The model each model is compared against: within stage A the previous addition, otherwise the
    previous stage's winner. Raises KeyError for a model outside the sequence."""
    if model in STAGE_A_PARENT:
        return STAGE_A_PARENT[model]
    if model in STAGE_B:
        return CHOICE_A
    # The stage-B winner; replaced by m07_novelty when the novelty bonus is kept.
    b_winner = next((m for m, b in STAGE_B.items() if b == CHOICE_B), None)
    if model == "m07_novelty":
        return b_winner
    if CHOICE_B2:
        b_winner = "m07_novelty"
    if model in ("m08_trend_beta", "m09_trend_bias"):
        return b_winner
    # Every ITI model is compared with the stage-C winner.
    if model in STAGE_D:
        return {"beta_trend": "m08_trend_beta", "bias_trend": "m09_trend_bias"}.get(CHOICE_C, b_winner)
    raise KeyError(f"no stage parent for {model!r}")


# Register the sequence on import, so that SPECS, PARAMS and STAGES are ready for other modules.
register_models()


# ---------------------------------------------------------------------------
# The engine, vectorised over S parameter samples. The learning state is a dict (see _init_state).
# On each trial, _decision_values gives the decision values of the three options, and _update
# learns from the outcome.
# ---------------------------------------------------------------------------
def _init_state(S):
    """Initial learning state for S parameter samples. Entries:
      q (S, 3)          values, start at Q_INIT
      C (S, 3)          choice-kernel traces, start at INIT
      prev              the last choice (-1 = none yet)
      prev_w, prev_r    coded ITI and outcome of the last trial
      tb                trend covariate of the current trial
      last_trial (3,)   trial number at which each option was last chosen (0 = never, i.e. before
                        trial 1)
      n                 current trial number
      Ns (3,)           run counts (m12): consecutive choices of each option up to the last choice
                        (0 for every option but the one chosen last)
      u_prev            u = 1 / (1 + n) of the option chosen last, read before it was counted (m12)"""
    return dict(q=np.full((S, 3), Q_INIT), C=np.full((S, 3), INIT), prev=-1, prev_w=0.0, prev_r=0.0, tb=0.0,
                last_trial=np.zeros(3), n=1.0, Ns=np.zeros(3), u_prev=0.0)


def _gain(spec, p, st):
    """Gain g_t on the choice-bias weight.

    The trend exp(bias_slope * tb) (m09) times the ITI gain exp(gamma * w_{t-1}) (ITI form "bias",
    m10)."""
    g = np.ones_like(next(iter(p.values())))
    if spec["bias_trend"]:
        g = g * np.exp(p["bias_slope"] * st["tb"])
    if spec["iti"] == "bias":
        g = g * np.exp(p["gamma"] * st["prev_w"])
    return g


def _decision_values(spec, p, st, scored):
    """Decision values (S, 3) of the three options for the coming choice, from the current state.

    The value part depends on the ITI form; the novelty bonus and the choice-bias term act on top.
    `scored` is False for a choice right after a missed trial, which gets no stay bias."""
    S = st["q"].shape[0]
    dv = np.zeros((S, 3))
    if spec["intercepts"]:
        dv += np.column_stack([p["c1"], p["c2"], np.zeros(S)])
    if spec["learn"]:
        # Inverse temperature, with the log-linear trend of m08 if switched on.
        beta = p["beta"] * np.exp(p["beta_slope"] * st["tb"]) if spec["beta_trend"] else p["beta"]
        if spec["iti"] == "compete":
            # m12: after a long wait a share pi = kappa * u * L of beta moves from the values to the wait
            # just taken, which belongs to the option chosen last (L = w_{t-1}, u of that option).
            pi = p["kappa"] * st["prev_w"] * st["u_prev"]
            dv += (beta * (1.0 - pi))[:, None] * st["q"]
            if st["prev"] >= 0:
                dv[:, st["prev"]] -= beta * pi
        else:
            dv += beta[:, None] * st["q"]
    if spec["novelty"]:
        # Constant bonus on the option(s) chosen furthest back (never chosen = before trial 1).
        is_c = st["last_trial"] == st["last_trial"].min()
        dv[:, is_c] += p["omega"][:, None]
    if spec["bias"] == "none":
        return dv
    g = _gain(spec, p, st)
    if spec["bias"] == "choice_kernel":
        dv += (g * p["beta_c"])[:, None] * st["C"]
    elif st["prev"] >= 0 and scored:                  # stay bias, not for a choice right after a miss
        dv[:, st["prev"]] += g * p["stick"]
    return dv


def _update(spec, p, st, c, r, w_t, scored):
    """Learn from trial t, in place: choice c (0-indexed), outcome r (+1 win / -1 loss), and w_t, the
    coded ITI that followed the trial ({-1, +1} centred, or {0, 1} long_only).

    `scored` is not used by the updates (a choice right after a miss still updates learning and the
    traces); it is kept so that the signature matches _decision_values. Order: choice kernel, value
    update (with the ITI in the target in m11), decay of unchosen values, the run counts of m12, then
    the record of the last choice."""
    if spec["bias"] == "choice_kernel":
        # Choice kernel: the trace moves toward 1 for the chosen option and toward 0 for the others.
        x = np.zeros(3); x[c] = 1.0
        st["C"] = st["C"] + p["alpha_c"][:, None] * (x[None, :] - st["C"])
    if spec["learn"]:
        # Value target y: the outcome, minus the cost of the wait that followed it in m11.
        y = r - p["cost"] * w_t if spec["iti"] == "target" else r
        # Learning rate: alpha after a win (or the only rate), alpha_r after a loss.
        lr = p["alpha"] if (spec["lr"] == "one" or r > 0.5) else p["alpha_r"]
        # The value update of the chosen option.
        st["q"][:, c] += lr * (y - st["q"][:, c])
        # Decay of the unchosen values toward the neutral value.
        if spec["decay"]:
            for k in range(3):
                if k != c:
                    st["q"][:, k] = DECAY_TARGET + p["decay"] * (st["q"][:, k] - DECAY_TARGET)
    if spec["iti"] == "compete":
        # m12: u = 1 / (1 + n_c) of the option just chosen, read before it is counted; then its run
        # count goes up by 1 and every other option's count is reset to 0.
        st["u_prev"] = 1.0 / (1.0 + st["Ns"][c])
        nc = st["Ns"][c] + 1.0; st["Ns"][:] = 0.0; st["Ns"][c] = nc
    # Remember this trial for the next decision (novelty, stay bias, ITI terms).
    st["last_trial"][c] = st["n"]
    st["prev"] = c; st["prev_w"] = w_t; st["prev_r"] = r


def loglik(model: str, theta, choices, rewards, itis, return_trialwise: bool = False, return_probs: bool = False, trials=None):
    """Log-likelihood of one participant's responded choices.

    theta (S, P): S parameter vectors, with columns in the order of PARAMS[model].
    choices (T,): 0-indexed.
    rewards: as stored (+1 win / -1 loss); reward_code maps them to +1 / -1 (any value above 0 is a
    win).
    itis: in seconds, the ITI that FOLLOWED each trial.
    trials: the trial numbers of the responded choices (None scores every choice and uses the
    responded-trial count for the trend and the novelty bonus).
    Returns the total log-likelihood (S,); with return_trialwise, (S, T) trial-wise values (0 for
    unscored choices); with return_probs, (S, T, 3) choice probabilities on every trial (this takes
    precedence)."""
    spec = parse(model)
    theta = np.atleast_2d(np.asarray(theta, dtype=float)); S = theta.shape[0]
    # p maps each parameter name to its (S,) column of theta.
    p = {n: theta[:, i] for i, n in enumerate(PARAMS[model])}
    T = len(choices); w = coded_itis(spec, itis); r = reward_code(rewards)
    sc = scored_mask(trials) if trials is not None else np.ones(T, dtype=bool)
    # Trend covariate, and the trial number of each choice (or the responded-trial count).
    tb = beta_trend_covariate(trials, T); ntr = np.asarray(trials, dtype=float) if trials is not None else np.arange(T, dtype=float) + 1.0
    st = _init_state(S)
    out = np.zeros((S, T, 3)) if return_probs else (np.zeros((S, T)) if return_trialwise else np.zeros(S))
    # For each trial: decision values, the softmax log-probability of the actual choice (if scored),
    # then learning.
    for t in range(T):
        c = int(choices[t]); st["tb"] = tb[t]; st["n"] = ntr[t]
        dv = _decision_values(spec, p, st, bool(sc[t]))
        lse = logsumexp(dv, axis=1)
        if return_probs:
            out[:, t, :] = np.exp(dv - lse[:, None])
        elif sc[t]:
            if return_trialwise:
                out[:, t] = dv[:, c] - lse
            else:
                out += dv[:, c] - lse
        _update(spec, p, st, c, r[t], w[t], bool(sc[t]))
    return out


def simulate_choices(model: str, theta1d, probs: np.ndarray, itis: np.ndarray, rng: np.random.Generator, trials=None):
    """Play one participant's environment with a synthetic agent, using the same engine as loglik.

    theta1d: one parameter vector in the order of PARAMS[model]. probs (T, 3): the reward probability
    of each option on each trial. itis (T,): the ITIs in seconds. Returns (choices 0-indexed, rewards
    +1 / -1 as stored in the data). With `trials`, a choice that follows one of the participant's
    missed trials gets no stay bias, as in the likelihood."""
    spec = parse(model)
    theta = np.asarray(theta1d, dtype=float)[None, :]
    p = {n: theta[:, i] for i, n in enumerate(PARAMS[model])}
    T = len(itis); w = coded_itis(spec, itis)
    sc = scored_mask(trials) if trials is not None else np.ones(T, dtype=bool)
    tb = beta_trend_covariate(trials, T); ntr = np.asarray(trials, dtype=float) if trials is not None else np.arange(T, dtype=float) + 1.0
    st = _init_state(1); choices = np.zeros(T, int); rewards = np.zeros(T)
    for t in range(T):
        st["tb"] = tb[t]; st["n"] = ntr[t]
        # Softmax choice probabilities; draw the choice, then the outcome from that option's reward
        # probability on this trial.
        dv = _decision_values(spec, p, st, bool(sc[t]))[0]
        pr = np.exp(dv - dv.max()); pr /= pr.sum()
        c = int(rng.choice(3, p=pr)); win = rng.random() < probs[t, c]
        choices[t] = c; rewards[t] = 1.0 if win else -1.0
        _update(spec, p, st, c, 1.0 if win else -1.0, w[t], bool(sc[t]))
    return choices, rewards
