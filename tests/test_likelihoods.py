"""Tests of the likelihood engine (analysis_common/likelihoods.py) against independent loop-by-loop implementations.

Coding: outcomes +1 / -1, values start at 0 and unchosen values decay toward 0, and the choice-kernel traces start at
0.5. The inter-trial interval (ITI) is centred in the tests of the m10 and m11 mechanisms (w = -1 after a 1 s ITI,
+1 after 4 s); m12 uses w = 1 after a 4 s ITI and 0 after 1 s, as in the paper.

What is tested:
  1. Every mechanism matches an independent, loop-by-loop implementation (_reference), on real Study 3 participants,
     including participants with missed trials (a choice right after a miss is not scored and gets no stay-type
     bias).
  2. Every model reduces to the model it extends when its added parameters take their null values.
  3. The PyTensor likelihood used for NUTS (No-U-Turn sampler) fits equals the numpy one, and the simulator runs.
  4. The choice right after a missed trial is not scored.
  5. m12 (ITI competes with value) matches its own independent implementation (_reference_compete), reduces to
     m08 when kappa = 0, and the PyTensor likelihood equals numpy for the stage-D models.
Because the stage choices of the sequence are made on the data, the tests register every combination the later
stages can take (each choice-bias term x each trend x each of the m10 and m11 ITI mechanisms).

Data: five Study 3 participants from the main sample (sessions that reached trial 90): three with missed trials
and two without.

Run: python3 tests/test_likelihoods.py   (from the repository root; PYTENSOR_FLAGS=cxx= avoids the C compiler)
"""
import sys
from pathlib import Path
# Put the repository root on the import path for analysis_common.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import itertools
import numpy as np
import analysis_common.likelihoods as L
from analysis_common.likelihoods import loglik, PARAMS, SUPPORT, register, parse, simulate_choices
from analysis_common.data import load_study, subjects_for_fitting


def draw_theta(model, S, rng):
    """S random parameter vectors for a model, one column per parameter in PARAMS order.

    Unit-interval parameters are uniform on (0.05, 0.95), positive ones uniform on (0.3, 4.0), and real-valued ones
    normal with mean 0 and SD 0.8."""
    cols = []
    for n in PARAMS[model]:
        sup = SUPPORT[n]
        cols.append(rng.uniform(0.05, 0.95, S) if sup == "unit" else rng.uniform(0.3, 4.0, S) if sup == "pos" else rng.normal(0, 0.8, S))
    return np.column_stack(cols)


def _reference(spec, th, choices, rewards, itis, trials):
    """Independent implementation of the summed log-likelihood: one participant, one parameter vector (dict th).

    Coding: r +1 win / -1 loss; w -1 after a 1 s ITI / +1 after a 4 s ITI; choice-kernel traces
    1 chosen / 0 not. Values start at 0 and unchosen values decay toward 0; traces start at 0.5. itis[t] is the ITI
    that followed trial t, so it enters the value update of trial t and the choice-bias gain of trial t + 1.
    spec is the parsed model specification (likelihoods.parse)."""
    # Q = option values; C = choice-kernel traces; prev, prev_w = previous choice and its ITI code.
    Q = np.full(3, 0.0); C = np.full(3, 0.5); prev = None; prev_w = 0.0; last_trial = None; ll = 0.0
    last_chosen = {0: 0.0, 1: 0.0, 2: 0.0}                     # trial each option was last chosen (0 = never)
    for c, r_pm, iti, tr in zip(choices, rewards, itis, trials):
        r = 1.0 if r_pm > 0 else -1.0
        consecutive = last_trial is None or tr == last_trial + 1   # False right after a missed trial
        tb = (tr - 50.5) / 50.0; w = -1.0 if iti == 1 else 1.0   # tb = trend covariate; w = centred ITI code
        # Decision values: intercepts, inverse temperature (beta, with optional trend) times values.
        v = np.zeros(3)
        if spec["intercepts"]:
            v += [th["c1"], th["c2"], 0.0]
        if spec["learn"]:
            v += th["beta"] * (np.exp(th["beta_slope"] * tb) if spec["beta_trend"] else 1.0) * Q
        # gain multiplies the choice-bias weight: trend over the session and/or the ITI just elapsed (prev_w).
        gain = 1.0
        if spec["bias_trend"]:
            gain *= np.exp(th["bias_slope"] * tb)
        if spec["iti"] == "bias":
            gain *= np.exp(th["gamma"] * prev_w)
        if spec["novelty"]:                                    # the option(s) whose last choice is furthest back
            # a constant bonus omega
            oldest = min(last_chosen.values())
            for a in range(3):
                if last_chosen[a] == oldest:
                    v[a] += th["omega"]
        # Choice kernel: a bonus on every option, scored on every trial. Stay bias (stick): a bonus on the previous
        # choice, only when this trial follows it directly.
        if spec["bias"] == "choice_kernel":
            v += gain * th["beta_c"] * C
        elif spec["bias"] != "none" and prev is not None and consecutive:
            v[prev] += gain * th["stick"]
        # Log softmax probability of the choice made; a choice right after a miss is not scored.
        if consecutive:
            ll += v[c] - np.log(np.exp(v).sum())
        # Updates.
        x = np.zeros(3); x[c] = 1.0
        if spec["bias"] == "choice_kernel":
            C += th["alpha_c"] * (x - C)
        # Value update: target y = r, or r - cost * w for the ITI as learning target; alpha after a win (or the only
        # rate), alpha_r after a loss; unchosen values are multiplied by decay, i.e. decay toward 0.
        if spec["learn"]:
            y = r
            if spec["iti"] == "target":
                y = r - th["cost"] * w
            lr = th["alpha"] if (spec["lr"] == "one" or r == 1.0) else th["alpha_r"]
            Q[c] += lr * (y - Q[c])
            if spec["decay"]:
                for k in range(3):
                    if k != c:
                        Q[k] = th["decay"] * Q[k]
        prev = c; prev_w = w; last_trial = tr; last_chosen[c] = float(tr)
    return ll


def _reference_compete(th, choices, rewards, itis, trials):
    """Independent implementation of m12, ITI competes with value, one participant, one parameter
    vector (dict th).

    The m08 base (two learning rates, decay to 0, choice kernel, constant novelty bonus, trend in beta) where a long
    ITI moves a share pi = kappa * u of beta from the values to the wait just taken, a cost of 1 on the option chosen
    last. u = 1 / (1 + n) of that option, with n the number of consecutive choices of it just before that choice."""
    # Q = values; C = choice-kernel traces; run = consecutive choices of the last option; u_last and long_last = u of
    # the previous choice and whether the ITI after it was long; prev = the previous choice.
    Q = np.zeros(3); C = np.full(3, 0.5); run = 0; u_last = 0.0; long_last = 0.0; prev = None
    last_chosen = {0: 0.0, 1: 0.0, 2: 0.0}; last_trial = None; ll = 0.0
    for c, r_pm, iti, tr in zip(choices, rewards, itis, trials):
        r = 1.0 if r_pm > 0 else -1.0; long_ = 1.0 if iti == 4 else 0.0
        beta = th["beta"] * np.exp(th["beta_slope"] * (tr - 50.5) / 50.0)
        share = th["kappa"] * long_last * u_last   # pi: nonzero only after a long ITI
        v = beta * (1.0 - share) * Q + th["beta_c"] * C
        if prev is not None:
            v[prev] -= beta * share
        oldest = min(last_chosen.values())
        for a in range(3):
            if last_chosen[a] == oldest:
                v[a] += th["omega"]
        if last_trial is None or tr == last_trial + 1:
            ll += v[c] - np.log(np.exp(v).sum())
        onehot = np.zeros(3); onehot[c] = 1.0; C += th["alpha_c"] * (onehot - C)
        Q[c] += (th["alpha"] if r == 1.0 else th["alpha_r"]) * (r - Q[c])
        for k in range(3):
            if k != c:
                Q[k] *= th["decay"]
        n_before = run if c == prev else 0         # consecutive earlier choices of c, before this one is counted
        u_last = 1.0 / (1.0 + n_before); run = n_before + 1
        long_last = long_; prev = c
        last_trial = tr; last_chosen[c] = float(tr)
    return ll


def test_compete():
    """m12: the engine equals its independent implementation; kappa = 0 is m08; PyTensor = numpy."""
    from analysis_common import pymc_models as PM
    L.register_models(); subs = _subjects(); rng = np.random.default_rng(21); S = 3; m = "m12_iti_compete"
    for x in subs:
        th = draw_theta(m, S, rng)
        mine = loglik(m, th, x["choices"], x["rewards"], x["itis"], trials=x["trial"])
        ref = [_reference_compete(dict(zip(PARAMS[m], th[i])), x["choices"], x["rewards"], x["itis"], x["trial"]) for i in range(S)]
        assert np.allclose(mine, ref), (mine, ref)
        # At kappa = 0, m12 must equal m08 (whose parameters are the first columns of m12's).
        th0 = th.copy(); th0[:, PARAMS[m].index("kappa")] = 0.0
        assert np.allclose(loglik(m, th0, x["choices"], x["rewards"], x["itis"], trials=x["trial"]),
                           loglik("m08_trend_beta", th[:, :len(PARAMS["m08_trend_beta"])], x["choices"], x["rewards"], x["itis"], trials=x["trial"]))
    worst = max(PM.check_against_numpy(mm, subs, S=3, seed=7) for mm in L.STAGES[-1][1])   # last stage = stage D
    assert worst < 1e-6, worst
    print(f"  m12 matches its independent implementation and reduces to m08 at kappa = 0; PyTensor = numpy for the stage-D models (max {worst:.1e})")


def _all_combinations():
    """Register every model the sequence can contain and return their names.

    Names are t_<lr>_<decay 0/1>[n]_<bias>_<trend>_<iti>, with n marking the novelty bonus.
    Two kinds of combination are skipped: a trend in, or an ITI scaling of, a choice-bias weight that does not
    exist (bias "none"). The registered stage-A models are added by the callers."""
    names = []
    for lr, dec, nov in (("one", False, False), ("two", False, False), ("two", True, False), ("two", True, True)):
        for bias in L.BIASES:
            for trend in ("none", "beta", "bias"):
                if bias == "none" and trend == "bias":
                    continue
                for iti in ("none", "target", "bias"):
                    if bias == "none" and iti == "bias":
                        continue
                    kw = dict(lr=lr, decay=dec, novelty=nov, bias=bias, beta_trend=trend == "beta", bias_trend=trend == "bias", iti=iti)
                    names.append(register(f"t_{lr}_{int(dec)}{'n' if nov else ''}_{bias}_{trend}_{iti}", **kw))
    return names


def _subjects():
    """Five Study 3 participants from the main sample: the first three with missed trials (gaps in their trial
    numbers) and the first two without."""
    trials, P, _ = load_study("study_3", exclusion="min90")
    subs = subjects_for_fitting(trials)
    gaps = [x for x in subs if np.any(np.diff(x["trial"]) > 1)]
    return gaps[:3] + [x for x in subs if not np.any(np.diff(x["trial"]) > 1)][:2]


def test_reference():
    """1. The engine equals the independent implementation for every combination (and for the registered models)."""
    L.register_models(); combos = _all_combinations()
    subs = _subjects(); rng = np.random.default_rng(1); S = 3
    for m in list(L.STAGES[0][1]) + combos:
        spec = parse(m); th = draw_theta(m, S, rng)
        for x in subs:
            mine = loglik(m, th, x["choices"], x["rewards"], x["itis"], trials=x["trial"])
            ref = [_reference(spec, dict(zip(PARAMS[m], th[i])), x["choices"], x["rewards"], x["itis"], x["trial"]) for i in range(S)]
            assert np.allclose(mine, ref), (m, mine, ref)
    print(f"  {len(combos) + len(L.STAGES[0][1])} models match the independent implementation (participants with missed trials included)")


def test_reductions():
    """2. Each addition reduces to its parent at its null value."""
    # One participant with missed trials; a = choices, rewards, ITIs.
    L.register_models(); _all_combinations()
    x = _subjects()[0]; rng = np.random.default_rng(2); S = 4
    a = (x["choices"], x["rewards"], x["itis"])

    def same(child, parent, fill):
        """Assert that child and parent give the same log-likelihood.

        The parent's parameters are drawn at random. Each child parameter is copied from the parent parameter of
        the same name; otherwise fill gives either the name of a parent parameter to copy or a constant."""
        pc, pp = PARAMS[child], PARAMS[parent]
        thp = draw_theta(parent, S, rng)
        thc = np.column_stack([thp[:, pp.index(k)] if k in pp else (thp[:, pp.index(fill[k])] if isinstance(fill.get(k), str) else np.full(S, fill[k]))
                               for k in pc])
        assert np.allclose(loglik(child, thc, *a, trials=x["trial"]), loglik(parent, thp, *a, trials=x["trial"])), (child, parent, fill)
    # Stage A: equal learning rates give one rate; decay = 1 means no decay. Novelty bonus off at omega = 0.
    same("m03_rw_2lr", "m02_rw_1lr", dict(alpha_r="alpha"))
    same("m04_rw_2lr_decay", "m03_rw_2lr", dict(decay=1.0))
    same("t_two_1n_choice_kernel_none_none", "t_two_1_choice_kernel_none_none", dict(omega=0.0))
    same("t_two_1n_choice_kernel_beta_target", "t_two_1_choice_kernel_beta_target", dict(omega=0.0))
    # Each choice-bias term at zero weight equals no bias (beta_c has positive support, so a tiny positive value
    # stands in for 0); each trend at slope 0 and each ITI term at 0 equal the model without them.
    for bias, fill in (("stick", dict(stick=0.0)), ("choice_kernel", dict(alpha_c=0.3, beta_c=1e-300))):
        same(f"t_two_1_{bias}_none_none", "t_two_1_none_none_none", fill)
        same(f"t_two_1_{bias}_beta_none", f"t_two_1_{bias}_none_none", dict(beta_slope=0.0))
        same(f"t_two_1_{bias}_bias_none", f"t_two_1_{bias}_none_none", dict(bias_slope=0.0))
        for trend in ("beta", "bias"):
            base = f"t_two_1_{bias}_{trend}_none"
            same(f"t_two_1_{bias}_{trend}_target", base, dict(cost=0.0))
            same(f"t_two_1_{bias}_{trend}_bias", base, dict(gamma=0.0))
    print("  every addition reduces to its parent at its null value (incl. the novelty bonus at omega = 0)")


def test_pytensor_and_simulator():
    """3. PyTensor = numpy for every combination; the simulator runs."""
    from analysis_common import pymc_models as PM
    L.register_models(); combos = _all_combinations()
    subs = _subjects(); rng = np.random.default_rng(3); worst = 0.0
    # For every model: PyTensor vs numpy on the five participants (2 parameter draws), and one simulated session
    # in the first participant's environment.
    for m in list(L.STAGES[0][1]) + combos:
        worst = max(worst, PM.check_against_numpy(m, subs, S=2, seed=5))
        simulate_choices(m, draw_theta(m, 1, rng)[0], subs[0]["probs"], subs[0]["itis"], rng, trials=subs[0]["trial"])
    assert worst < 1e-6, worst
    print(f"  PyTensor = numpy for {len(combos) + len(L.STAGES[0][1])} models (max abs difference {worst:.1e}); the simulator runs")


def test_missed_trials():
    """The choice right after a miss is not scored: changing it leaves the stay-type likelihood of the next choices unchanged.

    What the code checks: in the trial-wise log-likelihood (constant stay bias model), the first choice after a miss
    contributes exactly 0."""
    L.register_models(); _all_combinations()
    x = [s for s in _subjects() if np.any(np.diff(s["trial"]) > 1)][0]
    i = int(np.where(np.diff(x["trial"]) > 1)[0][0]) + 1              # first choice after a miss
    th = draw_theta("t_two_1_stick_none_none", 1, np.random.default_rng(4))
    tw = loglik("t_two_1_stick_none_none", th, x["choices"], x["rewards"], x["itis"], trials=x["trial"], return_trialwise=True)[0]
    assert tw[i] == 0.0, "a choice right after a miss must not be scored"
    print("  a choice made right after a missed trial is not scored")


# Run all tests in order; any failure raises an AssertionError.
if __name__ == "__main__":
    test_reference()
    test_reductions()
    test_pytensor_and_simulator()
    test_missed_trials()
    test_compete()
    print("ALL LIKELIHOOD TESTS PASSED")
