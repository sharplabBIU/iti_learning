"""Generative simulators (posterior predictive checks, model recovery) for every model that
likelihoods.py can evaluate (same engine, so the simulator cannot drift from the likelihood).

A posterior predictive check (PPC) asks whether a fitted model, run as an agent, reproduces the
behaviour seen in the data, such as more switching after a 4 s than after a 1 s inter-trial
interval (ITI). Model recovery fits models to such simulated data to check that the generating
model is identified.

simulate_dataset(model, thetas, subjects, rng) plays each participant's own task environment
(their reward-probability trajectories p1..p3 and their ITI sequence) with a synthetic agent whose
parameters are that participant's fitted values, and returns a tidy trial table in the same format
as data/trials.csv (so every behavioural function in behavior.py applies unchanged).

`subjects` is the output of data.subjects_for_fitting, so only the participant's responded trials
are replayed, under their original trial numbers. Every simulated trial has a response
(valid = True), and rt is left empty. Nothing is read from or written to disk here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .likelihoods import simulate_choices


def simulate_subject(model: str, theta1d, probs: np.ndarray, itis: np.ndarray, rng: np.random.Generator, trials=None):
    """Simulate one participant (likelihoods.simulate_choices).

    Returns (choices 0-indexed, rewards +1 / -1).

    theta1d: the participant's parameter vector; probs: reward probabilities (trials x 3);
    itis: ITIs in seconds; trials: actual trial numbers, so that a choice following a missed trial
    is treated as in the likelihood.
    """
    return simulate_choices(model, theta1d, probs, itis, rng, trials=trials)


def simulate_dataset(model: str, thetas: np.ndarray, subjects: list[dict], rng: np.random.Generator,
                     rep: int = 0) -> pd.DataFrame:
    """thetas: (N, P) parameter vectors aligned with `subjects`. Returns a tidy trial table.

    N = number of participants, P = number of model parameters. rep labels the replicate
    (column "rep") when several simulated data sets are stacked.
    """
    # One simulated session per participant, in the layout of data/trials.csv (choices back to 1-3)
    frames = []
    for i, s in enumerate(subjects):
        ch, rw = simulate_subject(model, thetas[i], s["probs"], s["itis"], rng, trials=s.get("trial"))
        frames.append(pd.DataFrame({
            "subject_id": s["subject_id"], "trial": s["trial"], "choice": ch + 1, "rt": np.nan,
            "win": (rw > 0).astype(int), "reward": rw, "iti": s["itis"],
            "p1": s["probs"][:, 0], "p2": s["probs"][:, 1], "p3": s["probs"][:, 2], "rep": rep}))
    # Three of the derived columns of data.tidy_trials: valid, best_option and chose_best
    t = pd.concat(frames, ignore_index=True)
    t["valid"] = True
    t["best_option"] = t[["p1", "p2", "p3"]].to_numpy().argmax(axis=1) + 1
    t["chose_best"] = (t["choice"] == t["best_option"]).astype(int)
    return t
