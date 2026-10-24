"""Project-wide configuration: paths, study definitions, the inclusion rule and model labels.

The other modules and the analysis scripts read their settings from here, so this file is the place
to look up
  * where the data live (PROJECT_ROOT, DATA_ROOT, and PRIVATE_ROOT for the raw exports, which are
    not part of the repository);
  * which participants are analysed (the main sample);
  * the names under which the computational models appear in the text and on figure axes
    (MODEL_SHORT, MODEL_ABBR);
  * the definition of each study (STUDIES): its raw file, participant-ID column, questionnaires and
    task variant;
  * the standard folder layout of one study (study_paths).

Key definitions
  * ITI = inter-trial interval: 1 s or 4 s in Studies 1-4.
  * Study 1 = unfilled ITI (n = 96). Study 2 = ITI filled with a random-dot-motion task (n = 96).
    Study 3 = preregistered online replication (n = 439).
    Study 4 = the task of Study 3 with a new ITI sequence, built against the hypothesis (n = 117).
  * Main sample = sessions that reached trial 90 of the 100.

Importing the module only defines constants, a small dict class and one helper function; it reads
and writes no files.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]        # the repository root (this package lives directly inside it)
DATA_ROOT = PROJECT_ROOT / "data"                            # data/study_N/{data,analysis}
# The raw exports hold Prolific IDs and demographics and are kept outside the repository, with the map
# from Prolific IDs to the pseudonyms of the shared tables. Only 00_prepare_data.py reads them; every
# other script starts from the tidy tables in data/study_N/data/. Set ITI_PRIVATE_DATA to use another
# folder.
PRIVATE_ROOT = Path(os.environ.get("ITI_PRIVATE_DATA", PROJECT_ROOT.parent / "private_data"))
ID_MAP_CSV = PRIVATE_ROOT / "participant_id_map.csv"

# ---------------------------------------------------------------------------
# Inclusion rule
# ---------------------------------------------------------------------------
# No exclusion criterion was preregistered. Every analysis uses the participants whose session reached
# trial 90 of the 100 (rule "min90"); nobody is dropped for their behaviour, and analyses that need a
# lagged pair of trials simply skip participants who contribute no such pair. Rule "none" (everyone with
# trial data) is used only where the sample does not matter (the reward trajectories of Fig. 1b).
EXCLUSION_MAIN = "min90"
MIN_TRIALS_REACHED = 90          # main sample: the session reached trial 90 (90% of the task)
N_TASK_TRIALS = 100              # trials in the main task, every study

# ---------------------------------------------------------------------------
# Computational models (analysis_common/likelihoods.py)
# ---------------------------------------------------------------------------
# MODEL_SHORT holds the name used in the text and MODEL_ABBR the short label used on figure axes; the
# internal codes are never shown to readers. Each model is named by what it adds to the model it builds
# on (Methods; Supplementary Table 1). m01 = intercepts only (fixed preferences); m02 = Rescorla-Wagner
# learning with one learning rate; m03 = separate learning rates after wins and losses; m04 = plus decay
# of unchosen values; m05 = constant stay bias; m06 = choice kernel; m07 = novelty bonus; m08/m09 =
# trends over the session; m10, m11, m12 = the three ITI models. m12 (ITI competes with value) is the
# winning model.
MODEL_SHORT = {
    "m01_intercepts": "Intercepts only", "m02_rw_1lr": "Rescorla-Wagner, one learning rate",
    "m03_rw_2lr": "Separate learning rates for wins and losses", "m04_rw_2lr_decay": "Value decay",
    "m05_stick": "Constant stay bias", "m06_choice_kernel": "Choice kernel", "m07_novelty": "Novelty bonus",
    "m08_trend_beta": "Exploration trend", "m09_trend_bias": "Choice-bias trend",
    "m10_iti_bias": "ITI on choice bias", "m11_iti_target": "ITI as learning target",
    "m12_iti_compete": "ITI competes with value"}
MODEL_ABBR = {
    "m01_intercepts": "Intercepts", "m02_rw_1lr": "RW", "m03_rw_2lr": "RW ±α", "m04_rw_2lr_decay": "Decay",
    "m05_stick": "Stay bias", "m06_choice_kernel": "Choice kernel",
    "m07_novelty": "Novelty", "m08_trend_beta": "Explore trend", "m09_trend_bias": "Bias trend",
    "m10_iti_bias": "ITI → choice bias", "m11_iti_target": "ITI → learning target",
    "m12_iti_compete": "ITI competes with value"}


class _LabelDict(dict):
    """A dict that returns the key itself when a model has no label, so it keeps its code name."""
    def __missing__(self, key):
        return key


MODEL_SHORT = _LabelDict(MODEL_SHORT)
MODEL_ABBR = _LabelDict(MODEL_ABBR)

# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
# One entry per study. Fields:
#   key, label, subtitle    identifier, name in the text, and a one-line description of the design.
#   dir                     the study's folder (data/study_N), laid out as in study_paths() below.
#   raw_kind, raw_file      format ("csv" or "json") and file name of the raw export in
#                           PRIVATE_ROOT/raw/study_N/ (read only by 00_prepare_data.py).
#   id_col                  participant-ID column in the raw export (renamed to subject_id).
#   questionnaires          trait scores analysed in the paper: column name, label, and the
#                           preregistered direction of the correlation with the ITI effect (+1 or -1),
#                           used as the one-sided direction in behavior.correlate(). Studies 1 and 2
#                           collected other questionnaires, which are not analysed.
#   has_rdm                 the study has random-dot-motion (RDM) task data (Study 2 only).
#   rt_floor_ms             optional: choices faster than this many ms are treated as missed
#                           (see data.load_study).
# BIS-11 = Barratt Impulsiveness Scale; SHAPS = Snaith-Hamilton Pleasure Scale.
_TRAITS = {"bis": dict(col="bis_total", label="BIS-11 impulsivity (8 items)", prereg_direction=+1),
           "shaps": dict(col="pleasure_total", label="SHAPS anhedonia", prereg_direction=-1)}
STUDIES = {
    "study_1": dict(
        key="study_1", label="Study 1", subtitle="unfilled ITI (fixation cross)",
        dir=DATA_ROOT / "study_1", raw_kind="csv", raw_file="merged_data_with_surveys.csv", id_col="prolific_id",
        questionnaires={}, has_rdm=False,
    ),
    "study_2": dict(
        key="study_2", label="Study 2", subtitle="ITI filled with a random-dot-motion task",
        dir=DATA_ROOT / "study_2", raw_kind="csv", raw_file="merged_data_with_surveys_new.csv", id_col="subject_id",
        questionnaires={}, has_rdm=True,
        rt_floor_ms=250,   # choices faster than 250 ms are late Q+T+P answers to the dot-motion display that
                           # land on the choice screen (Q+T+P = the Q, T and P keys pressed together, the
                           # dot-motion "down" response; the same three keys choose the options); they are
                           # treated as missed trials (Methods; Fig. 2d-f)
    ),
    "study_3": dict(
        key="study_3", label="Study 3", subtitle="preregistered replication and individual differences (unfilled ITI)",
        dir=DATA_ROOT / "study_3", raw_kind="json", raw_file="trials_combined.json", id_col="subject_id",
        questionnaires=_TRAITS, has_rdm=False,
    ),
    "study_4": dict(
        key="study_4", label="Study 4", subtitle="the task of Study 3 with a new ITI sequence (unfilled ITI)",
        dir=DATA_ROOT / "study_4", raw_kind="json", raw_file="trials_combined.json", id_col="subject_id",
        questionnaires=_TRAITS, has_rdm=False,
    ),
}


def study_paths(key: str) -> dict:
    """Return the standard folders and files of one study (key such as "study_3").

    data/study_N/data/ holds the tidy tables written by data.prepare_study() (trials.csv,
    participants.csv, exclusion_flow.csv); data/study_N/analysis/ holds the scripts and their results/
    folder. raw_dir is the study's folder of raw exports outside the repository.
    """
    d = STUDIES[key]["dir"]
    return dict(
        study_dir=d,
        data_dir=d / "data",
        raw_dir=PRIVATE_ROOT / "raw" / key,
        analysis_dir=d / "analysis",
        results_dir=d / "analysis" / "results",
        trials_csv=d / "data" / "trials.csv",
        participants_csv=d / "data" / "participants.csv",
        flow_csv=d / "data" / "exclusion_flow.csv",
    )
