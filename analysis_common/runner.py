"""Small helper used by every analysis script in data/study_N/analysis/.

Each analysis script lives in data/study_N/analysis/ and starts by creating a Runner. The Runner
works out which study the script belongs to, loads the main sample (sessions that reached trial 90),
gives the folder to write results to, and sets common plot defaults. The module also holds small
shared helpers: p-value formatting (fmt_p), a two-bar plot of a paired inter-trial interval (ITI)
test (paired_bar) and a Markdown table writer (md_table).

    R = Runner(__file__)          # infers the study from the folder name (study_1 ... study_4)
    trials, P = R.load()          # the main sample
    out = R.out("02_switching")   # results/02_switching/
    R.save_json(out, "main_effect", d)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Put the repository root on the import path, so any script folder can import `analysis_common`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_common import config as C                       # noqa: E402
from analysis_common.data import load_study, subjects_for_fitting  # noqa: E402

BLUE, RED, GREY, GREEN, ORANGE = "#4C72B0", "#C44E52", "#8c8c8c", "#1b9e77", "#F58518"   # plot colours


def fmt_p(p) -> str:
    """Format a p value for figures: "p = n/a" if missing, else scientific below 1e-4, else 3-4 decimals."""
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p < 1e-4:
        return f"p = {p:.1e}"
    return f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}"


class Runner:
    """Study, sample and output folders for one analysis script (usage: see the module docstring)."""

    def __init__(self, script_file: str):
        """Set up a run of the script `script_file` (pass the script's __file__). Creates the results
        folder and sets plot defaults."""
        # The study is the folder two levels above the script: data/<study>/analysis/script.py
        self.script = Path(script_file).resolve()
        self.key = self.script.parents[1].name
        if self.key not in C.STUDIES:
            raise RuntimeError(f"cannot infer study from {self.script}")
        self.study = C.STUDIES[self.key]
        self.paths = C.study_paths(self.key)
        self.results_dir = self.paths["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Plot without a display ("Agg" backend), with shared font size, axes and resolution
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                             "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
        self.plt = plt

    @property
    def label(self):
        """The study's name in the text, e.g. "Study 3"."""
        return self.study["label"]

    def load(self, exclusion: str = C.EXCLUSION_MAIN):
        """(trials, participants) of this study, by default the main sample."""
        trials, P, _ = load_study(self.key, exclusion)
        return trials, P

    def subjects(self):
        """(per-participant arrays for model fitting, participants) of the main sample; see
        data.subjects_for_fitting."""
        trials, P = self.load()
        return subjects_for_fitting(trials), P

    def out(self, section: str) -> Path:
        """The output folder results/<section>/, created if needed."""
        d = self.results_dir / section
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def save_json(out_dir: Path, name: str, obj) -> Path:
        """Write obj to out_dir/<name>.json; numpy and pandas values are converted (_json_default)."""
        path = Path(out_dir) / f"{name}.json"
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        return path

    def banner(self, title: str):
        """Print a header with the study and the analysis title (marks the start of a run in the log)."""
        print("=" * 88); print(f"{self.label} ({self.study['subtitle']}) -- {title}"); print("=" * 88, flush=True)


def _json_default(o):
    """Convert what json cannot write (numpy values, pandas tables, paths) to plain Python, else to str."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_dict()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def paired_bar(ax, res: dict, labels=("1 s", "4 s"), colors=(BLUE, RED), ylabel="p(switch)", title=None):
    """Two-bar plot of mean +/- SEM with the paired-test annotation.

    res is the summary dict of behavior.paired_iti_test (short = 1 s, long = 4 s by default). The
    annotation gives n, the paired t statistic with its degrees of freedom and the two-sided p value.
    SEM = standard error of the mean.
    """
    m = [res["mean_short"], res["mean_long"]]
    e = [res.get("sem_short", 0), res.get("sem_long", 0)]
    ax.bar([0, 1], m, yerr=e, color=colors, width=0.6, capsize=6, edgecolor="black", linewidth=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.97, f"n = {res['n']}\nt({res['df']}) = {res['t']:.2f}\n{fmt_p(res['p_two_sided'])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    # y range: 35% headroom above the tallest bar + error bar, capped at 1 and at least 0.05
    ax.set_ylim(0, max(0.05, min(1.0, max(np.array(m) + np.array(e)) * 1.35)))


def md_table(df: pd.DataFrame, floatfmt: int = 4) -> str:
    """Minimal GitHub-markdown table (no tabulate dependency).

    Floats are written with `floatfmt` decimals and trailing zeros removed (general format for very
    small or large values; missing values as empty cells). "|" and line breaks in text are escaped.
    """
    df = df.copy()
    cols = list(df.columns)

    def cell(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return ""
            if abs(v) < 1e-3 or abs(v) >= 1e5:
                return f"{v:.{floatfmt}g}"
            txt = f"{v:.{floatfmt}f}"
            return txt.rstrip("0").rstrip(".") if "." in txt else txt       # never strip the zeros of a whole number ("100")
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)
