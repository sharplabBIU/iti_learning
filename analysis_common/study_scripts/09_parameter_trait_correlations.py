"""09 - Do the parameters of the winning computational model relate to impulsivity and anhedonia? (Methods, Individual
differences; Results, the parameter-trait correlations.) Run from the Study 3 folder; it pools Studies 3 and 4.

The winning model is the ITI-competes-with-value model (m12_iti_compete). Each participant's posterior means, from the fit
of their own study (06_fit_models_nuts.py), are correlated with their impulsivity (BIS-11 8-item subset) and anhedonia
(SHAPS) totals by Spearman correlations over the participants of Studies 3 and 4 together, with Benjamini-Hochberg
false-discovery-rate (FDR) correction over the full set of tests (every parameter x both questionnaires). The same
correlations within each study are written alongside.

Inputs: data/study_{3,4}/data/participants.csv; data/study_{3,4}/analysis/results/06_model_fits/m12_iti_compete_subjects.csv
Outputs (results/09_parameter_traits/ in the Study 3 folder): parameter_trait_correlations.csv (pooled, with FDR-corrected
p) and parameter_trait_correlations_by_study.csv
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                     # repository root, so analysis_common can be imported
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from analysis_common import config as C
from analysis_common.likelihoods import PARAMS

ROOT = HERE.parents[1]
WINNER = "m12_iti_compete"
STUDIES = ("study_3", "study_4")
qs = C.STUDIES["study_3"]["questionnaires"]                  # the same two questionnaires in both studies
out = ROOT / "study_3/analysis/results/09_parameter_traits"; out.mkdir(parents=True, exist_ok=True)

# Participant posterior means of the winner joined with the questionnaire totals, one block per study.
frames = []
for s in STUDIES:
    fit = pd.read_csv(ROOT / s / "analysis/results/06_model_fits" / f"{WINNER}_subjects.csv")
    P = pd.read_csv(ROOT / s / "data/participants.csv")
    fit["subject_id"] = fit["subject_id"].astype(str); P["subject_id"] = P["subject_id"].astype(str)
    frames.append(fit.merge(P[["subject_id"] + [q["col"] for q in qs.values()]], on="subject_id", how="left").assign(study=s))
D = pd.concat(frames, ignore_index=True)


def spearman_table(d):
    rows = []
    for p in PARAMS[WINNER]:
        for qk, q in qs.items():
            dd = d[[f"{p}_mean", q["col"]]].dropna()
            rho, pv = stats.spearmanr(dd.iloc[:, 0], dd.iloc[:, 1])
            rows.append(dict(parameter=p, questionnaire=qk, n=len(dd), df=len(dd) - 2, spearman_rho=float(rho), p=float(pv)))
    return pd.DataFrame(rows)


pooled = spearman_table(D)
pooled["p_fdr"] = multipletests(pooled["p"], method="fdr_bh")[1]
pooled.to_csv(out / "parameter_trait_correlations.csv", index=False)
pd.concat([spearman_table(D[D.study == s]).assign(study=s) for s in STUDIES], ignore_index=True).to_csv(
    out / "parameter_trait_correlations_by_study.csv", index=False)
print(f"{C.MODEL_SHORT[WINNER]}: Studies 3 and 4 pooled (n = {D.subject_id.nunique()}), Spearman correlations, FDR over {len(pooled)} tests:")
print(pooled.round(4).to_string(index=False))
print("wrote", out)
