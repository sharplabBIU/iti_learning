# The external dataset: Tomov et al. (2020)

Is a switch away from the previous choice more likely after a longer inter-trial interval (ITI) in a public fMRI bandit
dataset in which the ITI was jittered for imaging and never analysed as a task parameter? (Results [59]; Methods, External
dataset; Fig. 6a.)

Tomov, Truong, Hundia & Gershman (2020, Nature Communications): 31 participants, two scanned sessions of a two-armed bandit in
which each ten-trial run presented a new pair of arms; the blank interval after feedback was scheduled uniformly at random
between 4.8 and 7.3 s. The data come from the authors' repository, github.com/tomov/Exploration-Data-Analysis (`data/*.csv`),
which has no licence file, so they are analysed here but not redistributed: `reproduce.sh` downloads them into
`data/tomov2020/subjects/` (one file per participant and session) when they are missing.

| script (`code/`) | what it does | output |
|---|---|---|
| `prepare_tomov2020.py` | one row per trial: option identity from the arm colour, the scheduled ITI before each choice (the realised ITI depended on the participant's own response time and is not used), rewards binarised within each run | `data/tomov2020/tidy.csv` (not shared) |
| `tomov_iti_glmm.py` | the trial table (valid choice on the trial and on the previous trial of the same run; no participant excluded) and the mixed model of our own studies, `is_shift ~ iti_c + trial_c + (1 + iti_c + trial_c \| participant)`, with the ITI centred on each participant's median and the trend over the ten trials of each pair of arms | `results/tomov2020_iti_glmm_summary.json` / `.txt` (the reported slope), `results/tomov2020_iti_glmm_fixef.csv`, `_subject.csv`, `.json` |
| `plot_fig6a_tomov.py` | Fig. 6a: switching by quintile of the ITI and over the ten trials of a pair of arms, with the model's curves | `results/fig6a_tomov.png` (pasted into Fig. 6 by `figures/make_generalization_figure.py`) |
