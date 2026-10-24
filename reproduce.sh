#!/usr/bin/env bash
# Reproduce every analysis, figure and table of the main text and the Supplementary Information from the shared data.
#
# Usage:  ./reproduce.sh [--skip-recovery] [--serial]
#   (default)        everything, including the parameter and model recovery of Fig. 4d,e
#   --skip-recovery  everything except the recovery (about a day of computing on its own); the recovery results already
#                    in data/study_3/analysis/results/07_recovery/ are kept and used by Fig. 4
#   --serial         run the lanes below one after another instead of side by side (for machines with fewer than about
#                    16 cores or 24 GB of memory; slower)
#
# The analyses start from the tidy tables in data/study_N/data/ (participant IDs pseudonymized). The raw exports hold
# Prolific IDs and demographics and are not shared, so the step that built the tables (00_prepare_data.py) is not run here.
# The Tomov et al. (2020) data are not redistributed: they are downloaded from the authors' repository if missing.
#
# Lanes (run side by side unless --serial; the time is that of an 18-core Apple laptop):
#   behaviour   Studies 1-4: learning, ITI effects (paired tests and mixed models), each half of the session, earlier
#               ITIs, the dot-motion task, the fixed ITI sequence, response times, individual differences, the HMM
#               check of the exploration marker (about 1.5 h)
#   models3     Study 3: the model sequence (12 models, NUTS) and its posterior predictive checks (about 6 h)
#   models4     Study 4: the same (the exploration-trend and winning models with longer chains, as in the paper) (2 h)
#   models2     Study 2: the model without an ITI term, for Fig. 2e (10 min)
#   external    Tomov et al. (2020): the mixed model of switching on the jittered ITI and Fig. 6a (10 min)
#   agents      the language-model agents: scoring of their runs, and the tests of the likelihood code (10 min)
# Then, once the model lanes are done: the exploration marker with its posterior predictive check (Studies 3 and 4), the
# parameter-trait correlations, the recovery (unless --skip-recovery), every figure and Supplementary Table 1, and
# check_reported_numbers.py, which compares the regenerated results with every number reported in the paper and writes
# reproduction_report.md.
#
# Logs: logs/<step>.log, and logs/status.tsv (step, ok/FAILED, seconds). A failed step does not stop the other lanes; the
# script exits with status 1 if any step failed.

set -u
cd "$(dirname "$0")"
RECOVERY=1; SERIAL=0
for a in "$@"; do
  case $a in
    --skip-recovery) RECOVERY=0 ;;
    --serial) SERIAL=1 ;;
    *) echo "unknown option $a (use --skip-recovery, --serial)"; exit 2 ;;
  esac
done
PY=${PYTHON:-python3}
# PyTensor's C backend is not needed (the samplers are compiled by numba through nutpie); an empty compiler avoids
# requiring Xcode or gcc. Set PYTENSOR_FLAGS yourself to override.
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS-cxx=}"
LOG=logs; mkdir -p $LOG; STATUS=$LOG/status.tsv; : > $STATUS
S() { echo "data/study_$1/analysis"; }

step() {   # step NAME COMMAND...: run COMMAND with its output in logs/NAME.log and record the outcome
  local name=$1; shift
  local t0; t0=$(date +%s)
  echo "$(date '+%F %T')  start   $name"
  "$@" > "$LOG/$name.log" 2>&1; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then printf '%s\tok\t%d\n' "$name" $dt >> $STATUS; echo "$(date '+%F %T')  done    $name ($((dt / 60)) min)"
  else printf '%s\tFAILED\t%d\n' "$name" $dt >> $STATUS; echo "$(date '+%F %T')  FAILED  $name (exit $rc; see $LOG/$name.log)"; fi
  return $rc
}

run_lanes() {   # run the given lane functions side by side (or one after another with --serial) and wait for all
  if [ $SERIAL -eq 1 ]; then for l in "$@"; do $l; done
  else for l in "$@"; do $l & done; wait; fi
}

# ---- 0. the environment, and the per-study script copies -----------------------------------------------------------
$PY - > $LOG/environment.txt 2>&1 <<'EOF'
import platform, sys
print("python", sys.version.split()[0], platform.platform())
for m in ("numpy", "scipy", "pandas", "statsmodels", "matplotlib", "pymc", "pytensor", "nutpie", "arviz", "patsy", "PIL"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
EOF
cat $LOG/environment.txt
# Every data/study_N/analysis/*.py must be an exact copy of analysis_common/study_scripts/ (the scripts take the study
# from the folder they sit in).
bad=0
for f in data/study_*/analysis/*.py; do cmp -s "$f" "analysis_common/study_scripts/$(basename "$f")" || { echo "differs from the master copy: $f"; bad=1; }; done
[ $bad -eq 0 ] || { echo "update the copies (cp analysis_common/study_scripts/<script> data/study_N/analysis/) and rerun"; exit 1; }

# ---- 1. the lanes -------------------------------------------------------------------------------------------------
lane_behaviour() {
  for n in 1 2 3 4; do
    step s${n}_01_learning          $PY $(S $n)/01_learning_check.py
    step s${n}_02_iti_switching     $PY $(S $n)/02_iti_switching.py
    step s${n}_02b_by_half          $PY $(S $n)/02b_iti_switching_phase_by_outcome.py
    step s${n}_16_earlier_itis      $PY $(S $n)/16_position_controlled_contrasts.py
  done
  step s2_11_dot_motion             $PY $(S 2)/11_rdm_task.py
  for n in 3 4; do
    step s${n}_13_iti_sequence      $PY $(S $n)/13_design_confound.py
    step s${n}_17_response_times    $PY $(S $n)/17_response_times.py
  done
  step s3_05_individual_differences $PY $(S 3)/05_individual_differences.py
  step s34_20_exploration_hmm       $PY $(S 3)/20_exploration_proxy_hmm.py --studies 3 4
}
lane_models3() {
  step s3_06_model_fits $PY $(S 3)/06_fit_models_nuts.py &&
  step s3_10_ppc        $PY $(S 3)/10_posterior_predictive_checks.py
}
lane_models4() {
  # As in the paper: every model with 4 chains x 1000 draws after 2000 warm-up iterations, except the exploration-trend
  # model (m08) and the winning model (m12), with 4 x 2000 after 4000, because the spread of the choice-kernel learning
  # rate mixed slowly.
  step s4_06_model_fits      $PY $(S 4)/06_fit_models_nuts.py --models m01_intercepts m02_rw_1lr m03_rw_2lr m04_rw_2lr_decay \
                                 m05_stick m06_choice_kernel m07_novelty m09_trend_bias m10_iti_bias m11_iti_target &&
  step s4_06_model_fits_long $PY $(S 4)/06_fit_models_nuts.py --models m08_trend_beta m12_iti_compete --tune 4000 --draws 2000 &&
  step s4_10_ppc             $PY $(S 4)/10_posterior_predictive_checks.py
}
lane_models2() {
  step s2_06_model_fit_m08 $PY $(S 2)/06_fit_models_nuts.py --models m08_trend_beta
}
lane_external() {
  local d=fMRI_study_generalization
  if ! ls $d/data/tomov2020/subjects/uep*.csv > /dev/null 2>&1; then
    step tomov_download bash -c "set -e; t=\$(mktemp -d); git clone --depth 1 https://github.com/tomov/Exploration-Data-Analysis \$t/repo; \
      mkdir -p $d/data/tomov2020/subjects; cp \$t/repo/data/uep*.csv $d/data/tomov2020/subjects/; rm -rf \$t" || return 1
  fi
  step tomov_prepare $PY $d/code/prepare_tomov2020.py &&
  step tomov_glmm    $PY $d/code/tomov_iti_glmm.py &&
  step tomov_fig6a   $PY $d/code/plot_fig6a_tomov.py
}
lane_agents() {
  step agents_analysis   $PY ai_agents/analyze_agents.py
  step tests_likelihoods $PY tests/test_likelihoods.py
}
run_lanes lane_behaviour lane_models3 lane_models4 lane_models2 lane_external lane_agents

# ---- 2. what needs the fitted models ----------------------------------------------------------------------------------
lane_marker() {
  step s34_19_exploration_marker $PY $(S 3)/19_exploration_marker_ppc.py --studies 3 4
  step s34_09_parameter_traits   $PY $(S 3)/09_parameter_trait_correlations.py
}
lane_recovery() {   # the runs of the paper (Study 3): parameter recovery of the winning model, model recovery of the ITI models
  rm -rf $(S 3)/results/07_recovery
  step s3_07_parameter_recovery $PY $(S 3)/07_recovery_nuts.py --part parameter --model m12_iti_compete
  for g in m10_iti_bias m11_iti_target m12_iti_compete; do
    step s3_07_model_recovery_$g $PY $(S 3)/07_recovery_nuts.py --part model --gen $g
  done
  step s3_07_collect $PY $(S 3)/07_recovery_nuts.py --collect-only
}
if [ $RECOVERY -eq 1 ]; then run_lanes lane_marker lane_recovery; else lane_marker; fi

# ---- 3. figures, Supplementary Table 1 and the check against the paper ------------------------------------------------
step fig1_2_3_4    $PY figures/make_manuscript_figures.py
step fig5_study4   $PY figures/make_study4_figure.py
step fig6_generalization $PY figures/make_generalization_figure.py
step supp_fig1     $PY figures/make_supp_fig1_earlier_iti.py
step supp_fig2     $PY figures/make_supp_fig2_lag_profile.py
step supp_fig3     $PY figures/make_supp_fig3_schedule_expectation.py
step supp_table1   $PY figures/make_supp_table1.py
step check_reported_numbers $PY check_reported_numbers.py

echo; echo "step summary (logs/status.tsv):"; column -t -s $'\t' $STATUS
if grep -q FAILED $STATUS; then echo "some steps FAILED"; exit 1; fi
echo "all steps completed; see reproduction_report.md"
