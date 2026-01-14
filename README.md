Here is a `README.md` structured to detail your pilot findings and outline the new experimental task.

---

# iti_learning

We investigate how time for thinking, over short time differences, impacts learning in classic bandit tasks. Specifically, we examine whether Inter-Trial Interval (ITI) duration modulates simple heuristic behaviors like "lose-shift" or "win-stay."

## Repository Structure

The project is organized into two main components:

* **`pilot/`**: Contains the data and analysis from the initial proof-of-concept study.
* **`task/`**: Contains the code and assets for the main experiment currently under development.

---

## 1. Pilot Study (`/pilot`)

We conducted an initial pilot study to test the effects of ITI duration on decision-making strategies.

### Experimental Design

* **Task Type:** Stable Multi-Armed Bandit.
* **Options:** 3 Arms with stable reward probabilities (20% / 40% / 60%).
* **Duration:** 2 blocks of 42 trials each (84 trials total).
* **Manipulation:** ITI duration (Short vs. Long).

### Key Findings

Analysis of the pilot data revealed a significant behavioral shift based on the time available between trials:

* **Main Result:** Participants exhibited significantly higher **Lose-Shift** behavior under **Long ITI** conditions compared to Short ITI conditions.
* *Interpretation:* This suggests that longer pauses may facilitate a shift away from perseveration or random responding, potentially allowing for more cognitive evaluation of negative outcomes (or conversely, increased noise/exploration depending on the model interpretation).

*See the Jupyter Notebook in this folder for the full statistical analysis and visualization of these results.*

---

## 2. Main Experiment (`/task`)

Building on the pilot results, we are developing a new paradigm to test this effect more robustly in a dynamic environment.

### Task Specifications

* **Task Type:** Restless Bandit (non-stationary reward probabilities).
* **Options:** 3 Arms.
* **Duration:** 1 Block of 100 trials.
* **Objective:** To determine if the ITI-dependent increase in lose-shift behavior persists or adapts when reward contingencies are constantly fluctuating.

---

### Usage

To replicate the pilot analysis:

1. Navigate to `pilot/`.
2. Open the Jupyter Notebook (e.g., `analysis.ipynb`).
3. Ensure required dependencies (pandas, seaborn, etc.) are installed.

