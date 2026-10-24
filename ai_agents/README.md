# Language-model agents as participants in the Study 4 task

Claude agents (Claude Opus 5, Sonnet 5 and Haiku 4.5) took the live Study 4 task exactly as a participant would: consent,
instructions, age, 6 practice trials, 100 task trials, birth year, the BIS-11 items with the attention check, and the SHAPS.
The page's own code saved their data in the same format as the human participants (Results [60]-[61]; Methods, Language-model
agents; Fig. 6b-f).

## What is here

| path | contents |
|---|---|
| `run_agents.py`, `prompts.py` | the harness (Playwright drives the live page; one fresh API conversation per agent) and the only researcher-written text the agents saw |
| `runs/` | the complete record of every agent: every prompt, every reply, the summarised reasoning, token use and the outcome of each trial; `runs/images/` holds the two screenshots each agent received |
| `pull_ai_data.py` | pulls the agents' task data from the database into `data/` |
| `data/` | `trials.csv`, `participants.csv`, `questionnaires.csv` and `trials_combined.json`, in the format of Study 4 |
| `analyze_agents.py` | the measures of the paper per agent: the chose-best rate, switching after 1 s and 4 s ITIs after wins and losses, the stay rates, the questionnaire answers and reply quality, each with its percentile among the Study 4 participants |
| `results/` | the outputs of `analyze_agents.py` (`agent_metrics.csv`, `summary.md`) |

`reproduce.sh` runs `analyze_agents.py`; `figures/make_generalization_figure.py` draws Fig. 6b-f from the run records and the
data. Fourteen agents were run, plus a mock record (`AIAGENT_mock_...`, random replies, used to test saving) that every
analysis leaves out. The two pilot agents (`..._pilot_...`) were run before a harness fix and sometimes replied with text
beyond the answer (Methods); their key presses were read correctly and their data are kept.

## Why the agents are naive

- Every agent is a fresh API conversation; it knows only what is put in it.
- What is put in it: the text of each screen, read from the live page at run time (consent form, instructions, trial counter,
  "You won $1" / "You lost $1", "TOO SLOW!", error messages, every questionnaire item and option), two screenshots of the three
  pictures (first practice trial and first task trial), and the short formatting lines in `prompts.py`. Nothing in it mentions
  the hypotheses or the ITI manipulation.

## Timing and conditions

The page gives 2 s per choice; a model needs longer. The harness freezes the page's clock while the agent decides, then moves
it on: 600 ms to the key press, the 1.25 s feedback, then the fixation ITI until the next trial. The ITI sequence, feedback and
trial order are therefore exactly those of Study 4, but the recorded response times are always 600 ms and are not analysed.
Two conditions differ in one sentence after each trial:
- `time_blind`: "The images disappear and a fixation cross (+) is shown." Any ITI effect here can only come from the task
  structure (ITI sequence, reward schedule), not from waiting.
- `time_told`: the same, plus "... for 1 second." / "... for 4 seconds." (measured on the page clock).
One sentence added to the opening instructions (`--persona`) varied between runs: none, "Play like a human would.", or "Play
like a human that is sensitive to time costs."

## Identifiers

`PROLIFIC_PID = AIAGENT_<blind|told>_<model>_<tag>_<nnn>` (never a real Prolific ID), `STUDY_ID = AI_AGENTS`,
`SESSION_ID = <model>_<condition>`. The harness refuses a PID that already exists in the database (unless `--overwrite`).
Agents stop at the end screen; nothing is sent to Prolific.

## Rerunning the agents

Rerunning needs the live task page, its Firebase web API key in the environment variable `FIREBASE_WEB_API_KEY` (not stored
in this repository), and an Anthropic API key in `ANTHROPIC_API_KEY`. `setup.sh` makes a Python environment with `anthropic`
and `playwright` and installs Playwright's Chromium. Then, from this folder:

```bash
python run_agents.py --mock --dry-run --n 1                     # test the browser flow: random replies, no API calls, nothing saved
python run_agents.py --n 2 --tag pilot                          # 2 agents, one per condition
python run_agents.py --n 40 --model claude-opus-5,claude-sonnet-5 --parallel 4 --tag r1
python pull_ai_data.py                                          # the agents' data into data/
python analyze_agents.py
```

Options (`--help` lists them all): `--model` one ID or a comma list; `--condition time_blind|time_told|both`; `--effort`
(reasoning depth); `--persona` (one sentence added to the instructions); `--no-images`; `--parallel N`; `--start N`;
`--show` to watch the browser; `--dry-run` blocks all database traffic and stops before the final Submit. Agents of the same
model are not independent people: their variability comes only from sampling.
