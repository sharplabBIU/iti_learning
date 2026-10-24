# The browser tasks

The three-armed restless bandit of the four studies, as single-page web apps hosted on Firebase (Methods, Task).

| folder | study | notes |
|---|---|---|
| `task_studies_1_and_3/` | Studies 1 and 3 (unfilled ITI) | the Study 3 build; its three changes from Study 1 are listed in the Methods. `study_1_standalone_html/game_v2.html` is the original single-page Study 1 version |
| `task_v2_rdm_study_2/` | Study 2 | the ITI is filled with the random-dot-motion task |
| `task_study4_newITIsequence/` | Study 4 | the Study 3 task with a new fixed ITI sequence; `iti_schedule_study4.csv` lists the ITI after every trial under the Study 4 sequence (`iti`) and under the sequence of Studies 1-3 (`iti_studies_1_3`), used by Fig. 5c |

In each folder, `public/index.html` is the task (the reward trajectories and the ITI sequence are built in the page from fixed
seeds, so they are identical for every participant), the `p*.png` and `r*.png` files are the pictures, and
`public/probability_graph.html` and `public/trial_probabilities_table.html` show the reward trajectories.
`scripts/export-prob-curves.mjs` and `scripts/generate-prob-canvas.mjs` regenerate the trajectories outside the page.

To run a task, create a Firebase project with a Firestore database, put its web configuration in the `firebaseConfig` block
of `public/index.html` (replacing the `YOUR_...` placeholders) and its hosting site in `firebase.json`, then
`firebase deploy`. `firestore.rules` is the file that was deployed with the studies: it lets the browser write participant
data without signing in, and it also allows anyone to read the database, so restrict reads (for example
`allow read: if false; allow create: if true;`) before collecting data. The scripts that read the collected data from the
database are not part of this repository.
