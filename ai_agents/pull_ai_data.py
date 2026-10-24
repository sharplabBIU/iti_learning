"""Pull the AI-agent runs (STUDY_ID = AI_AGENTS) from Firestore and join them with the local run records in runs/.

Uses the Firestore REST API (the database rules allow unauthenticated reads) and only the Python standard library.
Mock/test agents (PIDs starting AIAGENT_mock_) are left out unless --include-mock.

Outputs (data/):
  trials_combined.json    the Study 3/4 raw format ({participants, allTrials}), with model and condition added
  trials.csv              one row per trial: the page's own trial record + model, condition, run tag
  participants.csv        one row per agent: Firestore metadata + model, condition, status, requests, estimated cost
  questionnaires.csv      item-level BIS-11 / SHAPS answers as saved by the page
Usage: python pull_ai_data.py [--include-mock]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.request
from pathlib import Path

PROJECT = "bandit-task-v2"
API_KEY = os.environ.get("FIREBASE_WEB_API_KEY", "")   # the web API key of the task page (not stored in the repository)
DB = f"projects/{PROJECT}/databases/(default)/documents"
BASE = f"https://firestore.googleapis.com/v1/{DB}"
STUDY_ID = "AI_AGENTS"
HERE = Path(__file__).resolve().parent


def _post(url, body):
    req = urllib.request.Request(f"{url}?key={API_KEY}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _value(v):
    for k in ("stringValue", "booleanValue", "timestampValue"):
        if k in v:
            return v[k]
    if "integerValue" in v:
        return int(v["integerValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    return None


def _fields(doc):
    return {k: _value(v) for k, v in doc.get("fields", {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mock", action="store_true")
    args = ap.parse_args()
    body = {"structuredQuery": {"from": [{"collectionId": "participants"}],
                                "where": {"fieldFilter": {"field": {"fieldPath": "study_id"}, "op": "EQUAL",
                                                          "value": {"stringValue": STUDY_ID}}}}}
    metas = [(d["document"]["name"].rsplit("/", 1)[1], _fields(d["document"])) for d in _post(f"{BASE}:runQuery", body) if "document" in d]
    metas = [(pid, m) for pid, m in metas if args.include_mock or not pid.startswith("AIAGENT_mock_")]
    print(f"{len(metas)} AI-agent participant documents (study_id = {STUDY_ID})")
    paths = [f"{DB}/participants/{pid}/data/{n}" for pid, _ in metas for n in ("trials", "pswq")]
    sub = {}
    for i in range(0, len(paths), 100):
        for item in _post(f"{BASE}:batchGet", {"documents": paths[i:i + 100]}):
            if "found" in item:
                sub[item["found"]["name"][len(DB) + 1:]] = _fields(item["found"])

    participants, trials, qrows = [], [], []
    for pid, meta in sorted(metas):
        run = HERE / "runs" / f"{pid}.json"
        r = json.load(open(run)) if run.exists() else {}
        tag = dict(model=r.get("model", ""), condition=r.get("condition", ""), run_tag=pid.split("_")[-2] if pid.count("_") >= 4 else "")
        rows = list(csv.DictReader(io.StringIO(sub.get(f"participants/{pid}/data/trials", {}).get("csv_data", ""))))
        participants.append(dict(subject_id=pid, prolific_id=pid, **tag, age=meta.get("age"), birth_year=meta.get("birth_year"),
                                 trial_count=len(rows), bis_total=meta.get("bis_total"), pleasure_total=meta.get("pleasure_total"),
                                 bis_attention_passed=meta.get("bis_attention_passed"), timestamp=meta.get("timestamp"),
                                 status=r.get("status", "no local run record"), requests=r.get("usage", {}).get("requests"),
                                 cost_usd_estimate=r.get("cost_usd_estimate"), effort=r.get("effort"), persona=r.get("persona")))
        for row in rows:
            t = dict(row); t.update(subject_id=pid, prolific_id=pid, **tag)
            for k in ("trial", "choice", "win"):
                t[k] = int(t[k])
            for k in ("rt", "reward", "p1", "p2", "p3"):
                t[k] = float(t[k])
            trials.append(t)
        q = sub.get(f"participants/{pid}/data/pswq", {}).get("csv_data", "")
        for row in csv.DictReader(io.StringIO(q)):
            row.update(tag); qrows.append(row)

    out = HERE / "data"; out.mkdir(exist_ok=True)
    json.dump({"participants": participants, "allTrials": trials}, open(out / "trials_combined.json", "w"), indent=2)
    for name, rows in (("trials.csv", trials), ("participants.csv", participants), ("questionnaires.csv", qrows)):
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    by = {}
    for p in participants:
        by.setdefault((p["model"], p["condition"]), []).append(p["trial_count"])
    for (m, c), n in sorted(by.items()):
        print(f"  {m or '?'} / {c or '?'}: {len(n)} agents, {sum(x >= 100 for x in n)} with all 100 trials")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
