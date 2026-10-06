"""Turn the generated alerts into Strands Decider training examples and stratified splits.

Each alert yields two examples with the same wording the DevOps example sends at serving time:
a `choice` over the five runbooks and a `noul` for "destructive". Splits are stratified by
runbook: train 70 %, calib 15 % (temperature fitting), test 15 %. The 12 hand-written alerts of
examples/g_devops_self_healing.py are written separately as `holdout` and never trained on.

Usage:
    python finetune/build_examples.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "finetune"
SEED = 7

RUNBOOKS = {
    "restart_service": "the process crashed, hung, was OOMKilled or its health check keeps failing; a restart fixes it",
    "scale_out": "traffic spike, CPU saturation, queue backlog or latency growing with load; more replicas fix it",
    "rollback_deploy": "errors started right after a new deployment or release version",
    "clear_disk": "disk or volume almost full, no space left on device, logs filling the disk",
    "escalate": "anything else: unclear cause, data or database problems, security incidents, or several possible causes",
}
RUNBOOK_QUESTION = "Which runbook should handle this production alert?"
DESTRUCTIVE_QUESTION = ("Could fixing this alert be destructive or irreversible, for example by touching data, "
                        "databases or migrations?")
DESTRUCTIVE_OPTIONS = [
    ["false", "the fix is safe to undo, like restarting or adding replicas"],
    ["true", "the fix could lose or corrupt data, or cannot be undone"],
]


def state(alert: dict) -> dict:
    return {"service": alert["service"], "alert": alert["text"]}


def examples_for(alert: dict, task: str) -> list[dict]:
    names = list(RUNBOOKS)
    runbook = {"kind": "choice", "state": state(alert), "instructions": RUNBOOK_QUESTION,
               "options": [[name, RUNBOOKS[name]] for name in names],
               "label": names.index(alert["runbook"]), "task": f"{task}_runbook"}
    rows = [runbook]
    if "destructive" in alert:
        rows.append({"kind": "noul", "state": state(alert), "instructions": DESTRUCTIVE_QUESTION,
                     "options": DESTRUCTIVE_OPTIONS, "label": int(alert["destructive"]),
                     "task": f"{task}_destructive"})
    return rows


def holdout_alerts() -> list[dict]:
    sys.path.insert(0, str(ROOT / "examples"))
    source = (ROOT / "examples" / "g_devops_self_healing.py").read_text(encoding="utf-8")
    start = source.index("ALERTS = [")
    end = source.index("]\n", start) + 1
    namespace: dict = {}
    exec(source[start:end], namespace)
    return [{"service": a["service"], "text": a["text"], "runbook": a["expected"]} for a in namespace["ALERTS"]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    alerts = [json.loads(line) for line in (DATA / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    holdout_texts = {a["text"].lower() for a in holdout_alerts()}
    alerts = [a for a in alerts if a["text"].lower() not in holdout_texts]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for alert in alerts:
        by_label[alert["runbook"]].append(alert)
    splits: dict[str, list[dict]] = {"train": [], "calib": [], "test": []}
    rng = random.Random(SEED)
    for rows in by_label.values():
        rng.shuffle(rows)
        n_train, n_calib = int(len(rows) * 0.70), int(len(rows) * 0.15)
        splits["train"] += rows[:n_train]
        splits["calib"] += rows[n_train:n_train + n_calib]
        splits["test"] += rows[n_train + n_calib:]
    for name, rows in splits.items():
        write_jsonl(DATA / f"{name}_alerts.jsonl", rows)
        write_jsonl(DATA / f"{name}.jsonl", [ex for alert in rows for ex in examples_for(alert, "devops")])
    write_jsonl(DATA / "holdout_alerts.jsonl", holdout_alerts())
    print(json.dumps({name: len(rows) for name, rows in splits.items()} | {"holdout": 12}))


if __name__ == "__main__":
    main()
