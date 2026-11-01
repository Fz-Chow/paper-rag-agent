from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

CONDITIONS = ("necessary", "concentrated", "dispersed")

def load_tasks(path: Path) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, groups, fact_sets = set(), {}, {}
    if not tasks:
        raise ValueError("Empty dataset")
    for t in tasks:
        for key in ("id", "fact_group", "split", "question", "answer", "aliases", "required_ids", "evidence", "human_review"):
            if key not in t:
                raise ValueError(f"Missing {key}")
        if not isinstance(t["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", t["id"]):
            raise ValueError("Task ID must contain only letters, digits, underscore or hyphen")
        if t["id"] in ids or t["split"] not in {"exploration", "validation"}:
            raise ValueError("Duplicate task ID or invalid split")
        ids.add(t["id"])
        group = t["fact_group"]
        if group in groups and groups[group] != t["split"]:
            raise ValueError("Fact group leaks across splits")
        groups[group] = t["split"]
        evidence = t["evidence"]
        eids = [e["id"] for e in evidence]
        if any(not isinstance(eid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", eid) for eid in eids):
            raise ValueError("Invalid evidence ID")
        if len(eids) != len(set(eids)) or len(eids) < 4 or len(eids) % 2:
            raise ValueError("Use unique evidence IDs and an even number of >=4 blocks")
        if len(set(t["required_ids"])) != 2 or not set(t["required_ids"]) <= set(eids):
            raise ValueError("Exactly two necessary evidence IDs required")
        if not isinstance(t["aliases"], list) or not t["answer"] or not t["question"]:
            raise ValueError("Invalid question or answer aliases")
        if any(not e.get("source") or not e.get("text") for e in evidence):
            raise ValueError("Evidence text and source required")
        facts = tuple(sorted(e["text"] for e in evidence if e["id"] in t["required_ids"]))
        if facts in fact_sets and fact_sets[facts] != t["split"]:
            raise ValueError("Same facts leak across splits")
        fact_sets[facts] = t["split"]
        review = t["human_review"]
        if review.get("status") not in {"pending", "verified"}:
            raise ValueError("Invalid human review status")
        if review["status"] == "verified" and (not review.get("reviewer") or not review.get("reviewed_at")):
            raise ValueError("Verified tasks require reviewer and reviewed_at")
    return tasks


def build_context(task, condition, rotation=0, reverse=False):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    by_id = {e["id"]: e for e in task["evidence"]}
    required = list(task["required_ids"])
    if reverse:
        required.reverse()
    distractors = [e["id"] for e in task["evidence"] if e["id"] not in required]
    if condition == "necessary":
        order = required
    else:
        n = len(by_id)
        rotation %= n//2
        slots = (2*rotation, 2*rotation+1) if condition == "concentrated" else (rotation, rotation+n//2)
        order = [None]*n
        for slot, eid in zip(slots, required):
            order[slot] = eid
        remaining = iter(distractors[rotation:] + distractors[:rotation])
        order = [next(remaining) if eid is None else eid for eid in order]
    blocks = [f"[{eid}] 来源：{by_id[eid]['source']}\n{by_id[eid]['text']}" for eid in order]
    return {"text": "\n\n".join(blocks), "order": order,
            "required_positions": {eid: order.index(eid) for eid in required},
            "material_hash": hashlib.sha256(json.dumps(sorted(blocks), ensure_ascii=False).encode()).hexdigest(),
            "condition": condition, "rotation": rotation, "reverse": reverse}


def measure_context(context, task, generator):
    result = dict(context)
    prompt = generator.prompt(task["question"], context["text"])
    result["prompt"] = prompt
    result["input_tokens"] = generator.count_tokens(prompt)
    result["context_tokens"] = generator.count_tokens(context["text"])
    result["evidence_token_offsets"] = {}
    for eid in context["order"]:
        start = prompt.index(f"[{eid}] 来源：")
        result["evidence_token_offsets"][eid] = generator.count_tokens(prompt[:start])
    return result
