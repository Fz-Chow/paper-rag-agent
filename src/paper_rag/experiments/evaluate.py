from __future__ import annotations
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ERRORS = {"evidence_omission", "association_error", "unsupported_answer", "output_format_error"}

def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def score(record, review=None):
    task = record["task"]
    result = {"run_id": record["run_id"], "task_id": task["id"], "split": task["split"],
              "condition": record["condition"], "rotation": record["rotation"], "reverse": record["reverse"],
              "status": record["status"], "answer_correct": False, "format_valid": False,
              "necessary_coverage": 0.0, "citation_support": None,
              "error_type": None, "error_type_source": None}
    if record["status"] != "ok":
        result["error_type"] = "runtime_error"
        return result
    output = record["generation"]["answer"]
    try:
        parsed = json.loads(output)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
            raise ValueError()
        citations = parsed["citations"]
        if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        result.update(error_type="output_format_error", error_type_source="automatic")
        return result
    required = set(task["required_ids"])
    cited = set(citations)
    available = set(record["context"]["order"])
    result.update(format_valid=True,
        answer_correct=normalize(parsed["answer"]) in {normalize(x) for x in [task["answer"], *task["aliases"]]},
        necessary_coverage=len(required & cited)/len(required),
        cited_ids=sorted(cited), invalid_citations=sorted(cited-available))
    if review:
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError("Citation review requires reviewer and reviewed_at")
        supported = review.get("supported_citations", {})
        if set(supported) != cited or any(type(x) is not bool for x in supported.values()):
            raise ValueError("Review must label every cited ID with true/false")
        if any(supported[c] for c in cited-available):
            raise ValueError("Unknown citations cannot be supported")
        result["citation_support"] = sum(supported.values())/len(cited) if cited else None
        label = review.get("error_type")
        if label is not None and label not in ERRORS:
            raise ValueError("Unknown human error category")
        result.update(error_type=label, error_type_source="human" if label else None, human_review=review)
    return result


def load_records(directory):
    return [json.loads(p.read_text()) for p in sorted((Path(directory)/"records").glob("*.json"))]


def evaluate(records, reviews=None):
    reviews = reviews or {}
    details = [score(r, reviews.get(r["run_id"])) for r in records]
    grouped = defaultdict(list)
    for r, s in zip(records, details):
        grouped[(s["split"], s["condition"])].append((r, s))
    summaries = []
    for (split, condition), pairs in sorted(grouped.items()):
        ok = [(r,s) for r,s in pairs if r["status"] == "ok"]
        def mean_metric(key):
            values = [r["generation"][key] for r,s in ok if r["generation"].get(key) is not None]
            return sum(values)/len(values) if values else None
        support = [s["citation_support"] for r,s in ok if s["citation_support"] is not None]
        task_scores = defaultdict(list)
        for r,s in pairs:
            task_scores[s["task_id"]].append(s["answer_correct"])
        summaries.append({"split": split, "condition": condition, "runs": len(pairs),
            "tasks": len(task_scores), "failures": len(pairs)-len(ok),
            "accuracy": sum(sum(v)/len(v) for v in task_scores.values())/len(task_scores),
            "necessary_coverage": sum(s["necessary_coverage"] for r,s in pairs)/len(pairs),
            "citation_support": sum(support)/len(support) if support else None,
            "support_reviewed_runs": len(support), "mean_input_tokens": mean_metric("input_tokens"),
            "mean_output_tokens": mean_metric("output_tokens"), "mean_seconds": mean_metric("elapsed_seconds"),
            "max_peak_allocated_mb": max((r["generation"].get("gpu", {}).get("peak_allocated_mb", 0)
                for r,s in ok if r["generation"].get("gpu")), default=None)})
    by_pair = defaultdict(dict)
    for s in details:
        by_pair[(s["task_id"],s["rotation"],s["reverse"])][s["condition"]] = s
    changes = []
    for (tid,rotation,reverse), conditions in sorted(by_pair.items()):
        if {"concentrated","dispersed"} <= conditions.keys():
            a,b = conditions["concentrated"], conditions["dispersed"]
            changes.append({"task_id":tid,"rotation":rotation,"reverse":reverse,
                "concentrated_correct":a["answer_correct"],"dispersed_correct":b["answer_correct"],
                "complete_pair":a["status"]==b["status"]=="ok",
                "concentrated_run_id":a["run_id"],"dispersed_run_id":b["run_id"]})
    return {"summary":summaries,"details":details,"paired_changes":changes,
            "notes":"Accuracy averages permutations within each task, then tasks. Failures count as incorrect. Citation coverage checks IDs only; support and reasoning error categories require human review. Permutations are not independent samples."}
