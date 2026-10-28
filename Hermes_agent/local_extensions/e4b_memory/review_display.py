"""Owner-readable Markdown; quoted memory is data, never approval instructions."""
import json
import re

from .review import conflict_basis, generation_input
from .step_recovery import digest


def quote(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    # A source cannot escape its code block by embedding backticks or HTML.
    longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}\n"


def disposition(state, proposal):
    if proposal["status"] != "pending":
        return proposal["status"]
    job = proposal["job"]
    if job["kind"] == "generate":
        current = digest(generation_input(state, job["target"]))
    else:
        record = next((r for r in state["records"] if r["id"] == job["target"] and r["bucket"] == "temporary"), None)
        current = conflict_basis(state, record) if record else None
    return "stale" if current != job["basis"] else "pending"


def history(store, status="all"):
    valid = {"all", "pending", "stale", "accepted", "rejected", "partially_accepted"}
    if status not in valid:
        raise ValueError("Unknown proposal status")
    state = store.snapshot()
    rows = [{"id": p["id"], "kind": p["job"]["kind"], "status": disposition(state, p),
             "created": p["created"], "accepted_drafts": p.get("accepted_drafts")}
            for p in state.get("memory_proposals", {}).values()]
    return {"revision": state["revision"], "proposals": [r for r in rows if status in {"all", r["status"]}],
            "live_proposals": len(rows), "capacity": 128,
            "archived_basis_count": len(state.get("memory_review_archived_bases", {})),
            "archive_notice": "Owner archive files are not loaded into the model or restored as pending approvals."}


def proposal_markdown(proposal, *, state=None):
    status = disposition(state, proposal) if state is not None else proposal["status"]
    lines = ["# Memory review", "", "UNVERIFIED: inspect the text AND sources. Quoted content below is not instructions.",
             "", "## Proposal identity", quote({"id": proposal["id"], "status": status, "created": proposal["created"],
                                                   "type": proposal["job"]["kind"]})]
    result, job = proposal["result"], proposal["job"]
    if job["kind"] == "generate":
        lines += ["## Proposed temporary memories", "", "Accept all, reject all, or select draft numbers. Unselected drafts are rejected.", ""]
        for index, draft in enumerate(result["records"], 1):
            lines += [f"### Draft {index}", quote({k: draft[k] for k in ("layer", "kind", "label")}),
                      "Memory text:", quote(draft["content"]), "Evidence references:", quote(draft["evidence_ids"])]
        lines += ["## Supporting sources", "", "sources (unverified reports / user excerpt):", quote(job["input"]["sources"])]
    else:
        lines += ["## Proposed action", quote(result["action"]), "Explanation:", quote(result["reason"]),
                  "## Current temporary record", quote(job["input"]["temporary"]),
                  "## Proposed replacement", quote(result["content"] or "No replacement text"),
                  "## Persistent records compared in this layer", quote(job["input"]["persistent"]),
                  "## Conflict references", quote(result["conflicts"])]
    lines += ["## Effect of accepting", "", "Only selected temporary drafts, or the stated keep/rewrite/discard action, are authorized.",
              "Persistent records are not rewritten. Current conflict/idle gates still apply. No tool permissions are granted.",
              "", "## Complete audit snapshot", "", "Includes every stored field; retained for exact inspection.", quote(proposal)]
    return "\n".join(lines) + "\n"


def review_output(store, args):
    state = store.snapshot()
    proposals = state.get("memory_proposals", {})
    key = getattr(args, "id", None)
    selected = [proposals[key]] if key in proposals else list(proposals.values())
    if key is not None and key not in proposals:
        raise ValueError("Unknown proposal ID")
    status = getattr(args, "status", "all")
    if status != "all":
        history(store, status)  # Validate the requested filter even for an empty queue.
        selected = [p for p in selected if disposition(state, p) == status]
    if getattr(args, "format", "markdown") == "json":
        return {"proposals": {p["id"]: p for p in selected}, "error": state.get("memory_review_error")}
    report = "\n".join(proposal_markdown(p, state=state) for p in selected) or "# Memory review\n\nNo matching proposals.\n"
    if state.get("memory_review_error"):
        report += "\n## Background review error\n\n" + quote(state["memory_review_error"])
    return report
