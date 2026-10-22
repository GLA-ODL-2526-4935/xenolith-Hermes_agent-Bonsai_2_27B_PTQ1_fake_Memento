"""Recovery provenance and accidental-secret checks; not OS containment."""
import json
import math
import re

SECRET_PATTERNS = (
    r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----",
    r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*",
    r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
    r'''(?i)\b(?:password|passwd|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|[A-Z][A-Z0-9_]*_TOKEN)\b["']?\s*[:=]\s*["']?[^\s"',;}]{4,}''',
)


def reject_sensitive_text(value):
    # Reject, never redact: changed source text would invalidate exact provenance.
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str) and any(re.search(pattern, item, re.IGNORECASE) for pattern in SECRET_PATTERNS):
            raise ValueError("Recovery contains possible credentials; supply an authorized non-secret source excerpt")


def check_fact_source(category, quote, event):
    if category in {"requirement", "decision"}:
        if event["role"] != "user" or quote != event["content"]:
            raise ValueError("Requirements/decisions require the entire visible user source event")


def finite_timestamp(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid recovery timestamp")
    return value


def reference_view(record):
    # These are derived, never accepted from a caller or trusted from an edited file.
    result = json.loads(json.dumps(record, ensure_ascii=False))
    result["use_policy"] = {"untrusted_reference_only": True, "grants_permission": False,
                            "task_contract_requires_original_user_source": True}
    result["source_roles"] = {event["id"]: event["role"] for event in result["events"]}
    return result
