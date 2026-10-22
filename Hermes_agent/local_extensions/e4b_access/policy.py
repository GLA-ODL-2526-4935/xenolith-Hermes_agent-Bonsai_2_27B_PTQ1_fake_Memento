"""Small capability vocabulary; no executable browser or shell instructions."""
import hashlib
import ipaddress
import json
import re
from urllib.parse import urlsplit, urlunsplit

WEB_ACTIONS = ("open_page", "download", "upload", "submit_form", "send_message", "purchase", "account_change")
MAX_TEXT = 65536


def bounded(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
        raise ValueError("Expected nonempty bounded UTF-8 text")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError("Control characters are forbidden")
    return value


def public_https_url(value):
    """Syntactic screening only: DNS rebinding/subrequests require OS egress control."""
    bounded(value, 4096)
    if any(char.isspace() for char in value) or "\\" in value:
        raise ValueError("Ambiguous URL")
    url = urlsplit(value)
    if url.scheme != "https" or url.username is not None or url.password is not None or url.port not in (None, 443):
        raise ValueError("Only credential-free public HTTPS URLs on port 443 are allowed")
    host = (url.hostname or "").encode("idna").decode("ascii").lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if (len(labels) < 2 or host.endswith((".localhost", ".local", ".internal", ".lan", ".home"))
                or not re.fullmatch(r"[a-z]{2,63}|xn--[a-z0-9-]+", labels[-1])
                or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) for part in labels)):
            raise ValueError("Invalid or local hostname")
    else:
        if not address.is_global:
            raise ValueError("Private/local addresses are forbidden")
        host = f"[{host}]" if address.version == 6 else host
    return urlunsplit(("https", host, url.path or "/", url.query, url.fragment))


def validate_proposal(proposal):
    if not isinstance(proposal, dict):
        raise ValueError("Expected proposal object")
    kind = proposal.get("kind")
    if kind == "text_export":
        if set(proposal) != {"kind", "label", "text"}:
            raise ValueError("Text exports accept no paths, filenames, commands or extra fields")
        return {"kind": kind, "label": bounded(proposal["label"], 120), "text": bounded(proposal["text"], MAX_TEXT)}
    if kind in WEB_ACTIONS:
        if set(proposal) != {"kind", "url", "details"}:
            raise ValueError("Web proposals require exactly kind, url and details")
        return {"kind": kind, "url": public_https_url(proposal["url"]),
                "details": bounded(proposal["details"], 4000)}
    raise PermissionError("Capability is not allowed")


def digest(proposal):
    return hashlib.sha256(json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
