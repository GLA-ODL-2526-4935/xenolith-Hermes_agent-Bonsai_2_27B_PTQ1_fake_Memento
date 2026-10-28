"""Owner-side approval session; never imported by a model tool or controller API."""
import re
import secrets
import time

from .review import decide
from .review_display import history, proposal_markdown
from .step_recovery import digest


class RemoteReview:
    """One owner, DM only, complete previews and one-use five-minute receipts."""

    def __init__(self, store, owner_id, *, clock=time.monotonic):
        if not isinstance(owner_id, str) or not re.fullmatch(r"[1-9][0-9]{0,19}", owner_id):
            raise ValueError("Explicit numeric Discord owner ID required")
        self.store = store
        self.owner_id = owner_id
        self.clock = clock
        self.receipts = {}

    def authorized(self, user_id, *, direct, bot=False, webhook=False):
        return direct is True and not bot and not webhook and str(user_id) == self.owner_id

    def prepare(self, user_id, command, *, direct, bot=False, webhook=False):
        """Return bounded transport output. Authorization precedes every store read."""
        if not self.authorized(user_id, direct=direct, bot=bot, webhook=webhook):
            return None
        if not isinstance(command, str) or len(command) > 200:
            raise ValueError("Invalid command")
        parts = command.split()
        if parts == ["!memory", "pending"]:
            ids = [p["id"] for p in history(self.store, "pending")["proposals"]]
            return {"text": "Pending proposal IDs (first 20):\n" + ("\n".join(ids[:20]) or "None")}
        if len(parts) < 3 or parts[0] != "!memory" or not re.fullmatch(r"[a-f0-9]{32}", parts[2]):
            raise ValueError("Use !memory pending, !memory show ID, or !memory accept|reject ID CODE")
        action, key = parts[1:3]
        if action == "show" and len(parts) == 3:
            return self._preview(key)
        if action in {"accept", "reject"} and len(parts) in {4, 5}:
            selected = None
            if len(parts) == 5:
                if action != "accept" or not re.fullmatch(r"[1-4](,[1-4])*", parts[4]):
                    raise ValueError("Use comma-separated draft numbers on acceptance only")
                selected = [int(i) for i in parts[4].split(",")]
            return self._decision(key, parts[3], action == "accept", selected)
        raise ValueError("Unsupported memory review command")

    def _preview(self, key):
        state = self.store.snapshot()
        proposal = state.get("memory_proposals", {}).get(key)
        if not proposal or proposal["status"] != "pending":
            raise ValueError("No pending proposal")
        payload = proposal_markdown(proposal, state=state).encode("utf-8")
        if len(payload) > 128000:
            raise ValueError("Too large for remote review; use owner CLI")
        # Only the latest preview is actionable; receipts are not written into model-readable state.
        code = secrets.token_hex(16)
        self.receipts = {key: {"code": code, "digest": digest(proposal),
                               "expires": self.clock() + 300, "delivered": False}}
        return {"text": "Read the attached FULL proposal and evidence. Acceptance may add, rewrite or discard temporary memory. "
                        "It does not verify truth or grant tool permissions. Within five minutes reply:\n"
                        f"!memory accept {key} {code}\nOR\n!memory reject {key} {code}\n"
                        "For selected generation drafts, append numbers, e.g. 1,3. All unselected drafts are rejected.",
                "attachment": payload, "receipt": (key, code)}

    def delivered(self, receipt, success):
        key, code = receipt
        current = self.receipts.get(key)
        if current and secrets.compare_digest(current["code"], code):
            if success:
                current["delivered"] = True
            else:
                self.receipts.pop(key, None)

    def _decision(self, key, code, accept, selected=None):
        receipt = self.receipts.get(key)
        if (not receipt or not receipt["delivered"] or self.clock() >= receipt["expires"]
                or not re.fullmatch(r"[a-f0-9]{32}", code)
                or not secrets.compare_digest(receipt["code"], code)):
            raise PermissionError("Missing, expired or invalid review receipt; show the proposal again")
        # Consume before I/O, including failure: ambiguous disk/readback failures must not retry.
        self.receipts.pop(key)
        result = decide(self.store, key, accept, expected_digest=receipt["digest"], selected_drafts=selected)
        return {"text": f"Proposal {key}: {result['status']}. No permission grant. "
                        "Accepted drafts still require conflict review before persistent promotion."}


async def handle_message(review, message, *, direct, send):
    """Transport seam: send(text, attachment) must complete before receipt activation."""
    if not review.authorized(message.author.id, direct=direct, bot=message.author.bot,
                             webhook=bool(message.webhook_id)):
        return
    if not message.content.startswith("!memory"):
        return
    response = None
    try:
        response = review.prepare(message.author.id, message.content, direct=direct)
        await send(response["text"], response.get("attachment"))
    except (ValueError, PermissionError, OSError):
        if response and "receipt" in response:
            review.delivered(response["receipt"], False)
        await send("Review refused or not confirmed. Check owner CLI state; do not blindly repeat a decision. "
                   "Commands: !memory pending; !memory show ID; !memory accept|reject ID CODE.", None)
        return
    if "receipt" in response:
        review.delivered(response["receipt"], True)
