"""Owner-controlled request inbox. Keep its directory outside the agent sandbox."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from .policy import digest, validate_proposal


class Broker:
    def __init__(self, root, *, clock=time.time):
        self.root = Path(root).resolve(strict=True)
        self.database = self.root / "requests.sqlite3"
        self.clock = clock
        if not self.database.is_file() or self.database.is_symlink():
            raise PermissionError("Initialize a dedicated broker directory first")
        with sqlite3.connect(self.database) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(settings)")}
            self.scope = db.execute("SELECT scope FROM settings").fetchone()[0] if "scope" in columns else "manual"
        if self.scope not in {"manual", "research"}:
            raise PermissionError("Unknown broker scope")

    @classmethod
    def create(cls, root, *, clock=time.time, scope="manual"):
        if scope not in {"manual", "research"}:
            raise ValueError("Unknown broker scope")
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (root / "exports").mkdir(mode=0o700)
        token = secrets.token_urlsafe(32)
        with sqlite3.connect(root / "requests.sqlite3") as db:
            db.executescript("CREATE TABLE settings (token_hash TEXT NOT NULL, scope TEXT NOT NULL);"
                             "CREATE TABLE requests (id TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL, "
                             "created REAL NOT NULL, expires REAL NOT NULL, status TEXT NOT NULL);")
            db.execute("INSERT INTO settings VALUES (?, ?)", (hashlib.sha256(token.encode()).hexdigest(), scope))
        os.chmod(root / "requests.sqlite3", 0o600)
        return cls(root, clock=clock), token

    @contextmanager
    def transaction(self):
        with sqlite3.connect(self.database, timeout=5) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            yield db

    def authenticate(self, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 200:
            raise PermissionError("Invalid client credential")
        with self.transaction() as db:
            expected = db.execute("SELECT token_hash FROM settings").fetchone()[0]
        if not secrets.compare_digest(expected, hashlib.sha256(token.encode()).hexdigest()):
            raise PermissionError("Invalid client credential")

    def submit(self, proposal):
        proposal = validate_proposal(proposal)
        self._check_scope(proposal)
        key, now = uuid.uuid4().hex, self.clock()
        with self.transaction() as db:
            count = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if count >= 1000:
                raise PermissionError("Inbox full; owner maintenance required")
            db.execute("INSERT INTO requests VALUES (?, ?, ?, ?, ?, 'pending')",
                       (key, json.dumps(proposal, ensure_ascii=False), digest(proposal), now, now + 3600))
        return {"id": key, "status": "pending", "digest": digest(proposal), "expires": now + 3600}

    def status(self, key):
        with self.transaction() as db:
            row = self._row(db, key)
            state = "expired" if row["status"] == "pending" and self.clock() >= row["expires"] else row["status"]
            return {"id": key, "status": state, "digest": row["digest"]}

    def inspect(self, key):
        with self.transaction() as db:
            return dict(self._row(db, key))

    def list_pending(self):
        with self.transaction() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, digest, created, expires FROM requests WHERE status='pending' AND expires>?", (self.clock(),))]

    def decide(self, key, expected_digest, *, approve):
        with self.transaction() as db:
            row = self._row(db, key)
            if row["status"] != "pending" or self.clock() >= row["expires"]:
                raise PermissionError("Request is expired or already decided")
            payload = json.loads(row["payload"])
            if expected_digest != row["digest"] or digest(payload) != expected_digest:
                raise PermissionError("Approval must match the exact reviewed payload digest")
            result = {"id": key, "status": "rejected"}
            if approve:
                payload = validate_proposal(payload)
                self._check_scope(payload)
                result = self._approved(key, payload)
            db.execute("UPDATE requests SET status=? WHERE id=?", (result["status"], key))
            return result

    def _approved(self, key, payload):
        if payload["kind"] != "text_export":
            return {"id": key, "status": "approved_manual", "url": payload["url"],
                    "details": payload["details"], "notice": "Owner handoff only; no browser/download/action executed"}
        exports = self.root / "exports"
        if exports.is_symlink() or exports.resolve(strict=True) != self.root / "exports":
            raise PermissionError("Export directory must not be redirected")
        destination = exports / (key + ".txt")
        content = payload["text"].encode("utf-8")
        # Exclusive creation refuses overwrite, symlinks and path selection by model text.
        try:
            with destination.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            # Recover a crash after a successful write but before DB commit; never change the file.
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != content:
                raise PermissionError("Existing export differs; owner intervention required") from None
        return {"id": key, "status": "exported", "path": str(destination)}

    def _check_scope(self, payload):
        if self.scope == "research" and payload["kind"] != "open_page":
            raise PermissionError("Research scope permits public page handoffs only; no files or consequential actions")

    @staticmethod
    def _row(db, key):
        if not isinstance(key, str) or len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid request ID")
        row = db.execute("SELECT * FROM requests WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Unknown request")
        return row
