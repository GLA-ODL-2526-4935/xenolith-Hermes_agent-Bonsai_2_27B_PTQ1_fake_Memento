"""Owner-issued single-owner client credentials; no issuance/revocation over HTTP."""
import hashlib
import secrets
import sqlite3
import uuid


class Credentials:
    def __init__(self, root):
        self.path = root / "credentials.sqlite3"

    def initialize(self):
        if self.path.exists():
            raise FileExistsError("Credential store already exists")
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE clients (id TEXT PRIMARY KEY, model TEXT, controller TEXT, revoked INTEGER)")
        self.path.chmod(0o600)

    def issue(self):
        client = uuid.uuid4().hex
        tokens = {role: secrets.token_urlsafe(32) for role in ("model", "controller")}
        with sqlite3.connect(self.path) as db:
            if db.execute("SELECT COUNT(*) FROM clients").fetchone()[0] >= 32:
                raise PermissionError("Client capacity reached; owner maintenance required")
            db.execute("INSERT INTO clients VALUES (?, ?, ?, 0)",
                       (client, hashed(tokens["model"]), hashed(tokens["controller"])))
        return {"client_id": client, **tokens}

    def resolve(self, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 200:
            raise PermissionError("Invalid credential")
        value = hashed(token)
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, model, controller FROM clients WHERE revoked=0").fetchall()
        for client, model, controller in rows:
            for role, expected in (("model", model), ("controller", controller)):
                if secrets.compare_digest(value, expected):
                    return client, role
        raise PermissionError("Invalid credential")

    def revoke(self, client):
        with sqlite3.connect(self.path) as db:
            if db.execute("UPDATE clients SET revoked=1 WHERE id=? AND revoked=0", (client,)).rowcount != 1:
                raise ValueError("Unknown or revoked client")


def hashed(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
