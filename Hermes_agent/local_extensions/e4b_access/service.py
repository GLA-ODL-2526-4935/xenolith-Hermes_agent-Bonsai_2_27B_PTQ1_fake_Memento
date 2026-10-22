"""Loopback proposal-only transport; approvals are deliberately absent from HTTP."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from local_extensions.e4b_http import deadline_reader, read_json, request_headers

MAX_BODY = 80000


def dispatch(broker, token, body):
    broker.authenticate(token)
    if not isinstance(body, dict):
        raise ValueError("Expected object")
    if body.get("action") == "submit" and set(body) == {"action", "proposal"}:
        return broker.submit(body["proposal"])
    if body.get("action") == "status" and set(body) == {"action", "id"}:
        return broker.status(body["id"])
    raise PermissionError("Only submit/status are available; owner decisions are offline")


def handler_for(broker):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(5)
            super().setup()
            self.rfile = deadline_reader(self.request, self.rfile)

        def do_POST(self):
            try:
                if self.path != "/v1/requests":
                    raise PermissionError("Invalid request path")
                token, length = request_headers(self.headers, self.server.server_port, MAX_BODY)
                broker.authenticate(token)
                body = read_json(self.rfile, length)
                result = dispatch(broker, token, body)
                self.reply(200, {"success": True, **result})
            except (ValueError, PermissionError, TypeError, OSError):
                self.reply(400, {"success": False, "error": "Request refused"})

        def reply(self, code, body):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            return  # Never log request payloads/credentials to default stderr logs.

    return Handler


def serve(broker, port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an unprivileged fixed port")
    with HTTPServer(("127.0.0.1", port), handler_for(broker)) as server:
        server.timeout = 5
        server.serve_forever()
