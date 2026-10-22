"""Explicit loopback server. No approvals/admin methods; no automatic startup."""
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

from local_extensions.e4b_http import deadline_reader, read_json, request_headers

logger = logging.getLogger(__name__)


def handler_for(runtime):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(5)
            super().setup()
            self.rfile = deadline_reader(self.request, self.rfile)

        def do_POST(self):
            try:
                if self.path != "/v1/memory":
                    raise PermissionError("Invalid request path")
                token, length = request_headers(self.headers, self.server.server_port, 32000)
                runtime.credentials.resolve(token)
                body = read_json(self.rfile, length)
                result = runtime.dispatch(token, body)
                self.reply(200, {"success": True, **result})
            except (ValueError, PermissionError, TypeError, KeyError, OSError):
                self.reply(400, {"success": False, "error": "Memory request refused"})

        def reply(self, code, value):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            if len(payload) > 100000:
                code, payload = 400, b'{"success":false,"error":"Response too large"}'
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    return Handler


def serve(runtime, port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an unprivileged port")
    with HTTPServer(("127.0.0.1", port), handler_for(runtime)) as server:
        server.timeout = 30
        while True:
            server.handle_request()
            try:
                runtime.maintain()
            except (OSError, ValueError, KeyError, TypeError, PermissionError):
                logger.exception("Memory maintenance failed; temporary records retained")
