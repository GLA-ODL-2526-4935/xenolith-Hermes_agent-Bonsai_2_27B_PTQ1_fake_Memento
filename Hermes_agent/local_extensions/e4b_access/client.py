"""Proposal-only client: no filesystem, arbitrary HTTP, redirects or owner endpoint."""
import json
import re
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PermissionError("Broker redirects are forbidden")


class Client:
    def __init__(self, endpoint, token):
        match = re.fullmatch(r"http://127\.0\.0\.1:([0-9]{4,5})/v1/requests", endpoint or "")
        if not match or not 1024 <= int(match[1]) <= 65535:
            raise ValueError("Broker must be an explicit numeric-loopback request endpoint")
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,200}", token):
            raise PermissionError("Client credential is missing or invalid")
        self.endpoint, self.token = endpoint, token

    def call(self, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(data) > 80000:
            raise ValueError("Request exceeds size limit")
        request = Request(self.endpoint, data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=5) as response:
            payload = response.read(4097)
        if len(payload) > 4096:
            raise ValueError("Broker response exceeds limit")
        result = json.loads(payload)
        if not isinstance(result, dict) or result.get("success") is not True:
            raise PermissionError("Broker refused request")
        return result
