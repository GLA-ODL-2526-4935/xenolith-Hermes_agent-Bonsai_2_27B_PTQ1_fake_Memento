"""Exact service endpoint guard for the memory-only launcher; not OS containment."""
import ipaddress
from urllib.parse import urlsplit


class EndpointPolicy:
    def __init__(self, urls):
        allowed = set()
        for url in urls:
            parts = urlsplit(url)
            host = parts.hostname
            if (parts.scheme != "http" or parts.username or parts.password
                    or parts.query or parts.fragment or not host):
                raise ValueError("Expected credential-free numeric loopback services")
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("Only numeric loopback services are supported")
            allowed.add((str(ipaddress.ip_address(host)), parts.port or 80))
        if not allowed:
            raise ValueError("At least one explicit service endpoint is required")
        self.allowed = frozenset(allowed)

    def __call__(self, event, args):
        if event in {"socket.bind", "socket.sendto", "socket.sendmsg"}:
            raise PermissionError("Agent listening/datagram networking is disabled")
        if event == "socket.connect":
            address = args[1]
            if not isinstance(address, tuple) or len(address) not in (2, 4):
                raise PermissionError("Agent Unix/unknown socket access is disabled")
            host, port = address[:2]
        elif event == "socket.getaddrinfo":
            host, port = args[:2]
        else:
            return
        if type(port) is not int:
            raise PermissionError("An explicit configured service port is required")
        try:
            endpoint = (str(ipaddress.ip_address(host)), port)
        except (ValueError, TypeError):
            raise PermissionError("Hostname resolution is disabled in the agent") from None
        if endpoint not in self.allowed:
            raise PermissionError("Only the configured model/memory endpoints are permitted")


def profile_endpoints(config):
    from .sorting import local_endpoint
    urls = [local_endpoint(config["model"]["base_url"])]
    memory = config.get("memory", {}).get("e4b_memory", {})
    if memory.get("backend", "local") == "service":
        from .remote import validate_endpoint
        urls.append(validate_endpoint(memory.get("service_url")))
    return EndpointPolicy(urls)
