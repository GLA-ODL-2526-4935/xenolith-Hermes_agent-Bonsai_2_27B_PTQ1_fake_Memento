"""Strict request parsing shared by the opt-in local approval/memory services."""
import json
import io
import time


class DeadlineReader(io.RawIOBase):
    """One underlying read per call, with a total request-read deadline.

    Socket inactivity timeouts alone restart on each received fragment. read1()
    performs at most one underlying raw read, so each receive gets the remaining
    total budget rather than a fresh five seconds. This does not bound dispatch.
    """
    def __init__(self, connection, source, *, seconds=5, clock=time.monotonic):
        super().__init__()
        if not 0 < seconds <= 30:
            raise ValueError("Invalid request-read budget")
        self.connection, self.source, self.clock = connection, source, clock
        self.deadline = clock() + seconds

    def readable(self):
        return True

    def readinto(self, buffer):
        if not buffer:
            return 0
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Request-read deadline exceeded")
        self.connection.settimeout(remaining)
        payload = self.source.read1(len(buffer))
        buffer[:len(payload)] = payload
        return len(payload)

    def close(self):
        try:
            self.source.close()
        finally:
            super().close()


def deadline_reader(connection, source):
    return io.BufferedReader(DeadlineReader(connection, source))


def single_header(headers, name):
    values = headers.get_all(name, [])
    if len(values) != 1:
        raise ValueError("Expected exactly one security-sensitive header")
    return values[0]


def request_headers(headers, port, maximum):
    # Presence, not truthiness: an empty forbidden header is still forbidden.
    if "Origin" in headers or "Transfer-Encoding" in headers:
        raise PermissionError("Invalid request origin/framing")
    if single_header(headers, "Host") != f"127.0.0.1:{port}":
        raise PermissionError("Invalid host")
    size = single_header(headers, "Content-Length")
    if not size.isascii() or not size.isdecimal() or len(size) > 6:
        raise ValueError("Invalid request size")
    length = int(size)
    if not 0 < length <= maximum:
        raise ValueError("Invalid request size")
    if single_header(headers, "Content-Type") != "application/json":
        raise ValueError("Expected JSON")
    auth = single_header(headers, "Authorization")
    if not auth.startswith("Bearer "):
        raise PermissionError("Client credential required")
    return auth[7:], length


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def reject_constant(_value):
    raise ValueError("Non-finite JSON numbers are forbidden")


def check_depth(value):
    depth, quoted, escaped = 0, False, False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting limit exceeded")
        elif char in "]}":
            depth -= 1


def read_json(reader, length):
    payload = reader.read(length)
    if len(payload) != length:
        raise ValueError("Incomplete request body")
    value = payload.decode("utf-8", errors="strict")
    check_depth(value)
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
