"""Bounded, session-private page cache. Caller owns turn resets, never the model."""
import json
import uuid
from collections import OrderedDict

from local_extensions.e4b_access.policy import public_https_url

from .page import extract


def size(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def bounded(value, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("Expected bounded nonempty text")
    return value


class BrowserSession:
    def __init__(self, backend, *, turn_bytes=8000, response_bytes=3000):
        if type(turn_bytes) is not int or not 512 <= turn_bytes <= 16000:
            raise ValueError("Invalid owner turn budget")
        if type(response_bytes) is not int or not 512 <= response_bytes <= min(turn_bytes, 4096):
            raise ValueError("Invalid owner response budget")
        self.backend = backend
        self.turn_bytes, self.response_bytes = turn_bytes, response_bytes
        self.remaining = turn_bytes
        self.pages, self.cursors = OrderedDict(), OrderedDict()

    def begin_turn(self, available_bytes=None):
        """Controller-only hook; budget must account for existing prompt/context overhead."""
        if available_bytes is None:
            available_bytes = self.turn_bytes
        if type(available_bytes) is not int or available_bytes < 0:
            raise ValueError("Invalid available-context allowance")
        self.remaining = min(self.turn_bytes, available_bytes)

    def deliver(self, value):
        amount = size(value)
        if amount > min(self.remaining, self.response_bytes):
            raise ValueError("Browser output budget exhausted; controller must provide more context allowance")
        self.remaining -= amount
        return value

    def cursor(self, state):
        key = uuid.uuid4().hex
        self.cursors[key] = state
        while len(self.cursors) > 128:
            self.cursors.popitem(last=False)
        return key

    def position(self, cursor, prefix):
        if cursor is None:
            return 0
        state = self.cursors.get(cursor)
        if state is None or state[:-1] != prefix:
            raise ValueError("Unknown, expired or mismatched continuation")
        return state[-1]

    def page(self, page_id):
        if page_id not in self.pages:
            raise ValueError("Unknown or evicted page; open it again")
        return self.pages[page_id]

    def search(self, query, limit=5):
        bounded(query, 500)
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("Invalid search result limit")
        if min(self.remaining, self.response_bytes) < 512:
            raise ValueError("Browser output budget exhausted")
        raw = self.backend.search(query, limit)
        if not isinstance(raw, list) or len(raw) > limit:
            raise ValueError("Invalid adapter results")
        results = [{"url": public_https_url(item["url"]), "title": bounded(item["title"], 2000)[:160]}
                   for item in raw]
        # Never silently omit results; caller can retry with a smaller explicit limit.
        return self.deliver({"results": results})

    def open(self, url):
        url = public_https_url(url)
        if min(self.remaining, self.response_bytes) < 512:
            raise ValueError("Browser output budget exhausted")
        source = self.backend.open(url)
        final_url = public_https_url(source.url)
        sections = extract(source.html)
        page_id = uuid.uuid4().hex
        result = {"page_id": page_id, "url": final_url, "title": bounded(source.title, 2000)[:160],
                  "section_count": len(sections), "next": "outline"}
        self.deliver(result)
        self.pages[page_id] = sections
        while len(self.pages) > 4:
            self.pages.popitem(last=False)
        return result

    def outline(self, page_id, query="", cursor=None, limit=8):
        sections = self.page(page_id)
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Invalid section query")
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("Invalid outline limit")
        terms = query.casefold().split()
        matches = [(i, title, text) for i, (title, text) in enumerate(sections)
                   if not terms or all(term in (title + " " + text).casefold() for term in terms)]
        prefix = ("outline", page_id, query)
        start = self.position(cursor, prefix)
        count = min(limit, len(matches) - start)
        while count >= 0:
            end = start + count
            more = end < len(matches)
            result = {"sections": [{"id": f"s{i}", "title": title, "characters": len(text)}
                                   for i, title, text in matches[start:end]],
                      "matching_sections": len(matches), "cursor": "x" * 32 if more else None}
            if size(result) <= min(self.remaining, self.response_bytes) and (count or not more):
                result["cursor"] = self.cursor((*prefix, end)) if more else None
                return self.deliver(result)
            count -= 1
        raise ValueError("Browser output budget exhausted")

    def read(self, page_id, section_id, cursor=None, max_chars=1800):
        sections = self.page(page_id)
        if not isinstance(section_id, str) or not section_id.startswith("s") or not section_id[1:].isdecimal():
            raise ValueError("Invalid section ID")
        index = int(section_id[1:])
        if index >= len(sections) or section_id != f"s{index}":
            raise ValueError("Unknown section")
        if type(max_chars) is not int or not 1 <= max_chars <= 3000:
            raise ValueError("Invalid chunk size")
        text = sections[index][1]
        prefix = ("read", page_id, section_id)
        start = self.position(cursor, prefix)
        low, high = 0, min(max_chars, len(text) - start)
        # Fit serialized UTF-8, including JSON escaping and metadata, not character guesses.
        while low < high:
            count = (low + high + 1) // 2
            value = {"text": text[start:start + count], "cursor": "x" * 32 if start + count < len(text) else None}
            if size(value) <= min(self.remaining, self.response_bytes):
                low = count
            else:
                high = count - 1
        if low == 0 and start < len(text):
            raise ValueError("Browser output budget exhausted")
        end = start + low
        result = {"text": text[start:end], "cursor": self.cursor((*prefix, end)) if end < len(text) else None}
        return self.deliver(result)
