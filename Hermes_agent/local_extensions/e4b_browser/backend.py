"""One injected browser connection; no filesystem, execution or general debugging API."""
from dataclasses import dataclass
from typing import Protocol

from local_extensions.e4b_access.policy import public_https_url


@dataclass(frozen=True)
class PageSource:
    url: str
    title: str
    html: str


class BrowserBackend(Protocol):
    def search(self, query: str, limit: int) -> list[dict]: ...

    def open(self, url: str) -> PageSource: ...


class FixtureBrowser:
    """Owner-supplied fixtures only. Never launches, fetches or resolves anything."""
    def __init__(self, pages):
        self.pages = {public_https_url(page.url): page for page in pages}

    def search(self, query, limit):
        terms = query.casefold().split()
        matches = [page for page in self.pages.values()
                   if any(term in (page.title + " " + page.html).casefold() for term in terms)]
        return [{"url": page.url, "title": page.title} for page in matches[:limit]]

    def open(self, url):
        return self.pages[public_https_url(url)]
