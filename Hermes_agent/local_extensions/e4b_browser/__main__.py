"""Owner-only offline fixture preview. Does not register tools or contact a browser."""
import argparse
import json
from pathlib import Path

from .backend import FixtureBrowser, PageSource
from .session import BrowserSession
from .surface import BrowserSurface


def preview(fixture, url):
    if fixture.stat().st_size > 2_000_000:
        raise ValueError("Fixture is too large")
    entries = json.loads(fixture.read_text(encoding="utf-8"))
    pages = [PageSource(**entry) for entry in entries]
    session = BrowserSession(FixtureBrowser(pages))
    surface = BrowserSurface(session, {})
    surface.dispatch({"action": "describe", "section": "navigation", "name": "open"})
    page = surface.dispatch({"action": "call", "section": "navigation", "name": "open", "arguments": {"url": url}})
    surface.dispatch({"action": "describe", "section": "reading", "name": "outline"})
    outline = surface.dispatch({"action": "call", "section": "reading", "name": "outline",
                                "arguments": {"page_id": page["page_id"]}})
    return {"page": page, "outline": outline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="Owner-supplied JSON array of url/title/html fixtures")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    print(json.dumps(preview(args.fixture, args.url), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
