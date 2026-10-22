"""Deterministic HTML-to-text sections, with no model, summaries or trust filtering."""
from html.parser import HTMLParser

MAX_HTML_BYTES = 2_000_000
MAX_TEXT_CHARS = 400_000
MAX_SECTIONS = 2000
HARD_SECTION_CHARS = 8000
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCKS = HEADINGS | {"p", "div", "li", "br", "tr", "pre", "blockquote", "section", "article"}


class TextSections(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.heading = False
        self.heading_parts = []
        self.title = "Page"
        self.parts = []
        self.sections = []
        self.count = 0

    def flush(self):
        value = "".join(self.parts).strip()
        self.parts = []
        if value:
            # Bound even a heading-free, single-paragraph document without losing text.
            for start in range(0, len(value), HARD_SECTION_CHARS):
                self.sections.append((self.title[:160], value[start:start + HARD_SECTION_CHARS]))
                if len(self.sections) > MAX_SECTIONS:
                    raise ValueError("Page section limit exceeded")

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag in HEADINGS:
            self.flush()
            self.heading = True
            self.heading_parts = []
        elif tag in BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if tag in HEADINGS and self.heading:
            self.title = "".join(self.heading_parts).strip() or "Section"
            self.parts.append(self.title + "\n")
            self.heading = False
        elif tag in BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.hidden:
            return
        self.count += len(data)
        if self.count > MAX_TEXT_CHARS:
            raise ValueError("Page text limit exceeded; narrower extraction required")
        if self.heading:
            self.heading_parts.append(data)
        else:
            self.parts.append(data)


def extract(html):
    if not isinstance(html, str) or len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise ValueError("Page source limit exceeded; narrower extraction required")
    parser = TextSections()
    parser.feed(html)
    parser.close()
    if parser.heading:
        parser.parts.extend(parser.heading_parts)
    parser.flush()
    return parser.sections
