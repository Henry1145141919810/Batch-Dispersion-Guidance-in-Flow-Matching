"""Report how much of the paper the MAIN TEXT occupies, against the 5-page limit.

The assignment limits the main text to 5 pages; references and the appendix do
not count.  The main text ends where the `References` heading begins, which is
usually part-way down a page, so this reports fractional pages rather than just
the page index.

Usage (from paper/):   ../.venv/Scripts/python.exe tools/pagecheck.py
"""
import re
import sys
import pathlib

try:
    import pymupdf
except ImportError:  # pragma: no cover
    sys.exit("pymupdf missing: .venv/Scripts/python.exe -m pip install pymupdf")

PDF = pathlib.Path(__file__).resolve().parent.parent / "main.pdf"
LOG = PDF.with_suffix(".log")
LIMIT = 5.0

TOP, BOTTOM = 72.0, 752.0          # neurips text block, in points


def main() -> int:
    if not PDF.exists():
        sys.exit(f"{PDF} not found; build the paper first")
    doc = pymupdf.open(PDF)

    page_idx, y = None, None
    for i, page in enumerate(doc):
        for b in page.get_text("blocks"):
            if re.match(r"^\s*References\s*$", b[4].strip()):
                page_idx, y = i, b[1]
                break
        if page_idx is not None:
            break

    if page_idx is None:
        main_pages = float(doc.page_count)
        print("references heading not found; counting the whole document")
    else:
        # Full pages before, plus the fraction of the last one that is main text.
        # A heading sitting within the first line of the text block starts that
        # page, so the sliver of white space above it is not main text.
        frac = max(0.0, min(1.0, (y - TOP) / (BOTTOM - TOP)))
        if frac < 0.02:
            frac = 0.0
        main_pages = page_idx + frac

    print(f"total pages     : {doc.page_count}")
    print(f"MAIN TEXT       : {main_pages:.2f} pages  (limit {LIMIT:.0f})")

    if LOG.exists():
        log = LOG.read_text(encoding="utf-8", errors="ignore")
        print(f"overfull hboxes : {len(re.findall(r'Overfull .hbox', log))}")
        print(f"undefined cites : {len(re.findall(r'Citation .* undefined', log))}")

    todos = sum(p.get_text().count("[v3") + p.get_text().count("[not run")
                + p.get_text().count("[restate") + p.get_text().count("[better")
                for p in doc)
    print(f"visible TODOs   : {todos}")

    if main_pages > LIMIT:
        over = main_pages - LIMIT
        print(f"\nOVER BY {over:.2f} PAGES  (~{int(over * 780)} words of prose)")
        return 1
    print(f"\nwithin the limit, {LIMIT - main_pages:.2f} pages to spare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
