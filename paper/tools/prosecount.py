"""Count main-text prose words, excluding tables, equations, figures and markup.

Used to hold the 5-page limit: the paper's prose budget is roughly 2,100 words
once the figure, five tables and the displayed equations are accounted for.

Usage (from paper/):  ../.venv/Scripts/python.exe tools/prosecount.py
"""
import re
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "body.tex"


def strip(body: str) -> str:
    for env in ("table", "equation", "figure", "algorithm", "algorithmic"):
        body = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", " ",
                      body, flags=re.S)
    body = re.sub(r"(?m)^\s*%.*$", " ", body)          # comment lines
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)         # control sequences
    body = re.sub(r"[{}\[\]$~^_&\\]", " ", body)        # markup leftovers
    return re.sub(r"\s+", " ", body).strip()


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    body = src
    words = strip(body).split()
    print(f"main-text prose words: {len(words)}")

    # Per-section breakdown, so it is obvious which block to cut next.
    parts = re.split(r"\\(?:sub)?section\{([^}]*)\}", body)
    if len(parts) > 1:
        print("\nper section:")
        for name, chunk in zip(parts[1::2], parts[2::2]):
            n = len(strip(chunk).split())
            if n:
                print(f"  {n:5d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
