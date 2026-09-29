"""Pre-submission integrity scan of the manuscript body.

Checks the structural elements the rubric looks for, the house-style bans, and the
scope rule that this paper reports the q50 target only.

Usage (from paper/):  ../.venv/Scripts/python.exe tools/check.py
"""
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.tex").read_text(encoding="utf-8")
BODY = MAIN.split("% MAIN_TEXT_START", 1)[1].split("% MAIN_TEXT_END", 1)[0]
# The main text, tables and appendix all live in main.tex.
BOTH = MAIN

# Style checks run on rendered text only. A LaTeX comment cannot render an em
# dash or a banned phrase, and TikZ section banners like "% ---- panel (a)"
# otherwise trip the em-dash rule.
PROSE = re.sub(r"(?<!\\)%.*", "", BODY)

# Phrases our own adversarial review refuted; none may appear.
BANNED = [
    "variance control",
    "holds its setpoint",
    "state of the art",
    "we can prove",
    "by definition",
    "no existing arm",
]
# House-style bans.
STYLE = ["rather than", "instead,", "not only", "delve", "leverage", "robust",
         "groundbreaking", "novel approach"]


def main() -> int:
    fail = 0

    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', BODY, re.S).group(1)
    abstract_words = len(abstract.split())
    print(f'abstract words     : {abstract_words} (200-250 required)')
    if not 200 <= abstract_words <= 250:
        fail = 1
    intro = BODY.split(r'\section{Introduction}', 1)[1].split(r'\begin{itemize}', 1)[0]
    paragraphs = [p for p in re.split(r'\n\s*\n', intro.strip()) if p.strip()]
    print(f'introduction paras : {len(paragraphs)} (3 required)')
    if len(paragraphs) != 3:
        fail = 1

    discussion = BODY.split(r'\section{Discussion}', 1)[1]
    discussion_paras = [p for p in re.split(r'\n\s*\n', discussion.strip()) if p.strip()]
    print(f'discussion paras   : {len(discussion_paras)} (3 required)')
    if len(discussion_paras) != 3:
        fail = 1

    cites = sorted(set(re.findall(r"\\citep\{([^}]*)\}", BODY)))
    keys = sorted({k.strip() for group in cites for k in group.split(",")})
    print(f"distinct citations : {len(keys)}")
    bibkeys = []
    for name in ('citation.bib', 'refs_extra.bib'):
        bibkeys.extend(re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,',
                                  (ROOT / name).read_text(encoding='utf-8')))
    duplicates = sorted({k for k in bibkeys if bibkeys.count(k) > 1})
    missing_cites = sorted(set(keys) - set(bibkeys))
    print(f'bibliography keys  : {len(duplicates)} duplicates, {len(missing_cites)} missing')
    if duplicates or missing_cites:
        print('  duplicate / missing:', duplicates, missing_cites)
        fail = 1
    print(f"TODO markers       : {BODY.count(chr(92) + 'TODO{')}")
    print(f"labels / refs      : {BODY.count(chr(92) + 'label{')}"
          f" / {BODY.count(chr(92) + 'ref{')}")
    print(f"tables / figures   : {BODY.count(chr(92) + 'begin{table}')}"
          f" / {BODY.count(chr(92) + 'begin{figure}')}")
    print(f"equations          : {BODY.count(chr(92) + 'begin{equation}')}")

    # Every float must be both labelled and referenced somewhere, and every
    # reference must resolve.  A reserved placeholder nobody points at is the
    # same defect as a dangling reference.
    labels = set(re.findall(r"\\label\{([^}]*)\}", BOTH))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", BOTH))
    dangling = sorted(refs - labels)
    unused = labels - refs
    # A table, figure or algorithm nobody points at reads as filler, so that is
    # a failure.  A displayed equation introduced by the sentence above it is
    # normal typesetting, so those are reported and not failed on.
    orphans = sorted(l for l in unused
                     if l.split(":")[0] in ("tab", "fig", "alg"))
    loose_eq = sorted(l for l in unused if l.startswith("eq:"))
    print(f"dangling refs      : {len(dangling)}"
          + (f"  -> {dangling}" if dangling else ""))
    print(f"unreferenced floats: {len(orphans)}"
          + (f"  -> {orphans}" if orphans else ""))
    print(f"equations in flow  : {len(loose_eq)}"
          + (f"  -> {loose_eq}" if loose_eq else "") + "  (not a failure)")
    if dangling or orphans:
        fail = 1

    # Placeholders.  Their presence is fine; a BARE one is not.  Every pending
    # number belongs in a table cell (\pend) or a sized box (\phfig), so a
    # \TODO surviving in the appendix means a placeholder was never built out.
    app = MAIN.split(chr(92) + "appendix", 1)[-1]
    bare = app.count(chr(92) + "TODO{")
    print(f"appendix \\pend      : {app.count(chr(92) + 'pend')}")
    print(f"appendix \\phfig     : {app.count(chr(92) + 'phfig{')}")
    print(f"bare appendix TODOs: {bare}  (must be 0; build the placeholder out)")
    if bare:
        fail = 1

    # required sections
    required = ["sec:data", "sec:fm", "sec:diff", "sec:guidance", "sec:bdg",
                "sec:transfer", "sec:overview", "sec:protocol", "sec:fmvd",
                "sec:guidworks", "sec:ablations", "sec:recent", "sec:m2",
                "sec:failure"]
    missing = [r for r in required if ("\\label{" + r + "}") not in BODY]
    print(f"required sections  : {len(required) - len(missing)}/{len(required)}")
    if missing:
        print("  MISSING:", ", ".join(missing))
        fail = 1

    # contribution bullets (the template asks for four or five)
    items = BODY.count(chr(92) + "item")
    print(f"contribution bullets: {items}")
    if not 4 <= items <= 5:
        print("  template asks for four or five")
        fail = 1

    em = PROSE.count("\u2014") + PROSE.count("---")
    print(f"em dashes          : {em}")
    if em:
        fail = 1

    q90 = len(re.findall(r"q90", BODY, re.I))
    print(f"q90 mentions       : {q90}  (must be 0; this study reports q50 only)")
    if q90:
        fail = 1

    low = BODY.lower()
    hits = [p for p in BANNED if p in low]
    print(f"refuted claims     : {len(hits)}" + (f"  -> {hits}" if hits else ""))
    if hits:
        fail = 1

    shits = [p for p in STYLE if p in low]
    print(f"style bans         : {len(shits)}" + (f"  -> {shits}" if shits else ""))
    if shits:
        fail = 1

    print("\nFAIL" if fail else "\nall checks pass")
    return fail


if __name__ == "__main__":
    sys.exit(main())
