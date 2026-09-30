"""Assemble the submission folder, sanitised, from the working repository.

The folder is built from scratch every run so it can never drift from the repo.
Three things it does that a plain copy does not:

  1. Scrubs cluster paths. The SLURM scripts and some notes carry absolute paths
     that contain PennKeys (<path redacted> <path redacted>).
     Those are replaced by $PROJECT_ROOT, which the scripts already accept.
  2. Drops files that reference an unrelated project. CLUSTER_BETTY_GUIDE.md is a
     general cluster how-to that names another lab project's data paths; it is
     not needed to grade or reproduce this work.
  3. Excludes what is regenerated rather than shipped: the 430 MB QM9 build, the
     619 MB raw cell tree, caches, and every build artefact. The recipe for each
     is in the README instead.

Usage:
  python proj1/scripts/build_submission.py
  python proj1/scripts/build_submission.py --out SUBMISSION_CIS6270_Project1_Group2
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Absolute cluster paths embed PennKeys and, in a shared lab area, other people's
# project directories. Matching them by prefix proved too narrow: the first build
# surfaced a second lab root carrying two unrelated projects. So any absolute
# /vast or /scratch path is rewritten. A path reaching into THIS project keeps its
# tail, because the scripts already read $PROJECT_ROOT and stay runnable; anything
# else under those roots is someone else's area and is dropped whole.
SCRUB = [
    (re.compile(r"/(?:vast|scratch\d*)/[^\s\"')]*?/cis6270-project1-group2"),
     "$PROJECT_ROOT"),
    (re.compile(r"/(?:vast|scratch\d*)/[^\s\"')]*?/cgm"), "$PROJECT_ROOT"),
    (re.compile(r"/(?:vast|scratch\d*)/[^\s\"')]*"), "<path redacted>"),
    (re.compile(r"\b[a-z][a-z0-9]{2,}@(?:sas\.)?upenn\.edu\b"), "<email redacted>"),
    # bare login names in prose and in shell prompts
    (re.compile(r"\b(?:<user>|<user>|<user>)\b"), "<user>"),
]

TEXT_SUFFIXES = {".py", ".md", ".tex", ".sh", ".slurm", ".bib", ".sty", ".txt",
                 ".json", ".cfg", ".toml", ".yaml", ".yml", ".gitignore"}

# Never copied: regenerated, vendored, or belonging to another project.
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git",
                 ".DS_Store", "Thumbs.db"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".aux", ".log", ".out", ".blg", ".bbl",
                    ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"}
EXCLUDE_FILES = {
    # names another lab project's data paths; not needed to grade or reproduce
    "docs/protocol/CLUSTER_BETTY_GUIDE.md",
}

# A third party's released weights, fetched by our own script rather than
# redistributed here. The paper names the checkpoint and its md5, and
# code/scripts/fetch_tfg_assets.py retrieves it.
EXCLUDE_DIRS = {"EDMsecond"}


def wanted(rel: pathlib.PurePosixPath) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_NAMES or parts & EXCLUDE_DIRS:
        return False
    if rel.suffix in EXCLUDE_SUFFIXES:
        return False
    if str(rel) in EXCLUDE_FILES:
        return False
    return True


def copy_tree(src: pathlib.Path, dst: pathlib.Path, label: str,
              stats: dict) -> None:
    """Copy src to dst, scrubbing text files as they are written."""
    if not src.exists():
        stats.setdefault("missing", []).append(label)
        return
    for s in sorted(src.rglob("*")):
        if not s.is_file():
            continue
        rel = pathlib.PurePosixPath(s.relative_to(src).as_posix())
        if not wanted(pathlib.PurePosixPath(
                (pathlib.PurePosixPath(label) / rel).as_posix())):
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        if not wanted(rel):
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if s.suffix in TEXT_SUFFIXES:
            try:
                text = s.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                shutil.copy2(s, out)
                stats["binary"] = stats.get("binary", 0) + 1
                continue
            scrubbed = text
            for pat, repl in SCRUB:
                scrubbed, n = pat.subn(repl, scrubbed)
                if n:
                    stats["scrubbed_hits"] = stats.get("scrubbed_hits", 0) + n
                    stats.setdefault("scrubbed_files", set()).add(
                        str(pathlib.PurePosixPath(label) / rel))
            out.write_text(scrubbed, encoding="utf-8")
        else:
            shutil.copy2(s, out)
            stats["binary"] = stats.get("binary", 0) + 1
        stats["copied"] = stats.get("copied", 0) + 1


def copy_files(pairs: list[tuple[str, str]], out_root: pathlib.Path,
               stats: dict) -> None:
    for src_rel, dst_rel in pairs:
        s = ROOT / src_rel
        if not s.exists():
            stats.setdefault("missing", []).append(src_rel)
            continue
        d = out_root / dst_rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.suffix in TEXT_SUFFIXES:
            text = s.read_text(encoding="utf-8")
            for pat, repl in SCRUB:
                text = pat.sub(repl, text)
            d.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(s, d)
        stats["copied"] = stats.get("copied", 0) + 1


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="SUBMISSION_CIS6270_Project1_Group2")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    stats: dict = {}

    # --- the paper, its source and its gates ----------------------------
    copy_files([
        ("paper/main.pdf", "paper/main.pdf"),
        ("paper/main.tex", "paper/main.tex"),
        ("paper/body.tex", "paper/body.tex"),
        ("paper/citation.bib", "paper/citation.bib"),
        ("paper/refs_extra.bib", "paper/refs_extra.bib"),
        ("paper/neurips_2026.sty", "paper/neurips_2026.sty"),
        ("paper/README.md", "paper/README.md"),
        ("paper/versions/CHANGELOG.md", "paper/CHANGELOG.md"),
    ], out, stats)
    copy_tree(ROOT / "paper/figs", out / "paper/figs", "paper/figs", stats)
    copy_tree(ROOT / "paper/tools", out / "paper/tools", "paper/tools", stats)

    # --- the code -------------------------------------------------------
    for sub in ("src", "scripts", "tests", "m2", "cluster"):
        copy_tree(ROOT / "proj1" / sub, out / "code" / sub,
                  f"proj1/{sub}", stats)

    # --- the measured results, as the scripts wrote them ----------------
    copy_tree(ROOT / "docs/results", out / "results_and_docs/results",
              "docs/results", stats)
    copy_tree(ROOT / "docs/protocol", out / "results_and_docs/protocol",
              "docs/protocol", stats)
    copy_tree(ROOT / "docs/methods", out / "results_and_docs/methods",
              "docs/methods", stats)
    copy_tree(ROOT / "docs/status", out / "results_and_docs/status",
              "docs/status", stats)

    # --- the weights and their protocol ---------------------------------
    copy_tree(ROOT / "weights", out / "weights", "weights", stats)

    # --- the entry point ------------------------------------------------
    # Kept in the repository rather than written into the folder, because the
    # folder is rebuilt from scratch every run and would otherwise lose it.
    copy_files([("proj1/scripts/submission_README.md", "README.md")], out, stats)

    # --- the assignment and its rubric ----------------------------------
    for cand in ("course", "."):
        d = ROOT / cand
        for pdf in sorted(d.glob("CIS_6270_*.pdf")):
            dst = out / "assignment_and_rubric" / pdf.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, dst)
            stats["copied"] = stats.get("copied", 0) + 1

    # --- a manifest of what shipped -------------------------------------
    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    lines = ["# Manifest", "",
             f"{len(files)} files, {total/1e6:.1f} MB. "
             "Generated by `code/scripts/build_submission.py`.", "",
             "SHA-256 of the graded artefacts:", ""]
    for key in ("paper/main.pdf", "paper/main.tex"):
        p = out / key
        if p.exists():
            lines.append(f"- `{key}`  `{sha256(p)}`")
    lines += ["", "Top-level contents:", ""]
    for d in sorted(x for x in out.iterdir() if x.is_dir()):
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        sz = sum(x.stat().st_size for x in d.rglob("*") if x.is_file())
        lines.append(f"- `{d.name}/` — {n} files, {sz/1e6:.1f} MB")
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"built {out.name}: {len(files)} files, {total/1e6:.1f} MB")
    print(f"  copied {stats.get('copied',0)}, skipped {stats.get('skipped',0)}")
    print(f"  scrubbed {stats.get('scrubbed_hits',0)} path/email hits in "
          f"{len(stats.get('scrubbed_files', set()))} files")
    if stats.get("missing"):
        print("  MISSING:", ", ".join(map(str, stats["missing"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
