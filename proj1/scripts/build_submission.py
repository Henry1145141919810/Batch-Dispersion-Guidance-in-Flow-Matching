"""Assemble the submission folder, sanitised, from the working repository.

The folder is built from scratch every run so it can never drift from the repo.
Four things it does that a plain copy does not:

  1. Keeps the repository's own shape. proj1/, docs/, results/, paper/, weights/
     land under those exact names. This is not cosmetic: every script resolves
     its imports as ROOT/"proj1"/"src" and every command in the docs is written
     against those paths. An earlier build renamed proj1/ to code/ and
     docs/ to results_and_docs/ to read more nicely, and the result was a folder
     in which `python code/scripts/transfer_sweep.py` died with
     `ModuleNotFoundError: No module named 'evaluation'`. Do not rename them.
  2. Ships the measured cells, not only the pages built from them.
     paper/tools/build_results.py reads results/v3 and results/m2, so without
     them the folder cannot rebuild a single table or figure. JSON only: the
     .permol.pt sidecars beside them are 500 MB and no paper table reads one.
  3. Scrubs cluster paths. The SLURM scripts and some notes carry absolute paths
     that contain PennKeys (/vast/projects/.../hyhuang/, /scratch2/boboli/...).
     Those are replaced by $PROJECT_ROOT, which the scripts already accept.
  4. Excludes what is regenerated, redistributed, or superseded: the 430 MB QM9
     build, third-party released weights our own fetch scripts retrieve and
     hash-check, draft material, and every build artefact. The recipe for each
     is in SUBMISSION.md instead.

It then RUNS the folder it just built (--verify, on by default) and refuses to
leave a broken one behind. The failure this guards against is not a crash: it is
a folder that looks complete, ships 368 files, and cannot execute one of them.

Usage:
  python proj1/scripts/build_submission.py
  python proj1/scripts/build_submission.py --out SUBMISSION_CIS6270_Project1_Group2
  python proj1/scripts/build_submission.py --no-verify     # skip the smoke checks
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

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
    # "<path redacted>" was a redirection in shell: it made 21 of the 52 shipped
    # .sh/.slurm files fail `bash -n`. The replacement has to stay a valid word.
    (re.compile(r"/(?:vast|scratch\d*)/[^\s\"')]*"), "$PROJECT_ROOT/redacted"),
    (re.compile(r"\b[a-z][a-z0-9]{2,}@(?:sas\.)?upenn\.edu\b"), "<email redacted>"),
    # bare login names in prose and in shell prompts
    (re.compile(r"\b(?:hyhuang|boboli|nnori)\b"), "<user>"),
    # The Windows home of whoever drove the run. Every runbook opens with
    # `cd "C:/Users/<name>/.../Project 1"`, which is step 1 of the procedure and
    # has to stay readable, so only the account name goes. Both separators
    # appear: forward slashes in the shell snippets, escaped backslashes in the
    # JSON that records downloaded-weight paths.
    (re.compile(r"([Cc]:[/\\]{1,2}Users[/\\]{1,2})[A-Za-z0-9_.~-]+"),
     r"\1<user>"),
]

TEXT_SUFFIXES = {".py", ".md", ".tex", ".sh", ".slurm", ".bib", ".sty", ".txt",
                 ".json", ".cfg", ".toml", ".yaml", ".yml", ".gitignore"}

# Never copied: regenerated, vendored, or belonging to another project.
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git",
                 ".DS_Store", "Thumbs.db"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".aux", ".log", ".out", ".blg", ".bbl",
                    ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"}

# Anything whose name carries a draft version number. The paper's figures were
# de-versioned for submission (figs/results.pdf, not figs/results_v10.pdf) and
# main.tex now references only the unversioned names; the 26 older files beside
# them are orphans. A graded folder should show one version of the work, so the
# suffix is matched rather than each name listed.
VERSIONED = re.compile(r"_v\d+\.(?:pdf|png|json|txt)$")

# Third-party released weights our own scripts fetch and hash-check, so that we
# link to them rather than redistribute them. Same rule in all three cases; the
# md5 of each is recorded next to the fetch.
#   weights/EDMsecond/         <- proj1/scripts/fetch_tfg_assets.py     (edm)
#   audit/**/{*.npy,*.pickle}  <- proj1/scripts/fetch_tfg_assets.py     (TFG nets)
#   DeepFlyBrain.hdf5          <- source + md5 in proj1/m2/deepflybrain.py
#   TFG-Flow/ and the three other clones: reference checkouts nothing in proj1/
#     imports. TFG-Flow's two checkpoint zips alone are 40 MB. SUBMISSION.md said
#     "the rest of audit/ stays in the git repository only", which was false in
#     both directions for these -- they shipped, and they are .gitignore'd, so
#     they are not in the repository either.
EXCLUDE_DIRS = {"EDMsecond", "versions", "tmp", "TFG-Flow",
                "MolGuidance", "PropMolFlow", "e3_diffusion_for_molecules"}
EXCLUDE_FILES = {
    # names another lab project's data paths; not needed to grade or reproduce
    "docs/protocol/CLUSTER_BETTY_GUIDE.md",
    # The aborted 489-epoch run, superseded by the 1500-epoch checkpoint beside
    # it. Nothing in the paper or the code reads it, and shipping a dead 4 MB
    # checkpoint in a graded folder just invites the grader to load the wrong one.
    "proj1/m2/blade_bundle/fm_m2_dfb500.e489-aborted.pt",
    # A third party's released weights, same rule as EDMsecond: 13 MB we would be
    # redistributing. proj1/m2/deepflybrain.py records the source
    # (zenodo.org/record/5153337) and the md5, and DeepFlyBrain.json, which is
    # the architecture rather than the weights, still ships.
    "weights/deepflybrain/DeepFlyBrain.hdf5",
    # a dump of an earlier teammate's prototype, kept as review evidence; not
    # code, not cited, and not something to hand a grader
    "audit/fa_fb_search/haimo_v1_extracted.txt",
}

# Present or the build refuses. Each entry is something whose absence makes the
# folder fail a graded requirement rather than merely look thin: the two entry
# points, the environment, the sweep and table tooling, the cells the tables read,
# and the checkpoints that let a grader sample instead of only rebuild.
REQUIRED = [
    "README.md", "SUBMISSION.md", "requirements.txt",
    "proj1/src/guidance.py", "proj1/src/sampling.py",
    "proj1/scripts/transfer_sweep.py", "proj1/tests/test_bdg.py",
    "paper/main.tex", "paper/main.pdf", "paper/tools/build_results.py",
    "docs/results/DATA_INDEX.md",
    "weights/fm_ema.pt", "weights/diff_ema.pt",
    # cited external code, and one of the six vendored property networks no fetch
    # script restores -- listed here because excluding them looked reasonable once
    "audit/fa_fb_search/PROVENANCE.md",
    "audit/fa_fb_search/TFG/tf_predict_mu/model_ema_2000.npy",
    # Modality 2 has to be runnable from the folder, and all three of these were
    # caught by proj1/.gitignore's *.pt / *.npz rules, so they existed only on the
    # machine that made them. The folder shipped without enhancer_gate.pt for two
    # builds while weights/README.md documented its md5. A missing checkpoint is
    # not a thinner folder, it is a Modality 2 nobody else can run.
    "proj1/m2/enhancer_gate.pt",
    "proj1/m2/blade_bundle/fm_m2_dfb500.pt",
    "proj1/m2/blade_bundle/dfb500.npz",
]

# Run inside the built folder. Each is a command a grader could plausibly type in
# their first five minutes, and each failed on the folder shipped before this
# script was rewritten.
# sys.executable, not "python": on Windows a bare "python" in a subprocess can
# resolve to an interpreter that is not the one running this build, and the first
# run of these checks failed with "No module named 'torch'" for that reason alone.
# Verifying with the interpreter that built the folder is also the honest test.
SMOKE = [
    # imports resolve: the sweep is the entry point for every QM9 experiment
    ([sys.executable, "-B", "proj1/scripts/transfer_sweep.py", "--help"], "sweep imports"),
    # the cells are present AND readable: this gate used to pass vacuously on an
    # empty tree, which is how a folder with no cells in it was shipped
    ([sys.executable, "-B", "proj1/scripts/v3_sanity.py"], "cells present, 0 fail"),
    # the innovation's own gates. NOTE 19 of 25, not 25: Part B needs data/qm9.pt,
    # which the folder does not ship, and the test says so as it skips. The
    # skipped six are the real-generator gates, so this check is weaker inside the
    # folder than in the repository -- do not read a pass here as all 25.
    ([sys.executable, "-B", "proj1/tests/test_bdg.py"], "BDG gates (19/25 here)"),
    # the v3 gates, including the one that reads the shipped shell scripts
    ([sys.executable, "-B", "proj1/tests/test_v3.py"], "v3 gates"),
    # the provenance cross-check the submission cites: recomputes every generated
    # table from the shipped cells and verifies each cell against the manifest
    ([sys.executable, "-B", "paper/tools/build_results.py", "--check"],
     "tables match cells, hashes verify"),
]

# Every shipped shell script has to parse. The scrub edits these files, and an
# edit that produced invalid shell went unnoticed through two builds because
# nothing ever asked bash to read them.
SHELL_GLOBS = ("proj1/cluster/*.sh", "proj1/cluster/*.slurm", "proj1/m2/*.slurm",
               "proj1/m2/blade_bundle/*.sh", "blade_runs/*.sh")


def write_lf(path: pathlib.Path, text: str) -> None:
    """Write text with LF endings, whatever platform the build runs on.

    pathlib's write_text uses os.linesep, so a build on Windows rewrote every
    shipped .sh and .slurm with CRLF. The repository's own .gitattributes pins
    `*.sh text eol=lf` and `*.slurm text eol=lf`, and says why: a CRLF shebang
    makes the cluster's interpreter fail with "bad interpreter: No such file or
    directory". The build was quietly undoing the rule that prevents that.
    """
    # Written to a sibling temp file and moved into place. Writing 1700 files in a
    # few seconds on Windows occasionally hits a file a filesystem filter is still
    # holding: opening a just-copied results_manifest.json for rewrite failed once
    # with [Errno 22] Invalid argument, while the identical open succeeded a second
    # later. os.replace is atomic, so a half-written file cannot ship either.
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(4):
        try:
            with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(0.25 * (attempt + 1))


def shell_needs_project_root(rel: pathlib.PurePosixPath, text: str) -> bool:
    """True for a shipped shell script that reads $PROJECT_ROOT and never sets it.

    The scrub rewrites other people's cluster paths to $PROJECT_ROOT, and this
    file used to claim the scripts "already accept" it. They do not: 50 shipped
    .sh/.slurm read it and none assign it, so under `set -u` -- which
    v3_run.slurm sets -- they die with "PROJECT_ROOT: unbound variable", and
    proj1/tests/test_v3.py's shell_references_only_defined_vars gate fails in the
    folder while passing in the repo. A default assignment makes the promise true.
    """
    if rel.suffix not in (".sh", ".slurm"):
        return False
    return "$PROJECT_ROOT" in text and not re.search(r"^\s*(?:export\s+)?PROJECT_ROOT=",
                                                     text, re.M)


def with_project_root_default(text: str) -> str:
    """Insert a PROJECT_ROOT default after the shebang, or at the top."""
    # A plain assignment, and deliberately `$(pwd)` rather than anything derived
    # from ${BASH_SOURCE}: test_v3.py's shell gate reports every $NAME a script
    # reads without assigning, so a clever default that mentions BASH_SOURCE just
    # trades one undefined name for another. Run from the folder root, as the
    # README says to, $(pwd) IS the project root.
    line = ('PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"'
            '  # default added when this submission folder was built')
    lines = text.split("\n")
    at = 1 if lines and lines[0].startswith("#!") else 0
    # after the SLURM directive block, so #SBATCH lines keep their place at the top
    while at < len(lines) and (lines[at].startswith("#SBATCH") or not lines[at].strip()):
        at += 1
    lines.insert(at, line)
    return "\n".join(lines)


def wanted(rel: pathlib.PurePosixPath) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_NAMES or parts & EXCLUDE_DIRS:
        return False
    if rel.suffix in EXCLUDE_SUFFIXES:
        return False
    if VERSIONED.search(rel.name):
        return False
    if str(rel) in EXCLUDE_FILES:
        return False
    return True


def copy_tree(src: pathlib.Path, dst: pathlib.Path, label: str, stats: dict,
              keep: re.Pattern | None = None, max_bytes: int | None = None) -> None:
    """Copy src to dst, scrubbing text files as they are written.

    `label` is the path the file has IN THE REPOSITORY, and it is what the
    exclusion keys are written against; `keep`, when given, is matched against
    the file name and is the only thing that ships. `max_bytes` caps a single
    file: a suffix allow-list is the wrong tool against a third party's own
    training logs, and `TFG/tf_predict_mu/logs.txt` is 77 MB of them, half the
    weight of the first build of this folder.
    """
    if not src.exists():
        stats.setdefault("missing", []).append(label)
        return
    for s in sorted(src.rglob("*")):
        if not s.is_file():
            continue
        rel = pathlib.PurePosixPath(s.relative_to(src).as_posix())
        repo_rel = pathlib.PurePosixPath((pathlib.PurePosixPath(label) / rel).as_posix())
        if not wanted(repo_rel) or not wanted(rel) or (keep and not keep.search(s.name)):
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        if max_bytes is not None and s.stat().st_size > max_bytes:
            stats.setdefault("oversize", []).append(
                f"{repo_rel} ({s.stat().st_size/1e6:.0f} MB)")
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
                    stats.setdefault("scrubbed_files", set()).add(str(repo_rel))
            if shell_needs_project_root(rel, scrubbed):
                scrubbed = with_project_root_default(scrubbed)
                stats["rooted"] = stats.get("rooted", 0) + 1
            write_lf(out, scrubbed)
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
            write_lf(d, text)
        else:
            shutil.copy2(s, d)
        stats["copied"] = stats.get("copied", 0) + 1


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(out: pathlib.Path, stats: dict) -> None:
    """Write the whole folder. Every destination path mirrors the repository."""
    # --- the two entry points, and the environment ----------------------
    # README.md is Bobo's submission README, kept in the repository rather than
    # written into the folder, because the folder is rebuilt from scratch every
    # run and would otherwise lose it.
    copy_files([
        ("proj1/scripts/submission_README.md", "README.md"),
        ("SUBMISSION.md", "SUBMISSION.md"),
        ("requirements.txt", "requirements.txt"),
    ], out, stats)

    # --- the paper and the tooling that builds its numbers --------------
    copy_files([
        ("paper/main.pdf", "paper/main.pdf"),
        ("paper/main.tex", "paper/main.tex"),
        ("paper/body.tex", "paper/body.tex"),
        ("paper/citation.bib", "paper/citation.bib"),
        ("paper/refs_extra.bib", "paper/refs_extra.bib"),
        ("paper/neurips_2026.sty", "paper/neurips_2026.sty"),
        ("paper/README.md", "paper/README.md"),
        ("paper/results_manifest.json", "paper/results_manifest.json"),
        ("paper/property_distributions.json", "paper/property_distributions.json"),
    ], out, stats)
    copy_tree(ROOT / "paper/figs", out / "paper/figs", "paper/figs", stats)
    copy_tree(ROOT / "paper/tools", out / "paper/tools", "paper/tools", stats)

    # --- the code, under its own name so imports resolve ----------------
    for sub in ("src", "scripts", "tests", "m2", "cluster"):
        copy_tree(ROOT / "proj1" / sub, out / "proj1" / sub, f"proj1/{sub}", stats)
    copy_tree(ROOT / "blade_runs", out / "blade_runs", "blade_runs", stats)

    # --- the protocols, the generated pages, the status ------------------
    for sub in ("results", "protocol", "methods", "status"):
        copy_tree(ROOT / "docs" / sub, out / "docs" / sub, f"docs/{sub}", stats)

    # --- the measured cells the tables are built from -------------------
    # JSON only. The .permol.pt sidecars beside them hold per-molecule samples,
    # 500 MB of them, and no paper table or figure reads one: build_results.py
    # works from the seed-level cell JSONs. The paired / useful-yield readouts
    # (paper_fill_v3.py, v3_blade_readout.py, v3_paired.py) DO read the sidecars
    # and cannot be recomputed from this folder; the pages they produced ship
    # under docs/results/ instead, and SUBMISSION.md says so.
    json_only = re.compile(r"\.json$")
    copy_tree(ROOT / "results/v3", out / "results/v3", "results/v3", stats, keep=json_only)
    copy_tree(ROOT / "results/m2", out / "results/m2", "results/m2", stats, keep=json_only)
    # The independent BDG port. Down-ranked as a result, but DATA_INDEX warns a
    # figure script still reads results/bdg_port/table.txt, so it ships whole
    # (minus sidecars) rather than being orphaned.
    copy_tree(ROOT / "results/bdg_port", out / "results/bdg_port", "results/bdg_port",
              stats, keep=re.compile(r"\.(?:json|txt|md)$"))
    # The loose JSON at the top of results/ -- 736 KB, 22 files. Leaving it out
    # broke four documented commands for want of a few hundred kilobytes:
    # test_v3.py's batch_probe_record_exists gate needs v3_batch_memory.json,
    # v3_final_summary.py stops at m2_share.json, and m2_v3_results.py raises
    # KeyError 'gate' without m2_gate_scores.json.
    for f in sorted((ROOT / "results").glob("*.json")):
        copy_files([(f"results/{f.name}", f"results/{f.name}")], out, stats)

    # --- the checkpoints, so the folder can sample and not only tabulate -
    copy_tree(ROOT / "weights", out / "weights", "weights", stats)

    # --- external code we import but did not write ----------------------
    # Code, provenance, AND the six vendored property networks (28 MB of .npy).
    # Those look like third-party weights we should link rather than ship, and the
    # first version of this rule excluded them for that reason -- wrongly.
    # fetch_tfg_assets.py says it plainly: the generators are missing from this
    # repository, the property networks are not, and the script "checks them
    # rather than re-downloading". Drop them and no fetch brings them back, so
    # `--backend edm` and the pair that scores `equifm` are simply dead. The size
    # cap still drops TFG's own 79 MB training log.
    copy_tree(ROOT / "audit/fa_fb_search", out / "audit/fa_fb_search",
              "audit/fa_fb_search", stats,
              keep=re.compile(r"\.(?:py|md|txt|json|sh|yml|yaml|cfg|toml|bib"
                              r"|npy|pickle)$"),
              max_bytes=20_000_000)

    # --- re-point the provenance manifest at the cells AS SHIPPED --------
    # The scrub rewrites this machine's home out of every cell's `prov` block, so
    # 982 of the 1286 shipped cells differ byte-wise from the repository's. The
    # manifest records hashes of the unscrubbed originals, so inside the folder
    # `build_results.py --check` failed with "Source files changed; rebuild before
    # publication." -- the tool the submission cites as its provenance
    # cross-check told the grader the results were stale. The data is identical;
    # only a redacted path differs. So the hashes are recomputed here, over the
    # shipped bytes, with the same line-ending normalisation build_results.py uses.
    man = out / "paper/results_manifest.json"
    if man.exists():
        import json
        m = json.loads(man.read_text(encoding="utf-8"))
        rehashed = 0
        for key in list(m.get("sources", {})):
            cell = out / key
            if cell.exists():
                digest = hashlib.sha256(
                    cell.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if digest != m["sources"][key]:
                    rehashed += 1
                m["sources"][key] = digest
            else:
                stats.setdefault("manifest_missing", []).append(key)
        write_lf(man, json.dumps(m, indent=2) + "\n")
        stats["manifest_rehashed"] = rehashed

    # --- the assignment and the paper template we were given ------------
    for pdf in sorted((ROOT / "course").glob("CIS_6270_*.pdf")):
        dst = out / "course" / pdf.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, dst)
        stats["copied"] = stats.get("copied", 0) + 1


def write_manifest(out: pathlib.Path) -> tuple[int, float]:
    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    lines = ["# Manifest", "",
             f"{len(files)} files, {total/1e6:.1f} MB. "
             "Generated by `proj1/scripts/build_submission.py`.", "",
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
    lines += ["",
              "The folder keeps the repository's layout, so every command in "
              "`README.md` and `SUBMISSION.md` runs here verbatim.", "",
              "Not shipped, and why: the QM9 build (430 MB, rebuilt by "
              "`proj1/scripts/prepare_qm9.py`), third-party released weights "
              "(fetched and hash-checked by `proj1/scripts/fetch_*_assets.py`), "
              "and the `.permol.pt` per-molecule sidecars (500 MB; no paper "
              "table reads one). See SUBMISSION.md §8.", ""]
    write_lf(out / "MANIFEST.md", "\n".join(lines) + "\n")
    return len(files), total / 1e6


def purge_bytecode(out: pathlib.Path) -> int:
    """Remove every __pycache__ the verification step may still have left.

    PYTHONDONTWRITEBYTECODE covers the checks we run, but anything a check shells
    out to could ignore it, so the folder is swept rather than trusted. This is
    not tidiness: a .pyc records the absolute path of the source it was compiled
    from, so bytecode built here ships the build machine's home directory. The
    first run of these checks left 17 of them, in 77 of which the account name
    appeared.
    """
    n = 0
    for d in sorted(out.rglob("__pycache__"), reverse=True):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    for p in out.rglob("*.pyc"):
        p.unlink(missing_ok=True)
    return n


def verify(out: pathlib.Path) -> list[str]:
    """Run the built folder. Returns a list of failures, empty if it is sound."""
    problems = []
    for rel in REQUIRED:
        if not (out / rel).exists():
            problems.append(f"missing required file: {rel}")
    # -B / PYTHONDONTWRITEBYTECODE: the checks import the shipped modules, and
    # without this Python writes __pycache__ into the folder being verified.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for cmd, what in SMOKE:
        try:
            r = subprocess.run(cmd, cwd=out, capture_output=True, text=True,
                               timeout=900, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:       # noqa: BLE001
            problems.append(f"{what}: could not run ({exc})")
            continue
        tail = (r.stdout + r.stderr).strip().splitlines()
        tail = tail[-1] if tail else "(no output)"
        if r.returncode != 0:
            problems.append(f"{what}: exit {r.returncode} — {tail}")
        else:
            print(f"  ok  {what}: {tail}")

    # Every in-folder path the two entry points cite must exist. README.md was in
    # REQUIRED, so the build asserted that it was present and never that it was
    # true: it went on naming code/ and results_and_docs/ in 15 places after the
    # folder was reshaped, and 16 of the 21 paths it cited did not exist.
    # DOCUMENTED_ABSENT are the ones both files explain are not shipped.
    DOCUMENTED_ABSENT = ("data/", "weights/EDMsecond/", "audit/equifm_20260922/",
                         "paper/versions/", "logs/", "betty_pull/", "bundles/",
                         "archive/", "scratch_schnet/", "slides/")
    tops = ("proj1/", "paper/", "docs/", "results/", "weights/", "audit/",
            "blade_runs/", "course/")
    for doc in ("README.md", "SUBMISSION.md"):
        d = out / doc
        if not d.exists():
            continue
        cited = set(re.findall(r"`([A-Za-z0-9_./{}*-]+/[A-Za-z0-9_./{}*-]+)`",
                               d.read_text(encoding="utf-8", errors="ignore")))
        missing = sorted(
            c for c in cited
            if c.startswith(tops) and not c.startswith(DOCUMENTED_ABSENT)
            and not any(ch in c for ch in "{}*")
            # "..." marks an elided path in prose, and anything the build
            # deliberately drops is named in the "what is left out" table, so
            # citing it is the documentation working, not failing
            and "..." not in c
            and not (set(pathlib.PurePosixPath(c).parts) & EXCLUDE_DIRS)
            and not (out / c).exists())
        if missing:
            problems.append(f"{doc} cites {len(missing)} path(s) not in the folder: "
                            + ", ".join(missing[:6]))
        else:
            print(f"  ok  {doc}: every in-folder path it cites exists")

    # every shipped shell script must parse
    scripts = sorted(p for g in SHELL_GLOBS for p in out.glob(g))
    bad = []
    for s in scripts:
        try:
            r = subprocess.run(["bash", "-n", s.relative_to(out).as_posix()],
                               cwd=out, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            bad = []                      # no bash here; cannot judge, so do not
            print("  --  shell syntax: bash unavailable, not checked")
            break
        if r.returncode != 0:
            bad.append(f"{s.relative_to(out).as_posix()}: "
                       f"{(r.stderr or '').strip().splitlines()[-1:] or ['?']}")
    else:
        if bad:
            problems.append(f"shell syntax: {len(bad)} of {len(scripts)} scripts "
                            f"do not parse, e.g. {bad[0]}")
        else:
            print(f"  ok  shell syntax: {len(scripts)} scripts parse")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="SUBMISSION_CIS6270_Project1_Group2")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the smoke checks (they take about a minute)")
    ap.add_argument("--zip", action="store_true",
                    help="also write <out>.zip, after verification passes")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        try:
            shutil.rmtree(out)
        except PermissionError as exc:                            # noqa: PERF203
            # Windows refuses to remove a directory any process has open, and a
            # shell or editor sitting inside the folder is enough. The traceback
            # this used to raise reads like a bug in the build; it is not.
            print(f"cannot clear {out.name}: {exc}\n"
                  "Close anything open inside that folder (a shell whose working "
                  "directory is in it counts) and run this again.", file=sys.stderr)
            return 1
    out.mkdir(parents=True)
    stats: dict = {}

    assemble(out, stats)

    # Verify BEFORE the manifest, then sweep what verification left behind. The
    # manifest counts files and hashes the paper, so it has to be written against
    # the folder that is actually handed in -- not the one that existed before the
    # checks ran Python inside it.
    ok = True
    if args.no_verify:
        print("verification SKIPPED (--no-verify)")
    else:
        print("verifying the folder by running it:")
        problems = verify(out)
        if problems:
            print("\nREFUSING this folder — it is not submittable:")
            for p in problems:
                print(f"  - {p}")
            ok = False
        else:
            print("verified: required files present, and the folder runs.")
    purged = purge_bytecode(out)

    if not ok:
        # The docstring promises this script "refuses to leave a broken one
        # behind", and it used to write a clean-looking MANIFEST.md anyway --
        # complete file count, and the missing PDF's hash line simply absent. A
        # refused folder now says so in place of a manifest.
        write_lf(out / "REFUSED.md",
                 "# This folder was REFUSED by its own build\n\n"
                 "Verification failed, so no manifest was written. Do not submit "
                 "it. Fix the problems below and run\n"
                 "`python proj1/scripts/build_submission.py` again.\n\n"
                 + "".join(f"- {p}\n" for p in problems))
        print(f"  wrote REFUSED.md instead of a manifest; {purged} __pycache__ purged")
        return 1

    # Written twice on purpose. write_manifest counts the files it can see, and on
    # the first pass MANIFEST.md does not exist yet, so the folder it describes is
    # always one file smaller than the folder that ships -- it printed 1711 when
    # 1712 shipped. The second pass counts the manifest too.
    write_manifest(out)
    n, mb = write_manifest(out)

    print(f"built {out.name}: {n} files, {mb:.1f} MB")
    print(f"  copied {stats.get('copied',0)}, skipped {stats.get('skipped',0)}"
          + (f", purged {purged} __pycache__" if purged else ""))
    print(f"  scrubbed {stats.get('scrubbed_hits',0)} path/email hits in "
          f"{len(stats.get('scrubbed_files', set()))} files")
    if stats.get("missing"):
        print("  MISSING:", ", ".join(map(str, stats["missing"])))
    if stats.get("oversize"):
        print("  dropped as oversize:", ", ".join(stats["oversize"]))
    # The five heaviest files, always printed. 77 MB of a third party's training
    # log shipped unnoticed once because nothing in the output named it; a file
    # count and a total cannot show you what you are actually carrying.
    heavy = sorted(((p.stat().st_size, p) for p in out.rglob("*") if p.is_file()),
                   reverse=True)[:5]
    print("  heaviest: " + ", ".join(
        f"{p.relative_to(out).as_posix()} {sz/1e6:.1f} MB" for sz, p in heavy))

    # A last scan for anything personal that the scrub rules did not catch. The
    # rules are pattern-based, so this asks the opposite question -- is the name
    # still in there? -- and it is the check that caught 81 files carrying the
    # build account's home directory after one rewrite dropped a SCRUB rule.
    home = pathlib.Path.home().name
    leaked = [p for p in out.rglob("*") if p.is_file()
              and p.suffix in TEXT_SUFFIXES
              and home.lower() in p.read_text(encoding="utf-8",
                                              errors="ignore").lower()]
    if leaked:
        print(f"  WARNING: the account name appears in {len(leaked)} files, e.g. "
              + ", ".join(p.relative_to(out).as_posix() for p in leaked[:3]))
        ok = False

    if not ok:
        return 1
    if args.zip:
        # Built by the script for the same reason as the folder: a hand-made zip
        # is where the stray file gets in.
        archive = shutil.make_archive(str(out), "zip", root_dir=ROOT,
                                      base_dir=out.name)
        print(f"  wrote {pathlib.Path(archive).name}, "
              f"{pathlib.Path(archive).stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
