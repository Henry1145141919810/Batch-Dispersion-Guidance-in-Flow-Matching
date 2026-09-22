"""D18: stop the mutation test from ever touching the working tree. Run once.

It mutated proj1/src/guidance.py and proj1/src/sampling.py in place and
restored them in a `finally`. A SIGKILL, a scheduler kill or a power loss
between the write and the restore leaves the repository holding a deliberately
broken implementation -- with passing-looking source, since the mutations are
syntactically valid.

Now: proj1/{src,scripts,tests} is copied to a temp tree once, mutations are
applied to the COPY, and the copy's own test files are run. The gate files
resolve their imports from their own location, so the copy is self-contained
and the working tree is never opened for writing at all.
"""
import io

NL = chr(10)
p = "results/bench/mutation_test.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''SRC = os.path.abspath("proj1/src/guidance.py")
SAMPLING = os.path.abspath("proj1/src/sampling.py")
PY = os.path.abspath(".venv/Scripts/python.exe")
TESTS = [os.path.abspath("proj1/tests/test_arms_exact.py"),
         os.path.abspath("proj1/tests/test_v2_arms.py")]'''
NEW = '''PY = os.path.abspath(".venv/Scripts/python.exe")

# Paths RELATIVE to a project root, resolved against the sandbox copy below.
# Nothing here ever names the working tree, which is the point of D18.
SRC_REL = "proj1/src/guidance.py"
SAMPLING_REL = "proj1/src/sampling.py"
TEST_REL = ["proj1/tests/test_arms_exact.py", "proj1/tests/test_v2_arms.py"]

SANDBOX = None          # set by main(); a temp copy of proj1/{src,scripts,tests}


def _p(rel):
    return os.path.join(SANDBOX, rel)


SRC = SRC_REL           # mutation entries refer to these two names
SAMPLING = SAMPLING_REL'''
assert s.count(OLD) == 1, "paths anchor"
s = s.replace(OLD, NEW)

OLD_RT = '''def run_test():
    """True only if EVERY gate file passes. A mutation is caught when any fails."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    for t in TESTS:
        r = subprocess.run([PY, t], capture_output=True, text=True, env=env,
                           timeout=900)
        if r.returncode != 0 or "ALL PASS" not in r.stdout:
            return False
    return True'''
NEW_RT = '''def run_test():
    """True only if EVERY gate file passes. A mutation is caught when any fails.

    Runs the SANDBOX copy's test files, which resolve their imports from their
    own location, so the working tree is never involved."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    for t in TEST_REL:
        r = subprocess.run([PY, _p(t)], capture_output=True, text=True,
                           env=env, timeout=900)
        if r.returncode != 0 or "ALL PASS" not in r.stdout:
            return False
    return True'''
assert s.count(OLD_RT) == 1, "run_test anchor"
s = s.replace(OLD_RT, NEW_RT)

OLD_MAIN = '''def main():
    files = sorted({m[3] if len(m) > 3 else SRC for m in MUTATIONS})
    original = {f: io.open(f, encoding="utf-8").read() for f in files}
    backups = {}
    for f in files:
        backups[f] = tempfile.mktemp(suffix=".py")
        shutil.copy2(f, backups[f])
    caught = survived = 0
    try:'''
NEW_MAIN = '''def main():
    global SANDBOX
    SANDBOX = tempfile.mkdtemp(prefix="mutation_sandbox_")
    for sub in ("src", "scripts", "tests"):
        shutil.copytree(os.path.abspath(os.path.join("proj1", sub)),
                        os.path.join(SANDBOX, "proj1", sub),
                        ignore=shutil.ignore_patterns("__pycache__"))
    print("sandbox: %s" % SANDBOX)
    print("the working tree is never written to by this script\\n")

    files = sorted({m[3] if len(m) > 3 else SRC for m in MUTATIONS})
    original = {f: io.open(_p(f), encoding="utf-8").read() for f in files}
    caught = survived = 0
    try:'''
assert s.count(OLD_MAIN) == 1, "main anchor"
s = s.replace(OLD_MAIN, NEW_MAIN)

OLD_W = '''            io.open(tgt, "w", encoding="utf-8", newline="\\n").write(
                src.replace(old, new, 1))'''
NEW_W = '''            io.open(_p(tgt), "w", encoding="utf-8", newline="\\n").write(
                src.replace(old, new, 1))'''
assert s.count(OLD_W) == 1, "write anchor"
s = s.replace(OLD_W, NEW_W)

OLD_RS = '''            io.open(tgt, "w", encoding="utf-8", newline="\\n").write(src)
    finally:
        for f in files:
            shutil.copy2(backups[f], f)
            os.remove(backups[f])'''
NEW_RS = '''            io.open(_p(tgt), "w", encoding="utf-8", newline="\\n").write(src)
    finally:
        shutil.rmtree(SANDBOX, ignore_errors=True)'''
assert s.count(OLD_RS) == 1, "restore anchor"
s = s.replace(OLD_RS, NEW_RS)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("mutation_test.py: runs against a sandbox copy (D18)")
