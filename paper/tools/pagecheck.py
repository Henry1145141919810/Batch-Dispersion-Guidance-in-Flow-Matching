"""Check the compiled v3 paper, including floats, against the five-page limit.

Run after the final bibliography and LaTeX passes. Requires pypdf.
References begin on a fresh page in main.tex; every preceding page counts.
"""
from pathlib import Path
import re
from pypdf import PdfReader

root = Path(__file__).resolve().parent.parent
reader = PdfReader(root / 'main.pdf')
pages = [p.extract_text() or '' for p in reader.pages]
ref = next((i for i, text in enumerate(pages) if re.match(r'^\s*References\b', text)), None)
if ref is None:
    raise SystemExit('FAIL: References page not found')
log = (root / 'main.log').read_text(encoding='utf-8', errors='replace')
overfull = len(re.findall(r'Overfull \\[hv]box', log))
undefined = len(re.findall(r'(?:Citation|Reference).*undefined', log))
print(f'Total pages: {len(pages)}; main text including floats: {ref}; limit: 5')
print(f'Overfull boxes: {overfull}; undefined references/citations: {undefined}')
print('Pending results are intentionally marked P, not measured zeros.')
raise SystemExit(0 if ref <= 5 and not overfull and not undefined else 1)
