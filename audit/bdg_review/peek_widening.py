import json
import sys

d = json.load(open(sys.argv[1]))
print("PROV:", json.dumps(d["prov"])[:800])
print()
print("POOLED:", json.dumps(d["pooled"])[:1200])
print()
c = d["cells"]
print("n cells", len(c))
if isinstance(c, list):
    print("keys:", list(c[0]))
    for row in c[:12]:
        print({k: row[k] for k in list(row)[:10]})
else:
    ks = list(c)
    print("keys:", ks[:6])
    print(json.dumps(c[ks[0]])[:800])
