"""Does the central complex hold opponent read-outs onto descending neurons?

    ./.venv/bin/python pipeline/cx-opponent.py [permutations]

pipeline/dn-opponent.py found nothing distinguishable from chance below the mushroom
body -- but it never showed its detector could find anything. This one starts from a
case where the answer is known and only continues if the detector recovers it.

The axis here is HEMISPHERE, not valence: the central complex has no PAM/PPL1 split,
and shuffling transmitter signs would destroy a structural property of the tissue
and manufacture a positive. Steering in Drosophila is a left/right opponency --
PFL3 cells from the two sides drive the left and right DNa02 differently, and the
difference between those two descending neurons is the turn.

Metric, fixed before running:
    for each CX cell i and a descending type with one left and one right cell,
    s_i = (E[i, DN_left] - E[i, DN_right]) / (|E[i, DN_left]| + |E[i, DN_right]|)
    r   = correlation between soma side (+1 left, -1 right) and s_i across the type
    |r| near 1: the type is split by hemisphere into two groups driving opposite sides.
Null: shuffle soma side labels WITHIN the CX type, keeping its left/right counts.

Stage 1, positive control: PFL3 onto DNa02 must reach p < 0.01, or the script stops.
Stage 2: every CX type with >= 4 cells per side, against every descending type with
exactly one left and one right cell, with Benjamini-Hochberg across all tests.

A hit is a two-alternative, left/right opponent axis. It is not a ten-way selector,
and it is not evidence of a basal ganglia.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/out"
PERMS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
rng = np.random.default_rng(0)

cx = json.loads((OUT / "cx_circuit.json").read_text())
dn = json.loads((OUT / "dn_circuit.json").read_text())
co = json.loads((OUT / "cx_out.json").read_text())
nC, nD, nR = len(cx["neurons"]), len(dn["neurons"]), len(co["relays"])

D = np.zeros((nC, nD))
for i, j, w in co["cx_dn"]: D[i, j] += w
A = np.zeros((nC, nR))
for i, c, w in co["cx_relay"]: A[i, c] += w
B = np.zeros((nR, nD))
for c, j, w in co["relay_dn"]: B[c, j] += w
tot = np.abs(A).sum(0); tot[tot == 0] = 1
E = D + (A / tot) @ B

side = np.array([1.0 if s == "left" else -1.0 if s == "right" else 0.0 for s in co["cx_side"]])
ctype = [n["type"] or "unknown" for n in cx["neurons"]]
by_type = defaultdict(list)
for i, t in enumerate(ctype): by_type[t].append(i)
dn_pairs = defaultdict(dict)
for j, n in enumerate(dn["neurons"]): dn_pairs[n["t"]].setdefault(n["side"], []).append(j)
dn_pairs = {t: (s["left"][0], s["right"][0]) for t, s in dn_pairs.items()
            if len(s.get("left", [])) == 1 and len(s.get("right", [])) == 1 and t != "unknown"}

def test(cells, jl, jr):
    cells = np.array(cells)
    L, R = E[cells, jl], E[cells, jr]
    denom = np.abs(L) + np.abs(R)
    live = denom > 0
    if live.sum() < 6: return None
    s = (L - R)[live] / denom[live]; sd = side[cells][live]
    if (sd > 0).sum() < 3 or (sd < 0).sum() < 3 or s.std() == 0: return None
    r = abs(np.corrcoef(sd, s)[0, 1])
    null = np.empty(PERMS)
    for k in range(PERMS):
        null[k] = abs(np.corrcoef(rng.permutation(sd), s)[0, 1])
    return r, (np.sum(null >= r) + 1) / (PERMS + 1), int(live.sum()), float(np.abs(s).mean())

print(f"CX {nC}, DN {nD}, relays {nR}; {PERMS} side shuffles per test\n")
print("stage 1 -- positive control, known steering opponency")
ok = True
for dty in ["DNa02", "DNa03"]:
    jl, jr = dn_pairs[dty]
    res = test(by_type["PFL3"], jl, jr)
    if res is None:
        print(f"  PFL3 -> {dty}: too few cells reach it"); 
        if dty == "DNa02": ok = False
        continue
    r, p, n, sel = res
    print(f"  PFL3 -> {dty}: |r| = {r:.3f}, p = {p:.4f}, {n} cells, mean |side preference| = {sel:.2f}")
    if dty == "DNa02" and p >= 0.01: ok = False
if not ok:
    print("\nPositive control FAILED. The detector cannot recover the known case; stopping.")
    sys.exit(0)
print("  passed: the detector recovers the known case\n")

print("stage 2 -- every CX type x every left/right descending pair")
tests = []
for t, cells in by_type.items():
    sd = side[cells]
    if (sd > 0).sum() < 4 or (sd < 0).sum() < 4: continue
    for dty, (jl, jr) in dn_pairs.items():
        if np.abs(E[cells][:, [jl, jr]]).sum() == 0: continue
        res = test(cells, jl, jr)
        if res: tests.append((t, dty, *res))
p = np.array([x[3] for x in tests]); m = len(p)
order = np.argsort(p); q = np.empty(m); prev = 1.0
for rank, k in enumerate(order[::-1]):
    prev = min(prev, p[k] * m / (m - rank)); q[k] = prev
hits = sorted([(q[k], *tests[k]) for k in range(m) if q[k] < 0.05])
# FDR gets stricter as the number of tests grows, and the two populations run very
# different numbers of tests. The raw rate below does not depend on the pool size.
raw = float((p <= 0.001).mean())
print(f"  raw p <= 0.001 (independent of how many tests were run): {int((p <= 0.001).sum())} of {m} = {raw*100:.1f}%")
print(f"  tests run: {m}  (CX types with >= 4 cells per side, DN types with one cell per side)")
print(f"  surviving FDR q < 0.05: {len(hits)}\n")
fam = defaultdict(list)
for h in hits: fam[h[1]].append(h)
for t in sorted(fam, key=lambda t: min(h[0] for h in fam[t])):
    hs = fam[t]
    print(f"  {t:<10} -> " + ", ".join(f"{h[2]} (|r|={h[3]:.2f})" for h in hs[:6])
          + (f"  +{len(hs)-6} more" if len(hs) > 6 else ""))
print(f"\n  CX types with at least one hit: {len(fam)} of "
      f"{sum(1 for t, c in by_type.items() if (side[c]>0).sum()>=4 and (side[c]<0).sum()>=4)} eligible")
print(f"  DN types reached by a hit: {len({h[2] for h in hits})} of {len(dn_pairs)} left/right pairs")
