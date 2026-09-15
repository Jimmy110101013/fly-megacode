"""Does mushroom-body valence reach the central complex's side-split output cells?

    ./.venv/bin/python pipeline/cx-valence.py [permutations]

pipeline/cx-opponent.py found a hemisphere push-pull stage in the central complex's
output onto descending neurons. Whether that matters for the megacode depends on
something it did not test: can the mushroom body's approach/avoidance signal steer
it? If approach and avoidance MBONs land on the same cells indiscriminately, the
opponent stage sits downstream of a signal that cannot drive it apart.

Side-split types are the 18 that survived FDR q < 0.05 in cx-opponent.py (listed
below, copied from that run, not re-selected here).

Statistics, both fixed before running, each on two pathways (direct MBON -> CX, and
through one CX -> CX recurrent hop), so four p-values; Bonferroni threshold 0.0125:

  segregation  1 - cosine between approach-group and avoidance-group drive vectors
               across the side-split cells. High = the two valences reach
               different cells, so they could route to different outputs.
  lateral      |correlation| between (approach - avoidance) drive and soma side.
               High = valence difference lines up with the left/right axis itself.

Null: MBON valence shuffled across MBON types, left/right copies together -- the
null validated in dn-opponent.py.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/out"
PERMS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SPLIT_TYPES = {"EPG", "FC2B", "FC2C", "FS4A", "FS4C", "PFL3", "FR2", "PFL1", "PFR",
               "FC1C,FC1E", "FC2A", "FS1B", "PEG", "FR1", "FC1D", "FB5V", "FS2", "FS1A"}
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}

mb = json.loads((OUT / "mb_circuit_both.json").read_text())
cx = json.loads((OUT / "cx_circuit.json").read_text())
co = json.loads((OUT / "cx_out.json").read_text())
nM, nC = len(mb["mbon"]), len(cx["neurons"])
assert cx["n_mbon"] == nM, "MBON index spaces disagree -- the MBON->CX bug again"

pam = np.zeros(nM); ppl = np.zeros(nM)
for d, m, w in mb["dan_mbon"]:
    t = mb["dan"][d]["type"] or ""
    if t.startswith("PAM"): pam[m] += w
    elif t.startswith("PPL"): ppl[m] += w
val = np.where(ppl > pam, 1.0, np.where(pam > ppl, -1.0, 0.0))
mtype = [m["type"] for m in mb["mbon"]]
msign = np.array([SIGN.get(m["nt"], 1.0) for m in mb["mbon"]])

M = np.zeros((nM, nC))
for i, c, w in cx["mbon_cx"]: M[i, c] += w * msign[i]
W = np.zeros((nC, nC))
for a, b, w in cx["edges"]:
    if w >= cx.get("min_syn", 2): W[a, b] += w * cx["neurons"][a]["sign"]
indeg = np.abs(W).sum(0); indeg[indeg == 0] = 1
M1 = M + M @ (W / indeg)          # plus one recurrent hop, weighted by share of input

cells = np.array([i for i, n in enumerate(cx["neurons"]) if n["type"] in SPLIT_TYPES])
side = np.array([1.0 if s == "left" else -1.0 if s == "right" else 0.0 for s in co["cx_side"]])[cells]

def stats(Mx, v):
    ap = np.abs(Mx[v > 0][:, cells]).sum(0); av = np.abs(Mx[v < 0][:, cells]).sum(0)
    seg = 1 - ap @ av / (np.linalg.norm(ap) * np.linalg.norm(av) + 1e-12)
    diff = Mx[v > 0][:, cells].sum(0) - Mx[v < 0][:, cells].sum(0)
    lat = abs(np.corrcoef(side, diff)[0, 1]) if diff.std() > 0 else 0.0
    return seg, lat

utypes = sorted(set(mtype)); tval = {t: val[mtype.index(t)] for t in utypes}
rng = np.random.default_rng(0)
shuffles = []
for _ in range(PERMS):
    m = dict(zip(utypes, rng.permutation([tval[t] for t in utypes])))
    shuffles.append(np.array([m[t] for t in mtype]))

reach = np.abs(M[:, cells]).sum() / (np.abs(M).sum() + 1e-12)
print(f"side-split CX cells: {len(cells)} of {nC} ({len(cells)/nC*100:.1f}% of the central complex)")
print(f"share of direct MBON->CX drive landing on them: {reach*100:.1f}%\n")
print("pathway     statistic      measured   shuffled mean   p")
for label, Mx in (("direct", M), ("+1 hop", M1)):
    s0, l0 = stats(Mx, val)
    ns = np.array([stats(Mx, v) for v in shuffles])
    for k, name, obs in ((0, "segregation", s0), (1, "lateral", l0)):
        pv = (np.sum(ns[:, k] >= obs) + 1) / (PERMS + 1)
        print(f"{label:<11} {name:<14} {obs:8.3f}   {ns[:, k].mean():13.3f}   {pv:.4f}")
print("\nBonferroni threshold for four tests: 0.0125")
