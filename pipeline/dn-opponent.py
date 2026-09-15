"""Is there a basal-ganglia-like output stage anywhere downstream of the mushroom body?

    ./.venv/bin/python pipeline/dn-opponent.py [permutations]

Humans select actions through opponent pathways -- a "go" and a "no-go" route
converging on each action channel -- which is exactly what the descending-neuron
experiment lacked (docs/descending-neurons.md). The mushroom body has the opponent
part: compartments owned by PPL1 hold approach-driving MBONs, compartments owned by
PAM hold avoidance-driving ones, and behaviour is roughly the difference.

So the signature to look for is a descending neuron that READS THAT DIFFERENCE:
approach-group MBONs push it one way and avoidance-group MBONs push it the other,
through direct or one-relay paths, signed by transmitter.

    consistency  c_j = sum_i sign_i * E[i,j] / sum_i |E[i,j]|     in [-1, 1]
    balance      b_j = min(approach share, avoidance share) / max

|c| near 1 with b well above 0 means the DN is wired like one pan of a balance.

The lesson of the 914-relay mistake applies: a count means nothing without a null.
MBON valence labels are shuffled at the level of MBON TYPE (left/right copies move
together) and the whole analysis is re-run each time.

Caveats that bound what this can show: transmitter identity is a prediction, and
glutamate is treated as inhibitory although it is not always; gap junctions and
receptor subtypes are not in the connectome; this is wiring, not function.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/out"
PERMS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TOP = 300                      # DNs with the most mushroom-body influence
C_MIN, B_MIN = 0.6, 0.25

mb = json.loads((OUT / "mb_circuit_both.json").read_text())
dn = json.loads((OUT / "dn_circuit.json").read_text())
nM, nD, nR = dn["n_mbon"], len(dn["neurons"]), dn["n_relay"]

# ---- valence of each MBON, from which dopamine population owns its compartment
pam = np.zeros(nM); ppl = np.zeros(nM)
for d, m, w in mb["dan_mbon"]:
    t = mb["dan"][d]["type"] or ""
    if t.startswith("PAM"): pam[m] += w
    elif t.startswith("PPL"): ppl[m] += w
sign = np.where(ppl > pam, 1.0, np.where(pam > ppl, -1.0, 0.0))   # +1 approach, -1 avoidance
types = [m["type"] for m in mb["mbon"]]

# ---- effective signed MBON -> DN influence, direct plus through one relay
D = np.zeros((nM, nD))
for i, j, w in dn["mbon_dn"]: D[i, j] += w
A = np.zeros((nM, nR))
for i, c, w in dn["mbon_relay"]: A[i, c] += w
B = np.zeros((nR, nD))
for c, j, w in dn["relay_dn"]: B[c, j] += w
share = np.abs(A).sum(0); share[share == 0] = 1
E = D + (A / share) @ B            # relay contribution weighted by each MBON's share of it

infl = np.abs(E).sum(0)
cand = np.argsort(-infl)[:TOP]
lab = sign != 0

def score(sg):
    Ec = E[np.ix_(lab, cand)]; s = sg[lab][:, None]
    c = (s * Ec).sum(0) / (np.abs(Ec).sum(0) + 1e-12)
    ap = np.abs(Ec[s[:, 0] > 0]).sum(0); av = np.abs(Ec[s[:, 0] < 0]).sum(0)
    b = np.minimum(ap, av) / (np.maximum(ap, av) + 1e-12)
    return c, b

c, b = score(sign)
hit = (np.abs(c) >= C_MIN) & (b >= B_MIN)

# ---- null: shuffle valence across MBON types, keep left/right copies together
rng = np.random.default_rng(0)
utypes = sorted(set(t for t, l in zip(types, lab) if l))
tsign = {t: sign[types.index(t)] for t in utypes}
null_counts, ge = [], np.zeros(TOP)
for _ in range(PERMS):
    vals = rng.permutation([tsign[t] for t in utypes])
    m = dict(zip(utypes, vals))
    sg = np.array([m.get(t, 0.0) if l else 0.0 for t, l in zip(types, lab)])
    cn, bn = score(sg)
    null_counts.append(int(((np.abs(cn) >= C_MIN) & (bn >= B_MIN)).sum()))
    ge += np.abs(cn) >= np.abs(c)
p = (ge + 1) / (PERMS + 1)

# Benjamini-Hochberg across the candidates
order = np.argsort(p); q = np.empty(TOP); prev = 1.0
for rank, k in enumerate(order[::-1]):
    r = TOP - rank
    prev = min(prev, p[k] * TOP / r); q[k] = prev

nc = np.array(null_counts)
print(f"MBONs with a valence: {int(lab.sum())}/{nM} "
      f"(approach {int((sign>0).sum())}, avoidance {int((sign<0).sum())}), "
      f"{len(utypes)} MBON types shuffled")
print(f"descending neurons examined: top {TOP} of {nD} by mushroom-body influence\n")
print(f"DNs wired like a balance (|c| >= {C_MIN}, balance >= {B_MIN}):")
print(f"  measured wiring      {int(hit.sum())}")
print(f"  shuffled valence     mean {nc.mean():.1f}, 95th percentile {np.percentile(nc,95):.0f}, "
      f"max {nc.max()}  ({PERMS} shuffles)")
print(f"  measured beats       {(nc < hit.sum()).mean()*100:.1f}% of shuffles\n")

sig = [k for k in np.argsort(q) if q[k] < 0.1 and b[k] >= B_MIN]
print(f"individual DNs surviving FDR q < 0.1 with both groups present: {len(sig)}")
for k in sig[:15]:
    j = cand[k]; n = dn["neurons"][j]
    print(f"  {n['t']:<12} {n['side']:<6} c={c[k]:+.2f}  balance={b[k]:.2f}  p={p[k]:.4f}  q={q[k]:.3f}")

# ---- are the balance-like DNs separate channels, or one pool?
pick = [cand[k] for k in sig] if len(sig) >= 2 else [cand[k] for k in np.where(hit)[0]]
def cosmed(cols):
    X = E[:, cols]; X = X / (np.linalg.norm(X, axis=0) + 1e-12)
    S = X.T @ X; iu = np.triu_indices(len(cols), 1)
    return float(np.median(S[iu])) if len(iu[0]) else float("nan")
if len(pick) >= 2:
    rand = [cosmed(list(rng.choice(cand, len(pick), replace=False))) for _ in range(500)]
    print(f"\nmedian input-pattern cosine among those DNs: {cosmed(pick):.3f} "
          f"(random DN sets of the same size: {np.median(rand):.3f})")
    print("  lower = more like separate per-action channels, higher = one shared pool")
