"""Negative control for pipeline/cx-opponent.py: is the central complex special,
or does the side-split metric just detect that neurons are lateralised?

    ./.venv/bin/python pipeline/lateral-control.py [permutations] [n_types]

cx-opponent.py passed its positive control (PFL3 -> DNa02, |r| = 1.000) and then
flagged 275 of 644 CX-type x descending-pair tests at FDR q < 0.05. That rate is too
high to take at face value: almost any bilateral cell type projects mostly to one
side, and the metric would score that as a perfect split.

So the identical metric, null and FDR are run on central-brain cell types that are
NOT in the central complex (and not mushroom body), drawn at random with a fixed
seed BEFORE any result is seen. If they hit at a similar rate, the metric measures
lateralisation, and the CX result says nothing about opponent action selection.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
PERMS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
NTYPES = int(sys.argv[2]) if len(sys.argv) > 2 else 90
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}

ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False, dtype={"root_id": "string"})
ann = ann[ann.root_id.notna()].copy(); ann["root_id"] = ann.root_id.astype("int64")
ann["ty"] = ann.cell_type.fillna(ann.hemibrain_type)
nt = dict(zip(ann.root_id, ann.top_nt.fillna("acetylcholine")))
sg = lambda r: SIGN.get(nt.get(r, "acetylcholine"), 1.0)

central = ann[ann.super_class.eq("central") & ann.side.isin(["left", "right"])
              & ~ann.cell_class.isin(["CX", "MBON", "DAN", "Kenyon_Cell"])]
counts = central.groupby("ty").side.value_counts().unstack(fill_value=0)
eligible = sorted(counts[(counts.get("left", 0) >= 4) & (counts.get("right", 0) >= 4)].index)
pick = list(np.random.default_rng(12345).choice(eligible, size=min(NTYPES, len(eligible)), replace=False))
print(f"[select] {len(eligible)} eligible non-CX central types; sampled {len(pick)} with seed 12345",
      file=sys.stderr)

cells = central[central.ty.isin(pick)]
src_ids = cells.root_id.to_numpy(dtype=np.int64)
src_ix = {int(r): i for i, r in enumerate(src_ids)}
src_side = np.array([1.0 if s == "left" else -1.0 for s in cells.side])
src_type = list(cells.ty)

dn = json.loads((OUT / "dn_circuit.json").read_text())
dn_ids = np.array([int(n["id"]) for n in dn["neurons"]], dtype=np.int64)
dn_ix = {int(r): i for i, r in enumerate(dn_ids)}
relay_ids = central.root_id.to_numpy(dtype=np.int64)          # same relay pool definition

S, Dn, Rl = np.sort(src_ids), np.sort(dn_ids), np.sort(relay_ids)
direct, s_rel, rel_dn = [], [], []
with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as f:
    rd = ipc.open_file(f)
    for b in range(rd.num_record_batches):
        t = rd.get_batch(b).select(["pre_pt_root_id", "post_pt_root_id", "syn_count"])
        pre = t.column(0).to_numpy(); post = t.column(1).to_numpy(); syn = t.column(2).to_numpy()
        k = syn >= 2; pre, post, syn = pre[k], post[k], syn[k]
        ps = np.isin(pre, S); qd = np.isin(post, Dn); qr = np.isin(post, Rl); pr = np.isin(pre, Rl)
        for m, sink in ((ps & qd, direct), (ps & qr, s_rel), (pr & qd, rel_dn)):
            sink.extend(zip(pre[m].tolist(), post[m].tolist(), syn[m].tolist()))

on = {q for _, q, _ in s_rel} & {p for p, _, _ in rel_dn}
rix = {r: k for k, r in enumerate(sorted(on))}
nS, nD, nR = len(src_ids), len(dn_ids), len(rix)
E = np.zeros((nS, nD)); A = np.zeros((nS, nR)); B = np.zeros((nR, nD))
for p, q, w in direct: E[src_ix[p], dn_ix[q]] += w * sg(p)
for p, q, w in s_rel:
    if q in rix: A[src_ix[p], rix[q]] += w * sg(p)
for p, q, w in rel_dn:
    if p in rix: B[rix[p], dn_ix[q]] += w * sg(p)
tot = np.abs(A).sum(0); tot[tot == 0] = 1
E = E + (A / tot) @ B
print(f"[pathway] {len(direct)} direct, {nR} relays on a path", file=sys.stderr)

rng = np.random.default_rng(0)
pairs = defaultdict(dict)
for j, n in enumerate(dn["neurons"]): pairs[n["t"]].setdefault(n["side"], []).append(j)
pairs = {k: (v["left"][0], v["right"][0]) for k, v in pairs.items()
         if len(v.get("left", [])) == 1 and len(v.get("right", [])) == 1 and k != "unknown"}
by = defaultdict(list)
for i, t in enumerate(src_type): by[t].append(i)

tests = []
for ty, cl in by.items():
    cl = np.array(cl); sd = src_side[cl]
    for d, (jl, jr) in pairs.items():
        L, R = E[cl, jl], E[cl, jr]; den = np.abs(L) + np.abs(R); live = den > 0
        if live.sum() < 6: continue
        s = (L - R)[live] / den[live]; s2 = sd[live]
        if (s2 > 0).sum() < 3 or (s2 < 0).sum() < 3 or s.std() == 0: continue
        r = abs(np.corrcoef(s2, s)[0, 1])
        null = np.array([abs(np.corrcoef(rng.permutation(s2), s)[0, 1]) for _ in range(PERMS)])
        tests.append((ty, d, r, (np.sum(null >= r) + 1) / (PERMS + 1), int(live.sum())))
p = np.array([x[3] for x in tests]); m = len(p)
order = np.argsort(p); q = np.empty(m); prev = 1.0
for rank, k in enumerate(order[::-1]):
    prev = min(prev, p[k] * m / (m - rank)); q[k] = prev
hit = q < 0.05
# FDR gets stricter as the number of tests grows, and the two populations run very
# different numbers of tests. The raw rate below does not depend on the pool size.
raw = float((p <= 0.001).mean())
print(f"  raw p <= 0.001 (independent of how many tests were run): {int((p <= 0.001).sum())} of {m} = {raw*100:.1f}%")
nlive = np.array([x[4] for x in tests]); big = nlive >= 12
print(f"  cells per test: median {int(np.median(nlive))}; tests with >= 12 cells: {int(big.sum())}, "
      f"raw p <= 0.001 among them: {(p[big] <= 0.001).mean()*100:.1f}%")
types_hit = {tests[k][0] for k in range(m) if hit[k]}
types_tested = {x[0] for x in tests}
print(f"non-CX central types sampled: {len(pick)}; with at least one testable DN pair: {len(types_tested)}")
print(f"tests run: {m}; surviving FDR q < 0.05: {int(hit.sum())} ({hit.mean()*100:.1f}%)")
print(f"types with at least one hit: {len(types_hit)} of {len(types_tested)} testable")
print("\nfor comparison, central complex (cx-opponent.py): 275 of 644 tests (42.7%), "
      "18 of 90 eligible types")
