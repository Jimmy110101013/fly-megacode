"""Traced morphology for the mushroom body, from the FlyWire v783 skeletons.

Input   data/raw/sk_lod1_783.parquet   Schlegel et al. 2024, Zenodo 10877326, CC BY 4.0
        data/raw/annotations.tsv
Output  data/out/atlas.json            polylines for both hemispheres, quantised

Both hemispheres are extracted and both are wired into the model, so both light up.
An earlier version used the right hemisphere alone and drew the left as dim
anatomical context that never reacted, because showing it lit would have implied
it computes; `more-brain.mjs` measures what the second one is actually worth.
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
UNIT = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0    # nm per stored unit
RDP_EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1100.0  # nm, straightness tolerance
# A skeleton splits into ~370 unbranched runs, most of them two-node twigs, and
# RDP cannot take a run below two points -- so a total-vertex budget is unreachable
# and you end up with two million straight sticks. Keep the longest runs instead,
# which carry the shape, and spend the detail budget on their curvature.
MAX_RUNS_PER_NEURON = int(sys.argv[2]) if len(sys.argv) > 2 else 25
CACHE = Path(__file__).resolve().parent.parent / "data/raw/mb_skel_cache.pkl"

CLASSES = ["KC", "MBON", "DAN", "APL"]


def pick_neurons():
    ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                      dtype={"root_id": "string"})
    ann = ann[ann.root_id.notna()].copy()
    ann["root_id"] = ann.root_id.astype("int64")
    is_apl = ann.cell_type.fillna("").str.startswith("APL")
    sel = ann.cell_class.isin(["Kenyon_Cell", "MBON", "DAN"]) | is_apl
    ann = ann[sel & ann.side.isin(["left", "right"])]

    def cls(r):
        if str(r.cell_type).startswith("APL"): return 3
        return {"Kenyon_Cell": 0, "MBON": 1, "DAN": 2}[r.cell_class]

    meta = {int(r.root_id): (cls(r), 1 if r.side == "right" else 0,
                             r.cell_type if pd.notna(r.cell_type) else "")
            for r in ann.itertuples()}
    return meta


def rdp(pts, eps):
    """Ramer-Douglas-Peucker in 3D. pts is (n,3)."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]
        L = np.linalg.norm(seg)
        sub = pts[a + 1:b]
        if L < 1e-6:
            d = np.linalg.norm(sub - pts[a], axis=1)
        else:
            d = np.linalg.norm(np.cross(sub - pts[a], seg / L), axis=1)
        i = int(np.argmax(d))
        if d[i] > eps:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m)); stack.append((m, b))
    return pts[keep]


def paths_of(node_ids, parents, xyz):
    """Split a skeleton tree into unbranched runs, each edge used exactly once."""
    idx = {n: i for i, n in enumerate(node_ids)}
    children = defaultdict(list)
    roots = []
    for i, p in enumerate(parents):
        j = idx.get(int(p))
        if j is None:
            roots.append(i)
        else:
            children[j].append(i)

    out = []
    for r in roots:
        stack = [(r, [r])]
        while stack:
            node, run = stack.pop()
            kids = children.get(node, [])
            if len(kids) == 1:
                run = run + [kids[0]]
                stack.append((kids[0], run))
            else:
                if len(run) > 1:
                    out.append(xyz[run])
                for k in kids:
                    stack.append((k, [node, k]))
    return out


def main():
    meta = pick_neurons()
    want = set(meta)
    print(f"[select] {len(want)} neurons "
          f"({sum(1 for v in meta.values() if v[1] == 1)} right / "
          f"{sum(1 for v in meta.values() if v[1] == 0)} left)", file=sys.stderr)

    if CACHE.exists():
        print(f"[cache] reusing {CACHE.name}", file=sys.stderr)
        chunks = pickle.loads(CACHE.read_bytes())
        seen = set(chunks)
        return finish(meta, chunks, seen)

    cols = ["node_id", "parent_id", "x", "y", "z", "neuron"]
    want_sorted = np.array(sorted(want), dtype=np.int64)
    pf = pq.ParquetFile(RAW / "sk_lod1_783.parquet")
    chunks = defaultdict(list)
    seen, rows = set(), 0
    first = True
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
        neu = batch.column("neuron").to_numpy(zero_copy_only=False)
        # np.isin against a sorted array; a per-row Python membership test over
        # 268M rows is the difference between minutes and never finishing.
        mask = np.isin(neu, want_sorted, assume_unique=False)
        if first:
            assert mask.any(), "no skeleton rows matched -- check the int64 join"
            print(f"[join] first batch matched {mask.sum()} rows", file=sys.stderr)
            first = False
        if mask.any():
            idx = np.flatnonzero(mask)
            sub_neu = neu[idx]
            # group by neuron with one sort instead of one full scan per neuron
            order = np.argsort(sub_neu, kind="stable")
            idx, sub_neu = idx[order], sub_neu[order]
            bounds = np.flatnonzero(np.diff(sub_neu)) + 1
            node = batch.column("node_id").to_numpy(zero_copy_only=False)[idx]
            par = batch.column("parent_id").to_numpy(zero_copy_only=False)[idx]
            xs = batch.column("x").to_numpy(zero_copy_only=False)[idx]
            ys = batch.column("y").to_numpy(zero_copy_only=False)[idx]
            zs = batch.column("z").to_numpy(zero_copy_only=False)[idx]
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(idx)]):
                rid = int(sub_neu[a])
                chunks[rid].append((node[a:b], par[a:b],
                                    np.stack([xs[a:b], ys[a:b], zs[a:b]], 1)))
                seen.add(rid)
        rows += len(neu)
        print(f"  {rows/1e6:7.1f}M rows scanned, {len(seen)} neurons found",
              file=sys.stderr, flush=True)
    print(f"[scan] {len(seen)}/{len(want)} neurons found", file=sys.stderr)
    CACHE.write_bytes(pickle.dumps(dict(chunks)))
    print(f"[cache] wrote {CACHE.name} ({CACHE.stat().st_size/1e6:.0f} MB)", file=sys.stderr)
    return finish(meta, chunks, seen)


def finish(meta, chunks, seen):
    neurons, flat, plens, offs = [], [], [], []
    lo = np.array([np.inf] * 3); hi = np.array([-np.inf] * 3)
    total_seg = 0
    for rid in sorted(seen):
        parts = chunks[rid]
        node_ids = np.concatenate([p[0] for p in parts])
        parents = np.concatenate([p[1] for p in parts])
        xyz = np.concatenate([p[2] for p in parts]).astype(np.float64)
        runs = paths_of(node_ids, parents, xyz)
        runs.sort(key=lambda r: np.linalg.norm(np.diff(r, axis=0), axis=1).sum(),
                  reverse=True)
        runs = [r for r in (rdp(r, RDP_EPS) for r in runs[:MAX_RUNS_PER_NEURON])
                if len(r) > 1]
        if not runs:
            continue
        c, side, ctype = meta[rid]
        offs.append(len(plens))
        for r in runs:
            q = np.rint(r / UNIT).astype(np.int32)
            lo = np.minimum(lo, q.min(0)); hi = np.maximum(hi, q.max(0))
            plens.append(len(q))
            # first vertex absolute, the rest as deltas: most are one or two
            # digits, which roughly halves the JSON
            d = q.copy()
            d[1:] = q[1:] - q[:-1]
            flat.extend(d.ravel().tolist())
            total_seg += len(q) - 1
        neurons.append({"id": str(rid), "c": c, "s": side, "t": ctype})
    offs.append(len(plens))

    print(f"[geometry] {len(neurons)} neurons · {len(plens)} polylines · "
          f"{total_seg} segments · {len(flat)//3} vertices", file=sys.stderr)
    print(f"[bbox] {lo.tolist()} .. {hi.tolist()} (0.25 um units)", file=sys.stderr)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "atlas.json"
    p.write_text(json.dumps({
        "source": "FlyWire FAFB v783 skeletons, Schlegel et al. 2024 (Zenodo 10877326, CC BY 4.0)",
        "unit_nm": UNIT, "classes": CLASSES, "encoding": "delta-per-polyline",
        "bbox": [*lo.astype(int).tolist(), *hi.astype(int).tolist()],
        "neurons": neurons, "offsets": offs, "plens": plens, "xyz": flat,
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
