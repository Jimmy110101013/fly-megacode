"""Traced morphology for the central complex, to sit beside the mushroom body.

    ./.venv/bin/python pipeline/extract_cx_skeletons.py [rdp_nm] [runs]

The reservoir holding the fly's working memory is 2,875 real neurons; drawing the
mushroom body without them would show the half of the loop that decides and hide
the half that remembers. Same polyline format and same coordinate space as
atlas.json, so the two can be rendered together.
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_skeletons import rdp, paths_of        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
RDP_EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
MAX_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
UNIT = 400.0                       # must match atlas.json
CACHE = RAW / "cx_skel_cache.pkl"


def scan(want):
    if CACHE.exists():
        print(f"[cache] reusing {CACHE.name}", file=sys.stderr)
        return pickle.loads(CACHE.read_bytes())
    cols = ["node_id", "parent_id", "x", "y", "z", "neuron"]
    want_sorted = np.array(sorted(want), dtype=np.int64)
    pf = pq.ParquetFile(RAW / "sk_lod1_783.parquet")
    chunks, rows = defaultdict(list), 0
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
        neu = batch.column("neuron").to_numpy(zero_copy_only=False)
        mask = np.isin(neu, want_sorted)
        if mask.any():
            idx = np.flatnonzero(mask)
            sub = neu[idx]
            order = np.argsort(sub, kind="stable")
            idx, sub = idx[order], sub[order]
            bounds = np.flatnonzero(np.diff(sub)) + 1
            col = {c: batch.column(c).to_numpy(zero_copy_only=False)[idx] for c in cols[:5]}
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(idx)]):
                chunks[int(sub[a])].append((
                    col["node_id"][a:b], col["parent_id"][a:b],
                    np.stack([col["x"][a:b], col["y"][a:b], col["z"][a:b]], 1)))
        rows += len(neu)
        print(f"  {rows/1e6:7.1f}M rows, {len(chunks)} neurons", file=sys.stderr, flush=True)
    CACHE.write_bytes(pickle.dumps(dict(chunks)))
    return dict(chunks)


def main():
    cx = json.loads((OUT / "cx_circuit.json").read_text())
    order = [int(n["root_id"]) for n in cx["neurons"]]        # index must match the reservoir
    chunks = scan(set(order))
    print(f"[scan] {len(chunks)}/{len(order)} central-complex neurons found", file=sys.stderr)

    neurons, flat, plens, offs = [], [], [], []
    lo = np.array([np.inf] * 3); hi = np.array([-np.inf] * 3)
    seg = 0
    for cxi, rid in enumerate(order):
        parts = chunks.get(rid)
        if not parts:
            continue
        xyz = np.concatenate([p[2] for p in parts]).astype(np.float64)
        runs = paths_of(np.concatenate([p[0] for p in parts]),
                        np.concatenate([p[1] for p in parts]), xyz)
        runs.sort(key=lambda r: np.linalg.norm(np.diff(r, axis=0), axis=1).sum(), reverse=True)
        runs = [r for r in (rdp(r, RDP_EPS) for r in runs[:MAX_RUNS]) if len(r) > 1]
        if not runs:
            continue
        offs.append(len(plens))
        for r in runs:
            q = np.rint(r / UNIT).astype(np.int32)
            lo = np.minimum(lo, q.min(0)); hi = np.maximum(hi, q.max(0))
            plens.append(len(q))
            d = q.copy(); d[1:] = q[1:] - q[:-1]
            flat.extend(d.ravel().tolist())
            seg += len(q) - 1
        # `cx` is the index into the reservoir's state vector, so the page can light
        # each cell by what that neuron is actually holding
        neurons.append({"id": str(rid), "cx": cxi,
                        "t": cx["neurons"][cxi]["family"]})
    offs.append(len(plens))

    print(f"[geometry] {len(neurons)} neurons · {len(plens)} polylines · "
          f"{seg} segments · {len(flat)//3} vertices", file=sys.stderr)
    p = OUT / "cx_atlas.json"
    p.write_text(json.dumps({
        "source": "FlyWire FAFB v783 skeletons (Zenodo 10877326, CC BY 4.0)",
        "unit_nm": UNIT, "encoding": "delta-per-polyline",
        "bbox": [*lo.astype(int).tolist(), *hi.astype(int).tolist()],
        "neurons": neurons, "offsets": offs, "plens": plens, "xyz": flat,
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
