"""A whole-brain silhouette to sit behind the mushroom body.

    ./.venv/bin/python pipeline/extract_context.py [neurons] [rdp_nm] [runs]

The atlas so far shows 5,606 cells, which is 4% of the brain and gives no sense of
where in the animal any of it sits. This samples neurons from everywhere -- optic
lobes, central complex, everything -- at low detail, so the mushroom body can glow
inside an actual brain instead of floating in black.

Output: data/out/context.json, same polyline format as atlas.json.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_skeletons import rdp, paths_of        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
RDP_EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 3000.0
MAX_RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
UNIT = 400.0


def main():
    ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                      dtype={"root_id": "string"})
    ann = ann[ann.root_id.notna()].copy()
    ann["root_id"] = ann.root_id.astype("int64")
    # Spread the sample across cell classes so the optic lobes, which are most of
    # the brain by count, do not drown out everything else.
    rng = np.random.default_rng(7)
    groups = ann.groupby(ann.cell_class.fillna("unknown"))
    per = max(1, N_SAMPLE // max(1, len(groups)))
    picks = []
    for _, g in groups:
        take = min(len(g), per * 3 if len(g) > per * 3 else len(g))
        picks.append(g.root_id.to_numpy()[rng.permutation(len(g))[:take]])
    want = np.unique(np.concatenate(picks))
    if len(want) > N_SAMPLE:
        want = want[rng.permutation(len(want))[:N_SAMPLE]]
    print(f"[select] {len(want)} neurons sampled across {len(groups)} classes",
          file=sys.stderr)

    chunks = defaultdict(list)
    cols = ["node_id", "parent_id", "x", "y", "z", "neuron"]
    pf = pq.ParquetFile(RAW / "sk_lod1_783.parquet")
    rows = 0
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
        neu = batch.column("neuron").to_numpy(zero_copy_only=False)
        mask = np.isin(neu, want)
        if mask.any():
            idx = np.flatnonzero(mask)
            sub = neu[idx]
            order = np.argsort(sub, kind="stable")
            idx, sub = idx[order], sub[order]
            bounds = np.flatnonzero(np.diff(sub)) + 1
            cell = {c: batch.column(c).to_numpy(zero_copy_only=False)[idx] for c in cols[:5]}
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(idx)]):
                chunks[int(sub[a])].append((
                    cell["node_id"][a:b], cell["parent_id"][a:b],
                    np.stack([cell["x"][a:b], cell["y"][a:b], cell["z"][a:b]], 1)))
        rows += len(neu)
        print(f"  {rows/1e6:7.1f}M rows, {len(chunks)} neurons", file=sys.stderr, flush=True)

    neurons, flat, plens, offs = [], [], [], []
    lo = np.array([np.inf] * 3); hi = np.array([-np.inf] * 3)
    seg = 0
    for rid in sorted(chunks):
        parts = chunks[rid]
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
        neurons.append({"id": str(rid), "c": 0, "s": 0, "t": "context"})
    offs.append(len(plens))

    print(f"[geometry] {len(neurons)} neurons · {len(plens)} polylines · "
          f"{seg} segments · {len(flat)//3} vertices", file=sys.stderr)
    p = OUT / "context.json"
    p.write_text(json.dumps({
        "source": "FlyWire FAFB v783 skeletons (Zenodo 10877326, CC BY 4.0), sampled",
        "unit_nm": UNIT, "encoding": "delta-per-polyline",
        "bbox": [*lo.astype(int).tolist(), *hi.astype(int).tolist()],
        "neurons": neurons, "offsets": offs, "plens": plens, "xyz": flat,
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
