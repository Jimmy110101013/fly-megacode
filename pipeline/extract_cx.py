"""The central complex, as a recurrent circuit.

    ./.venv/bin/python pipeline/extract_cx.py [side]

The mushroom body stores associations but holds no state between decisions: in the
model so far the engine hands the fly "two shocks given, epinephrine three minutes
ago" as input, which means the engine is doing the remembering. The central complex
is where a fly would hold that itself -- the ellipsoid body and protocerebral
bridge implement a ring attractor that sustains a heading in darkness, which is
persistent activity maintained by recurrent wiring.

So this extracts the central complex's own recurrent connectivity, signed by
neurotransmitter, to be used as a fixed reservoir. Nothing in it is trained: in the
animal that wiring is anatomy, and the plastic part is the mushroom body reading it.

Output: data/out/cx_circuit.json
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
SIDE = sys.argv[1] if len(sys.argv) > 1 else "both"

# Excitatory in Drosophila: acetylcholine. Glutamate and GABA are inhibitory here.
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}


def main():
    ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                      dtype={"root_id": "string"})
    ann = ann[ann.root_id.notna()].copy()
    ann["root_id"] = ann.root_id.astype("int64")

    m = ann.cell_class.eq("CX")
    m &= ann.side.isin(["left", "right"]) if SIDE == "both" else ann.side.eq(SIDE)
    cx = ann[m]
    ids = cx.root_id.to_numpy()
    ix = {int(r): i for i, r in enumerate(ids)}
    sign = np.array([SIGN.get(t, 1.0) for t in cx.top_nt.fillna("acetylcholine")])

    types = cx.cell_type.fillna(cx.hemibrain_type.fillna("unknown")).astype(str).to_numpy()
    fam = np.array([t.rstrip("0123456789_")[:6] or "other" for t in types])
    print(f"[select] {len(ids)} central-complex neurons (side={SIDE})", file=sys.stderr)
    from collections import Counter
    print("[families]", dict(Counter(fam).most_common(8)), file=sys.stderr)

    # The mushroom body's own output pathway into the central complex: what the
    # model drives the reservoir with, rather than an invented projection.
    mbon = ann[ann.cell_class.eq("MBON") &
               (ann.side.isin(["left", "right"]) if SIDE == "both" else ann.side.eq(SIDE))]
    mb_ix = {int(r): i for i, r in enumerate(mbon.root_id.to_numpy())}

    edges, mbon_cx = [], []
    with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as src:
        r = ipc.open_file(src)
        for b in range(r.num_record_batches):
            t = r.get_batch(b).select(["pre_pt_root_id", "post_pt_root_id", "syn_count"])
            pre = t.column(0).to_numpy()
            post = t.column(1).to_numpy()
            syn = t.column(2).to_numpy()
            for a, z, w in zip(pre, post, syn):
                i, j = ix.get(int(a)), ix.get(int(z))
                if i is not None and j is not None:
                    edges.append([i, j, int(w)])
                    continue
                mi = mb_ix.get(int(a))
                if mi is not None and j is not None:
                    mbon_cx.append([mi, j, int(w)])
    print(f"[edges] {len(edges)} recurrent CX->CX connections", file=sys.stderr)
    print(f"[drive] {len(mbon_cx)} MBON->CX connections from {len(mb_ix)} MBONs",
          file=sys.stderr)

    # A rough look at how strongly it feeds back on itself -- a reservoir needs
    # recurrence, and a feed-forward chain would not hold anything.
    deg = defaultdict(int)
    for i, j, w in edges:
        deg[i] += w
    print(f"[recurrence] mean outgoing synapses per neuron inside CX: "
          f"{np.mean(list(deg.values())):.0f}", file=sys.stderr)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cx_circuit.json"
    p.write_text(json.dumps({
        "dataset": "FlyWire FAFB v783 central complex",
        "side": SIDE,
        "neurons": [{"root_id": str(int(r)), "type": str(t), "family": str(f),
                     "sign": float(s)}
                    for r, t, f, s in zip(ids, types, fam, sign)],
        "edges": edges,
        "mbon_cx": mbon_cx,
        "n_mbon": len(mb_ix),
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
