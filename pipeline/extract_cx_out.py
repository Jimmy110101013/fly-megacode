"""Central complex output: how it reaches the descending neurons.

    ./.venv/bin/python pipeline/extract_cx_out.py

The central complex is where Drosophila is actually shown to select behaviour --
PFL3 compares heading with goal and drives turning through DNa02/DNa03. To ask
whether it holds an opponent, per-action read-out that the mushroom-body-to-DN path
lacked, its output pathway has to be extracted the same way that path was: direct
CX -> DN, and CX -> central interneuron -> DN, signed by transmitter.

Index order matches cx_circuit.json (CX) and dn_circuit.json (DN), so the arrays can
be combined with the existing reservoir and MBON->CX drive.

Output: data/out/cx_out.json
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
MIN_SYN = 2
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}

ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False, dtype={"root_id": "string"})
ann = ann[ann.root_id.notna()].copy(); ann["root_id"] = ann.root_id.astype("int64")
nt = dict(zip(ann.root_id, ann.top_nt.fillna("acetylcholine")))
side_of = dict(zip(ann.root_id, ann.side.fillna("?")))
type_of = dict(zip(ann.root_id, ann.cell_type.fillna(ann.hemibrain_type).fillna("unknown")))

cx = json.loads((OUT / "cx_circuit.json").read_text())
dn = json.loads((OUT / "dn_circuit.json").read_text())
cx_ids = np.array([int(n["root_id"]) for n in cx["neurons"]], dtype=np.int64)
dn_ids = np.array([int(n["id"]) for n in dn["neurons"]], dtype=np.int64)
cx_ix = {int(r): i for i, r in enumerate(cx_ids)}
dn_ix = {int(r): i for i, r in enumerate(dn_ids)}
central = ann[ann.super_class.eq("central") & ann.side.isin(["left", "right"])
              & ~ann.cell_class.isin(["CX", "MBON", "DAN", "Kenyon_Cell"])]
cen_ids = central.root_id.to_numpy(dtype=np.int64)
print(f"[select] {len(cx_ids)} CX, {len(dn_ids)} DN, {len(cen_ids)} candidate relays", file=sys.stderr)

cx_s, dn_s, cen_s = np.sort(cx_ids), np.sort(dn_ids), np.sort(cen_ids)
direct, cx_cen, cen_dn = [], [], []
cols = ["pre_pt_root_id", "post_pt_root_id", "syn_count"]
with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as src:
    r = ipc.open_file(src)
    for b in range(r.num_record_batches):
        t = r.get_batch(b).select(cols)
        pre = t.column(0).to_numpy(); post = t.column(1).to_numpy(); syn = t.column(2).to_numpy()
        keep = syn >= MIN_SYN
        pre, post, syn = pre[keep], post[keep], syn[keep]
        pc = np.isin(pre, cx_s); qd = np.isin(post, dn_s)
        qc = np.isin(post, cen_s); pce = np.isin(pre, cen_s)
        for m, sink in ((pc & qd, direct), (pc & qc, cx_cen), (pce & qd, cen_dn)):
            for p, q, w in zip(pre[m], post[m], syn[m]):
                sink.append((int(p), int(q), int(w)))

on_path = {q for _, q, _ in cx_cen} & {p for p, _, _ in cen_dn}
relay_ids = sorted(on_path); relay_ix = {r: k for k, r in enumerate(relay_ids)}
sg = lambda r: SIGN.get(nt.get(r, "acetylcholine"), 1.0)
out = {
    "dataset": "FlyWire FAFB v783 central complex output",
    "source": "connectivity Zenodo 10676866, annotations Schlegel et al. 2024, CC BY 4.0",
    "min_syn": MIN_SYN,
    "cx_dn": [[cx_ix[p], dn_ix[q], w * sg(p)] for p, q, w in direct],
    "cx_relay": [[cx_ix[p], relay_ix[q], w * sg(p)] for p, q, w in cx_cen if q in relay_ix],
    "relay_dn": [[relay_ix[p], dn_ix[q], w * sg(p)] for p, q, w in cen_dn if p in relay_ix],
    "relays": [{"id": str(r), "t": str(type_of.get(r, "unknown")), "side": side_of.get(r, "?"),
                "sign": sg(r)} for r in relay_ids],
    "cx_side": [side_of.get(int(r), "?") for r in cx_ids],
}
print(f"[pathway] direct CX->DN {len(out['cx_dn'])} edges; relays on a path {len(relay_ids)}, "
      f"CX->relay {len(out['cx_relay'])}, relay->DN {len(out['relay_dn'])}", file=sys.stderr)
p = OUT / "cx_out.json"; p.write_text(json.dumps(out, separators=(",", ":")))
print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)
