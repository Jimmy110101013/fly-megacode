"""Extract the mushroom body learning circuit from the FlyWire FAFB v783 connectome.

Inputs  (data/raw/):
  annotations.tsv                      Schlegel et al. 2024 neuron annotations
  proofread_connections_783.feather    Dorkenwald et al. FlyWire connectivity, Zenodo 10676866
Outputs (data/out/):
  mb_circuit.json    KC -> MBON weight matrix + DAN compartment wiring, right hemisphere
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
SIDE = sys.argv[1] if len(sys.argv) > 1 else "right"   # "right", "left", or "both"
NT_COLS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]

ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                  dtype={"root_id": "string"})
ann = ann[ann.root_id.notna()].copy()
ann["root_id"] = ann.root_id.astype("int64")      # feather side is int64 -- must match

def pick(cell_class, side=SIDE):
    m = ann.cell_class.eq(cell_class)
    if side == "both":
        m &= ann.side.isin(["left", "right"])
    elif side:
        m &= ann.side.eq(side)
    return ann[m]

kc, mbon, dan = pick("Kenyon_Cell"), pick("MBON"), pick("DAN")
print(f"[annotations] KC={len(kc)} MBON={len(mbon)} DAN={len(dan)} (side={SIDE})", file=sys.stderr)
assert len(kc) > 1000 and len(mbon) > 20, "annotation filter collapsed -- check side/cell_class"

kc_ids   = kc.root_id.to_numpy()
mbon_ids = mbon.root_id.to_numpy()
dan_ids  = dan.root_id.to_numpy()

kc_ix   = {r: i for i, r in enumerate(kc_ids)}
mbon_ix = {r: i for i, r in enumerate(mbon_ids)}
dan_ix  = {r: i for i, r in enumerate(dan_ids)}
mb_all  = set(kc_ids) | set(mbon_ids) | set(dan_ids)

cls_of = dict(zip(ann.root_id, ann.cell_class.fillna("unknown")))

kc2mbon, dan2mbon, dan2kc, mbon2dan = [], [], [], []

cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"]
with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as src:
    reader = ipc.open_file(src)
    for b in range(reader.num_record_batches):
        t = reader.get_batch(b).select(cols)
        pre  = t.column("pre_pt_root_id").to_numpy()
        post = t.column("post_pt_root_id").to_numpy()
        syn  = t.column("syn_count").to_numpy()
        npil = t.column("neuropil").to_pylist()

        for i in range(len(pre)):
            a, z, w = int(pre[i]), int(post[i]), int(syn[i])
            if a not in mb_all and z not in mb_all:
                continue
            if a in kc_ix and z in mbon_ix:
                kc2mbon.append((kc_ix[a], mbon_ix[z], w, npil[i]))
            elif a in dan_ix and z in mbon_ix:
                dan2mbon.append((dan_ix[a], mbon_ix[z], w, npil[i]))
            elif a in dan_ix and z in kc_ix:
                dan2kc.append((dan_ix[a], kc_ix[z], w, npil[i]))
            elif a in mbon_ix and z in dan_ix:
                mbon2dan.append((mbon_ix[a], dan_ix[z], w))
        if b % 40 == 0:
            print(f"  batch {b}/{reader.num_record_batches} "
                  f"kc2mbon={len(kc2mbon)}", file=sys.stderr)

print(f"[edges] KC->MBON={len(kc2mbon)} DAN->MBON={len(dan2mbon)} "
      f"DAN->KC={len(dan2kc)} MBON->DAN={len(mbon2dan)}", file=sys.stderr)
assert len(kc2mbon) > 5000, "KC->MBON edge count implausible -- check the int64 join"

def meta(df, idx):
    return [{"root_id": str(r.root_id), "type": (r.cell_type if pd.notna(r.cell_type)
                                                 else r.hemibrain_type if pd.notna(r.hemibrain_type)
                                                 else "unknown"),
             "nt": r.top_nt if pd.notna(r.top_nt) else "unknown"}
            for r in df.itertuples()]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"mb_circuit_{SIDE}.json" if SIDE != "right" else OUT / "mb_circuit.json").write_text(json.dumps({
    "dataset": "FlyWire FAFB v783",
    "side": SIDE,
    "license": "CC BY 4.0 (Zenodo 10676866) / annotations CC BY 4.0",
    "kc": meta(kc, kc_ix), "mbon": meta(mbon, mbon_ix), "dan": meta(dan, dan_ix),
    "kc_mbon": [[a, z, w] for a, z, w, _ in kc2mbon],
    "dan_mbon": [[a, z, w] for a, z, w, _ in dan2mbon],
    "dan_kc":   [[a, z, w] for a, z, w, _ in dan2kc],
    "mbon_dan": [[a, z, w] for a, z, w in mbon2dan],
}, separators=(",", ":")))

print("[done]", *(f"{p.name}={p.stat().st_size/1e6:.1f}MB" for p in OUT.glob("*.json")),
      file=sys.stderr)
