"""Which mushroom-body lobe each plastic synapse sits in, and where the lobes are.

    ./.venv/bin/python pipeline/extract_lobes.py

Input   data/raw/proofread_connections_783.feather   FlyWire v783, Zenodo 10676866
        data/raw/annotations.tsv
        web/mb_circuit.json                           to align against, edge for edge
        web/neuropils.json                            JFRC2NP surfaces in FlyWire space
Output  data/out/mb_lobes.json

Two things, both for the atlas's memory layer:

  edge_lobe   one code per `kc_mbon` entry in mb_circuit.json. Each entry there is
              one row of the FlyWire table, and FlyWire labels every row with the
              neuropil its synapses fall in, so this is exact -- not a split of a
              connection across lobes, and not a nearest-neighbour guess.
  grid        the lobe surfaces voxelised, so the page can tell which lobe a traced
              neurite vertex is in without testing 780,000 points against meshes.

Deliberately lobe-level. Compartments (gamma1..gamma5 and so on) are not in the
connectivity table, and an MBON takes dopamine from about five DAN types, so
assigning a KC->MBON synapse to one compartment would be an estimate. The page shows
compartments through the DAN skeletons instead, which need no estimate.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT, WEB = ROOT / "data/raw", ROOT / "data/out", ROOT / "web"

LOBES = ["VL", "ML", "PED", "CA"]          # code = index + 1; 0 = outside the mushroom body
VOXEL_UNITS = 5                            # grid cell, in the neuropil file's units (2 um)


def edge_lobes(circuit):
    # Same selection, same order as extract_mb.py with side=both. Any drift shows up
    # in the edge-for-edge assert below, not as a silently shifted overlay.
    ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                      dtype={"root_id": "string"})
    ann = ann[ann.root_id.notna()].copy()
    ann["root_id"] = ann.root_id.astype("int64")
    both = ann.side.isin(["left", "right"])
    kc_ids = ann[ann.cell_class.eq("Kenyon_Cell") & both].root_id.to_numpy()
    mbon_ids = ann[ann.cell_class.eq("MBON") & both].root_id.to_numpy()
    assert [str(x) for x in kc_ids] == [n["root_id"] for n in circuit["kc"]], "KC order differs from mb_circuit.json"
    assert [str(x) for x in mbon_ids] == [n["root_id"] for n in circuit["mbon"]], "MBON order differs"
    kc_ix = {int(r): i for i, r in enumerate(kc_ids)}
    mbon_ix = {int(r): i for i, r in enumerate(mbon_ids)}

    kc_arr, mbon_arr = pa.array(kc_ids, pa.int64()), pa.array(mbon_ids, pa.int64())
    rows = []
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"]
    with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as src:
        reader = ipc.open_file(src)
        for b in range(reader.num_record_batches):
            t = reader.get_batch(b).select(cols)
            # A filter keeps row order, which is all the alignment depends on.
            t = t.filter(pc.and_(pc.is_in(t["pre_pt_root_id"], kc_arr),
                                 pc.is_in(t["post_pt_root_id"], mbon_arr)))
            if t.num_rows:
                rows.extend(zip(t["pre_pt_root_id"].to_pylist(), t["post_pt_root_id"].to_pylist(),
                                t["syn_count"].to_pylist(), t["neuropil"].to_pylist()))

    edges = circuit["kc_mbon"]
    assert len(rows) == len(edges), f"{len(rows)} rows vs {len(edges)} kc_mbon edges"
    codes = np.zeros(len(edges), np.uint8)
    for i, ((pre, post, syn, npil), (k, m, w)) in enumerate(zip(rows, edges)):
        assert (kc_ix[pre], mbon_ix[post], syn) == (k, m, w), f"edge {i} does not match"
        base = npil.rsplit("_", 1)[0] if npil.endswith(("_L", "_R")) else npil
        codes[i] = LOBES.index(base[3:]) + 1 if base.startswith("MB_") and base[3:] in LOBES else 0
    return codes


def voxelise(np_data):
    """Label a regular grid by which lobe surface encloses each cell centre.

    Ray parity along z: for every column, count how many triangles of a closed
    surface lie above each cell centre; odd means inside.
    """
    verts = np.asarray(np_data["verts"], np.float64).reshape(-1, 3)
    tris = np.asarray(np_data["tris"], np.int64).reshape(-1, 3)
    regions = [r for r in np_data["regions"] if r["name"].startswith("MB_")]
    lo = np.min([verts[r["v0"]:r["v0"] + r["nv"]].min(0) for r in regions], 0) - VOXEL_UNITS
    hi = np.max([verts[r["v0"]:r["v0"] + r["nv"]].max(0) for r in regions], 0) + VOXEL_UNITS
    dims = np.ceil((hi - lo) / VOXEL_UNITS).astype(int)
    # Centres nudged off the lattice so no ray runs exactly along a triangle edge.
    cx = lo[0] + (np.arange(dims[0]) + 0.5) * VOXEL_UNITS + 1e-3
    cy = lo[1] + (np.arange(dims[1]) + 0.5) * VOXEL_UNITS + 2e-3
    cz = lo[2] + (np.arange(dims[2]) + 0.5) * VOXEL_UNITS
    label = np.zeros(dims, np.uint8)

    for r in regions:
        code = LOBES.index(r["name"][3:].rsplit("_", 1)[0]) + 1
        parity = np.zeros(dims, np.uint8)
        for f in tris[r["f0"]:r["f0"] + r["nf"]]:
            a, b, c = verts[f]
            x0, x1 = np.searchsorted(cx, min(a[0], b[0], c[0])), np.searchsorted(cx, max(a[0], b[0], c[0]))
            y0, y1 = np.searchsorted(cy, min(a[1], b[1], c[1])), np.searchsorted(cy, max(a[1], b[1], c[1]))
            if x0 >= x1 or y0 >= y1:
                continue
            X, Y = np.meshgrid(cx[x0:x1], cy[y0:y1], indexing="ij")
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-12:
                continue
            l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / det
            l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / det
            l3 = 1 - l1 - l2
            hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not hit.any():
                continue
            z = l1 * a[2] + l2 * b[2] + l3 * c[2]
            above = (cz[None, None, :] < z[..., None]) & hit[..., None]
            parity[x0:x1, y0:y1] ^= above.astype(np.uint8)
        # Lobe surfaces touch where the peduncle meets the lobes; first claim wins,
        # and a handful of boundary cells is all that decides.
        label[(parity == 1) & (label == 0)] = code
    return lo, dims, label


def rle(flat):
    """[value, run, value, run, ...] -- the grid is mostly empty space."""
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.r_[0, change]
    runs = np.diff(np.r_[starts, len(flat)])
    return np.stack([flat[starts], runs], 1).ravel().tolist()


def check(codes, lo, dims, label, unit_nm):
    """Do traced KC axons pass through the lobes their synapses are labelled with?"""
    atlas = json.loads((WEB / "atlas.json").read_text())
    circuit_kc = {n["root_id"]: i for i, n in enumerate(json.loads((WEB / "mb_circuit.json").read_text())["kc"])}
    xyz = np.asarray(atlas["xyz"], np.int64).reshape(-1, 3)
    at = 0
    for ln in atlas["plens"]:                      # undo the per-polyline delta encoding
        xyz[at:at + ln] = np.cumsum(xyz[at:at + ln], 0)
        at += ln
    # atlas units -> neuropil units, which are 400 nm
    q = np.floor((xyz * (atlas["unit_nm"] / unit_nm) - lo) / VOXEL_UNITS).astype(int)
    inside = np.all((q >= 0) & (q < dims), 1)
    vlab = np.zeros(len(xyz), np.uint8)
    vlab[inside] = label[q[inside, 0], q[inside, 1], q[inside, 2]]

    circuit = json.loads((WEB / "mb_circuit.json").read_text())
    syn_lobes = [set() for _ in circuit["kc"]]
    for (k, _, _), c in zip(circuit["kc_mbon"], codes):
        if c: syn_lobes[k].add(int(c))
    agree = total = tagged = kcv = 0
    vat = 0
    for n, neuron in enumerate(atlas["neurons"]):
        nv = sum(atlas["plens"][atlas["offsets"][n]:atlas["offsets"][n + 1]])
        k = circuit_kc.get(neuron["id"])
        if k is not None:
            seen = set(vlab[vat:vat + nv].tolist()) - {0}
            tagged += int((vlab[vat:vat + nv] > 0).sum()); kcv += nv
            total += len(syn_lobes[k]); agree += len(syn_lobes[k] & seen)
        vat += nv
    print(f"[check] KC vertices inside a lobe: {tagged / kcv:.1%}", file=sys.stderr)
    print(f"[check] synapse lobes that the cell's own traced axon passes through: "
          f"{agree / max(1, total):.1%}", file=sys.stderr)


def main():
    circuit = json.loads((WEB / "mb_circuit.json").read_text())
    codes = edge_lobes(circuit)
    counts = np.bincount(codes, minlength=len(LOBES) + 1)
    print("[edges] " + " ".join(f"{n}={c}" for n, c in zip(["other"] + LOBES, counts)), file=sys.stderr)

    np_data = json.loads((WEB / "neuropils.json").read_text())
    lo, dims, label = voxelise(np_data)
    print(f"[grid] {dims.tolist()} cells at {VOXEL_UNITS * np_data['unit_nm'] / 1000:g} um, "
          + " ".join(f"{n}={int((label == i + 1).sum())}" for i, n in enumerate(LOBES)), file=sys.stderr)
    check(codes, lo, dims, label, np_data["unit_nm"])

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "mb_lobes.json"
    p.write_text(json.dumps({
        "source": "FlyWire v783 connection neuropil labels (Zenodo 10676866); JFRC2NP surfaces via fafbseg",
        "lobes": LOBES,
        # one digit per kc_mbon entry, in mb_circuit.json order
        "edge_lobe": "".join(map(str, codes.tolist())),
        "grid": {"unit_nm": np_data["unit_nm"] * VOXEL_UNITS, "origin_nm": (lo * np_data["unit_nm"]).tolist(),
                 "dims": dims.tolist(), "order": "x-major", "rle": rle(label.ravel())},
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size / 1e3:.0f} kB", file=sys.stderr)


if __name__ == "__main__":
    main()
