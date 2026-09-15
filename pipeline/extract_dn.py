"""Descending neurons: the fly's own output channel, instead of a menu of actions.

    ./.venv/bin/python pipeline/extract_dn.py [side]

Until now the action was an *input*: `encode(features, action)` built one Kenyon-cell
code per candidate action and the model scored all ten. That is a menu, and the fly
does not have one. In the animal, mushroom-body output reaches descending neurons,
and the descending neurons are what the brain actually sends to the body -- roughly
1,300 cells carrying everything the brain can ask the body to do.

So this extracts the real MBON -> DN pathway, signed by the MBON's transmitter, to be
used as a FIXED output layer. Nothing in it is trained. The action then falls out of
which descending neuron wins, rather than out of scoring labelled options.

What this does NOT remove: the names. Calling one descending neuron `amiodarone` is
still an assignment made by hand. It is fixed from anatomy before any training and
never tuned -- see pipeline/dn-separable.mjs, which asks first whether the measured
matrix can separate ten outputs at all.

Output: data/out/dn_circuit.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
SIDE = sys.argv[1] if len(sys.argv) > 1 else "both"
MIN_SYN = 2

# Excitatory in Drosophila: acetylcholine. Glutamate and GABA are inhibitory here.
# The sign belongs to the presynaptic MBON, which is why it is looked up per MBON.
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}


def main():
    ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                      dtype={"root_id": "string"})
    ann = ann[ann.root_id.notna()].copy()
    ann["root_id"] = ann.root_id.astype("int64")

    def pick(mask):
        m = mask & (ann.side.isin(["left", "right"]) if SIDE == "both" else ann.side.eq(SIDE))
        return ann[m]

    # Same filter and row order as extract_mb.py, so MBON index i means the same
    # cell in both files. If that ever drifts, the pathway silently reads the
    # wrong neuron -- which is exactly how the MBON->CX bug hid for a day.
    mbon = pick(ann.cell_class.eq("MBON"))
    dn = pick(ann.super_class.eq("descending"))
    mbon_ix = {int(r): i for i, r in enumerate(mbon.root_id.to_numpy())}
    dn_ix = {int(r): i for i, r in enumerate(dn.root_id.to_numpy())}
    sign = {int(r): SIGN.get(t, 1.0)
            for r, t in zip(mbon.root_id, mbon.top_nt.fillna("acetylcholine"))}
    print(f"[select] {len(mbon)} MBON, {len(dn)} descending (side={SIDE})", file=sys.stderr)

    # Most of the mushroom body's reach into the descending neurons is NOT direct:
    # MBONs project into CRE/SMP and a layer of central interneurons carries it on.
    # That layer is anatomy too -- extracted, fixed, never trained -- so it is pulled
    # out here rather than left out, which would understate the real pathway.
    central = pick(ann.super_class.eq("central") & ~ann.cell_class.isin(
        ["MBON", "DAN", "Kenyon_Cell", "CX"]))
    cen_ix = {int(r): i for i, r in enumerate(central.root_id.to_numpy())}
    cen_sign = {int(r): SIGN.get(t, 1.0)
                for r, t in zip(central.root_id, central.top_nt.fillna("acetylcholine"))}
    print(f"[select] {len(central)} candidate interneurons", file=sys.stderr)

    # Dopaminergic innervation of the relay layer. If MBON->relay is ever to be
    # treated as plastic, the gate has to be anatomy, the way the mushroom body's
    # compartments are -- not a global reinforcement signal sprayed everywhere.
    #
    # Two gates are extracted because they are different claims:
    #   dan_relay  only the mushroom body's own PAM/PPL1 cells, which are the ones
    #              that carry valence in this model. Narrow, but the signal's
    #              identity is not assumed.
    #   da_relay   any dopaminergic neuron in the brain. Broad, but 46.8% of relays
    #              qualify against a 44.6% baseline over all central interneurons,
    #              so this gate selects almost nothing and assumes that unrelated
    #              dopaminergic populations carry this task's reward signal.
    dan = pick(ann.cell_class.eq("DAN"))
    dan_type = dict(zip(dan.root_id.astype(int), dan.cell_type.fillna("")))
    dan_ids = set(dan_type)
    da_ids = set(ann[ann.top_nt.eq("dopamine")].root_id.astype(int))
    print(f"[select] {len(dan_ids)} mushroom-body DANs, "
          f"{len(da_ids)} dopaminergic neurons brain-wide", file=sys.stderr)

    edges, mbon_cen, cen_dn = [], [], []
    dan_cen, da_cen = [], []
    cols = ["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as src:
        reader = ipc.open_file(src)
        for b in range(reader.num_record_batches):
            t = reader.get_batch(b).select(cols)
            pre = t.column("pre_pt_root_id").to_numpy()
            post = t.column("post_pt_root_id").to_numpy()
            syn = t.column("syn_count").to_numpy()
            for p, q, w in zip(pre, post, syn):
                if w < MIN_SYN:
                    continue
                pi, qi = int(p), int(q)
                i, j = mbon_ix.get(pi), dn_ix.get(qi)
                if i is not None and j is not None:
                    edges.append([i, j, int(w) * sign[pi]])
                if i is not None:
                    c = cen_ix.get(qi)
                    if c is not None:
                        mbon_cen.append([i, c, int(w) * sign[pi]])
                if j is not None:
                    c = cen_ix.get(pi)
                    if c is not None:
                        cen_dn.append([c, j, int(w) * cen_sign[pi]])
                c = cen_ix.get(qi)
                if c is not None:
                    if pi in dan_ids:
                        # +1 for PAM (reward), -1 for PPL1 (punishment), as in the
                        # mushroom body, so the relay gate keeps the same valence
                        # structure the compartments have.
                        t = dan_type[pi]
                        pop = 1 if t.startswith("PAM") else -1 if t.startswith("PPL") else 0
                        if pop:
                            dan_cen.append([c, int(w), pop])
                    if pi in da_ids:
                        da_cen.append([c, int(w)])

    # Keep only interneurons that are actually on a path: driven by an MBON and
    # driving a descending neuron. Everything else is not part of this pathway.
    on_path = {e[1] for e in mbon_cen} & {e[0] for e in cen_dn}
    keep = {c: k for k, c in enumerate(sorted(on_path))}
    mbon_cen = [[i, keep[c], w] for i, c, w in mbon_cen if c in keep]
    cen_dn = [[keep[c], j, w] for c, j, w in cen_dn if c in keep]
    # Collapse to one PAM total and one PPL1 total per relay: the gate is how much
    # of each kind of dopamine that cell sits under, not which cell delivered it.
    pam = {}; ppl = {}; da = {}
    for c, w, pop in dan_cen:
        if c in keep:
            (pam if pop > 0 else ppl)[keep[c]] = (pam if pop > 0 else ppl).get(keep[c], 0) + w
    for c, w in da_cen:
        if c in keep:
            da[keep[c]] = da.get(keep[c], 0) + w
    print(f"[dopamine] relays under mushroom-body DANs: "
          f"{len(set(pam) | set(ppl))}/{len(keep)} "
          f"(PAM {len(pam)}, PPL1 {len(ppl)});  under any dopaminergic cell: "
          f"{len(da)}/{len(keep)}", file=sys.stderr)
    print(f"[relay] {len(keep)} interneurons on a MBON->x->DN path: "
          f"{len(mbon_cen)} in, {len(cen_dn)} out, "
          f"{len({e[1] for e in cen_dn})} descending neurons reached", file=sys.stderr)

    reached = sorted({e[1] for e in edges} | {e[1] for e in cen_dn})
    print(f"[pathway] {len(edges)} MBON->DN edges, "
          f"{len({e[0] for e in edges})} MBONs -> {len(reached)} descending neurons, "
          f"{sum(abs(e[2]) for e in edges)} synapses", file=sys.stderr)

    # Rank the reached DNs by how much mushroom-body input they integrate. The
    # action naming, when it happens, is taken off the top of this order -- an
    # anatomical criterion fixed before any learning, not a tuned choice.
    drive = np.zeros(len(dn))
    for i, j, w in edges:
        drive[j] += abs(w)
    for c, j, w in cen_dn:
        drive[j] += abs(w) * 0.25          # a relayed synapse counts, but for less
    order = [int(j) for j in np.argsort(-drive) if drive[j] > 0]

    neurons = [{"id": str(int(r)),
                "t": str(ct) if isinstance(ct, str) else "unknown",
                "side": str(sd)}
               for r, ct, sd in zip(dn.root_id, dn.cell_type.fillna("unknown"), dn.side)]

    p = OUT / "dn_circuit.json"
    p.write_text(json.dumps({
        "dataset": "FlyWire FAFB v783 descending neurons",
        "source": "connectivity Zenodo 10676866, annotations Schlegel et al. 2024, CC BY 4.0",
        "side": SIDE, "min_syn": MIN_SYN,
        "n_mbon": len(mbon),
        "neurons": neurons,
        "n_relay": len(keep),
        "mbon_dn": edges,
        "mbon_relay": mbon_cen,
        "relay_dn": cen_dn,
        "relay_pam": [[c, w] for c, w in sorted(pam.items())],
        "relay_ppl1": [[c, w] for c, w in sorted(ppl.items())],
        "relay_da": [[c, w] for c, w in sorted(da.items())],
        "by_mb_drive": order,
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
