"""Build a learning circuit out of some other part of the fly brain.

    ./.venv/bin/python pipeline/extract_region.py CX
    ./.venv/bin/python pipeline/extract_region.py --survey

The mushroom body has a specific shape: a large population of cells feeding a
small set of output neurons, with dopaminergic neurons innervating those outputs
in separable compartments. This script looks for that shape elsewhere and, where
it finds something, writes a circuit in the same format as mb_circuit.json so the
identical model can run on it.

The dopamine split is the generous part. The mushroom body has named reward (PAM)
and punishment (PPL1) populations; elsewhere there is no such labelling, so the
dopaminergic neurons are split into two groups by what they target -- the most
favourable reading an alternative region can be given.
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
SIDE = "right"
N_EXPANSION = 2597          # match the Kenyon cell count
N_OUTPUT = 48               # match the MBON count


def load_annotations():
    ann = pd.read_csv(RAW / "annotations.tsv", sep="\t", low_memory=False,
                      dtype={"root_id": "string"})
    ann = ann[ann.root_id.notna()].copy()
    ann["root_id"] = ann.root_id.astype("int64")
    return ann


def load_edges():
    cols = ["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    pre, post, syn = [], [], []
    with pa.memory_map(str(RAW / "proofread_connections_783.feather"), "rb") as src:
        r = ipc.open_file(src)
        for b in range(r.num_record_batches):
            t = r.get_batch(b).select(cols)
            pre.append(t.column(0).to_numpy())
            post.append(t.column(1).to_numpy())
            syn.append(t.column(2).to_numpy())
    return np.concatenate(pre), np.concatenate(post), np.concatenate(syn)


def build(region, ann, pre, post, syn, quiet=False):
    """Expansion layer -> its busiest targets -> the dopamine that reaches them."""
    side = ann.side.eq(SIDE)
    pool = ann[side & ann.cell_class.eq(region)]
    if len(pool) < 200:
        return None, f"only {len(pool)} neurons"

    exp_ids = pool.root_id.to_numpy()[:N_EXPANSION]
    exp_set = set(exp_ids.tolist())

    # Busiest downstream partners that are not themselves in the expansion layer.
    m = np.fromiter((p in exp_set for p in pre), bool, len(pre))
    tgt = defaultdict(int)
    for p, q, w in zip(pre[m], post[m], syn[m]):
        if int(q) not in exp_set:
            tgt[int(q)] += int(w)
    if not tgt:
        return None, "no downstream partners"
    out_ids = np.array([k for k, _ in sorted(tgt.items(), key=lambda kv: -kv[1])[:N_OUTPUT]])
    if len(out_ids) < 8:
        return None, f"only {len(out_ids)} output partners"

    exp_ix = {r: i for i, r in enumerate(exp_ids)}
    out_ix = {int(r): i for i, r in enumerate(out_ids)}
    exp_out = [[exp_ix[int(p)], out_ix[int(q)], int(w)]
               for p, q, w in zip(pre[m], post[m], syn[m]) if int(q) in out_ix]

    # Dopaminergic neurons reaching those outputs.
    da = ann[side & ann.top_nt.eq("dopamine") & ~ann.root_id.isin(exp_ids)]
    da_set = set(da.root_id.tolist())
    md = np.fromiter((p in da_set for p in pre), bool, len(pre))
    da_edges = [(int(p), int(q), int(w)) for p, q, w in zip(pre[md], post[md], syn[md])
                if int(q) in out_ix]
    da_ids = sorted({p for p, _, _ in da_edges})
    if len(da_ids) < 4:
        return None, f"only {len(da_ids)} dopaminergic neurons reach the outputs"

    # Split the dopamine into two populations by what they target -- the stand-in
    # for PAM and PPL1, chosen as favourably as the data allows.
    da_ix = {r: i for i, r in enumerate(da_ids)}
    prof = np.zeros((len(da_ids), len(out_ids)))
    for p, q, w in da_edges:
        prof[da_ix[p], out_ix[q]] += w
    prof /= np.maximum(prof.sum(1, keepdims=True), 1)
    c0, c1 = prof[0], prof[int(np.argmax(((prof - prof[0]) ** 2).sum(1)))]
    for _ in range(25):
        assign = ((prof - c1) ** 2).sum(1) < ((prof - c0) ** 2).sum(1)
        if assign.all() or not assign.any():
            break
        c0, c1 = prof[~assign].mean(0), prof[assign].mean(0)
    reward = set(np.array(da_ids)[~assign].tolist())

    meta = lambda ids, kind: [
        {"root_id": str(int(r)),
         "type": (f"{'PAM' if int(r) in reward else 'PPL'}-like" if kind == "dan" else region),
         "nt": "unknown"} for r in ids]

    circuit = {
        "dataset": f"FlyWire FAFB v783, {region} stand-in circuit",
        "side": SIDE,
        "kc": meta(exp_ids, "exp"),
        "mbon": meta(out_ids, "out"),
        "dan": meta(da_ids, "dan"),
        "kc_mbon": exp_out,
        "dan_mbon": [[da_ix[p], out_ix[q], w] for p, q, w in da_edges],
        "dan_kc": [], "mbon_dan": [],
    }
    if not quiet:
        print(f"  {region:16s} expansion={len(exp_ids):5d} outputs={len(out_ids):3d} "
              f"edges={len(exp_out):6d} dopamine={len(da_ids):4d} "
              f"({len(reward)} reward-like / {len(da_ids)-len(reward)} punish-like)",
              file=sys.stderr)
    return circuit, None


def main():
    ann = load_annotations()
    print("[edges] loading connectivity…", file=sys.stderr)
    pre, post, syn = load_edges()

    if "--survey" in sys.argv:
        counts = ann[ann.side.eq(SIDE)].cell_class.value_counts()
        cands = [c for c, n in counts.items() if n >= 400 and c != "Kenyon_Cell"][:12]
        print(f"[survey] testing {len(cands)} regions with >=400 neurons", file=sys.stderr)
        for region in cands:
            c, why = build(region, ann, pre, post, syn)
            if c is None:
                print(f"  {region:16s} unusable -- {why}", file=sys.stderr)
            else:
                (OUT / f"circuit_{region}.json").write_text(json.dumps(c, separators=(",", ":")))
        return

    region = sys.argv[1]
    c, why = build(region, ann, pre, post, syn)
    if c is None:
        print(f"{region} unusable: {why}", file=sys.stderr); sys.exit(1)
    p = OUT / f"circuit_{region}.json"
    p.write_text(json.dumps(c, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
