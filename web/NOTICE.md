# What the files in this directory are, and where they came from

Everything here is derived from public *Drosophila* connectome releases. None of
it is drawn by hand, and no neuron, synapse, or branch in it is invented. The
counts below are what each file actually contains; the hashes are of the files as
committed.

## Circuits — the wiring the fly computes with

| File | Contents |
| --- | --- |
| `mb_circuit.json` | Both mushroom bodies: 5,177 Kenyon cells, 96 MBONs, 331 DANs, and 89,315 KC→MBON, 3,147 DAN→MBON, 49,616 DAN→KC, 3,376 MBON→DAN connections |
| `cx_circuit.json` | Central complex: 2,875 neurons, 171,030 recurrent edges, plus the 1,368 real MBON→CX connections that drive it |

Source: FlyWire FAFB v783 connectivity, [Zenodo 10676866](https://zenodo.org/records/10676866),
CC BY 4.0. Cell types and neurotransmitter predictions from
[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
(Schlegel et al., *Nature* 2024). Edges below 2 synapses are dropped; the
threshold is recorded in each file as `min_syn`.

Only `kc_mbon` carries a plastic weight at run time. Every other edge, and the
entire central complex, is fixed at its measured strength.

## Morphology — the atlas you see

| File | Contents |
| --- | --- |
| `atlas.json` | 5,606 mushroom-body neurons, 140,150 polylines, 782,790 vertices |
| `atlas-lite.json` | The same neurons simplified for phones: 67,272 polylines |
| `cx_atlas.json` | 2,875 central-complex neurons, 57,500 polylines |
| `neuropils.json` | 78 JFRC2NP neuropil surfaces, 50,521 vertices, 100,892 triangles |
| `mb_lobes.json` | The lobe (VL, ML, PED, CA, or outside) of each of the 89,315 KC→MBON connections, from FlyWire's per-connection neuropil label; the eight lobe surfaces voxelised at 2 µm |

Skeletons: FlyWire FAFB v783, [Zenodo 10877326](https://zenodo.org/records/10877326),
CC BY 4.0 (multi-gigabyte parquet, reduced here by Ramer–Douglas–Peucker to
the polyline counts above). Neuropil surfaces: JFRC2NP registered into FlyWire space,
distributed with [fafbseg-py](https://github.com/navis-org/fafbseg-py).

Coordinates are quantised to a 400 nm grid (500 nm for the lite atlas) and stored
delta-encoded per polyline, which is what makes 782,790 vertices fit in 7 MB.

## Reproducing them

Download the two Zenodo archives into `data/raw/`, then:

```sh
python3 -m venv .venv && ./.venv/bin/pip install numpy pandas pyarrow

./.venv/bin/python pipeline/extract_mb.py both        # -> data/out/mb_circuit_both.json
./.venv/bin/python pipeline/extract_cx.py             # -> data/out/cx_circuit.json
./.venv/bin/python pipeline/extract_skeletons.py      # -> data/out/atlas.json
./.venv/bin/python pipeline/extract_cx_skeletons.py   # -> data/out/cx_atlas.json
./.venv/bin/python pipeline/extract_neuropils.py      # -> data/out/neuropils.json
./.venv/bin/python pipeline/extract_lobes.py          # -> data/out/mb_lobes.json (reads web/)
```

Then copy them into `web/`, `mb_circuit_both.json` landing as `mb_circuit.json`.
`atlas-lite.json` is the same skeleton script run a second time with a coarser
straightness tolerance, fewer runs per neuron, and a 500 nm grid — the arguments
are positional (`rdp_nm runs unit_nm`) and are a size/detail trade-off, not a
fixed recipe.

The skeleton passes stream the whole v783 skeleton table and take a few minutes
each; they cache the rows they matched, so a re-run at a different simplification
is fast.

## Checksums

```
a86fe2185b814f0232f3edc33e35610890db1f027007bcfe252a2f201b3fb226  atlas-lite.json
1353676586667085e9f9648cd79ac37d0ed9f624aef7f615befbbb8ffc856d34  atlas.json
cbd7d06e1eee5b92d79ac24d8fb17290b231b96b6a1a0e754e421caf31fc5bee  cx_atlas.json
226581260260d7532cf8a7d00bfbc2c724a86a873f4f6d1b262fd266a1b59667  cx_circuit.json
15bc799030e01ee192be1d3ea042c63f8f5ac66a6d6f145259216fddc3dc991a  mb_circuit.json
d8ccf7da2882adf4d3d148e1d6a0b646d7bfc05d42e27d26e2d36b74c6b9ea0f  mb_lobes.json
c0ecf5556fda567d0c1a1032bc69c5fe9a270cb92bfe64efdaa19dfef84c0826  neuropils.json
```
