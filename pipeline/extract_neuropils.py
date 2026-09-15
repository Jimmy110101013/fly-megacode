"""Neuropil volumes -- the translucent brain regions every good fly atlas draws.

    ./.venv/bin/python pipeline/extract_neuropils.py

Input   data/raw/np.zip   JFRC2NP.surf.fw.zip from fafbseg (navis-org/fafbseg-py),
                          156 neuropil surfaces already transformed into FlyWire
                          (FAFB14.1) space. Originals: doi 10.5281/zenodo.10567.
Output  data/out/neuropils.json

Meshes ship as 3D vertices and triangles so the page can project them itself and
keep both views; they are rasterised into the atlas's cached background, so the
triangle count costs nothing per frame.

Regions are grouped by the standard anatomical super-categories (Ito et al. 2014
nomenclature) rather than coloured one-by-one -- that grouping is what the
reference atlases key their palettes to.
"""
import json, re, struct, sys, zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/out"
UNIT = 400.0                  # match atlas.json: 0.4 um per stored unit

# Ito et al. 2014 super-categories. Order matters only for drawing back-to-front.
FAMILIES = [
    ("optic",      ["ME", "AME", "LO", "LOP", "LA"]),
    ("olfactory",  ["AL", "LH"]),
    ("mushroom",   ["MB_CA", "MB_ML", "MB_VL", "MB_PED"]),
    ("central",    ["FB", "EB", "PB", "NO", "AB"]),
    ("lateral",    ["LAL", "BU", "GA", "AOTU"]),
    ("ventrolat",  ["PLP", "PVLP", "AVLP", "WED"]),
    ("superior",   ["SLP", "SIP", "SMP"]),
    ("inferior",   ["CRE", "SCL", "ICL", "IB", "ATL"]),
    ("ventromed",  ["VES", "EPA", "GOR", "SPS", "IPS"]),
    ("periesoph",  ["SAD", "FLA", "PRW", "GNG", "CAN", "AMMC"]),
]


def family_of(name):
    stem = re.sub(r"_[LR]$", "", name)
    for fam, members in FAMILIES:
        if stem in members:
            return fam
    return "other"


def read_ply(buf):
    """Binary little-endian PLY with float32 xyz and an int-list face property."""
    head_end = buf.index(b"end_header\n") + len(b"end_header\n")
    head = buf[:head_end].decode("latin1")
    nv = int(re.search(r"element vertex (\d+)", head).group(1))
    nf = int(re.search(r"element face (\d+)", head).group(1))
    props = re.findall(r"property (\w+) (\w+)", head)
    vprops = [p for p in props if p[1] in ("x", "y", "z")]
    stride = 4 * len([p for p in props if p[0] in ("float", "float32", "int", "int32", "uint")
                      and p[1] not in ("vertex_indices",)])
    # These files carry exactly x, y, z per vertex.
    assert len(vprops) == 3 and stride == 12, f"unexpected vertex layout: {props}"

    v = np.frombuffer(buf, dtype="<f4", count=nv * 3, offset=head_end).reshape(nv, 3)
    at = head_end + nv * 12
    faces = np.empty((nf, 3), np.int32)
    for i in range(nf):
        (cnt,) = struct.unpack_from("<i", buf, at); at += 4
        idx = np.frombuffer(buf, dtype="<i4", count=cnt, offset=at); at += 4 * cnt
        faces[i] = idx[:3]          # every face here is a triangle
    return v, faces


def main():
    z = zipfile.ZipFile(RAW / "np.zip")
    # the archive carries macOS resource forks alongside the real meshes
    names = sorted(n for n in z.namelist()
                   if n.endswith(".ply") and not n.startswith("__MACOSX/"))
    regions, verts, tris = [], [], []
    vbase = 0
    lo = np.array([np.inf] * 3); hi = np.array([-np.inf] * 3)

    for n in names:
        v, f = read_ply(z.read(n))
        q = np.rint(v / UNIT).astype(np.int32)
        lo = np.minimum(lo, q.min(0)); hi = np.maximum(hi, q.max(0))
        stem = n[:-4]
        regions.append({
            "name": stem,
            "family": family_of(stem),
            "side": "R" if stem.endswith("_R") else "L" if stem.endswith("_L") else "C",
            "v0": vbase, "nv": len(q), "f0": len(tris) // 3, "nf": len(f),
        })
        verts.extend(q.ravel().tolist())
        tris.extend((f + vbase).ravel().tolist())
        vbase += len(q)

    fams = {}
    for r in regions:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"[meshes] {len(regions)} regions, {vbase:,} vertices, {len(tris)//3:,} triangles",
          file=sys.stderr)
    print(f"[families] {fams}", file=sys.stderr)
    print(f"[bbox] {lo.astype(int).tolist()} .. {hi.astype(int).tolist()} (0.4 um units)",
          file=sys.stderr)

    p = OUT / "neuropils.json"
    p.write_text(json.dumps({
        "source": "JFRC2NP surfaces in FlyWire space, via fafbseg (doi 10.5281/zenodo.10567)",
        "unit_nm": UNIT,
        "bbox": [*lo.astype(int).tolist(), *hi.astype(int).tolist()],
        "regions": regions, "verts": verts, "tris": tris,
    }, separators=(",", ":")))
    print(f"[done] {p.name} = {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
