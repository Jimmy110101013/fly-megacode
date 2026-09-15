# Third-party data, software, and sources

The MIT license in [LICENSE](LICENSE) covers the code written for this
repository. It does not relicense anything listed below.

## Connectome data redistributed in `web/`

**FlyWire FAFB v783 connectivity** — Dorkenwald et al., *Nature* 2024.
[Zenodo 10676866](https://zenodo.org/records/10676866), **CC BY 4.0**.
Derived files: `web/mb_circuit.json`, `web/cx_circuit.json`.

**FlyWire FAFB v783 skeletons** — Schlegel et al., *Nature* 2024.
[Zenodo 10877326](https://zenodo.org/records/10877326), **CC BY 4.0**.
Derived files: `web/atlas.json`, `web/atlas-lite.json`, `web/cx_atlas.json`.

**FlyWire annotations** — cell types, hemisphere assignment, and neurotransmitter
predictions. [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations),
Schlegel et al., *Nature* 2024, **CC BY 4.0**.

**JFRC2NP neuropil surfaces** — Ito et al., *Neuron* 2014 nomenclature,
registered into FlyWire space and distributed with
[fafbseg-py](https://github.com/navis-org/fafbseg-py) (GPL-3.0 for the library;
the surface meshes are the redistributed JFRC2NP data).
Derived file: `web/neuropils.json`.

If you build on any of this, cite the FlyWire papers, not this repository.
FlyWire asks that its data be used for non-commercial research and that
Dorkenwald et al. and Schlegel et al. be cited. `web/NOTICE.md` records the
exact contents, filters, and checksums of every derived file.

## Software loaded by the page

**three.js** r157, MIT, loaded from cdnjs at run time and not vendored here.

## Clinical source

The decision logic in `web/acls-engine.js` implements the procedure described in
the **American Heart Association 2025 Guidelines for CPR & ECC**, Part 9 (Adult
Advanced Life Support) and the Adult Cardiac Arrest Algorithm
(*Circulation*, 2025). The guideline text, figures, and algorithm diagrams are
copyrighted by the AHA and are **not** redistributed here — only the procedure is
implemented, from the published algorithm. Consult the AHA's own materials as the
authority: <https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines>

This project is not affiliated with, endorsed by, or reviewed by the American
Heart Association.
