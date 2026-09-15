# Fly Megacode

A *Drosophila* mushroom body, wired from the FlyWire connectome, learning the AHA 2025
cardiac arrest algorithm by trial and error.

[![checks](https://github.com/Jimmy110101013/fly-megacode/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimmy110101013/fly-megacode/actions/workflows/ci.yml)
![connectome](https://img.shields.io/badge/connectome-FlyWire_FAFB_v783-6f7fb8?labelColor=15181f)
![plastic synapses](https://img.shields.io/badge/plastic_synapses-KC%E2%86%92MBON_only-6f7fb8?labelColor=15181f)
[![licence](https://img.shields.io/badge/code-MIT-6f7fb8?labelColor=15181f)](LICENSE)

**[Run it in the browser →](https://jimmy110101013.github.io/fly-megacode/)** ·
[what we learned](#what-we-learned) · [retracted](#retracted) ·
[data provenance](web/NOTICE.md)

> **Not a clinical tool.** This is a neuroscience experiment. It is not for patient
> care, it is not ACLS training or certification material, and it is not affiliated
> with, endorsed by, or reviewed by the American Heart Association. Use the AHA's
> own published guidelines as the clinical authority.

## What this is

The question: what can the fly's own learning rule do, running inside the fly's own
wiring, on a task nothing like smelling?

```
ACLS engine (teacher)  ──state──▶  mushroom body (decides)  ──action──▶  engine grades
                                          ▲                                 │
                    central complex ──────┘                                 ▼
                    (holds the history)     dopamine: PAM if right, PPL1 if wrong
```

- **The fly sees only what a bystander could.** Rhythm, ETCO₂, whether a line and a
  tube are in. How many shocks, when epinephrine was given, which cycle this is: it
  has to remember those itself.
- **It is never told the right answer.** It scores ten candidate actions, commits to
  one, and gets one bit back as dopamine.
- **One kind of synapse learns.** KC→MBON, with the rule measured in real flies:
  dopamine depresses the synapses of the Kenyon cells that were active, with slow
  recovery. Nothing else is trained. The central complex is a fixed reservoir.

**Result:** 76.8% (71–82%) on unseen megacodes, 3 seeds, greedy, against 12.5% chance (`chance.mjs`).
Menu architecture, the fly holding its own history — the configuration the page runs.

| Real, from FlyWire FAFB v783 | Modelled |
| --- | --- |
| 5,177 Kenyon cells, 96 MBONs, 331 DANs (307 PAM / 24 PPL1) | the sensory interface: a clinical state as one-hot channels |
| 89,315 KC→MBON connections, and which DAN reaches which MBON | the ten-action menu |
| 2,875 central-complex neurons, 171,030 recurrent connections, the MBON→CX pathway | rate-based dynamics instead of spikes |
| every sign, every compartment | sparsity (1.5%), input sampling, gains |

## What we learned

**1. Working memory can live in a network nobody trained.** Take the history away
and the task collapses. Let the fly drive its own central complex and it recovers
92% of the gap. (`working-memory.mjs`, 3 seeds, 2,500 megacodes)

| who remembers | accuracy |
| --- | --- |
| the engine hands it over | 82.2% |
| nobody | 17.6% |
| the fly, in its central complex | 76.8% (71–82%) |

**2. The limit was the code, not the size of the brain.** A linear readout on the same
features reaches 100% (`upper-bound.mjs`). Both mushroom bodies score the same as one,
77.6% vs 78.4% over 4 seeds (`more-brain.mjs`). What bound it was overlap between
Kenyon-cell codes: in an earlier configuration, cutting activation from 5% to 1.5%
took accuracy from 37% to 88%.

**3. The shape of the circuit matters more than its address.** Build the same shape
(a large population converging on a few outputs) from other brain regions and run the
same rule: a piece of optic lobe reaches 55.0% against the mushroom body's 74.7%
(`other-regions.mjs`, 3 seeds). But it only does that with a reward/punishment split
the script had to invent. What the mushroom body uniquely has is the teaching signal.

**4. The fly's specific wiring is not being used. Its dopamine map is.**
- Initial synaptic weights do not matter: real, shuffled and random all converge to
  64–68%.
- Rewiring KC→MBON while keeping every cell's number of connections: 71.5% (62–78%)
  against 74.9% (69–82%), 5 seeds. A draw by a rule set before the run
  (`rewire-menu.mjs`).
- Rewiring which dopamine population reaches which MBON breaks the animal outright:
  it spends 78% of its choices on one action (27% for the real wiring) and every
  megacode runs to the step cap. With a depression-only rule, keeping reward and
  punishment in separate compartments is load-bearing.

Caveat: the input→Kenyon cell layer is modelled as random, so a Kenyon cell's identity
had no task meaning to lose in the first place.

**5. The menu does a lot of the work.** Take it away and let the action come out of the
fly's own descending neurons, and accuracy falls to 13.7%, near chance
([docs/descending-neurons.md](docs/descending-neurons.md)).

**Taken together:** this supports "a learner with the fly's shape and the fly's rule
can learn this task through an interface people built." It does not support "a fly
could run a code," and it does not yet show that the connectome's particular wiring
is needed.

## Retracted

Kept here on purpose.

- **70% — a grader exploit.** `correctActions()` accepted `rhythm_check` at every moment
  in a CPR block, so the fly checked the rhythm forever and never gave a drug.
- **96% — the engine was doing the remembering.** The history was handed over as input.
  The honest number is the one above.
- **A ten-point advantage for real synaptic weights** — the weight modes consumed
  different random draws. With a shared generator it disappeared.

## The page

Everything runs in the browser. The atlas draws 5,606 traced neurons from the FlyWire
skeletons plus the central complex; a cell lights up along its calyx→lobe axis when it
fires, and the central complex glows by how much it is holding. Hover a region for its
name and its role in the model, Ctrl/⌘-scroll or pinch to zoom. A second panel draws
every KC→MBON synapse on the path just taken, thinning as dopamine depresses it.

## Run

```bash
git clone https://github.com/Jimmy110101013/fly-megacode && cd fly-megacode

node pipeline/test-engine.mjs        # the grader — 16 checks on the ACLS state machine
node pipeline/test-plasticity.mjs    # the learning rule moves value the right way
node pipeline/working-memory.mjs     # result 1
python3 pipeline/serve.py            # then open http://localhost:8777/
```

Every number above has its own script in `pipeline/`. The connectome files the page
needs are committed; rebuilding them from the raw archives is described in
[`web/NOTICE.md`](web/NOTICE.md).

```
web/acls-engine.js        the grader: AHA 2025 state machine
web/mushroom-body.js      the fly: KC coding, MBON readout, dopamine plasticity
web/central-complex.js    the fixed reservoir
web/atlas3d.js            traced morphology on the GPU
pipeline/extract_*.py     connectome -> json
pipeline/*.mjs            one script per measurement
```

## Data

- connectivity: [Zenodo 10676866](https://zenodo.org/records/10676866), CC BY 4.0
- skeletons: [Zenodo 10877326](https://zenodo.org/records/10877326), CC BY 4.0
- annotations: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations),
  Schlegel et al., *Nature* 2024, CC BY 4.0
- neuropil surfaces: JFRC2NP in FlyWire space, via [fafbseg-py](https://github.com/navis-org/fafbseg-py)
- guidelines: AHA 2025 Guidelines for CPR & ECC (*Circulation*, 2025). The procedure is
  implemented in `web/acls-engine.js`; the guideline text is copyrighted by the AHA and
  is not redistributed here.

FlyWire data is released for non-commercial research; cite Dorkenwald et al. and
Schlegel et al., not this repository. Contents and checksums of every derived file:
[`web/NOTICE.md`](web/NOTICE.md). Licences: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Licence

Code: [MIT](LICENSE). The connectome data redistributed in `web/` stays CC BY 4.0 and is
not relicensed.
