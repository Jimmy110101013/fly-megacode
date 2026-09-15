# Fly Megacode

A real *Drosophila* mushroom body, wired from the FlyWire whole-brain connectome,
learning the AHA 2025 ACLS megacode by trial and error.

[![checks](https://github.com/Jimmy110101013/fly-megacode/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimmy110101013/fly-megacode/actions/workflows/ci.yml)
![connectome](https://img.shields.io/badge/connectome-FlyWire_FAFB_v783-6f7fb8?labelColor=15181f)
![plastic synapses](https://img.shields.io/badge/plastic_synapses-KC%E2%86%92MBON_only-6f7fb8?labelColor=15181f)
[![licence](https://img.shields.io/badge/code-MIT-6f7fb8?labelColor=15181f)](LICENSE)

**[Run it in the browser →](https://jimmy110101013.github.io/fly-megacode/)** ·
[what is real and what is modelled](#how-it-works) ·
[who does the remembering](#working-memory-who-is-doing-the-remembering) ·
[data provenance](web/NOTICE.md)

> **Not a clinical tool.** This is a neuroscience experiment. It is not for patient
> care, it is not ACLS training or certification material, and it is not affiliated
> with, endorsed by, or reviewed by the American Heart Association. Use the AHA's
> own published guidelines as the clinical authority.

## Status

The fly runs the megacode on its own memory. It is shown only what a bystander
could read off the patient — rhythm, ETCO₂, whether a line and a tube are in. How
many shocks, when epinephrine was last given, which cycle this is: those are
records of what has been done, and the fly holds them in its own central complex.

```
chance (engine accepts ~1.2 of 10 actions)   11.8%
after 150 megacodes                          57.4%
after 600 megacodes                          74.5%
after 2,200 megacodes                        79.9%
```

Those four numbers are one seed — the configuration the page ships — measured on
unseen scenarios. Across three seeds the same condition averages **76.8%
(71–82%)**; the table [further down](#working-memory-who-is-doing-the-remembering)
is the one to quote, because a single seed swings by thirty points on this task.

An earlier version scored 96%, but it was handed the history as input features,
which meant the scenario engine was acting as the animal's working memory. Taking
that away costs about sixteen points and is the point of the exercise: nothing
outside the fly is keeping track of anything now.

### It was the code, not the brain

An earlier version plateaued near 37% and the obvious reading was that 2,597
Kenyon cells are not enough for a task this structured. That reading was wrong,
and two measurements killed it:

- a **linear readout** on the identical feature vector reaches **100%** with 400
  parameters, against the mushroom body's 33,799 plastic synapses. The features
  fully determine the answer; nothing was missing from the input.
- **doubling the brain changes nothing.** Both hemispheres together (5,177 KC,
  89,315 synapses) score the same as one (77.6% vs 78.4%, 4 seeds each). Half a
  mushroom body is worse (62.9%), so size matters up to a point and then stops.

What actually bound it was **sparsity**. At the 5% activation the model started
with, 130 Kenyon cells carried every pattern and two clinically opposite states —
airway in, airway not in — produced codes that overlapped 80%. The fly learned an
average of the two. Dropping to 1.5% (39 cells) cut that overlap and took accuracy
from 37% to 88%; annealing exploration further took it to 96%.

Three changes, all inside the biological range: 5 state claws + 2 action claws
(real Kenyon cells carry 5–7), 1.5% activation, and a softmax floor of 0.06.

### An earlier 70% was not real either

The very first version scored 70%, and that number was an artifact of the grader,
not of the fly. `correctActions()` accepted `rhythm_check` at *every* moment inside
a CPR block, which made "check the rhythm" a permanently safe answer. The fly found
that degenerate policy and stopped: across 600 test megacodes it never once gave
epinephrine, established access, or intubated. Restricting the rhythm check to the
end of a block — which is also what "minimize interruptions in chest compressions"
requires — removed the exploit.

## How it works

```
ACLS engine (teacher)  ──state──▶  mushroom body (decides)  ──action──▶  engine grades
      AHA 2025                      real FlyWire wiring                   correct? 1 bit
   state machine                                                              │
        ▲                                                                     ▼
        └──────────────────── dopamine: PAM if right, PPL1 if wrong ──────────┘
```

The fly is never shown the right answer -- only whether its own choice was right,
after it commits. That is operant conditioning, not supervised learning.
Clinical correctness lives entirely in the engine; the fly never gates it.

### Real, taken from the connectome (FlyWire FAFB v783)

| | |
|---|---|
| Kenyon cells (right hemisphere) | 2,597 |
| MBONs | 48 |
| DANs | 165 (153 PAM reward / 12 PPL1 punishment) |
| KC→MBON synapses, used as initial weights | 33,799 |
| DAN→MBON compartment wiring | 831 |
| MBON→DAN feedback | 1,013 |

### Modelled, because a clinical state is not an odour

- the projection-neuron layer: each KC samples 5 clinical-context channels and 2
  action channels at random, and fires on their coincidence
- APL global inhibition, as k-winners-take-all at 5% (130 of 2,597 KCs per pattern)

### Learning rule

The fly's own: dopamine released in a compartment depresses the KC→MBON synapses
of exactly those Kenyon cells that were firing when it arrived, with slow recovery
toward the anatomical baseline. Depression only, in both directions -- reward
depresses avoidance compartments, punishment depresses approach compartments.

The decision reads the *balance* — how far each compartment has been pushed from
its anatomical resting weight — not absolute synaptic strength. A naive fly has no
innate preference between actions; learning is what tips the scale. Reading raw
weight instead bakes in a permanent anatomical bias toward whichever action happens
to sit on heavier synapses, and the model collapses onto two actions and stays there.

MBON behavioural polarity is taken from **which dopamine population owns the
compartment**, not from the MBON's transmitter. Transmitter identity gets the sign
backwards (MBON-γ1pedc is GABAergic yet drives approach) and with it the model sits
at chance forever.

## The atlas

The brain panel draws traced morphology, not an abstraction: 5,606 neurons from
the FlyWire v783 skeletons — both mushroom bodies, every Kenyon cell, MBON, DAN
and the two APL cells — simplified to 642,640 neurite segments.

- **activity propagates.** Each cell is split into five bands along the calyx →
  peduncle → lobe axis, and a wavefront crosses it over ~0.45 s. That is the
  direction the mushroom body's signal actually runs: dendrites in the calyx,
  axons down the peduncle, output synapses in the lobes. MBONs answer only once
  the wave has reached the lobes. Lighting the whole arbour at once, which is what
  the first version did, reads as a patch switching on rather than a cell firing.
- **brightness carries firing, never thickness** — thickness would read as a
  change in the anatomy.
- **both hemispheres compute.** The model runs on the bilateral circuit (5,177 KC,
  96 MBON, 331 DAN, 89,315 synapses), so both mushroom bodies fire. An earlier
  version modelled the right side only and dimmed the left, which just looked like
  a rendering fault.
- **the central complex is drawn too**, 2,875 more traced neurons between the two
  mushroom bodies, lit by how much each one is currently holding. Yellow is the
  mushroom body deciding; cyan-white is the central complex remembering.
- **memory overlay** colours each Kenyon cell by how far its KC→MBON synapses have
  been driven from the anatomical baseline: Σ(w₀ − w). That is the engram, drawn on
  the cells that hold it.
- **add the whole brain** drops in 78 neuropil volumes — the JFRC2 surfaces
  transformed into FlyWire space, shipped with `fafbseg` — coloured by the Ito et
  al. 2014 super-categories, which is the grouping published atlases key their
  palettes to. Thousands of overlapping triangles at very low alpha accumulate
  into a soft volume, the way a confocal stack does. Off by default: the optic
  lobes dwarf everything else.

Extraction is one pass over 268 million skeleton nodes, about 75 seconds, cached to
a pickle so the simplification can be retuned without rescanning. The first version
of that pass did a full array comparison per neuron per batch and would not have
finished; `pipeline/test-skeleton-logic.py` covers the tree splitting and RDP
simplification, which are the parts that are expensive to get wrong.

Three things made the first render unreadable, all of them mine:

- **the projection axes were swapped, then unflipped.** In FAFB nanometre space x
  is medial-lateral, y is anterior-posterior and z is dorsal-ventral, so a frontal
  view is x against z — and z grows toward the calyx, which is dorsal, so the
  vertical axis has to be flipped to draw the brain dorsal-up.
- **simplification was too aggressive.** At 2.6 µm tolerance, 66% of branches
  collapsed to two-point straight sticks, which read as noise. A skeleton also
  splits into ~370 unbranched runs and RDP cannot take a run below two points, so a
  total-vertex budget is unreachable — keeping the longest runs at fine tolerance
  is what works.
- **no additive blending, and one alpha for every class.** Thousands of faint
  neurites only become a glowing calyx if they accumulate. And there are 5,177
  Kenyon cells against 96 MBONs, but an MBON's processes are long, so equal weight
  hands the image to the 96.

The page is pure client-side — there is no server behind a published artifact — so
it all runs on the viewer's phone.

The atlas started as a 2D canvas that rasterised all 642,640 segments whenever
anything changed. Caching the silent brain as a bitmap made that affordable, but it
also made the view a fixed still image: nothing could be rotated, because rotating
invalidates the cache every frame. It now runs on the GPU instead — one indexed
`LineSegments` buffer uploaded once, with activity computed in the vertex shader
from a 5,177-pixel float texture holding each cell's wavefront position and
envelope. Per-frame CPU cost is a uniform and an 80 kB texture upload no matter how
much is on screen, the camera is free, and lines rasterise at native resolution
instead of being composited by hand.

Depth attenuation (near neurites brighter than far ones) is what makes rotation
informative rather than a flat mat of lines sliding around.

Dragging to orbit is enabled only above 760 px of viewport width; a phone gets the
preset frontal and dorsal views, a third of the geometry, and a capped pixel ratio.
The redraw interval still adapts to the frame rate the device actually sustains.

## Working memory: who is doing the remembering?

The mushroom body stores associations in its synapses but carries nothing between
decisions. Everywhere above, the scenario engine hands the fly "two shocks given,
epinephrine three minutes ago" as input features — which means **the engine has
been acting as the animal's working memory all along.**

The central complex is where a fly would hold that itself: the ellipsoid body and
protocerebral bridge sustain a heading in complete darkness, persistent activity
held by recurrent wiring rather than by changing synapses. So
`pipeline/extract_cx.py` pulls that circuit out — 2,875 neurons and 311,710
recurrent CX→CX connections signed by transmitter, of which the 171,030 carrying at
least two synapses are simulated — and runs it as a **fixed reservoir**.
Nothing in it is trained. The drive into it is the real MBON→CX pathway (1,368
connections) carrying the mushroom body's own output, and the only plastic synapses
in the whole loop stay where the fly actually has plasticity: KC→MBON.

Three conditions, identical mushroom body, identical learning rule, 3 seeds,
2,500 megacodes. Chance is 11.8%.

| who remembers | features | accuracy |
|---|---|---|
| the engine (every run above this section) | 41 | 82.2% |
| nobody — only what a bystander could see | 26 | **17.6%** |
| the fly, in its own central complex | 66 | **76.8%** |

Strip the history and the task collapses to near chance. Give the fly a reservoir
to hold it in and **it recovers 92% of the gap**, with a tighter spread across
seeds than the version that was handed the answer.

Observable means observable: rhythm, ETCO₂, whether a line and a tube are in — you
can see those. Shot count, drug timing, cycle number and whether the cause was
treated are records of what was done, and those are what the reservoir has to hold.

### Two bugs that made this look impossible first

`pipeline/cx-decodable.mjs` trains a linear decoder on the reservoir's readouts to
ask whether the history is in there at all — separating "the central complex isn't
holding it" from "the mushroom body can't read it". It found nothing, twice, for
two different reasons:

- **the drive carried no action identity.** The reservoir was being fed raw MBON
  activity, and the raw patterns for different actions have a cosine similarity of
  **0.967** — averaging over ~78 random Kenyon cells washes out a code whose
  overlap is only 6.5%. What differs between actions is the *deviation from the
  anatomical baseline*, which is also what the decision reads. Driving the CX with
  that instead is the fix.
- **half the drive pathway indexed off the end of an array.** MBON→CX was extracted
  bilaterally (96 MBONs) while the experiment ran one hemisphere (48), so half the
  edges read `undefined` and quietly filled the reservoir with NaN. The symptom was
  results that stayed *identical to the decimal* across parameter changes — which
  is the shape of a dead component, not an ineffective parameter.

With both fixed, shock count decodes from 16 hub readouts at 78.1% (53.8% for
always guessing the commonest class) and epinephrine timing at 94.3% (68.4%).

## Other brain regions

`pipeline/extract_region.py` looks for the mushroom body's shape elsewhere in the
brain — a large cell population converging on a small set of outputs, with
dopaminergic neurons innervating those outputs — and builds a same-sized circuit
wherever it finds one. `pipeline/other-regions.mjs` then runs the identical model
and learning rule on it.

The dopamine split is deliberately generous: the mushroom body has named reward
(PAM) and punishment (PPL1) populations, while elsewhere there is nothing to go on,
so the dopaminergic neurons are clustered into two groups by what they target.
That is the best case an alternative region can be given.

Identical model, identical rule, 3 seeds, 3,000 megacodes each. Chance is 11.8%.

| region | cells | synapses | dopamine (reward/punish) | accuracy |
|---|---|---|---|---|
| **mushroom body** | 2,597 | 33,799 | **153 / 12** | **74.7%** |
| lobula → lobula plate | 2,597 | 23,039 | 8 / 1 | 55.0% |
| medulla → lobula complex | 2,597 | 24,121 | 16 / 4 | 38.4% |
| medulla | 2,597 | 7,079 | 10 / 1 | 27.0% |
| mechanosensory | 1,301 | 13,197 | 13 / 4 | 21.1% |
| central complex | 1,437 | 7,504 | 10 / 1 | 20.0% |
| olfactory | 1,134 | 13,440 | 225 / 51 | 16.1% |
| ascending neurons | 1,206 | 4,767 | 63 / 30 | 2.8% |

Two things fall out of this.

**The shape matters more than the address.** Accuracy tracks the size of the
expansion layer times how densely it converges: the three circuits that happen to
have 2,597 cells feeding 48 outputs are the three that do best, and a chunk of
optic lobe reaches 55% despite having nothing to do with learning. Sparse
expansion onto a small readout is a good substrate for this rule wherever it
occurs, and the fly's visual system is full of it.

**What the mushroom body uniquely has is the teaching signal.** 165 dopaminergic
neurons reach its 48 outputs, in named reward and punishment populations. The next
best anywhere is 20, and the visual class has 2 — too few to build a circuit at
all. The optic lobe circuits above only score what they score because this script
*invented* a reward/punishment split for them by clustering; in the animal there
is no such organisation and no dopamine to drive it. The wiring elsewhere could
compute this. Nothing else in the brain could be told whether it was right.

The one region with plenty of dopamine, the olfactory class, scores 16.1% — barely
above chance — because its expansion layer is small and most of that dopamine is
the mushroom body's own PAM cluster spilling into the neighbourhood.

## Weight controls

Same topology, same learning rule, same draw sequence, 1,500 scenarios each:

| initial KC→MBON weights | start | end |
|---|---|---|
| real synapse counts | 52.3% | 68.4% |
| same multiset, reassigned across all edges | 43.3% | 63.8% |
| resampled from the same marginal distribution | 35.5% | 66.0% |
| uniform random | 45.6% | 68.3% |

Read honestly: **the initial weights do not matter.** Every mode converges to
64–68%; the spread in the starting column is initial-condition noise that the
plasticity rule erases within a few hundred scenarios. An earlier run appeared to
show a ten-point advantage for the real counts; that was an artifact of the weight
modes consuming different random draws, and it disappeared once they were given a
shared generator.

This does **not** say the connectome is irrelevant. Every mode above runs on the
real 33,799-edge KC→MBON structure — only the numbers on those edges changed. The
control that would actually test the wiring diagram, rewiring the graph while
preserving degree, has not been run.

## Layout

```
web/acls-engine.js     AHA 2025 Adult Cardiac Arrest Algorithm -- the teacher
web/mushroom-body.js   the fly
web/atlas3d.js         traced morphology of both mushroom bodies, on the GPU
pipeline/extract_mb.py connectome -> data/out/mb_circuit.json
pipeline/extract_skeletons.py  skeletons -> data/out/atlas.json
pipeline/train.mjs     headless learning curve
pipeline/test-plasticity.mjs  asserts the plasticity rule moves value the right way
pipeline/diagnose.mjs  per-boundary instrumentation
pipeline/ceiling.mjs   where the accuracy ceiling comes from, per action
pipeline/capacity.mjs  brain size vs code discriminability
pipeline/more-brain.mjs  half / one / two mushroom bodies, multi-seed
pipeline/upper-bound.mjs a linear readout on the same features -- the ceiling
pipeline/extract_region.py  build the same circuit shape from another brain region
pipeline/other-regions.mjs  run the identical model on those regions
pipeline/extract_cx.py      central complex as a fixed recurrent reservoir
web/central-complex.js      that reservoir
pipeline/working-memory.mjs who remembers: engine, nobody, or the fly
pipeline/cx-decodable.mjs   is the history in the reservoir at all?
pipeline/extract_cx_skeletons.py  central complex morphology for the atlas
pipeline/test-engine.mjs  16 guideline-conformance checks, each citing its box
web/NOTICE.md          what each connectome file contains, and how to rebuild it
```

## Run

The connectome files the page needs are committed, so nothing has to be downloaded
to run it or to reproduce the numbers.

```bash
git clone https://github.com/Jimmy110101013/fly-megacode && cd fly-megacode

node pipeline/test-engine.mjs        # the grader — 16 checks on the ACLS state machine
node pipeline/test-plasticity.mjs    # the learning rule moves value the right way
node pipeline/train.mjs 2500 real    # train and evaluate on unseen megacodes

python3 pipeline/serve.py            # then open http://localhost:8777/
```

Every measurement the README cites has its own script in `pipeline/`, listed above.

Rebuilding the connectome files themselves needs the raw Zenodo archives; the
commands are in [`web/NOTICE.md`](web/NOTICE.md).

## Data

- connectivity: [Zenodo 10676866](https://zenodo.org/records/10676866), CC BY 4.0
- skeletons: [Zenodo 10877326](https://zenodo.org/records/10877326), CC BY 4.0
- annotations: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations),
  Schlegel et al., *Nature* 2024, CC BY 4.0
- neuropil surfaces: JFRC2NP in FlyWire space, via
  [fafbseg-py](https://github.com/navis-org/fafbseg-py)
- guidelines: AHA 2025 Guidelines for CPR & ECC, Part 9 and the Adult Cardiac
  Arrest Algorithm (*Circulation*, 2025). The procedure is implemented in
  `web/acls-engine.js`; the guideline text is copyrighted by the AHA and is not
  redistributed here.

Exact contents, filters, and checksums of every derived file:
[`web/NOTICE.md`](web/NOTICE.md). Licences:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

FlyWire data is released for non-commercial research; cite Dorkenwald et al. and
Schlegel et al., not this repository, if you build on it.

## Licence

Code: [MIT](LICENSE). The connectome data redistributed in `web/` stays CC BY 4.0
and is not relicensed.
