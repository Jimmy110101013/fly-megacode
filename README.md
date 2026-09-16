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

**Result:** 47.3% (37–57%) on unseen megacodes, 5 seeds, greedy, against 12.3% chance (`chance.mjs`).
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
60% of the gap. (`working-memory.mjs`, 5 seeds, 2,500 megacodes)

| who remembers | accuracy |
| --- | --- |
| the engine hands it over | 66.6% (54–78%) |
| nobody | 18.6% (11–32%) |
| the fly, in its central complex | 47.3% (37–57%) |

> **Results 2 to 5 have not been corrected yet.** They were measured before the
> generator defect in [Retracted](#retracted), so their patients were VF and PEA only.
> Each compares conditions measured the same way, so the directions should survive, but
> the absolute numbers will move as result 1's did. They stay here, marked, until every
> script is rerun.

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
- The measured synapse counts give no advantage as initial weights. After training:
  real 74.9% (69–82%), shuffled 69.9% (64–81%), resampled 71.8% (63–75%), uniform
  random 76.0% (67–83%), 5 seeds (`weight-controls.mjs`). Shuffled trails real by
  5.1 points on 4 of 5 seeds, just past the bar set before the run, but random weights
  do as well as real, so the counts themselves are not what helps.
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
- **76.8% — measured on half the task.** `Megacode` spent the first draw of a raw
  xorshift on the starting rhythm, and every script seeded its scenarios consecutively
  (`s0 + e`). Those first draws never landed in the pVT or asystole bands, so every
  number this repository has published was trained and evaluated on VF and PEA alone —
  56% and 44% of patients, where the generator specifies 40/15/25/20. Mixing the seed
  before anything draws from it fixes the generator, and the headline falls to 47.3%
  (37–57%) over 5 seeds. Chance barely moves, 12.5% to 12.3%, so the margin over chance
  survives; "the fly recovers 92% of the working-memory gap" does not, it recovers 60%.
  Found by asking why the page showed 50–70% when the README claimed 77%.

## The page

Everything runs in the browser. The atlas draws 5,606 traced neurons from the FlyWire
skeletons plus the central complex; a cell lights up along its calyx→lobe axis when it
fires, and the central complex glows by how much it is holding. Hover a region for its
name and its role in the model, Ctrl/⌘-scroll or pinch to zoom; double-click returns to
the frontal view.

**Two speeds.** Demo takes one decision every couple of seconds and keeps the patient's
sequence on screen: every action in order, the wrong ones in red with what the guideline
wanted beside them. Training runs at whatever the speed slider allows and hides that
sequence, because at training speed what is worth watching is the accuracy curve rather
than any single decision. The animal is identical in both; only the clock changes.
The header's *recent accuracy* is the mean of the last sixty decisions with learning
still running, which is a pulse rather than a score: a sixty-decision window swings
tens of points on its own, and it is not the quantity the result at the top reports.

**Working memory layer.** The third atlas mode shows the central complex alone, the
mushroom body dimmed to context. Each reservoir cell is coloured by its state, above or
below rest, scaled to the 95th percentile and capped: scaled to the most active cell,
only a handful of cells ever showed. On every decision the cells the real MBON→CX
pathway reaches flash yellow in proportion to how hard they were driven (scaled the same
way, at the 90th percentile of the driven cells), and the eight
cells the mushroom body reads back are marked white. Which eight is modelled: they are
the reservoir's hub cells by in-degree, not an anatomical output. The state clears with
every new patient.

**Memory tape.** A panel beside the MBON vote draws, for the current patient, what those
eight readouts handed the Kenyon cells at every decision: one row per readout, shaded
by the level (0–4) the feature space actually encodes, with marks where a shock,
epinephrine or an antiarrhythmic was given. It is the fly's input, not a smoothed view
of the reservoir. A finished case stays on the tape until the next patient's first
decision. Expect the rows to climb together as the case goes on, several nearly
identical; whatever shock history they hold shows as the smaller differences between
them.

**Why it chose that.** The fly scores all ten candidate actions before committing to one,
and the page used to show only the commitment. This panel draws the scores: a bar per
action, longest first, the one taken filled in, a tick on each action the grader would
have accepted, and the lead over the runner-up as a number. It is the difference between
"it was wrong" and "it ranked the guideline's answer eighth out of ten" — and between a
confident mistake and a near-tie the softmax lost, which at the annealed temperature is
where nearly every departure from its own top pick happens.

**Rehearsal.** Below the atlas, one row per starting rhythm (VF, pVT, PEA, asystole),
each a fixed patient the fly never trains on. Whenever the fly has learned something new,
it runs that patient again as it is at that moment: greedy, for up to 14 steps, learning
nothing, with its history in a separate copy of the central complex. Each step shows the
action taken; where the grader disagrees it is struck through, with the guideline's
answer underneath. The fly holds no sequence anywhere. It holds what to do in a state,
and the order comes out one step at a time as the grader changes the patient and the
central complex keeps the record. A naive fly values every action at exactly zero, so
its rows say "no preference yet" instead of a confident first action. One patient per
rhythm is a sketch, not an evaluation; the evaluation is the result at the top.

**Memory layer.** Switched on, every Kenyon-cell axon is drawn lobe by lobe (vertical,
medial, peduncle) at (Σw / Σw₀)³ of that cell's KC→MBON connections there, so with the
animal paused a trained mushroom body is visibly darker than a naive one. On top of
that, the cells on the decision just taken light up at the same scale, which is what
shows where this particular path has been depressed. The calyx is drawn as context
only: it holds 3% of the connections but so much dendrite that at full brightness it
outshone the lobes that learn. The dopamine neurons flash on every verdict: PAM if
right, every other DAN if wrong, as in the model. Nothing brightens, because nothing in
the model potentiates. The lobe of
each connection is FlyWire's own per-connection neuropil label, not an estimate:
`pipeline/extract_lobes.py` checks it edge for edge against `mb_circuit.json`, and for
99.7% of those labels the cell's own traced axon does pass through that lobe. 27% of
connections are labelled with a neighbouring neuropil (mostly CRE and SIP, beside the
lobe tips) and are not drawn. It stops at lobes on purpose: the connectivity table does
not resolve compartments, and an MBON takes dopamine from about five DAN types, so
assigning a synapse to γ1 or γ5 would be a guess. The compartments show up where the
DAN skeletons light, which needs no guessing.

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
