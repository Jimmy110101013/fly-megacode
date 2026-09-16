# Working on Fly Megacode

A real *Drosophila* mushroom body, wired from FlyWire FAFB v783, learning the AHA 2025
ACLS megacode. The page runs at <https://jimmy110101013.github.io/fly-megacode/>.

## The one rule that matters

**Every number in the README has to be one this repository can reproduce, and every
claim about the fly has to be one the connectome supports.** This project has already
retracted three of its own headline results. Retracting a fourth is fine. Quietly
keeping one that turned out to be an artifact is not.

Concretely, before a number goes in the README:

- **Multi-seed or it did not happen.** A single seed swings thirty points on this
  task. Three seeds minimum, and report the spread, not just the mean.
- **Evaluate greedy on unseen scenarios.** Training accuracy includes exploration and
  is not the same number.
- **Say which architecture it belongs to.** The menu and descending-neuron
  architectures produce different numbers for the same task.
- **When a result gets retracted, the retraction stays in the README.** See "An
  earlier 70% was not real either".

## What is real and what is modelled

| | |
| --- | --- |
| **Real, from the connectome** | every neuron, every synapse, every sign. KC/MBON/DAN identity, compartment structure, MBON→CX, MBON→DN, the relay interneurons, the skeletons in the atlas |
| **Modelled, and labelled as such** | the sensory interface (a clinical state encoded as one-hot glomeruli — a fly has no eyes for an ECG), the action names, the eligibility signal in the descending architecture, rate-based dynamics instead of spikes |

The only plastic synapse is **KC→MBON**. The central complex is a fixed reservoir, the
MBON→DN pathway is fixed, and nothing else is trained anywhere. If a change would break
that sentence, it needs to be argued for in the README, not slipped in.

## Layout

```
web/                 the page. plain ES modules, no build step, no framework
  acls-engine.js       the grader: AHA state machine + feature space. owns clinical truth
  mushroom-body.js     the fly: KC coding, MBON readout, dopamine plasticity
  central-complex.js   fixed reservoir holding the resuscitation history
  descending.js        fixed MBON -> relay -> DN output pathway (experimental)
  atlas3d.js           WebGL atlas, indexed LineSegments + per-cell DataTexture
  *.json               the connectome. see web/NOTICE.md for contents and checksums

pipeline/
  extract_*.py         connectome -> json. needs data/raw/ (not committed, see NOTICE)
  test-engine.mjs      16 checks on the grader. run this first, always
  test-plasticity.mjs  the learning rule moves value the right way
  *.mjs                one script per measurement the README cites
```

`data/` and `refs/` are gitignored. `refs/` holds AHA guideline PDFs — the procedure is
implemented in `acls-engine.js`, but **the guideline text is copyrighted and must never
be committed or published.**

## Before you touch the model

```bash
node pipeline/test-engine.mjs        # the grader. if this is wrong, every number is
node pipeline/test-plasticity.mjs    # reward and punishment are differentiated
```

`test-plasticity.mjs` falls back to `web/mb_circuit.json` when `data/out/` is absent, so
it runs in a clean checkout and in CI.

## Failure modes this project has actually hit

Read these before debugging. Four of the six presented as "accuracy stuck at chance".

- **A dead component reads as an ineffective parameter.** MBON→CX was extracted
  bilaterally (96) while the experiment ran 48, so half the edges read `undefined` and
  filled the reservoir with NaN. The symptom was *results identical to the decimal
  across parameter changes*. If changing a knob changes nothing at all, the component
  is dead, not insensitive.
- **The grader can be exploited.** `correctActions()` once accepted `rhythm_check` at
  every moment, so "check the rhythm" was a permanently safe answer and the fly found
  it. The 70% was real arithmetic on a broken question.
- **Averaging washes out sparse codes.** Raw MBON patterns for different actions have a
  cosine similarity of 0.967; the *deviation from the anatomical baseline* is the
  signal. This bit twice, in the CX drive and again in the descending readout.
- **Adaptive gain eats slow learning.** The descending readout's per-channel gain, run
  at the central complex's adaptation rate, has a time constant of ~500 decisions and
  absorbs everything the plasticity does. Calibrate on the naive animal, then freeze.
- **Asymmetric dopamine locks the animal on its prior.** PPL1 × blame carried 48.9
  total drive where PAM × credit carried 3.0. Lowering the learning rate made it
  *worse*, because a lower rate cannot fix a ratio.
- **Code overlap is the usual ceiling, and sparsity is the usual lever** — 5% → 1.5%
  activation took accuracy from 37% to 88%. But it is not always the lever: in the
  descending architecture, driving 78 KCs down to 10 moved overlap 42% → 28% with no
  accuracy gain at all.

- **A seeded generator can quietly delete half the task.** `Megacode`'s rng was a raw
  xorshift and its first draw picks the rhythm; every script seeded scenarios
  consecutively, so pVT and asystole never appeared in any training or evaluation set
  and every published number described VF and PEA only. Mix a seed before anything
  draws from it, and check the distribution your scenarios actually have rather than
  the one the generator's source code claims.

- **Evaluating mid-run changes the animal.** `decide()` draws from the fly's own PRNG
  even when greedy, so a probe that evaluates between training blocks shifts every
  later draw and trains a different fly. Two scripts that differed only in that gave
  0% and 17% shock rates from the same seed, and two hours went into deciding which
  one had the bug. Neither did. Evaluate on a copy, or at the end.
- **This task has more than one attractor.** Of eight flies trained identically, three
  learn never to shock and five shock whenever it is called for, with nothing in
  between — and both groups land at the same accuracy, 51.2% against 49.6%. A mean
  over seeds can describe a population no individual belongs to, so look at the
  per-seed column before believing the mean.

**After three failed fixes, stop and question the architecture.** Do not attempt a
fourth without saying so out loud.

## Measurement habits that keep working

- **Upper bounds beat intuition.** A linear readout on the same features reached 100%,
  which killed "the fly needs more neurons" in one measurement. `pipeline/ceiling.mjs`,
  `upper-bound.mjs`, `dn-ceiling.mjs`.
- **Decodability probes separate two hypotheses.** `cx-decodable.mjs` asks whether the
  history is in the reservoir at all, which distinguishes "not held" from "held but
  unreadable".
- **Negative results that bear on the README's claims stay in the README.** Glomerular
  claw sampling made things worse and is still in `mushroom-body.js` as a documented
  option.

## Experiment records stay local

Exploratory probe scripts, result tables and write-ups go in `lab/`, which is excluded
through `.git/info/exclude`. **Do not commit or push them.** Only changes to the published
project — the page, the model, README-level results — go to GitHub, and only when asked.

## Style

Comments explain *why*, especially why something is the way it is rather than the
obvious alternative. Several comments in this repo exist to stop a future reader
re-introducing a bug. Do not strip them. Prose in the README is plain and specific.
The page's own copy is in English, academic and engineering in register, with a little
dry humour. The humour never makes a claim the README would not.

## Not a clinical tool

This is a neuroscience experiment. It is not for patient care, not ACLS training or
certification material, and not affiliated with or reviewed by the AHA. Any change that
makes the page look like a training device needs that disclaimer kept visible.
