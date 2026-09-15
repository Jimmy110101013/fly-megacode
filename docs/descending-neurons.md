# Taking the menu away

*A closed line of work. Archived here rather than deleted, with everything that was
measured. The page still runs the menu architecture.*

## The question

In the model the page ships, the action is an **input**. `encode(features, action)`
builds one Kenyon-cell code per candidate action, all ten are scored, and a softmax
picks one. A fly has no such menu. Mushroom-body output reaches descending neurons,
and the descending neurons are the brain's entire output to the body — roughly 1,300
cells for everything the brain can ask the body to do.

So: make the action an **output**. One state, one Kenyon-cell code, one MBON
population response, and the action is whichever descending channel wins.

## Gate: can the fly's own output wiring express ten actions?

`pipeline/dn-separable.mjs`. Granting perfect learning, is each of ten channels the
winner for *some* achievable MBON pattern?

| pathway | channels reachable |
| --- | --- |
| direct MBON→DN only, under depression-only plasticity | **5 / 10** |
| direct + relay interneurons | **10 / 10** |

Only 31 of 96 MBONs reach a descending neuron directly. Through the relay — 1,951
central interneurons carrying 7,916 connections in and 27,253 out — all 96 are on a
path. That matches what is known about the pathway: most of the mushroom body's reach
into the descending neurons is indirect.

So the experiment was on, but only through the relay. `pipeline/extract_dn.py` pulls
the whole thing out; `web/descending.js` runs it as fixed anatomy.

## It did not learn

3 seeds, 2,500 training megacodes, 300 unseen scenarios evaluated greedy, chance 11.8%.

| condition | accuracy |
| --- | --- |
| menu — scoring ten labelled options | **71.1%** (79-64-71) |
| descending — nothing downstream learns | 13.7% (23-8-10) |
| descending — MBON→relay plastic, any-dopamine gate | 8.6% (11-9-6) |
| descending — same, gated on PAM/PPL1 only | 11.7% (17-7-11) |

The bar was **33.1%**: what the best permutation of the ten channel names already
scores *untrained* (`pipeline/dn-naming.mjs`). Nothing came near it, so the
degree-preserving rewiring control in `pipeline/rewire-control.mjs` was never run —
with both arms at chance it could not have discriminated anything.

## Where the information actually stops

`pipeline/dn-ceiling.mjs` puts a linear decoder at each stage of the untrained
pathway and grants it perfect learning. 13,904 decisions, chance 11.7%.

| stage | inputs | decodable |
| --- | --- | --- |
| Kenyon-cell code, state only | 5,177 | 71.9% |
| MBON output, anatomical weights | 96 | 70.9% |
| descending channel drive | 10 | **67.1%** |

**Nothing in the connectome destroys the answer.** Not the 41.9% overlap between
state codes, not the 5,177 → 96 compression, not the fixed MBON→DN map. The answer is
still there at the ten channels and the fly gets 13.7% off it.

`dn-naming.mjs` splits the remaining gap:

| readout | accuracy |
| --- | --- |
| argmax, action *k* nailed to the *k*-th descending type by anatomical rank | 12.4% |
| argmax, best permutation of the same ten names | 33.1% |
| full 10×10 remix of the channel drives | 67.1% |

A third of the achievable performance is the **naming** alone, and ranking descending
types by mushroom-body drive turns out to be close to the worst assignment available.
The rest needs the channels **mixed**, which argmax over a fixed map cannot express.

## Why the plastic relay made it worse

Not a dead component and not an exploding one: 12.2% of the 7,916 MBON→relay edges
moved, all of them weakened as the rule requires, 94.3% of the anatomical weight
retained, largest single change 93.5%.

The rule is the problem. **Depression-only plasticity can only subtract, and the relay
is exactly what made 10 of 10 channels reachable.** Weakening it walks the model back
toward the direct-only pathway, which reaches 5. Measured: channel drive spread falls
from 2.335 to 1.866 over training.

In the mushroom body, subtraction works because the compartments are an **opponent
pair** and behaviour is *released* — depress the approach side and avoidance wins,
depress the avoidance side and approach wins. One-directional plasticity moves the
behaviour in both directions because the architecture supplies the opposition. A
feedforward relay has no opposing side, so subtraction there is not a choice between
two outcomes; it is just removing pathway.

## Two anatomical results that stand on their own

**The relay layer's dopamine is overwhelmingly punishment.** Of 1,951 relays, 468 sit
under mushroom-body DANs:

```
PAM   (reward)      33
PPL1  (punishment) 451        14x
```

PPL1 has 24 cells to PAM's 307, but projects broadly outside the lobes into CRE/SMP.
This is the same fact as the **16× functional asymmetry** measured independently in
the learning rule — PPL1 × blame carried 48.9 total drive where PAM × credit carried
3.0 — and it is why equalising the two arms was necessary rather than a fudge. The
anatomy predicted the failure mode before the failure mode was understood.

**The relay layer is not specially dopaminergic.** Receiving dopaminergic input at ≥2
synapses:

| population | share |
| --- | --- |
| relay interneurons | 46.8% (914/1,951) |
| **all central interneurons — baseline** | **44.6%** (10,616/23,814) |
| Kenyon cells | 99.7% (5,160/5,177) |

46.8% against a 44.6% baseline is nothing. An earlier note in this repository cited
the raw 914 as anatomical support for treating the layer as plastic; **that reading
was wrong** and is retracted here. Kenyon cells at 99.7% are what specific
dopaminergic innervation actually looks like.

## What would have to change

The architecture needs a readout that can *mix* channels, and depression-only
plasticity cannot build one in a feedforward layer. The ways forward all cost
something this project has so far refused to spend:

- **allow potentiation** — then it is no longer the fly's rule, and the best-evidenced
  property of KC→MBON is gone
- **find an opponent structure in the output pathway** — there may be one; nobody has
  shown it, and inventing it would be the same move as inventing the naming
- **let the fly learn the channel→action assignment** — that is the honest fix for the
  12.4% → 33.1% half, but it is a second plastic layer with the same problem

The menu architecture stays because it works and because what it assumes — that the
candidate action is presented — is at least stated plainly rather than smuggled in.

## Is there a basal ganglia down there?

Humans select actions through opponent pathways: a go route and a no-go route
converging on each action channel. That is exactly what this line of work lacked.
The mushroom body already has the opponent half — PPL1-owned compartments hold
approach-driving MBONs, PAM-owned compartments avoidance-driving ones — so
`pipeline/dn-opponent.py` looks for descending neurons wired to *read that difference*:
approach-group MBONs pushing one way, avoidance-group MBONs the other, through direct
or one-relay paths, signed by transmitter.

The null shuffles MBON valence across the 35 MBON types, keeping left/right copies
together.

| setting | balance-like DNs | shuffled mean | shuffled 95th pct | measured beats |
| --- | --- | --- | --- | --- |
| top 300 DNs, \|c\| ≥ 0.6, balance ≥ 0.25, 2,000 shuffles | 42 | 38.7 | 62 | 58% |

No individual descending neuron survives FDR q < 0.1. Sweeping the thresholds
(\|c\| 0.4/0.6/0.8, balance 0.1/0.25/0.4, top 100/300/600 DNs, direct-only and
relayed) gives "measured beats shuffled" from 14% to 96%, with one setting of 36
above 95% — what 36 heavily overlapping settings produce by chance. The measured
count sits a little above the shuffled mean in most settings, so there may be a
faint lean, but nothing here is evidence of a dedicated opponent read-out.

What this can and cannot rule out: it tests one signature — a descending neuron
integrating mushroom-body valence with opposite signs — using predicted transmitters,
glutamate counted as inhibitory, no gap junctions, no receptors. An opponent stage
built from recurrent loops, from the central complex, or deeper than one relay would
not show up.

## Looking in the central complex instead

In Drosophila it is the central complex, not the mushroom-body-to-DN path, that is
shown to select behaviour: PFL3 cells compare heading with goal and drive turning
through DNa02/DNa03. The central complex has no PAM/PPL1 split, so the opponent axis
tested is **hemisphere**. Shuffling transmitter signs instead would destroy a
structural property of the tissue and manufacture a positive.

`pipeline/cx-opponent.py`, metric fixed before running: for a CX cell type and a
descending type with one left and one right cell, how strongly does a cell's soma side
predict which of the two it drives harder (direct plus one central relay, signed)?
Null: soma side labels shuffled within the type, 10,000 times.

**Positive control first, and the scan only runs if it passes.**

| known case | \|r\| | p |
| --- | --- | --- |
| PFL3 → DNa02 | 1.000 | 0.0001 |
| PFL3 → DNa03 | 0.980 | 0.0001 |

**Then every CX type × every left/right descending pair — and the same metric on
central-brain types outside the central complex** (`pipeline/lateral-control.py`, 90
types drawn with a fixed seed before any result was seen). Without that second row
the scan could just be measuring that neurons are lateralised.

| population | tests | raw p ≤ 0.001 | FDR q < 0.05 | types with a hit |
| --- | --- | --- | --- | --- |
| central complex | 644 | **27.0%** | 275 | 18 / 90 |
| other central-brain types | 3,341 | **2.0%** | 68 | 4 / 85 |

The raw rate is the comparison that matters, because FDR grows stricter with the
number of tests and the two rows ran very different numbers. Thirteen-fold. The
central complex's output onto descending neurons is split by hemisphere far more than
ordinary central neurons are. Of the 222 strongest splits, 156 cross to the
contralateral descending neuron and 66 stay ipsilateral. The hits are concentrated in
the known premotor output types (PFL1, PFL3, PFR, FR) and the fan-shaped-body columnar
types (FC, FS); EPG, the compass neurons, also appear, presumably through relays.

**What it is and is not.** This is an opponent structure, and it is exactly the kind
the descending-neuron line of work lacked — but it is **one axis, left versus right**,
replicated across many descending pairs. It is a two-alternative steering read-out, not
a ten-way action selector, and nothing here shows that mushroom-body valence enters it.
It is wiring, with predicted transmitters and no gap junctions or receptors. What it
does establish is where in this connectome a push-pull output stage actually is.

## Scripts

```
pipeline/extract_dn.py        the pathway: MBON->DN, relay interneurons, dopamine gates
pipeline/dn-separable.mjs     can the measured wiring express ten actions at all?
pipeline/dn-ceiling.mjs       a linear decoder at each stage: where does the answer stop?
pipeline/dn-naming.mjs        how much of the gap is naming, how much is mixing
pipeline/no-menu.mjs          the four-condition comparison above
pipeline/rewire-control.mjs   degree-preserving rewiring control (written, never run)
pipeline/dn-opponent.py       search for descending neurons that read the valence balance
pipeline/extract_cx_out.py    central complex -> descending neurons, direct and via one relay
pipeline/cx-opponent.py       hemisphere opponency in CX output, gated on the PFL3 positive control
pipeline/lateral-control.py   the same metric on non-CX central types, as the negative control
web/descending.js             the output pathway, fixed or plastic
```
