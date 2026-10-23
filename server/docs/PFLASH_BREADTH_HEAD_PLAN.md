# PFlash breadth head: plan

Status: plan only. No training has run, and no code is in this PR. Training needs an explicit go after the gate below.

## Problem

With `--pflash on`, the unified view pages a long prompt down to the budget from the release keep curve. The curve sets
that budget from the prompt length alone: about 27 % of the droppable tokens between 32K and 64K, whatever the
question.

On the private dev split (93 fresh-r13 rows with gold evidence spans, Qwen3.8-27B target, R9700):

| | correct |
|---|---|
| unified view, release curve | 48 / 93 |
| full prefill | 60 / 93 |

The gap comes from the budget, not from the ranking:

- **Evidence kept (63 rows):** the unified view gets 48 right and full prefill 46. Compression costs nothing when the
  gold evidence fits.
- **Evidence dropped (30 rows):** the unified view gets 0 right and full prefill 14.
- **Which rows drop evidence:** broad questions, such as "find every passage that mentions X and return their ids".
  Rows with 3 or more gold spans drop evidence on 21 of 36; rows with fewer, on 9 of 57.
- **The query is not the cause:** the request probe covered the question on 29 of the 30 rows.

## Why the scores can't flag it

None of the signals from the scorer's output predicts lost evidence or a wrong answer. All score an AUROC between 0.40
and 0.65:
- the dropped share of score mass;
- score entropy;
- the top-1 margin;
- the edge-to-top ratio;
- the high-score cost.

A valve on these signals keeps no more evidence than the same extra tokens spent uniformly or on random rows. A broad
question produces many medium scores, and so does a vague question with one answer. Re-ranking doesn't help either:
ranking by peak score, or with lexical anchors, stays within ±3 of the score order.

Breadth shows in the question ("every", "all", "list the ids"), so it has to be read from the question.

## Oracle bound (offline, dev traces)

The **oracle budget** of a row is the smallest cap at which the shipped ranking keeps every gold-evidence unit. The
replay simulates `admit_paging`, including the preamble and the line markers. It agrees with the served selection to
within about one unit at the budget edge: 62 rows keep their evidence in the replay against 63 served.

A perfect breadth predictor gives each row the smallest tier that holds its evidence. The table compares it with a
uniform cap raise at the same mean optional tokens:

| tiers (× release cap) | perfect predictor: evidence kept | tokens | uniform raise, same tokens: evidence kept |
|---|---|---|---|
| none (release) | 62 / 93 | ×1.00 | — |
| 1, 2, full | 93 / 93 | ×1.79 | 70 / 93 |
| 0.5, 1, 2, full | 93 / 93 | ×1.49 | 68 / 93 |

With tiers 0.5, 1, 2 and full, the perfect predictor assigns:

| tier | rows | mean gold spans |
|---|---|---|
| 0.5 | 55 | 2.0 |
| 1 | 7 | 2.9 |
| 2 | 11 | 3.5 |
| full | 20 | 4.4 |

What the bound says:

- **Targeting:** at equal tokens, targeting by breadth keeps evidence on about 25 more rows than a uniform raise. It
  reaches all 16 rows where full prefill is right and the unified view is wrong; the uniform raise reaches 5 to 7.
- **Savings:** most questions need half the release budget or less, so a tier below ×1 pays for the broad ones.
- **Ceiling:** about full-prefill accuracy (60 / 93) at about ×1.5 tokens. Full prefill serves about ×3.6 tokens.

A real head will be noisier. The cost of a miss is asymmetric:
- under-predicting drops evidence, so the answer is wrong;
- over-predicting only spends tokens.

The tier mapping is therefore biased upward.

## Design

### Where it runs

A small extra output on the request probe. The probe already encodes the question on every compressed request, and
this head reads that pooled question state. The runtime cost is one small linear layer. It needs no extra drafter
pass and no regex or keyword rules.

### Label

For each training row: the oracle budget divided by the release cap, as a continuous ratio, from:
- the gold evidence spans of the row;
- the shipped scorer's per-unit costs and scores (the unified view's `trace_units`).

The label follows the shipped ranking, so it states how much this scorer needs, not how many spans exist. Rows with
no gold spans get no label.

### Output and action

- The head regresses log(oracle ratio), or uses ordinal classes {≤0.5, 1, 2, full}.
- Its prediction maps to a cap multiplier on the release curve: the region's `base_cap` × the tier, with "full"
  serving the region whole.
- The tier thresholds are calibrated on the dev split for a target miss rate, the share of rows put in a tier below
  their oracle. They are frozen before the confirm run.
- The widening uses the existing `budget_widened` path in `apply_unified_view`, so paging and recall see the same cap.

### Relation to stage 2

Stage 2 was trained to predict vague and look-back requests in multi-turn sessions. The release profile does not load
it (`stage2_not_loaded` in the traces). Breadth is a separate output with a separate label, and it sits next to the
request probe's query head.

## Data

- **Dev split** (93 rows): tuning and tier calibration only. It never trains the head.
- **Check split** (121 fresh-r13 rows): read once for the confirm run.
- **Training pool:** new rows with gold spans, disjoint from both splits:
  - fresh-r13 builds drawn from fresh seeds;
  - pflash-bench sources with Sol-labeled evidence spans.

  It needs varied, real phrasings of broad and narrow questions. At least one whole question family is held out of
  training and scored separately, so a head that learns only the benchmark's templates shows up as a gap on that
  family.
- **Size target:** at least 2,000 labeled rows, spread across 16K to 120K lengths and across code, prose, tables and
  mail. The pre-training check confirms the class balance.
- **Privacy:** private eval data stays off W&B, PRs and commits. PR text reports aggregates only.

## Gate before any training run

All of these must be met, each with written evidence:

1. **Oracle bound:** the perfect-predictor gain at matched tokens is at least +10 evidence rows on dev (measured:
   +23 to +25).
2. **Label pipeline:** the replay reproduces the served kept set to within the edge unit on at least 90 % of a fresh
   sample. The labels are recomputed on rows the head has never seen.
3. **Linear-probe check:** a frozen-feature logistic probe on the request probe's pooled state separates
   tier ≤ 1 from tier ≥ 2 with AUROC ≥ 0.75, using cross-validation on the training pool and checked on dev. If
   the pooled state does not carry breadth, the plan stops here or moves to a different input.
4. **Held-out family:** the probe's AUROC on the held-out family is no more than 0.05 below in-distribution.
5. **Budget:** the cost estimate for one full run fits the budget, with no pilots.

## Evaluation

All evaluation uses one frozen plan with a recorded sha, run once on the check split. Arms:

1. Unified view, release curve (×1).
2. Unified view with the breadth head.
3. Uniform curve raise matched to arm 2's mean served tokens.
4. Full prefill.

Report:
- correct per arm;
- named flips against arm 1 and arm 3;
- evidence kept;
- mean and median TTFT (compress + prefill);
- served tokens per tier.

**Ship rule:** arm 2 beats arm 3 by more than +3 correct, with mean TTFT at most that of arm 3.

## Steps

1. Build the label pipeline from the replay, with an agreement check against served selections.
2. Generate and label the training pool, with the held-out family set aside.
3. Run the linear-probe check (gate 3) and the held-out family check (gate 4).
4. Write the gate file and get a go.
5. Run one full training of the breadth head, then calibrate the tiers on dev and freeze them.
6. Runtime: load the head with the request probe, add the tier to the region cap, and report the tier in the
   `unified` stats. Fail closed: a missing head serves the release curve and says so in `/props`.
7. Run the check-split confirm, then write the PR text with aggregates only.

## Open questions

- Should a "full" prediction serve the region raw, or keep paging at a very wide cap so later turns still reuse the
  view?
- Should the 0.5 tier ship? It saves TTFT but turns miscalibration into wrong answers on narrow questions.
- Breadth across turns: should the head also read the newest turn's content kind (log, code, table), or only the
  question?
