# Independence check — results

Run of `experiments/independence_check.py` against the interpretation fixed in
`independence_preregistration.md` (committed first). Per-claim output:
`scored_llm.csv`. Small sample: **n = 40**, so all rates carry wide confidence
intervals — read directionally.

## Headline

| Metric | Result |
|---|---|
| NLI vs LLM agreement | **30/40 = 75%** |
| NLI vs human accuracy | 38/40 = 95% |
| LLM vs human accuracy | 32/40 = 80% |

Pre-registered reading: 75% < ~85% → **genuine independence**. Given identical
claims and identical evidence, a generative LLM verifier (claude-sonnet-4-6,
temperature 0) reaches a *different* verdict on 1 claim in 4. So the NLI judge is
not an LLM agreeing with itself — the headline faithfulness number is not an
artifact of self-consistency bias. That was the point of the check.

## The divergences are the result (10 of 40)

On the 10 claims where the two disagree, against the human labels:
**NLI right 8, LLM right 2, both wrong 0.**

**8/8 NLI wins are the same failure on the LLM's side: over-rejection of paraphrase
and abstraction.** All eight are `human=supported, NLI=supported, LLM=unsupported`.
The LLM, told a claim is supported only if the source "directly confirms" it,
rejects faithful *higher-level restatements* — "The speaker has a structured
creative process", "Systemic disadvantages are real", "treats landmark moments as
gratitude milestones". The NLI model, trained on entailment, correctly recognizes
these as entailed. This is systematic (one-directional), not noise.

**The 2 LLM wins are exactly the NLI judge's two already-characterized error cases** —
the independence check rediscovered both from the other direction:
- **Known FP:** `"...uses the phrase 'the muscle of examining hesitation honestly'"`
  — NLI supported (p=0.95), human+LLM unsupported. The verbatim-quotation meta-claim:
  NLI checks *meaning*, not exact quotes, so it certifies a paraphrase as the quote.
- **Known FN:** `"Tony Robbins describes death as a useful counselor that gave him
  drive."` — NLI unsupported (p=0.12), human+LLM supported. The one over-rejection.

So the two verifiers' disagreements localize precisely onto the two error cases
already documented from the human validation — two independent methods triangulating
the same boundary.

## Honest caveats

- **n = 40; the divergence tally is 10 cases.** "NLI wins 8:2" is a small sample —
  directional evidence, not a proven ratio.
- **75% is prompt-dependent.** The LLM verifier used a strict "directly confirms"
  instruction; a looser prompt would raise agreement and lower the LLM's
  over-rejection rate. The number characterizes *these two operationalizations*, not
  a universal independence constant. What is robust is the *direction*: the LLM is
  the more conservative verifier here, and its errors are over-rejections of
  faithful abstraction.
