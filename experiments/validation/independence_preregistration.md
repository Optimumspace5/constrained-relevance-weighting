# Independence check — pre-registration

**Committed BEFORE the run.** The git timestamp on this file precedes the commit that
adds any result, so the interpretation rule below was fixed in advance and not chosen
to flatter whatever number came out.

## What is measured

Per-claim agreement between two faithfulness verifiers on the **same 40 hand-labeled
validation claims**, holding the `(premise, claim)` pair **fixed** (the exact
`linked_text` premise the NLI judge saw in `scored.csv`) and varying **only the
verifier type**:

- **NLI judge** — committed `judge_label` in `scored.csv` (DeBERTa-v3 FEVER/ANLI,
  concatenated premise, entailment threshold 0.60).
- **LLM verifier** — `GENERATION_MODEL` (claude-sonnet-4-6), temperature 0, asked
  supported/unsupported on the identical premise + claim.

The verifier *type* is the only variable, so any agreement or divergence is
attributable to the method — not to different claims, premises, or evidence.

Also reported: NLI-vs-human accuracy and LLM-vs-human accuracy on the same 40 claims
(both scored against `my_label`), so "which verifier was right" is grounded in ground
truth, not in one verifier being treated as the reference.

## Interpretation rule (fixed in advance)

The agreement % is **not** the deliverable. The deliverable is the divergence table
read against the human labels. Committed reading:

- **High agreement (≥ ~85%): convergent validation.** Two independent methods — a
  discriminative entailment model and a generative LLM — reach the same verdicts, so
  the headline NLI faithfulness number is not an artifact of one model family or of
  self-consistency bias.
- **Low agreement (< ~85%): genuine independence.** The two verifiers catch different
  things. This is *not* a failure — it is the interesting case, and the divergences
  are the payload: every `NLI ≠ LLM` case is inspected against the human label to say
  **which verifier was right on that claim**, and why.

Either way, the conclusion comes from inspecting the disagreements against ground
truth. A high number alone does not "prove the judge good"; a low number alone does
not "prove it independent." The divergence analysis is the result.
