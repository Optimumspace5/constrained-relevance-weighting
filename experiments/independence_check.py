"""
independence_check.py — NLI judge vs LLM verifier, head-to-head on identical claims.

Pre-registered interpretation: experiments/validation/independence_preregistration.md
(committed BEFORE this was run).

Design: hold the (premise, claim) pair FIXED from the committed scored.csv — the same
40 hand-labeled validation claims, and the exact premise text the NLI judge saw (the
`linked_text` column) — and vary ONLY the verifier: NLI entailment model -> generative
LLM (GENERATION_MODEL, temperature=0). The verifier *type* is the sole variable, so
agreement/divergence is attributable to the method, not to different claims or evidence.

Outputs:
  experiments/validation/scored_llm.csv   scored.csv + an `llm_label` column
and prints:
  - NLI vs LLM agreement %
  - NLI vs human accuracy and LLM vs human accuracy (same 40 claims)
  - a divergence table (every NLI != LLM row) with the human label, so you can read
    WHICH verifier was right on each disagreement.

Usage:
  python -m experiments.independence_check
  python -m experiments.independence_check --in experiments/validation/scored.csv
"""
import argparse
import csv
import sys
import textwrap

# Importing the evaluator loads .env and builds the shared Anthropic client, and
# pins the LLM verifier to the same GENERATION_MODEL used elsewhere.
from src.evaluator import client, GENERATION_MODEL


def _norm_label(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("supported", "support", "s", "yes", "y", "1", "true"):
        return "supported"
    if v in ("unsupported", "unsupport", "u", "no", "n", "0", "false"):
        return "unsupported"
    return ""


def verify_claim_llm(premise: str, claim: str) -> str:
    """LLM verifier on a single (premise, claim). Same task as the NLI judge:
    is the claim entailed by ONLY this premise? Returns 'supported'/'unsupported'."""
    prompt = (
        "Determine whether the CLAIM is SUPPORTED or UNSUPPORTED based ONLY on the "
        "SOURCE text provided. A claim is SUPPORTED if the source contains evidence "
        "that directly confirms it. It is UNSUPPORTED if the source does not contain "
        "enough evidence to confirm it, even if you personally know it to be true "
        "from other knowledge. Answer with ONLY one word: supported or unsupported.\n\n"
        f"CLAIM: {claim}\n\n"
        f"SOURCE:\n{premise}"
    )
    resp = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=8,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    out = resp.content[0].text.strip().lower()
    if "unsupported" in out:
        return "unsupported"
    if "supported" in out or out.startswith("support"):
        return "supported"
    return "unsupported"   # conservative default if the model is off-format


def run(input_csv: str, output_csv: str) -> None:
    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows in {input_csv}.")
        return

    print(f"Scoring {len(rows)} claims with the LLM verifier "
          f"({GENERATION_MODEL}, temperature=0)...\n")
    for i, r in enumerate(rows, 1):
        r["llm_label"] = verify_claim_llm(r["linked_text"], r["claim"])
        print(f"  [{i:>2}/{len(rows)}] NLI={r['judge_label']:<11} "
              f"LLM={r['llm_label']:<11} human={r['my_label']}")

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {output_csv}")

    _report(rows)


def _report(rows: list) -> None:
    n = len(rows)
    nli_vs_llm = sum(1 for r in rows
                     if _norm_label(r["judge_label"]) == _norm_label(r["llm_label"]))

    # Accuracy vs human, over rows that carry a human label.
    labeled = [r for r in rows if _norm_label(r["my_label"]) in ("supported", "unsupported")]
    nli_acc = sum(1 for r in labeled
                  if _norm_label(r["judge_label"]) == _norm_label(r["my_label"]))
    llm_acc = sum(1 for r in labeled
                  if _norm_label(r["llm_label"]) == _norm_label(r["my_label"]))

    print("\n" + "=" * 70)
    print("INDEPENDENCE CHECK — NLI judge vs LLM verifier (identical claims+premises)")
    print("=" * 70)
    print(f"  NLI vs LLM agreement : {nli_vs_llm}/{n} = {nli_vs_llm / n:.1%}")
    print(f"  NLI vs human accuracy: {nli_acc}/{len(labeled)} = {nli_acc / len(labeled):.1%}")
    print(f"  LLM vs human accuracy: {llm_acc}/{len(labeled)} = {llm_acc / len(labeled):.1%}")

    # The payload: every disagreement, read against the human label.
    diverged = [r for r in rows
                if _norm_label(r["judge_label"]) != _norm_label(r["llm_label"])]
    print(f"\n  DIVERGENCES ({len(diverged)} of {n}) — who was right vs human:")
    if not diverged:
        print("    (none — the two verifiers agreed on every claim)")
    nli_right = llm_right = neither_known = 0
    for r in diverged:
        human = _norm_label(r["my_label"])
        nli = _norm_label(r["judge_label"])
        llm = _norm_label(r["llm_label"])
        if human == "":
            verdict = "(no human label)"
            neither_known += 1
        elif nli == human:
            verdict = "NLI right"
            nli_right += 1
        elif llm == human:
            verdict = "LLM right"
            llm_right += 1
        else:
            verdict = "both wrong"
        print("\n    " + "-" * 60)
        print(f"    {verdict}  |  human={r['my_label']}  NLI={r['judge_label']} "
              f"(p={r['judge_entailment_score']})  LLM={r['llm_label']}")
        print(f"    [{r['episode']}/{r['summary_type']}] {r['topic']}")
        print(textwrap.fill(f"    claim: {r['claim']}", width=78,
                            subsequent_indent="           "))
    print("\n    " + "-" * 60)
    print(f"    on divergences: NLI right {nli_right}, LLM right {llm_right}, "
          f"both wrong {len(diverged) - nli_right - llm_right - neither_known}, "
          f"unlabeled {neither_known}")
    print("=" * 70)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input_csv",
                    default="experiments/validation/scored.csv")
    ap.add_argument("--out", dest="output_csv",
                    default="experiments/validation/scored_llm.csv")
    args = ap.parse_args(argv)
    run(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
