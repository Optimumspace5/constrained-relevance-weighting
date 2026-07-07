"""
Validation harness for the NLI faithfulness judge — SCAFFOLD ONLY.

The whole point of this file is to let a human validate the judge against their
OWN labels. It is deliberately split into two independent steps so the human
stays in the loop at the two places that matter:

  1. build_validation_set(...)  — samples ~N claims across all episodes and all
     summary variants and extracts each claim together with the EXACT source
     segments the judge will use as its premise. It writes a CSV whose
     `my_label`, `judge_label`, and `judge_entailment_score` columns are BLANK.

        >>> This step NEVER assigns my_label. You label every row by hand. <<<

  2. score_validation_set(...) — AFTER you fill `my_label`, runs the judge AS
     WIRED (same extraction path, same narrow premise = seg.source_segment_indices)
     on the same claims, fills `judge_label` + `judge_entailment_score`, and
     reports agreement %, false-positive rate, false-negative rate, and a
     THRESHOLD-SWEEP table (agreement/FP/FN at each candidate threshold).

        >>> This step NEVER picks a threshold for you. It reports the sweep; the
            choice of operating threshold is yours to make and defend. <<<

Why this validates the *pipeline* and not the judge in isolation: it reuses the
evaluator's own _extract_claims (BULK_MODEL, temperature=0) and the narrow linked
premise, so the agreement number reflects the judge exactly as run_full_evaluation
runs it. Segmentation is deterministic (cached, temperature=0), so the segment
indices stored at build time are still valid at score time.

Usage:
    # Step 1 — build the sheet (generates summaries + extracts claims; costs API):
    python -m experiments.validate_judge build --n 40 \
        --out experiments/validation/claims.csv

    # ... open claims.csv and fill the `my_label` column with
    #     'supported' / 'unsupported' for each row (leave blank to skip a row) ...

    # Step 2 — score your labels (runs the local NLI judge; no generation):
    python -m experiments.validate_judge score \
        --in experiments/validation/claims.csv \
        --out experiments/validation/scored.csv
"""

import argparse
import csv
import os
import random

from src.loader import load_transcript
from src.segmenter import segment_transcript
from src.preferences import create_preferences
from src.summarizers import generate_all_summaries
from src.evidence import link_evidence, link_evidence_tfidf
from src.profiles import PROFILES
from src.nli_judge import load_nli_model, verify_claim
# Reuse the evaluator's shared extraction helper on purpose: the validation set
# must be built from the SAME claim extraction the evaluator uses, or the
# agreement number would not describe the wired pipeline.
from src.evaluator import _extract_claims


ALL_EPISODES = [
    "data/transcripts/episode1.txt",
    "data/transcripts/episode2.txt",
    "data/transcripts/episode3.txt",
]

# Candidate operating thresholds reported by the sweep. This harness only
# REPORTS agreement at each; it does not choose one.
SWEEP_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

CSV_COLUMNS = [
    "episode",
    "transcript_path",
    "summary_type",
    "topic",
    "claim",
    "linked_segments",           # ';'-joined source segment indices (the judge's premise)
    "my_label",                  # YOU fill this: 'supported' / 'unsupported'
    "judge_label",               # filled by score step
    "judge_entailment_score",    # filled by score step
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_label(value: str) -> str:
    """Normalize a human/judge label to 'supported' / 'unsupported' / '' (unknown)."""
    v = (value or "").strip().lower()
    if v in ("supported", "support", "s", "yes", "y", "1", "true"):
        return "supported"
    if v in ("unsupported", "unsupport", "u", "no", "n", "0", "false"):
        return "unsupported"
    return ""


# ---------------------------------------------------------------------------
# Step 1: build the (unlabeled) validation set
# ---------------------------------------------------------------------------

def _collect_claim_records(
    episodes: list[str],
    profile_name: str,
    link_method: str = "tfidf",
) -> list[dict]:
    """Generate summaries for each episode, evidence-link them, and extract every
    claim with its linked premise segments.

    link_method:
      - "tfidf": deterministic linking (reproducible sheets; free).
      - "llm":   matches the production experiment path (app.py / delta_sweep use
                 link_evidence). Non-deterministic at build time, but the chosen
                 indices are frozen into the CSV, so scoring stays reproducible.
    """
    records = []
    for path in episodes:
        if not os.path.exists(path):
            print(f"Skipping {path} — file not found")
            continue

        episode = os.path.splitext(os.path.basename(path))[0]
        segments = load_transcript(path)
        topics = segment_transcript(segments)   # cached + deterministic
        preferences = create_preferences(topics, PROFILES[profile_name](topics))

        summaries = generate_all_summaries(segments, topics, preferences)
        for summary_type, summary in summaries.items():
            # The judge premise is these linked segments (the narrow, wired path).
            if link_method == "llm":
                linked = link_evidence(summary, segments, topics)
            else:
                linked = link_evidence_tfidf(summary, segments, topics)
            for seg, claims in _extract_claims(linked):
                for claim in claims:
                    records.append({
                        "episode": episode,
                        "transcript_path": path,
                        "summary_type": summary_type,
                        "topic": seg.topic_name,
                        "claim": claim,
                        "linked_segments": ";".join(str(i) for i in seg.source_segment_indices),
                    })
    return records


def _stratified_sample(
    records: list[dict],
    n_claims: int,
    min_multi_segment: int,
    seed: int,
) -> list[dict]:
    """
    Sample ~n_claims spread evenly across (episode, summary_type) groups, and
    guarantee at least `min_multi_segment` claims whose support spans >= 2 linked
    segments — those are the cases where the narrow premise is most likely to
    over-reject, so they must be represented (per the reviewer's note). This
    biases *coverage*, never the label.
    """
    rng = random.Random(seed)

    # Ensure some multi-segment claims are present.
    multi = [r for r in records if len(r["linked_segments"].split(";")) >= 2 and r["linked_segments"]]
    rng.shuffle(multi)
    chosen = multi[:min_multi_segment]
    chosen_ids = {id(r) for r in chosen}

    # Round-robin across (episode, summary_type) buckets for the remainder.
    buckets: dict = {}
    for r in records:
        if id(r) in chosen_ids:
            continue
        buckets.setdefault((r["episode"], r["summary_type"]), []).append(r)
    for b in buckets.values():
        rng.shuffle(b)

    keys = list(buckets.keys())
    rng.shuffle(keys)
    while len(chosen) < n_claims and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                chosen.append(buckets[k].pop())
                if len(chosen) >= n_claims:
                    break

    rng.shuffle(chosen)
    return chosen[:n_claims]


def build_validation_set(
    episodes: list[str] = None,
    profile_name: str = "skewed_high",
    n_claims: int = 40,
    min_multi_segment: int = 4,
    seed: int = 0,
    link_method: str = "tfidf",
    output_csv: str = "experiments/validation/claims.csv",
) -> None:
    """Build an UNLABELED validation sheet. Never assigns my_label."""
    episodes = episodes or ALL_EPISODES
    records = _collect_claim_records(episodes, profile_name, link_method=link_method)
    if not records:
        print("No claims collected — check that the transcript files exist.")
        return

    sampled = _stratified_sample(records, n_claims, min_multi_segment, seed)
    _write_csv(output_csv, sampled)

    n_multi = sum(1 for r in sampled if len(r["linked_segments"].split(";")) >= 2)
    print(f"Wrote {len(sampled)} unlabeled claims to {output_csv} "
          f"({n_multi} span >= 2 source segments).")
    print("NEXT: open the CSV and fill the 'my_label' column with "
          "'supported' or 'unsupported' for each row, then run the 'score' command.")
    print("This step did NOT assign any labels — that is your job.")


# ---------------------------------------------------------------------------
# Step 2: score the human-labeled set with the judge (as wired) + report
# ---------------------------------------------------------------------------

def _report_metrics(rows: list[dict], thresholds: list[float]) -> None:
    """Report agreement / FP / FN at the judge's default threshold, plus a
    threshold sweep. Reports only — it does not choose a threshold."""
    labeled = [r for r in rows if _norm_label(r["my_label"]) in ("supported", "unsupported")]
    if not labeled:
        print("\nNo 'my_label' values filled in — nothing to score. "
              "Fill the my_label column and re-run.")
        return

    truly_sup = [r for r in labeled if _norm_label(r["my_label"]) == "supported"]
    truly_unsup = [r for r in labeled if _norm_label(r["my_label"]) == "unsupported"]

    def _stats_at(label_fn):
        """label_fn(row) -> 'supported'/'unsupported'. Returns (agreement, fp_rate, fn_rate)."""
        agree = sum(1 for r in labeled if label_fn(r) == _norm_label(r["my_label"]))
        # False positive: judge says supported when the human said unsupported.
        fp = sum(1 for r in truly_unsup if label_fn(r) == "supported")
        # False negative: judge says unsupported when the human said supported.
        fn = sum(1 for r in truly_sup if label_fn(r) == "unsupported")
        fp_rate = fp / len(truly_unsup) if truly_unsup else 0.0
        fn_rate = fn / len(truly_sup) if truly_sup else 0.0
        return agree / len(labeled), fp_rate, fn_rate

    print(f"\nLabeled claims: {len(labeled)}  "
          f"(human-supported: {len(truly_sup)}, human-unsupported: {len(truly_unsup)})")

    # At the judge's own default threshold (the judge_label already written out).
    agr, fpr, fnr = _stats_at(lambda r: r["judge_label"])
    print(f"\nAt the judge's default threshold:")
    print(f"  agreement:          {agr:.1%}")
    print(f"  false-positive rate:{fpr:>7.1%}   (judge 'supported' but you said 'unsupported' — missed hallucination)")
    print(f"  false-negative rate:{fnr:>7.1%}   (judge 'unsupported' but you said 'supported' — over-rejection)")

    # Threshold sweep — recompute the judge label from the stored entailment score.
    print(f"\nThreshold sweep (recomputed from judge_entailment_score):")
    print(f"  {'thresh':>7} {'agree':>7} {'FP rate':>8} {'FN rate':>8}")
    for th in thresholds:
        def _lbl(r, th=th):
            try:
                score = float(r["judge_entailment_score"])
            except (TypeError, ValueError):
                return "unsupported"
            return "supported" if score >= th else "unsupported"
        agr, fpr, fnr = _stats_at(_lbl)
        print(f"  {th:>7.2f} {agr:>7.1%} {fpr:>8.1%} {fnr:>8.1%}")

    print("\nNOTE: this is a REPORT of the sweep only. Choosing the operating "
          "threshold is your decision — pick it from this table and justify it "
          "(remember: false positives are the dangerous error for faithfulness).")


def score_validation_set(
    input_csv: str,
    output_csv: str = "experiments/validation/scored.csv",
    thresholds: list[float] = SWEEP_THRESHOLDS,
) -> None:
    """Run the judge AS WIRED on a human-labeled sheet; fill judge columns; report."""
    rows = _read_csv(input_csv)
    if not rows:
        print(f"No rows in {input_csv}.")
        return

    model = load_nli_model()
    segments_cache: dict = {}   # transcript_path -> segments (loaded once per episode)

    for r in rows:
        path = r["transcript_path"]
        if path not in segments_cache:
            segments_cache[path] = load_transcript(path)
        segments = segments_cache[path]

        indices = [int(x) for x in r["linked_segments"].split(";") if x.strip() != ""]
        # verify_claim builds the premise from exactly these linked segments —
        # the same narrow, wired behavior as evaluate_faithfulness_qa.
        result = verify_claim(r["claim"], segments, indices, model)
        r["judge_entailment_score"] = round(result["entailment_prob"], 4)
        r["judge_label"] = "supported" if result["supported"] else "unsupported"

    _write_csv(output_csv, rows)
    print(f"Wrote judge-scored claims to {output_csv}")
    _report_metrics(rows, thresholds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI judge validation harness (scaffold)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build an UNLABELED validation CSV (you label it by hand)")
    p_build.add_argument("--n", type=int, default=40, help="Approximate number of claims to sample")
    p_build.add_argument("--profile", default="skewed_high", choices=list(PROFILES.keys()))
    p_build.add_argument("--min-multi-segment", type=int, default=4,
                         help="Minimum claims whose support spans >= 2 segments")
    p_build.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducible sheets)")
    p_build.add_argument("--link", default="tfidf", choices=["tfidf", "llm"],
                         help="Evidence linking for the premise: 'tfidf' (deterministic) "
                              "or 'llm' (matches the production experiment path)")
    p_build.add_argument("--out", default="experiments/validation/claims.csv")

    p_score = sub.add_parser("score", help="Score a human-labeled CSV with the judge and report")
    p_score.add_argument("--in", dest="input_csv", required=True)
    p_score.add_argument("--out", default="experiments/validation/scored.csv")

    args = parser.parse_args()

    if args.command == "build":
        build_validation_set(
            profile_name=args.profile,
            n_claims=args.n,
            min_multi_segment=args.min_multi_segment,
            seed=args.seed,
            link_method=args.link,
            output_csv=args.out,
        )
    elif args.command == "score":
        score_validation_set(input_csv=args.input_csv, output_csv=args.out)
