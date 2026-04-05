"""
Delta sweep experiment — demonstrates CRW's core trade-off.

Generates constrained summaries at multiple delta values and compares:
- Proportion MAE: how closely the summary follows target proportions
- ROUGE scores: n-gram overlap with source (faithfulness proxy)
- Extractive overlap: fraction of summary words from the source
- Relevance score: preference-weighted coverage

Key insight this should show:
- Small delta → low proportion MAE (faithful to episode structure) but less personalization
- Large delta → high personalization but the summary may drift from the episode's balance

Usage:
    python -m experiments.delta_sweep
    python -m experiments.delta_sweep --transcript data/transcripts/episode1.txt
"""

import argparse
import json
import csv
import os
from datetime import datetime

from src.loader import load_transcript
from src.segmenter import segment_transcript
from src.preferences import create_preferences
from src.summarizers import (
    generate_constrained_summary,
    generate_baseline_summary,
    calculate_constrained_proportions,
)
from src.evaluator import (
    compute_rouge_scores,
    compute_extractive_overlap,
    evaluate_coverage,
    evaluate_relevance,
)
from src.config import CONSTRAINT_DELTAS
from src.profiles import PROFILES, PROFILE_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Delta values to sweep — includes the configured ones plus finer granularity.
SWEEP_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25]


def run_sweep(
    transcript_path: str,
    deltas: list[float] = SWEEP_DELTAS,
    profile_name: str = "skewed",
    output_dir: str = "experiments/results",
) -> list[dict]:
    """
    Run the delta sweep experiment.

    1. Load transcript and discover topics (one-time cost).
    2. Build preferences from the chosen profile.
    3. For each delta value, generate a constrained summary and evaluate it.
    4. Print a comparison table and save results to CSV.

    Returns the list of result dicts for programmatic use.
    """
    # --- Step 1: load and segment ---
    print(f"Loading transcript: {transcript_path}")
    segments = load_transcript(transcript_path)
    print(f"  {len(segments)} segments loaded.\n")

    print("Discovering topics...")
    topics = segment_transcript(segments)
    print()

    # --- Step 2: build preferences ---
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Choose from: {list(PROFILES.keys())}")

    ratings = PROFILES[profile_name](topics)
    preferences = create_preferences(topics, ratings)

    pref_lookup = {p.topic_name: p.weight for p in preferences}
    weight_to_label = {1.5: "high", 1.0: "medium", 0.5: "low"}

    print(f"Preference profile: {profile_name}")
    for topic in topics:
        w = pref_lookup.get(topic.name, 1.0)
        print(f"  {topic.name}: {weight_to_label.get(w, 'medium')} (base={topic.proportion:.1%})")
    print()

    # --- Step 3: generate baseline (once) ---
    print("Generating baseline summary...")
    baseline = generate_baseline_summary(segments, topics)
    baseline_rouge = compute_rouge_scores(baseline, segments)
    baseline_ext = compute_extractive_overlap(baseline, segments)
    baseline_coverage = evaluate_coverage(baseline, topics)
    baseline_relevance = evaluate_relevance(baseline, preferences, topics)
    print(f"  Baseline: ROUGE-1={baseline_rouge['rouge1']:.4f}, "
          f"Ext.Overlap={baseline_ext['unigram_overlap']:.4f}, "
          f"MAE={baseline_relevance['proportion_mae']:.4f}\n")

    # --- Step 4: sweep across deltas ---
    results = []

    # Add baseline as the first row.
    results.append({
        "delta": "baseline",
        "rouge1": baseline_rouge["rouge1"],
        "rouge2": baseline_rouge["rouge2"],
        "rougeL": baseline_rouge["rougeL"],
        "extractive_overlap": baseline_ext["unigram_overlap"],
        "bigram_overlap": baseline_ext["bigram_overlap"],
        "coverage_ratio": baseline_coverage["coverage_ratio"],
        "relevance_score": baseline_relevance["relevance_score"],
        "proportion_mae": baseline_relevance["proportion_mae"],
        "word_count": baseline.metadata.get("word_count", 0),
    })

    for delta in deltas:
        print(f"Generating constrained summary at delta={delta:.2f}...")
        summary = generate_constrained_summary(segments, topics, preferences, delta=delta)

        rouge = compute_rouge_scores(summary, segments)
        ext = compute_extractive_overlap(summary, segments)
        coverage = evaluate_coverage(summary, topics)
        relevance = evaluate_relevance(summary, preferences, topics)

        row = {
            "delta": delta,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "extractive_overlap": ext["unigram_overlap"],
            "bigram_overlap": ext["bigram_overlap"],
            "coverage_ratio": coverage["coverage_ratio"],
            "relevance_score": relevance["relevance_score"],
            "proportion_mae": relevance["proportion_mae"],
            "word_count": summary.metadata.get("word_count", 0),
        }
        results.append(row)

        print(f"  ROUGE-1={rouge['rouge1']:.4f}  "
              f"Ext.Overlap={ext['unigram_overlap']:.4f}  "
              f"MAE={relevance['proportion_mae']:.4f}  "
              f"Relevance={relevance['relevance_score']:.3f}  "
              f"Words={row['word_count']}")

    # --- Step 5: print comparison table ---
    print(f"\n{'='*90}")
    print(f"DELTA SWEEP RESULTS — Profile: {profile_name}")
    print(f"{'='*90}")

    col_w = 12
    headers = ["Delta", "ROUGE-1", "ROUGE-L", "Ext.Overlap", "Relevance", "Prop.MAE", "Words"]
    print("  ".join(f"{h:>{col_w}}" for h in headers))
    print("-" * (col_w * len(headers) + 2 * (len(headers) - 1)))

    for row in results:
        d = str(row["delta"]) if isinstance(row["delta"], str) else f"{row['delta']:.2f}"
        vals = [
            d,
            f"{row['rouge1']:.4f}",
            f"{row['rougeL']:.4f}",
            f"{row['extractive_overlap']:.4f}",
            f"{row['relevance_score']:.3f}",
            f"{row['proportion_mae']:.4f}",
            str(row["word_count"]),
        ]
        print("  ".join(f"{v:>{col_w}}" for v in vals))

    # --- Step 6: save to CSV ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"delta_sweep_{profile_name}_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta sweep experiment for CRW")
    parser.add_argument(
        "--transcript",
        default="data/transcripts/episode1.txt",
        help="Path to transcript file",
    )
    parser.add_argument(
        "--profile",
        default="skewed_high",
        choices=list(PROFILES.keys()),
        help="Preference profile to use (default: skewed_high)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory to save CSV results (default: experiments/results)",
    )
    args = parser.parse_args()

    run_sweep(args.transcript, profile_name=args.profile, output_dir=args.output_dir)
