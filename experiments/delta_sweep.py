"""
Delta sweep experiment — demonstrates CRW's core trade-off.

Generates constrained summaries at multiple delta values and compares:
- Proportion MAE: how closely the summary follows target proportions
- Faithfulness: LLM-as-judge score (1-5) per paragraph
- ROUGE scores: n-gram overlap with source
- Extractive overlap: fraction of summary words from the source
- Coverage and relevance metrics

Key insight this should show:
- Small delta → low proportion MAE (faithful to episode structure) but less personalization
- Large delta → high personalization but the summary may drift from the episode's balance

Usage:
    python -m experiments.delta_sweep
    python -m experiments.delta_sweep --all-episodes
    python -m experiments.delta_sweep --transcript data/transcripts/episode1.txt
"""

import argparse
import csv
import os
from datetime import datetime

from src.loader import load_transcript
from src.segmenter import segment_transcript
from src.preferences import create_preferences
from src.summarizers import (
    generate_constrained_summary,
    generate_unconstrained_summary,
    generate_generic_summary,
    generate_baseline_summary,
    calculate_constrained_proportions,
)
from src.evaluator import (
    evaluate_faithfulness,
    evaluate_faithfulness_qa,
    compute_rouge_scores,
    compute_extractive_overlap,
    evaluate_coverage,
    evaluate_relevance,
)
from src.evidence import link_evidence
from src.profiles import PROFILES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SWEEP_DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25]

ALL_EPISODES = [
    "data/transcripts/episode1.txt",
    "data/transcripts/episode2.txt",
    "data/transcripts/episode3.txt",
]


def evaluate_summary(summary, segments, topics, preferences, include_qags=False):
    """Run evaluations on a summary. Evidence-links first for accurate coverage."""
    # Evidence-link to get per-paragraph segments for accurate coverage measurement.
    linked = link_evidence(summary, segments, topics)
    rouge = compute_rouge_scores(linked, segments)
    ext = compute_extractive_overlap(linked, segments)
    coverage = evaluate_coverage(linked, topics)
    relevance = evaluate_relevance(linked, preferences, topics)
    result = {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "extractive_overlap": ext["unigram_overlap"],
        "bigram_overlap": ext["bigram_overlap"],
        "coverage_ratio": coverage["coverage_ratio"],
        "topics_covered": coverage["topics_covered"],
        "total_topics": coverage["total_topics"],
        "relevance_score": relevance["relevance_score"],
        "proportion_mae": relevance["proportion_mae"],
        "word_count": summary.metadata.get("word_count", 0),
    }
    if include_qags:
        qags = evaluate_faithfulness_qa(linked, segments)
        result["qags_precision"] = qags["precision"]
        result["qags_supported"] = qags["supported_claims"]
        result["qags_total"] = qags["total_claims"]
        result["qags_unsupported_count"] = len(qags["unsupported"])
    return result


def run_sweep(
    transcript_path: str,
    deltas: list[float] = SWEEP_DELTAS,
    profile_name: str = "skewed_high",
    include_faithfulness: bool = False,
    include_qags: bool = False,
    output_dir: str = "experiments/results",
) -> list[dict]:
    """
    Run the delta sweep experiment on a single transcript.

    1. Load transcript and discover topics.
    2. Build preferences from the chosen profile.
    3. Generate baseline, generic, unconstrained summaries.
    4. For each delta, generate a constrained summary and evaluate.
    5. Print comparison table and save to CSV.
    """
    episode_name = os.path.splitext(os.path.basename(transcript_path))[0]

    print(f"\n{'='*70}")
    print(f"EPISODE: {episode_name} | Profile: {profile_name}")
    print(f"{'='*70}")

    # --- Step 1: load and segment ---
    print(f"Loading transcript: {transcript_path}")
    segments = load_transcript(transcript_path)

    print("Discovering topics...")
    topics = segment_transcript(segments)

    # --- Step 2: build preferences ---
    ratings = PROFILES[profile_name](topics)
    preferences = create_preferences(topics, ratings)

    pref_lookup = {p.topic_name: p.weight for p in preferences}
    weight_to_label = {1.5: "high", 1.0: "medium", 0.5: "low"}
    for topic in topics:
        w = pref_lookup.get(topic.name, 1.0)
        print(f"  {topic.name}: {weight_to_label.get(w, 'medium')} (base={topic.proportion:.1%})")

    # --- Step 3: generate reference summaries ---
    results = []

    # Baseline
    print("\nGenerating baseline...")
    baseline = generate_baseline_summary(segments, topics)
    row = {"episode": episode_name, "summary_type": "baseline", "delta": "—"}
    row.update(evaluate_summary(baseline, segments, topics, preferences, include_qags=include_qags))
    if include_faithfulness:
        faith = evaluate_faithfulness(baseline, segments)
        row["faithfulness"] = faith["average_score"]
    results.append(row)

    # Generic
    print("Generating generic summary...")
    generic = generate_generic_summary(segments, topics)
    row = {"episode": episode_name, "summary_type": "generic", "delta": "—"}
    row.update(evaluate_summary(generic, segments, topics, preferences, include_qags=include_qags))
    if include_faithfulness:
        faith = evaluate_faithfulness(generic, segments)
        row["faithfulness"] = faith["average_score"]
    results.append(row)

    # Unconstrained
    print("Generating unconstrained summary...")
    unconstrained = generate_unconstrained_summary(segments, topics, preferences)
    row = {"episode": episode_name, "summary_type": "unconstrained", "delta": "—"}
    row.update(evaluate_summary(unconstrained, segments, topics, preferences, include_qags=include_qags))
    if include_faithfulness:
        faith = evaluate_faithfulness(unconstrained, segments)
        row["faithfulness"] = faith["average_score"]
    results.append(row)

    # --- Step 4: constrained at each delta ---
    for delta in deltas:
        print(f"Generating constrained at delta={delta:.2f}...")
        summary = generate_constrained_summary(segments, topics, preferences, delta=delta)
        row = {"episode": episode_name, "summary_type": "constrained", "delta": delta}
        row.update(evaluate_summary(summary, segments, topics, preferences, include_qags=include_qags))
        if include_faithfulness:
            faith = evaluate_faithfulness(summary, segments)
            row["faithfulness"] = faith["average_score"]
        results.append(row)

    # --- Step 5: print table ---
    header = f"{'Summary':<20} {'Delta':>6} {'ROUGE-1':>8} {'ROUGE-L':>8} {'Ext.Ovlp':>9} {'Coverage':>9} {'Rel.':>6} {'MAE':>7} {'Words':>6}"
    if include_faithfulness:
        header += f" {'Faith':>6}"
    if include_qags:
        header += f" {'QAGS':>10} {'Unsup.':>7}"
    print(f"\n{header}")
    sep_len = 85 + (7 if include_faithfulness else 0) + (18 if include_qags else 0)
    print("-" * sep_len)

    for r in results:
        d = str(r["delta"])
        cov = f"{r['topics_covered']}/{r['total_topics']}"
        line = (f"{r['summary_type']:<20} {d:>6} {r['rouge1']:>8.4f} {r['rougeL']:>8.4f} "
                f"{r['extractive_overlap']:>9.4f} {cov:>9} {r['relevance_score']:>6.3f} "
                f"{r['proportion_mae']:>7.4f} {r['word_count']:>6}")
        if include_faithfulness:
            line += f" {r.get('faithfulness', 0):>6.2f}"
        if include_qags:
            qp = f"{r.get('qags_supported', 0)}/{r.get('qags_total', 0)}"
            line += f" {qp:>10} {r.get('qags_unsupported_count', 0):>7}"
        print(line)

    # --- Step 6: save CSV ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"sweep_{episode_name}_{profile_name}_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {csv_path}")
    return results


def run_all_episodes(
    deltas: list[float] = SWEEP_DELTAS,
    profile_name: str = "skewed_high",
    include_faithfulness: bool = False,
    include_qags: bool = False,
    output_dir: str = "experiments/results",
) -> list[dict]:
    """Run the sweep across all episodes and save a combined CSV."""
    all_results = []
    for path in ALL_EPISODES:
        if not os.path.exists(path):
            print(f"Skipping {path} — file not found")
            continue
        results = run_sweep(
            path, deltas=deltas, profile_name=profile_name,
            include_faithfulness=include_faithfulness,
            include_qags=include_qags, output_dir=output_dir,
        )
        all_results.extend(results)

    # Save combined CSV.
    if all_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"sweep_all_episodes_{profile_name}_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCombined results saved: {csv_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta sweep experiment for CRW")
    parser.add_argument(
        "--transcript",
        default=None,
        help="Path to a single transcript file",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Run sweep across all episodes (episode1, episode2, episode3)",
    )
    parser.add_argument(
        "--profile",
        default="skewed_high",
        choices=list(PROFILES.keys()),
        help="Preference profile to use (default: skewed_high)",
    )
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="Include faithfulness evaluation (requires API calls per paragraph)",
    )
    parser.add_argument(
        "--qags",
        action="store_true",
        help="Include QAGS claim-level faithfulness evaluation (tracks unsupported claims)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory to save CSV results",
    )
    args = parser.parse_args()

    if args.all_episodes:
        run_all_episodes(
            profile_name=args.profile,
            include_faithfulness=args.faithfulness,
            include_qags=args.qags,
            output_dir=args.output_dir,
        )
    else:
        transcript = args.transcript or "data/transcripts/episode1.txt"
        run_sweep(
            transcript,
            profile_name=args.profile,
            include_faithfulness=args.faithfulness,
            include_qags=args.qags,
            output_dir=args.output_dir,
        )
