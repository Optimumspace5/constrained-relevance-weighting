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
import statistics
from collections import defaultdict
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


def _save_csv(path: str, rows: list[dict]) -> None:
    """Write rows to CSV using the union of all keys as the header."""
    if not rows:
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(rows: list[dict]) -> list[dict]:
    """
    Group rows by config (episode, summary_type, delta) and compute mean + std
    for every numeric metric across runs. Non-numeric fields and run_id are
    dropped. Returns one aggregated row per config with `<metric>_mean` /
    `<metric>_std` columns and an `n_runs` count.
    """
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["episode"], r["summary_type"], r["delta"])].append(r)

    aggregated = []
    for (episode, summary_type, delta), grp in groups.items():
        out = {
            "episode": episode,
            "summary_type": summary_type,
            "delta": delta,
            "n_runs": len(grp),
        }
        metric_keys = [
            k for k in grp[0]
            if k not in ("episode", "summary_type", "delta", "run_id")
        ]
        for k in metric_keys:
            vals = [
                r[k] for r in grp
                if isinstance(r[k], (int, float)) and not isinstance(r[k], bool)
            ]
            if len(vals) == len(grp) and vals:
                out[f"{k}_mean"] = round(statistics.mean(vals), 4)
                out[f"{k}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        aggregated.append(out)
    return aggregated


def _print_sweep_table(results, include_faithfulness, include_qags):
    """Print the detailed per-row table (one line per variant per run)."""
    header = (f"{'Summary':<20} {'Run':>4} {'Delta':>6} {'ROUGE-1':>8} {'ROUGE-L':>8} "
              f"{'Ext.Ovlp':>9} {'Coverage':>9} {'Rel.':>6} {'MAE':>7} {'Words':>6}")
    if include_faithfulness:
        header += f" {'Faith':>6}"
    if include_qags:
        header += f" {'QAGS':>10} {'Unsup.':>7}"
    print(f"\n{header}")
    sep_len = 90 + (7 if include_faithfulness else 0) + (18 if include_qags else 0)
    print("-" * sep_len)

    for r in results:
        d = str(r["delta"])
        cov = f"{r['topics_covered']}/{r['total_topics']}"
        line = (f"{r['summary_type']:<20} {r.get('run_id', 0):>4} {d:>6} {r['rouge1']:>8.4f} "
                f"{r['rougeL']:>8.4f} {r['extractive_overlap']:>9.4f} {cov:>9} "
                f"{r['relevance_score']:>6.3f} {r['proportion_mae']:>7.4f} {r['word_count']:>6}")
        if include_faithfulness:
            line += f" {r.get('faithfulness', 0):>6.2f}"
        if include_qags:
            qp = f"{r.get('qags_supported', 0)}/{r.get('qags_total', 0)}"
            line += f" {qp:>10} {r.get('qags_unsupported_count', 0):>7}"
        print(line)


def _evaluate_all_variants(
    episode_name: str,
    segments,
    topics,
    preferences,
    deltas: list[float],
    include_faithfulness: bool,
    include_qags: bool,
    run_id: int,
) -> list[dict]:
    """Generate and evaluate every summary variant once; tag each row with run_id."""
    rows = []

    def _row(summary_type, delta_label, summary):
        row = {
            "episode": episode_name,
            "summary_type": summary_type,
            "delta": delta_label,
            "run_id": run_id,
        }
        row.update(evaluate_summary(summary, segments, topics, preferences, include_qags=include_qags))
        if include_faithfulness:
            row["faithfulness"] = evaluate_faithfulness(summary, segments)["average_score"]
        return row

    print(f"\n[run {run_id}] Generating baseline...")
    rows.append(_row("baseline", "—", generate_baseline_summary(segments, topics)))

    print(f"[run {run_id}] Generating generic summary...")
    rows.append(_row("generic", "—", generate_generic_summary(segments, topics)))

    print(f"[run {run_id}] Generating unconstrained summary...")
    rows.append(_row("unconstrained", "—", generate_unconstrained_summary(segments, topics, preferences)))

    for delta in deltas:
        print(f"[run {run_id}] Generating constrained at delta={delta:.2f}...")
        summary = generate_constrained_summary(segments, topics, preferences, delta=delta)
        rows.append(_row("constrained", delta, summary))

    return rows


def run_sweep(
    transcript_path: str,
    deltas: list[float] = SWEEP_DELTAS,
    profile_name: str = "skewed_high",
    include_faithfulness: bool = False,
    include_qags: bool = False,
    runs: int = 1,
    output_dir: str = "experiments/results",
) -> list[dict]:
    """
    Run the delta sweep experiment on a single transcript.

    Load + segment once (segmentation is cached and deterministic), then generate
    and evaluate every variant `runs` times. Each row is tagged with run_id. When
    runs > 1, per-config mean and std are also written to a `_agg.csv`, capturing
    the variance introduced by non-deterministic summary generation.
    """
    episode_name = os.path.splitext(os.path.basename(transcript_path))[0]

    print(f"\n{'='*70}")
    print(f"EPISODE: {episode_name} | Profile: {profile_name} | Runs: {runs}")
    print(f"{'='*70}")

    print(f"Loading transcript: {transcript_path}")
    segments = load_transcript(transcript_path)

    print("Discovering topics...")
    topics = segment_transcript(segments)

    ratings = PROFILES[profile_name](topics)
    preferences = create_preferences(topics, ratings)

    pref_lookup = {p.topic_name: p.weight for p in preferences}
    weight_to_label = {1.5: "high", 1.0: "medium", 0.5: "low"}
    for topic in topics:
        w = pref_lookup.get(topic.name, 1.0)
        print(f"  {topic.name}: {weight_to_label.get(w, 'medium')} (base={topic.proportion:.1%})")

    # Repeat the full variant set `runs` times.
    results = []
    for run_id in range(runs):
        results.extend(_evaluate_all_variants(
            episode_name, segments, topics, preferences, deltas,
            include_faithfulness, include_qags, run_id,
        ))

    _print_sweep_table(results, include_faithfulness, include_qags)

    # Save raw per-run CSV.
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"sweep_{episode_name}_{profile_name}_{timestamp}.csv")
    _save_csv(csv_path, results)
    print(f"\nSaved raw runs: {csv_path}")

    # Save mean/std aggregates when there is more than one run.
    if runs > 1:
        agg = aggregate_runs(results)
        agg_path = os.path.join(output_dir, f"sweep_{episode_name}_{profile_name}_{timestamp}_agg.csv")
        _save_csv(agg_path, agg)
        print(f"Saved mean/std aggregates: {agg_path}")

    return results


def run_all_episodes(
    deltas: list[float] = SWEEP_DELTAS,
    profile_name: str = "skewed_high",
    include_faithfulness: bool = False,
    include_qags: bool = False,
    runs: int = 1,
    output_dir: str = "experiments/results",
) -> list[dict]:
    """Run the sweep across all episodes and save combined raw + aggregated CSVs."""
    all_results = []
    for path in ALL_EPISODES:
        if not os.path.exists(path):
            print(f"Skipping {path} — file not found")
            continue
        results = run_sweep(
            path, deltas=deltas, profile_name=profile_name,
            include_faithfulness=include_faithfulness,
            include_qags=include_qags, runs=runs, output_dir=output_dir,
        )
        all_results.extend(results)

    if all_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"sweep_all_episodes_{profile_name}_{timestamp}.csv")
        _save_csv(csv_path, all_results)
        print(f"\nCombined results saved: {csv_path}")

        if runs > 1:
            agg = aggregate_runs(all_results)
            agg_path = os.path.join(output_dir, f"sweep_all_episodes_{profile_name}_{timestamp}_agg.csv")
            _save_csv(agg_path, agg)
            print(f"Combined mean/std aggregates: {agg_path}")

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
        "--runs",
        type=int,
        default=1,
        help="Repeat each config N times and report per-metric mean + std (default: 1)",
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
            runs=args.runs,
            output_dir=args.output_dir,
        )
    else:
        transcript = args.transcript or "data/transcripts/episode1.txt"
        run_sweep(
            transcript,
            profile_name=args.profile,
            include_faithfulness=args.faithfulness,
            include_qags=args.qags,
            runs=args.runs,
            output_dir=args.output_dir,
        )
