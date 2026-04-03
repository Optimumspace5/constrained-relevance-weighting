import json
import anthropic
from dotenv import load_dotenv
from src.models import TranscriptSegment, Topic, UserPreference, Summary
from src.config import LLM_MODEL

load_dotenv()
client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# 1. evaluate_faithfulness
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    summary: Summary,
    segments: list[TranscriptSegment],
) -> dict:
    """
    Score how well each summary paragraph is supported by the source transcript.

    For each paragraph (SummarySegment), we send Claude:
    - The paragraph text
    - Up to 5 source transcript excerpts (first 200 words each), spread across the topic

    Claude rates support on a 1-5 scale and flags any unsupported claims.
    This catches hallucinations — things Claude may have said in the summary
    that aren't actually in the transcript.
    """
    paragraph_scores = []
    issues = []

    for seg in summary.segments:
        # Gather up to 5 source segments spread across the topic: first, 25%, middle, 75%, last.
        # This mirrors the _sample_topic_segments strategy in summarizers.py — spreading the
        # sample gives the faithfulness checker a better chance of finding the relevant text
        # regardless of where in the topic the claim appears.
        all_indices = seg.source_segment_indices if seg.source_segment_indices else []
        if len(all_indices) <= 5:
            source_indices = all_indices
        else:
            n = len(all_indices)
            source_indices = list(dict.fromkeys([
                all_indices[0],
                all_indices[n // 4],
                all_indices[n // 2],
                all_indices[(3 * n) // 4],
                all_indices[-1],
            ]))

        # Build the source excerpt block — first 200 words of each source segment.
        source_blocks = []
        for idx in source_indices:
            excerpt = " ".join(segments[idx].text.split()[:200])
            source_blocks.append(f"[Source {idx}]: {excerpt}")
        source_text = "\n\n".join(source_blocks) if source_blocks else "(no source segments available)"

        prompt = (
            f"You are evaluating summary faithfulness. "
            f"Given this summary paragraph and the source transcript excerpts, "
            f"rate how well the summary is supported by the source on a scale of 1-5 "
            f"where 1=completely unsupported/hallucinated, 3=partially supported, 5=fully supported. "
            f"Return ONLY a JSON object with 'score' (int) and 'issues' "
            f"(str, brief explanation of any unsupported claims). No other text.\n\n"
            f"SUMMARY PARAGRAPH:\n{seg.text}\n\n"
            f"SOURCE TRANSCRIPT EXCERPTS:\n{source_text}"
        )

        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"API call failed in evaluate_faithfulness: {e}") from e

        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse faithfulness JSON:\n{raw}\nError: {e}") from e

        paragraph_scores.append(result["score"])
        if result.get("issues"):
            issues.append(result["issues"])

    average_score = sum(paragraph_scores) / len(paragraph_scores) if paragraph_scores else 0.0

    return {
        "paragraph_scores": paragraph_scores,
        "average_score": round(average_score, 2),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# 2. evaluate_coverage
# ---------------------------------------------------------------------------

def evaluate_coverage(summary: Summary, topics: list[Topic]) -> dict:
    """
    Check how many of the identified topics appear in the summary.

    A topic is considered "covered" if at least one SummarySegment is attributed
    to it. Missing topics indicate the summary dropped something from the episode.
    """
    all_topic_names = {t.name for t in topics}

    # Collect which topic names appear across all summary segments.
    covered_topic_names = {seg.topic_name for seg in summary.segments}

    missing = sorted(all_topic_names - covered_topic_names)
    topics_covered = len(covered_topic_names & all_topic_names)   # intersection in case of stray names
    total = len(all_topic_names)

    return {
        "topics_covered": topics_covered,
        "total_topics": total,
        "coverage_ratio": round(topics_covered / total, 2) if total > 0 else 0.0,
        "missing_topics": missing,
    }


# ---------------------------------------------------------------------------
# 3. evaluate_relevance
# ---------------------------------------------------------------------------

def evaluate_relevance(
    summary: Summary,
    preferences: list[UserPreference],
    topics: list[Topic],
) -> dict:
    """
    Measure how well the summary's topic distribution matches user preferences.

    Approach:
    - Count paragraphs and words per topic in the summary.
    - A high-preference topic should have more coverage than a low-preference one.
    - Relevance score = sum(paragraph_count_for_topic * user_weight) / total_paragraphs
      This rewards coverage of high-weight topics and penalises over-coverage of low-weight ones.
    - topic_alignment flags whether each topic's coverage direction matched its preference
      (e.g. high pref + above-average paragraphs = aligned).
    """
    pref_lookup = {p.topic_name: p.weight for p in preferences}

    # Count paragraphs and words per topic in the summary.
    para_count: dict[str, int] = {}
    word_count: dict[str, int] = {}
    for seg in summary.segments:
        name = seg.topic_name
        para_count[name] = para_count.get(name, 0) + 1
        word_count[name] = word_count.get(name, 0) + len(seg.text.split())

    total_paragraphs = len(summary.segments)

    # Relevance score: weight each paragraph by its topic's user weight.
    relevance_score = 0.0
    if total_paragraphs > 0:
        for topic_name, count in para_count.items():
            weight = pref_lookup.get(topic_name, 1.0)
            relevance_score += (count * weight) / total_paragraphs

    # topic_alignment: did coverage direction match preference direction?
    # "Direction" = above or below the average paragraph count per topic.
    avg_paragraphs = total_paragraphs / len({t.name for t in topics}) if topics else 1.0

    topic_alignment: dict[str, bool] = {}
    for topic in topics:
        weight = pref_lookup.get(topic.name, 1.0)
        count = para_count.get(topic.name, 0)
        # High pref (>1.0) → expect above-average paragraphs; low pref (<1.0) → below average.
        if weight > 1.0:
            topic_alignment[topic.name] = count >= avg_paragraphs
        elif weight < 1.0:
            topic_alignment[topic.name] = count <= avg_paragraphs
        else:
            topic_alignment[topic.name] = True   # medium = always aligned

    return {
        "relevance_score": round(relevance_score, 3),
        "topic_alignment": topic_alignment,
    }


# ---------------------------------------------------------------------------
# 4. run_full_evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    generic: Summary,
    unconstrained: Summary,
    constrained: Summary,
    segments: list[TranscriptSegment],
    topics: list[Topic],
    preferences: list[UserPreference],
) -> dict:
    """
    Run all three evaluations on all three summary variants and print a comparison table.

    Returns the full nested results dict so callers (e.g. Streamlit) can display
    any part of the data they want.
    """
    summaries = {
        "generic": generic,
        "unconstrained": unconstrained,
        "constrained": constrained,
    }

    results = {}
    for name, summary in summaries.items():
        print(f"  Evaluating {name} summary...")
        faithfulness = evaluate_faithfulness(summary, segments)
        coverage     = evaluate_coverage(summary, topics)
        relevance    = evaluate_relevance(summary, preferences, topics)
        results[name] = {
            "faithfulness": faithfulness,
            "coverage": coverage,
            "relevance": relevance,
            "word_count": summary.metadata.get("word_count", 0),
        }

    # --- Print comparison table ---
    col_w = 15
    print(f"\n{'Summary Type':<15} {'Faithfulness':>{col_w}} {'Coverage':>{col_w}} {'Relevance':>{col_w}} {'Word Count':>{col_w}}")
    print("-" * (15 + col_w * 4 + 4))
    for name, r in results.items():
        faith  = f"{r['faithfulness']['average_score']:.2f} / 5"
        cov    = f"{r['coverage']['topics_covered']}/{r['coverage']['total_topics']} topics"
        rel    = f"{r['relevance']['relevance_score']:.3f}"
        words  = str(r["word_count"])
        print(f"{name:<15} {faith:>{col_w}} {cov:>{col_w}} {rel:>{col_w}} {words:>{col_w}}")

    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.loader import load_transcript
    from src.segmenter import segment_transcript
    from src.preferences import create_preferences
    from src.summarizers import (
        generate_generic_summary,
        generate_unconstrained_summary,
        generate_constrained_summary,
    )
    from src.evidence import link_evidence

    segments = load_transcript("data/transcripts/episode1.txt")
    topics = segment_transcript(segments)

    # Hardcoded test ratings: first 2 high, next 3 medium, rest low.
    test_ratings = {}
    for i, topic in enumerate(topics):
        if i < 2:
            test_ratings[topic.name] = "high"
        elif i < 5:
            test_ratings[topic.name] = "medium"
        else:
            test_ratings[topic.name] = "low"

    preferences = create_preferences(topics, test_ratings)

    # Generate all three summaries.
    print("\nGenerating summaries...")
    generic      = generate_generic_summary(segments, topics)
    unconstrained = generate_unconstrained_summary(segments, topics, preferences)
    constrained  = generate_constrained_summary(segments, topics, preferences)

    # Link evidence (paragraph-level traceability) before evaluating.
    # Faithfulness evaluation needs fine-grained SummarySegments to work well.
    print("Linking evidence...")
    generic      = link_evidence(generic,      segments, topics)
    unconstrained = link_evidence(unconstrained, segments, topics)
    constrained  = link_evidence(constrained,  segments, topics)

    # Run full evaluation and print comparison table.
    print("\nRunning evaluation...")
    results = run_full_evaluation(generic, unconstrained, constrained, segments, topics, preferences)
