import json
import anthropic
from rouge_score import rouge_scorer
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
    - The source segments that the summarizer actually used (from metadata["sampled_indices"])

    Claude rates support on a 1-5 scale and flags any unsupported claims.
    This catches hallucinations — things Claude may have said in the summary
    that aren't actually in the transcript.

    KNOWN LIMITATIONS:
    - LLM-as-judge: Claude is evaluating its own output, which introduces self-consistency
      bias (it may rate its own phrasing as well-supported). ROUGE and extractive overlap
      scores (compute_rouge_scores, compute_extractive_overlap) provide complementary
      non-LLM metrics to cross-validate.
    - Sampling: faithfulness is checked only against the sampled segments, not the full
      transcript. Claims paraphrased from unsampled segments may be flagged as unsupported.
    """
    paragraph_scores = []
    issues = []
    # Track per-topic scores for controlled comparison across summary types.
    topic_scores: dict[str, list[int]] = {}

    # Use the exact segments the summarizer sent to Claude, stored in metadata.
    # This ensures faithfulness is checked against the same evidence the summary was built from,
    # rather than re-sampling independently (which could pick different segments).
    sampled_indices = summary.metadata.get("sampled_indices", {})

    for seg in summary.segments:
        # Look up the segments that were actually used to generate this topic's summary.
        # Falls back to source_segment_indices if sampled_indices isn't available (backwards compat).
        source_indices = sampled_indices.get(seg.topic_name, [])
        if not source_indices:
            source_indices = seg.source_segment_indices[:7] if seg.source_segment_indices else []

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

        score = result["score"]
        paragraph_scores.append(score)
        if result.get("issues"):
            issues.append(result["issues"])

        # Accumulate per-topic scores.
        if seg.topic_name not in topic_scores:
            topic_scores[seg.topic_name] = []
        topic_scores[seg.topic_name].append(score)

    average_score = sum(paragraph_scores) / len(paragraph_scores) if paragraph_scores else 0.0

    # Compute per-topic averages.
    per_topic_avg = {
        topic: round(sum(scores) / len(scores), 2)
        for topic, scores in topic_scores.items()
    }

    return {
        "paragraph_scores": paragraph_scores,
        "average_score": round(average_score, 2),
        "issues": issues,
        "per_topic_scores": per_topic_avg,
    }


# ---------------------------------------------------------------------------
# 2. compute_rouge_scores
# ---------------------------------------------------------------------------

def compute_rouge_scores(
    summary: Summary,
    segments: list[TranscriptSegment],
) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L between the summary and its source text.

    Unlike the LLM-based faithfulness check, ROUGE is deterministic and reproducible.
    It measures n-gram overlap: how many words/phrases from the source actually appear
    in the summary. Higher scores indicate the summary is more grounded in the source.

    - ROUGE-1: unigram overlap (individual word matches)
    - ROUGE-2: bigram overlap (two-word phrase matches)
    - ROUGE-L: longest common subsequence (captures sentence-level structure)

    The source text is built from the segments the summarizer actually used (stored in
    metadata["sampled_indices"]), falling back to all source_segment_indices if unavailable.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Build the reference text from ALL source segments (not just sampled ones).
    # Using only sampled segments (3-5 excerpts) produces near-zero ROUGE scores
    # because the reference is too small relative to the summary.
    all_source_indices: set[int] = set()
    for seg in summary.segments:
        if seg.source_segment_indices:
            all_source_indices.update(seg.source_segment_indices)

    # Join all source segment texts into one reference string.
    reference_text = " ".join(
        segments[i].text for i in sorted(all_source_indices) if i < len(segments)
    )

    # The summary text — join all segment texts (handles both pre- and post-evidence-linking).
    summary_text = " ".join(seg.text for seg in summary.segments) if summary.segments else ""

    scores = scorer.score(reference_text, summary_text)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ---------------------------------------------------------------------------
# 3. compute_extractive_overlap
# ---------------------------------------------------------------------------

def compute_extractive_overlap(
    summary: Summary,
    segments: list[TranscriptSegment],
) -> dict:
    """
    Measure what fraction of the summary's content is directly extracted from the source.

    Unlike ROUGE (which is symmetric and uses F-measure), extractive overlap answers a
    directional question: "of everything in the summary, how much came verbatim from
    the transcript?" High overlap = the summary sticks close to the source wording.
    Low overlap = the LLM paraphrased or potentially hallucinated.

    Returns:
    - unigram_overlap: fraction of summary words found in the source
    - bigram_overlap: fraction of summary bigrams found in the source
    """
    # Build source text from ALL source segments (not just sampled ones).
    all_source_indices: set[int] = set()
    for seg in summary.segments:
        if seg.source_segment_indices:
            all_source_indices.update(seg.source_segment_indices)

    source_text = " ".join(
        segments[i].text for i in sorted(all_source_indices) if i < len(segments)
    )
    summary_text = " ".join(seg.text for seg in summary.segments) if summary.segments else ""

    # Tokenize: lowercase and split on whitespace for simple word-level comparison.
    source_words = source_text.lower().split()
    summary_words = summary_text.lower().split()

    if not summary_words:
        return {"unigram_overlap": 0.0, "bigram_overlap": 0.0}

    # Unigram overlap: what fraction of summary words appear in the source?
    source_unigrams = set(source_words)
    matching_unigrams = sum(1 for w in summary_words if w in source_unigrams)
    unigram_overlap = matching_unigrams / len(summary_words)

    # Bigram overlap: what fraction of summary bigrams appear in the source?
    def bigrams(words: list[str]) -> list[tuple[str, str]]:
        return [(words[i], words[i + 1]) for i in range(len(words) - 1)]

    source_bigrams = set(bigrams(source_words))
    summary_bigram_list = bigrams(summary_words)
    if summary_bigram_list:
        matching_bigrams = sum(1 for b in summary_bigram_list if b in source_bigrams)
        bigram_overlap = matching_bigrams / len(summary_bigram_list)
    else:
        bigram_overlap = 0.0

    return {
        "unigram_overlap": round(unigram_overlap, 4),
        "bigram_overlap": round(bigram_overlap, 4),
    }


# ---------------------------------------------------------------------------
# 4. evaluate_coverage (no API call — pure set comparison)
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

    Approach (improved — uses word counts, not paragraph counts):
    - Count words per topic in the summary to get actual word proportions.
    - Relevance score = sum(topic_word_proportion * user_weight) across all topics.
      This rewards giving more words to high-weight topics and fewer to low-weight ones.
    - Proportion MAE = mean absolute error between the summary's actual topic proportions
      and the target proportions (from calculate_constrained_proportions if available,
      otherwise from the original topic proportions). Lower MAE = better adherence to targets.
    - topic_alignment flags whether each topic's coverage direction matched its preference
      (e.g. high pref + above-average word share = aligned).
    """
    pref_lookup = {p.topic_name: p.weight for p in preferences}

    # Count words per topic in the summary.
    word_count: dict[str, int] = {}
    for seg in summary.segments:
        name = seg.topic_name
        word_count[name] = word_count.get(name, 0) + len(seg.text.split())

    total_words = sum(word_count.values())

    # Compute actual word proportions in the summary.
    actual_proportions: dict[str, float] = {}
    for topic in topics:
        words = word_count.get(topic.name, 0)
        actual_proportions[topic.name] = words / total_words if total_words > 0 else 0.0

    # Relevance score: weight each topic's word proportion by the user's preference weight.
    relevance_score = 0.0
    for topic_name, proportion in actual_proportions.items():
        weight = pref_lookup.get(topic_name, 1.0)
        relevance_score += proportion * weight

    # Proportion MAE: how far is the summary's actual distribution from the target?
    # Use constrained_proportions from metadata if available (constrained summary),
    # otherwise fall back to the original topic proportions (generic/unconstrained).
    target_proportions = summary.metadata.get("constrained_proportions", {})
    if not target_proportions:
        target_proportions = {t.name: t.proportion for t in topics}

    absolute_errors = []
    for topic in topics:
        actual = actual_proportions.get(topic.name, 0.0)
        target = target_proportions.get(topic.name, 0.0)
        absolute_errors.append(abs(actual - target))

    proportion_mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0

    # topic_alignment: did coverage direction match preference direction?
    # "Direction" = above or below the average word proportion per topic.
    avg_proportion = 1.0 / len(topics) if topics else 1.0

    topic_alignment: dict[str, bool] = {}
    for topic in topics:
        weight = pref_lookup.get(topic.name, 1.0)
        proportion = actual_proportions.get(topic.name, 0.0)
        # High pref (>1.0) → expect above-average proportion; low pref (<1.0) → below average.
        if weight > 1.0:
            topic_alignment[topic.name] = proportion >= avg_proportion
        elif weight < 1.0:
            topic_alignment[topic.name] = proportion <= avg_proportion
        else:
            topic_alignment[topic.name] = True   # medium = always aligned

    return {
        "relevance_score": round(relevance_score, 3),
        "proportion_mae": round(proportion_mae, 4),
        "actual_proportions": {k: round(v, 4) for k, v in actual_proportions.items()},
        "topic_alignment": topic_alignment,
    }


# ---------------------------------------------------------------------------
# 4. evaluate_faithfulness_qa  (QAGS-inspired, claim-level)
# ---------------------------------------------------------------------------

def evaluate_faithfulness_qa(
    summary: Summary,
    segments: list[TranscriptSegment],
) -> dict:
    """
    QA-based faithfulness evaluation inspired by QAGS (Wang, Cho & Lewis, 2020).

    Instead of asking Claude to subjectively rate faithfulness 1-5, this method:
    1. Extracts atomic factual claims from each summary paragraph.
    2. For each claim, generates a yes/no question.
    3. Answers the question using ONLY the source transcript segments.
    4. If the source can confirm the claim, it's "supported". Otherwise "unsupported".

    This is more stable than LLM-as-judge because the task is constrained:
    answering factual yes/no questions has far less variance than subjective rating.

    Returns:
        - total_claims: how many factual claims were extracted
        - supported_claims: how many were confirmed by the source
        - precision: supported / total (the faithfulness score)
        - unsupported: list of claims that could not be verified
        - per_topic: dict of topic_name → {total, supported, precision}
    """
    # Get the sampled indices so we check against the same evidence the summary used.
    sampled_indices = summary.metadata.get("sampled_indices", {})

    all_claims = []          # list of (claim_text, topic_name, is_supported)
    per_topic: dict[str, dict] = {}

    for seg in summary.segments:
        # --- Step 1: Extract atomic claims from this paragraph ---
        source_indices = sampled_indices.get(seg.topic_name, [])
        if not source_indices:
            source_indices = seg.source_segment_indices[:10] if seg.source_segment_indices else []

        extract_prompt = (
            "Extract every distinct factual claim from this summary paragraph. "
            "Each claim should be a single, specific, verifiable statement. "
            "Return ONLY a JSON array of strings — one claim per element. "
            "Do not include opinions, transitions, or vague statements.\n\n"
            f"PARAGRAPH:\n{seg.text}"
        )

        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": extract_prompt}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"API call failed in QAGS claim extraction: {e}") from e

        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            claims = json.loads(raw)
        except json.JSONDecodeError:
            # If parsing fails, treat the whole paragraph as one claim.
            claims = [seg.text]

        if not claims:
            continue

        # --- Step 2: Build source text for verification ---
        source_blocks = []
        for idx in source_indices:
            if idx < len(segments):
                excerpt = " ".join(segments[idx].text.split()[:200])
                source_blocks.append(f"[Segment {idx}]: {excerpt}")
        source_text = "\n\n".join(source_blocks) if source_blocks else "(no source segments)"

        # --- Step 3: Verify each claim against the source ---
        # Batch all claims into one API call for efficiency.
        claims_numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

        verify_prompt = (
            "For each numbered claim below, determine whether it is SUPPORTED or UNSUPPORTED "
            "based ONLY on the source transcript excerpts provided. "
            "A claim is SUPPORTED if the source contains evidence that directly confirms it. "
            "A claim is UNSUPPORTED if the source does not contain enough evidence to confirm it, "
            "even if you personally know it to be true from other sources. "
            "Return ONLY a JSON array of objects with 'claim_index' (int, 1-based) "
            "and 'verdict' (string: 'supported' or 'unsupported'). No other text.\n\n"
            f"CLAIMS:\n{claims_numbered}\n\n"
            f"SOURCE TRANSCRIPT EXCERPTS:\n{source_text}"
        )

        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": verify_prompt}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"API call failed in QAGS verification: {e}") from e

        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            verdicts = json.loads(raw)
        except json.JSONDecodeError:
            # If parsing fails, conservatively mark all as unsupported.
            verdicts = [{"claim_index": i+1, "verdict": "unsupported"} for i in range(len(claims))]

        # Build a lookup of verdicts.
        verdict_map = {v["claim_index"]: v["verdict"] for v in verdicts}

        # Record results.
        topic_name = seg.topic_name
        if topic_name not in per_topic:
            per_topic[topic_name] = {"total": 0, "supported": 0, "unsupported_claims": []}

        for i, claim in enumerate(claims):
            verdict = verdict_map.get(i + 1, "unsupported")
            is_supported = verdict.lower() == "supported"
            all_claims.append((claim, topic_name, is_supported))
            per_topic[topic_name]["total"] += 1
            if is_supported:
                per_topic[topic_name]["supported"] += 1
            else:
                per_topic[topic_name]["unsupported_claims"].append(claim)

    # Compute aggregate scores.
    total_claims = len(all_claims)
    supported_claims = sum(1 for _, _, s in all_claims if s)
    precision = supported_claims / total_claims if total_claims > 0 else 0.0

    unsupported = [claim for claim, _, s in all_claims if not s]

    # Per-topic precision.
    per_topic_precision = {}
    for topic_name, data in per_topic.items():
        t = data["total"]
        per_topic_precision[topic_name] = {
            "total": t,
            "supported": data["supported"],
            "precision": round(data["supported"] / t, 3) if t > 0 else 0.0,
            "unsupported_claims": data["unsupported_claims"],
        }

    return {
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "precision": round(precision, 3),
        "unsupported": unsupported,
        "per_topic": per_topic_precision,
    }


# ---------------------------------------------------------------------------
# 5. run_full_evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    generic: Summary,
    unconstrained: Summary,
    constrained: Summary,
    segments: list[TranscriptSegment],
    topics: list[Topic],
    preferences: list[UserPreference],
    baseline: Summary | None = None,
) -> dict:
    """
    Run all evaluations on all summary variants and print a comparison table.

    Returns the full nested results dict so callers (e.g. Streamlit) can display
    any part of the data they want. The optional baseline provides a lower bound.
    """
    summaries = {}
    if baseline is not None:
        summaries["baseline"] = baseline
    summaries["generic"] = generic
    summaries["unconstrained"] = unconstrained
    summaries["constrained"] = constrained

    results = {}
    for name, summary in summaries.items():
        print(f"  Evaluating {name} summary...")
        faithfulness = evaluate_faithfulness(summary, segments)
        print(f"    Running QAGS for {name}...")
        qags         = evaluate_faithfulness_qa(summary, segments)
        rouge        = compute_rouge_scores(summary, segments)
        extractive   = compute_extractive_overlap(summary, segments)
        coverage     = evaluate_coverage(summary, topics)
        relevance    = evaluate_relevance(summary, preferences, topics)
        results[name] = {
            "faithfulness": faithfulness,
            "qags": qags,
            "rouge": rouge,
            "extractive_overlap": extractive,
            "coverage": coverage,
            "relevance": relevance,
            "word_count": summary.metadata.get("word_count", 0),
        }

    # --- Print comparison table ---
    col_w = 15
    print(f"\n{'Summary Type':<15} {'Faith(1-5)':>{col_w}} {'QAGS Prec.':>{col_w}} {'ROUGE-1':>{col_w}} {'ROUGE-L':>{col_w}} {'Ext. Overlap':>{col_w}} {'Coverage':>{col_w}} {'Relevance':>{col_w}} {'Prop. MAE':>{col_w}} {'Words':>{col_w}}")
    print("-" * (15 + col_w * 9 + 9))
    for name, r in results.items():
        faith  = f"{r['faithfulness']['average_score']:.2f} / 5"
        qags_p = f"{r['qags']['supported_claims']}/{r['qags']['total_claims']} ({r['qags']['precision']:.0%})"
        r1     = f"{r['rouge']['rouge1']:.4f}"
        rl     = f"{r['rouge']['rougeL']:.4f}"
        ext    = f"{r['extractive_overlap']['unigram_overlap']:.4f}"
        cov    = f"{r['coverage']['topics_covered']}/{r['coverage']['total_topics']} topics"
        rel    = f"{r['relevance']['relevance_score']:.3f}"
        mae    = f"{r['relevance']['proportion_mae']:.4f}"
        words  = str(r["word_count"])
        print(f"{name:<15} {faith:>{col_w}} {qags_p:>{col_w}} {r1:>{col_w}} {rl:>{col_w}} {ext:>{col_w}} {cov:>{col_w}} {rel:>{col_w}} {mae:>{col_w}} {words:>{col_w}}")

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
