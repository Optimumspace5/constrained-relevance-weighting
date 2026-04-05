import json
import anthropic
from dotenv import load_dotenv
from src.models import TranscriptSegment, Topic, UserPreference, SummarySegment, Summary
from src.config import LLM_MODEL, MAX_SUMMARY_WORDS, DEFAULT_DELTA

load_dotenv()
client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Helper: pick representative segments for a topic
# ---------------------------------------------------------------------------

def _sample_topic_segments(
    topic: Topic,
    segments: list[TranscriptSegment],
    max_samples: int = 3,
) -> list[int]:
    """
    Pick up to `max_samples` segment indices spread across a topic's segments.
    Strategy: always take the first, last, and middle — this gives Claude
    a sense of how the topic begins, develops, and ends.

    LIMITATION: This is a sampling strategy, not full coverage. Claude only sees
    a subset of each topic's segments, meaning:
    - Details in unsampled segments will not appear in the summary.
    - Faithfulness evaluation is bounded by the same sample (Change 2/6 ensures
      the evaluator checks against these exact segments, not different ones).
    - Increasing max_samples improves coverage but increases prompt size and cost.
    This trade-off is inherent to prompt-size-constrained LLM summarization.
    """
    indices = topic.segment_indices
    if not indices:
        return []
    if len(indices) <= max_samples:
        return indices

    first = indices[0]
    last = indices[-1]
    middle = indices[len(indices) // 2]
    # Use a set to deduplicate in case first/middle/last overlap (small topics).
    return list(dict.fromkeys([first, middle, last]))


# ---------------------------------------------------------------------------
# 1. generate_generic_summary
# ---------------------------------------------------------------------------

def generate_generic_summary(
    segments: list[TranscriptSegment],
    topics: list[Topic],
) -> Summary:
    """
    Generate a balanced summary that covers each topic proportionally.

    Approach:
    - For each topic, sample up to 3 representative segments (first, middle, last).
    - Build a prompt that shows Claude each topic's name, proportion, and sample text.
    - Claude is instructed to mirror those proportions in the summary it writes.
    - Parse the response into SummarySegment objects — one per topic — so every
      sentence in the summary can be traced back to source segments.
    """

    # --- Build the topic + sample text block for the prompt ---
    topic_blocks = []
    sampled_indices: dict[str, list[int]] = {}   # track which segments were actually sent to Claude
    for topic in topics:
        sample_indices = _sample_topic_segments(topic, segments)
        sampled_indices[topic.name] = sample_indices
        sample_texts = [segments[i].text for i in sample_indices]

        # Join samples with a separator so Claude can tell them apart.
        combined = "\n---\n".join(sample_texts)

        topic_blocks.append(
            f"TOPIC: {topic.name} ({topic.proportion:.1%} of episode)\n"
            f"Description: {topic.description}\n"
            f"Sample transcript text:\n{combined}"
        )

    topic_content = "\n\n".join(topic_blocks)

    # --- System prompt: tells Claude its role and constraints ---
    system_prompt = (
        f"You are a podcast summarizer. Create a balanced summary of this podcast episode. "
        f"Cover each topic proportionally based on how much of the episode it represents. "
        f"Do not favor any topic over another. Write in clear, flowing paragraphs. "
        f"Target length: {MAX_SUMMARY_WORDS} words."
    )

    # --- User message: the actual transcript content ---
    user_message = (
        f"Here are the topics in this podcast episode with representative excerpts.\n"
        f"Summarize each topic in proportion to its share of the episode.\n\n"
        f"{topic_content}"
    )

    # --- Call the API ---
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        summary_text = response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed in generate_generic_summary: {e}") from e

    # --- Build SummarySegment objects — one per topic ---
    # We can't know exactly which sentence covers which topic, so we attribute
    # each topic's SummarySegment to all the source segments for that topic.
    # This gives downstream code full traceability even if it's coarse-grained.
    summary_segments = []
    for topic in topics:
        summary_segments.append(SummarySegment(
            text=summary_text,          # the full summary text is shared across all segments
            source_segment_indices=topic.segment_indices,
            topic_name=topic.name,
        ))

    # --- Assemble and return the Summary object ---
    word_count = len(summary_text.split())
    return Summary(
        segments=summary_segments,
        summary_type="generic",
        metadata={
            "word_count": word_count,
            "num_topics": len(topics),
            "sampled_indices": sampled_indices,
        },
    )


# ---------------------------------------------------------------------------
# 2. generate_unconstrained_summary
# ---------------------------------------------------------------------------

def generate_unconstrained_summary(
    segments: list[TranscriptSegment],
    topics: list[Topic],
    preferences: list[UserPreference],
) -> Summary:
    """
    Generate a preference-driven summary with no constraints on topic proportions.

    Unlike the generic summary (which mirrors the episode's actual proportions),
    this version lets user preferences fully override them. A high-rated topic
    gets rich coverage even if it was only 5% of the episode; a low-rated topic
    gets a brief mention even if it dominated the episode.

    The "unconstrained" label means there's no mathematical bound on how far
    the summary can deviate from the original proportions — Claude is simply
    told what the user wants and trusted to act on it.
    """

    # --- Build a preference lookup: topic_name → weight ---
    pref_lookup = {p.topic_name: p.weight for p in preferences}

    # Map weights back to human-readable labels for the prompt.
    weight_to_label = {1.5: "HIGH", 1.0: "MEDIUM", 0.5: "LOW"}

    # How many sample segments to pull per preference level.
    # High = more context so Claude can write richer coverage.
    samples_by_weight = {1.5: 5, 1.0: 3, 0.5: 1}

    # --- Build the topic + sample text block for the prompt ---
    topic_blocks = []
    sampled_indices: dict[str, list[int]] = {}   # track which segments were actually sent to Claude
    for topic in topics:
        weight = pref_lookup.get(topic.name, 1.0)   # default to medium if missing
        label = weight_to_label.get(weight, "MEDIUM")
        max_samples = samples_by_weight.get(weight, 3)

        sample_indices = _sample_topic_segments(topic, segments, max_samples=max_samples)
        sampled_indices[topic.name] = sample_indices
        sample_texts = [segments[i].text for i in sample_indices]
        combined = "\n---\n".join(sample_texts)

        topic_blocks.append(
            f"TOPIC: {topic.name} | User preference: {label}\n"
            f"Description: {topic.description}\n"
            f"Sample transcript text:\n{combined}"
        )

    topic_content = "\n\n".join(topic_blocks)

    # --- System prompt: user preferences take full priority ---
    system_prompt = (
        f"You are a personalized podcast summarizer. Create a summary tailored to the user's interests. "
        f"Topics the user rated HIGH should receive the most detailed coverage with rich explanation. "
        f"Topics rated MEDIUM should receive moderate coverage. "
        f"Topics rated LOW should receive minimal mention — just a brief sentence or two. "
        f"The user's preferences take full priority over the topic's original proportion in the episode. "
        f"Write in clear, flowing paragraphs. Target length: {MAX_SUMMARY_WORDS} words."
    )

    # --- User message: topic data with explicit preference labels ---
    user_message = (
        f"Here are the podcast topics with the user's preference ratings.\n"
        f"Adjust coverage depth according to each rating — HIGH gets the most, LOW gets the least.\n\n"
        f"{topic_content}"
    )

    # --- Call the API ---
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        summary_text = response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed in generate_unconstrained_summary: {e}") from e

    # --- Build SummarySegment objects — one per topic ---
    summary_segments = []
    for topic in topics:
        summary_segments.append(SummarySegment(
            text=summary_text,
            source_segment_indices=topic.segment_indices,
            topic_name=topic.name,
        ))

    # --- Assemble and return the Summary object ---
    word_count = len(summary_text.split())

    # Store the preferences in metadata so callers can see what weights were applied.
    prefs_dict = {p.topic_name: p.weight for p in preferences}

    return Summary(
        segments=summary_segments,
        summary_type="unconstrained",
        metadata={
            "word_count": word_count,
            "num_topics": len(topics),
            "preferences": prefs_dict,
            "sampled_indices": sampled_indices,
        },
    )


# ---------------------------------------------------------------------------
# Helper: calculate constrained proportions
# ---------------------------------------------------------------------------

def calculate_constrained_proportions(
    topics: list[Topic],
    preferences: list[UserPreference],
    delta: float,
) -> dict[str, float]:
    """
    Apply user preferences to topic proportions with a mathematical bound (delta).

    The core idea of Constrained Relevance Weighting:
    - Start with each topic's base proportion (how much of the episode it occupied).
    - Multiply by the user's weight to get a raw target (e.g. high = *1.5).
    - But clamp that target to [base - delta, base + delta] so no topic can be
      expanded or compressed by more than `delta` percentage points.
    - Also enforce a floor of 0.01 so every topic gets at least a 1% mention.
    - Finally, normalize all clamped values so they sum to exactly 1.0.

    Example with delta=0.15:
      Topic A: base=0.40, weight=1.5 → raw=0.60, clamped to 0.40+0.15=0.55
      Topic B: base=0.10, weight=0.5 → raw=0.05, clamped to 0.10-0.15=-0.05 → floor → 0.01
      → then normalize so all sum to 1.0

    Returns a dict mapping topic_name → constrained proportion (floats summing to 1.0).
    """
    pref_lookup = {p.topic_name: p.weight for p in preferences}

    clamped: dict[str, float] = {}
    for topic in topics:
        base = topic.proportion
        weight = pref_lookup.get(topic.name, 1.0)

        # Step 1: apply the user's weight to get an unconstrained target.
        raw_target = base * weight

        # Step 2: clamp to within delta of the base proportion.
        lower = base - delta
        upper = base + delta
        clamped_value = max(lower, min(raw_target, upper))

        # Step 3: enforce a minimum of 0.01 — every topic gets at least a brief mention.
        clamped_value = max(clamped_value, 0.01)

        clamped[topic.name] = clamped_value

    # Step 4: normalize so all proportions sum to 1.0.
    total = sum(clamped.values())
    return {name: value / total for name, value in clamped.items()}


# ---------------------------------------------------------------------------
# 3. generate_constrained_summary
# ---------------------------------------------------------------------------

def generate_constrained_summary(
    segments: list[TranscriptSegment],
    topics: list[Topic],
    preferences: list[UserPreference],
    delta: float = DEFAULT_DELTA,
) -> Summary:
    """
    Generate a preference-driven summary with mathematically bounded topic proportions.

    This is the full Constrained Relevance Weighting approach:
    - User preferences shift topic proportions (like unconstrained).
    - But the shift is capped at ±delta, so the summary stays anchored to the
      episode's actual content — it can't wildly over-represent a minor topic.

    The prompt shows Claude both the original proportion and the constrained target
    so it understands the direction of the adjustment and the ceiling it must respect.
    """

    # Calculate the bounded target proportions for each topic.
    constrained_proportions = calculate_constrained_proportions(topics, preferences, delta)
    pref_lookup = {p.topic_name: p.weight for p in preferences}
    weight_to_label = {1.5: "HIGH", 1.0: "MEDIUM", 0.5: "LOW"}

    # --- Determine sample count per topic based on constrained proportion ---
    # Topics with higher target proportions get more sample text in the prompt,
    # giving Claude richer material to draw from for more detailed coverage.
    # Scale: proportion >= 0.20 → 5 samples, >= 0.10 → 3, below → 1.
    def samples_for_proportion(p: float) -> int:
        if p >= 0.20:
            return 5
        if p >= 0.10:
            return 3
        return 1

    # --- Build the topic + sample text block for the prompt ---
    topic_blocks = []
    sampled_indices: dict[str, list[int]] = {}   # track which segments were actually sent to Claude
    for topic in topics:
        original = topic.proportion
        constrained = constrained_proportions[topic.name]
        weight = pref_lookup.get(topic.name, 1.0)
        label = weight_to_label.get(weight, "MEDIUM")
        max_samples = samples_for_proportion(constrained)

        sample_indices = _sample_topic_segments(topic, segments, max_samples=max_samples)
        sampled_indices[topic.name] = sample_indices
        sample_texts = [segments[i].text for i in sample_indices]
        combined = "\n---\n".join(sample_texts)

        topic_blocks.append(
            f"TOPIC: {topic.name} | User preference: {label}\n"
            f"  Original proportion: {original:.1%}  →  Target proportion: {constrained:.1%}\n"
            f"Description: {topic.description}\n"
            f"Sample transcript text:\n{combined}"
        )

    topic_content = "\n\n".join(topic_blocks)

    # --- System prompt: CRW framing ---
    system_prompt = (
        f"You are a personalized podcast summarizer using Constrained Relevance Weighting. "
        f"Each topic has an ORIGINAL proportion (how much it appeared in the episode) and a "
        f"TARGET proportion (adjusted for user preference within bounded limits). "
        f"Cover each topic according to its TARGET proportion, but ensure every topic receives "
        f"at least a brief mention. Do not fabricate information beyond what the transcript contains. "
        f"All claims must be traceable to the provided transcript excerpts. "
        f"Write in clear, flowing paragraphs. Target length: {MAX_SUMMARY_WORDS} words."
    )

    # --- User message ---
    user_message = (
        f"Here are the podcast topics with their original and target proportions.\n"
        f"Allocate your summary words according to the TARGET proportion for each topic.\n\n"
        f"{topic_content}"
    )

    # --- Call the API ---
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        summary_text = response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed in generate_constrained_summary: {e}") from e

    # --- Build SummarySegment objects ---
    summary_segments = []
    for topic in topics:
        summary_segments.append(SummarySegment(
            text=summary_text,
            source_segment_indices=topic.segment_indices,
            topic_name=topic.name,
        ))

    word_count = len(summary_text.split())
    prefs_dict = {p.topic_name: p.weight for p in preferences}
    original_proportions = {t.name: t.proportion for t in topics}

    return Summary(
        segments=summary_segments,
        summary_type="constrained",
        metadata={
            "word_count": word_count,
            "num_topics": len(topics),
            "delta": delta,
            "preferences": prefs_dict,
            "original_proportions": original_proportions,
            "constrained_proportions": constrained_proportions,
            "sampled_indices": sampled_indices,
        },
    )


# ---------------------------------------------------------------------------
# 4. generate_baseline_summary (no LLM — pure extractive)
# ---------------------------------------------------------------------------

def generate_baseline_summary(
    segments: list[TranscriptSegment],
    topics: list[Topic],
) -> Summary:
    """
    Generate a naive extractive baseline by pulling raw sentences from the transcript.

    No LLM is used — this just takes the first N words from each topic's segments,
    proportional to the topic's share of the episode. This provides a lower bound:
    if LLM-generated summaries don't score better than this, something is wrong.

    Approach:
    - Allocate MAX_SUMMARY_WORDS across topics based on their proportions.
    - For each topic, concatenate its segments' text and take the first allocated words.
    - Build one SummarySegment per topic with exact source traceability.
    """
    topic_texts = []
    summary_segments = []

    for topic in topics:
        # How many words this topic gets in the summary.
        word_budget = int(MAX_SUMMARY_WORDS * topic.proportion)
        word_budget = max(word_budget, 10)   # at least 10 words per topic

        # Concatenate all of this topic's segment texts.
        full_text = " ".join(segments[i].text for i in topic.segment_indices)

        # Take the first word_budget words.
        words = full_text.split()
        extracted = " ".join(words[:word_budget])

        # Track which segments contributed (all segments whose text was included).
        used_indices = []
        chars_used = 0
        for idx in topic.segment_indices:
            seg_len = len(segments[idx].text)
            if chars_used < len(extracted):
                used_indices.append(idx)
            chars_used += seg_len + 1   # +1 for the space we joined with

        topic_texts.append(f"**{topic.name}**: {extracted}")
        summary_segments.append(SummarySegment(
            text=extracted,
            source_segment_indices=used_indices,
            topic_name=topic.name,
        ))

    # Calculate total word count across all topic extracts.
    word_count = sum(len(seg.text.split()) for seg in summary_segments)

    return Summary(
        segments=summary_segments,
        summary_type="baseline",
        metadata={
            "word_count": word_count,
            "num_topics": len(topics),
        },
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.loader import load_transcript
    from src.segmenter import segment_transcript
    from src.preferences import create_preferences

    segments = load_transcript("data/transcripts/episode1.txt")
    topics = segment_transcript(segments)

    # Build hardcoded test ratings — first 2 topics high, next 3 medium, rest low.
    test_ratings = {}
    for i, topic in enumerate(topics):
        if i < 2:
            test_ratings[topic.name] = "high"
        elif i < 5:
            test_ratings[topic.name] = "medium"
        else:
            test_ratings[topic.name] = "low"

    preferences = create_preferences(topics, test_ratings)

    # --- Comparison table: original vs constrained proportions ---
    constrained_proportions = calculate_constrained_proportions(topics, preferences, DEFAULT_DELTA)
    pref_lookup = {p.topic_name: p.weight for p in preferences}
    weight_to_label = {1.5: "high", 1.0: "medium", 0.5: "low"}

    print(f"\n--- Proportion comparison (delta={DEFAULT_DELTA}) ---")
    print(f"  {'Topic':<35} {'Original':>9} {'Pref':>8} {'Target':>9}")
    print(f"  {'-'*35} {'-'*9} {'-'*8} {'-'*9}")
    for topic in topics:
        label = weight_to_label.get(pref_lookup.get(topic.name, 1.0), "medium")
        print(
            f"  {topic.name:<35} "
            f"{topic.proportion:>8.1%} "
            f"{label:>8} "
            f"{constrained_proportions[topic.name]:>8.1%}"
        )

    # --- Generic summary ---
    print("\nGenerating generic summary...")
    generic = generate_generic_summary(segments, topics)
    print(f"\n--- Generic Summary ({generic.metadata['word_count']} words) ---\n")
    print(generic.segments[0].text)

    # --- Unconstrained summary ---
    print("\nGenerating unconstrained summary...")
    unconstrained = generate_unconstrained_summary(segments, topics, preferences)
    print(f"\n--- Unconstrained Summary ({unconstrained.metadata['word_count']} words) ---\n")
    print(unconstrained.segments[0].text)

    # --- Constrained summary ---
    print("\nGenerating constrained summary...")
    constrained = generate_constrained_summary(segments, topics, preferences)
    print(f"\n--- Constrained Summary ({constrained.metadata['word_count']} words) ---\n")
    print(constrained.segments[0].text)
