import json
import anthropic
from dotenv import load_dotenv
from src.models import TranscriptSegment, Topic, UserPreference, SummarySegment, Summary
from src.config import LLM_MODEL, MAX_SUMMARY_WORDS

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
    for topic in topics:
        sample_indices = _sample_topic_segments(topic, segments)
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
    for topic in topics:
        weight = pref_lookup.get(topic.name, 1.0)   # default to medium if missing
        label = weight_to_label.get(weight, "MEDIUM")
        max_samples = samples_by_weight.get(weight, 3)

        sample_indices = _sample_topic_segments(topic, segments, max_samples=max_samples)
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

    # Build hardcoded test ratings so we don't need interactive CLI input.
    # Rule: first 2 topics → high, next 3 → medium, rest → low.
    test_ratings = {}
    for i, topic in enumerate(topics):
        if i < 2:
            test_ratings[topic.name] = "high"
        elif i < 5:
            test_ratings[topic.name] = "medium"
        else:
            test_ratings[topic.name] = "low"

    preferences = create_preferences(topics, test_ratings)
    print("\nTest preferences:")
    for pref in preferences:
        print(f"  {pref.topic_name}: {pref.weight}")

    # --- Generic summary (no preferences, proportional coverage) ---
    print("\nGenerating generic summary...")
    generic = generate_generic_summary(segments, topics)
    print(f"\n--- Generic Summary ({generic.metadata['word_count']} words) ---\n")
    print(generic.segments[0].text)

    # --- Unconstrained summary (preferences override proportions) ---
    print("\nGenerating unconstrained summary...")
    unconstrained = generate_unconstrained_summary(segments, topics, preferences)
    print(f"\n--- Unconstrained Summary ({unconstrained.metadata['word_count']} words) ---\n")
    print(unconstrained.segments[0].text)
