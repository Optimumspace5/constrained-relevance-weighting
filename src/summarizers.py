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
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.loader import load_transcript
    from src.segmenter import segment_transcript

    segments = load_transcript("data/transcripts/episode1.txt")
    topics = segment_transcript(segments)

    print("\nGenerating generic summary...")
    summary = generate_generic_summary(segments, topics)

    # Print the full summary text (it's the same across all SummarySegments,
    # so just print the first one).
    print(f"\n--- Generic Summary ({summary.metadata['word_count']} words) ---\n")
    print(summary.segments[0].text)
