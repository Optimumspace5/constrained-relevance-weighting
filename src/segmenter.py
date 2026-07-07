import json
import os
import hashlib
from dataclasses import asdict
import anthropic
from dotenv import load_dotenv
from src.models import TranscriptSegment, Topic
from src.config import BULK_MODEL, NUM_TOPICS
# Does two things
#1. Discover topics - what are the 8 main themes in this podcast
#2. Classify segments - which segments does each chunk belong to
# Analogy: Imagine you have a stack of 40 paragraphs from a podcast. This file is like hiring a librarian to first come up with 8 category labels, then sort every paragraph into the right category.

# Load ANTHROPIC_API_KEY from .env into the environment so the Anthropic client can find it.
load_dotenv()

# Initialise the Anthropic client once at module level.
# It automatically reads ANTHROPIC_API_KEY from the environment.
client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# 1. discover_topics
# ---------------------------------------------------------------------------

def discover_topics(segments: list[TranscriptSegment], num_topics: int = NUM_TOPICS) -> list[dict]:
    """
    Ask Claude to identify the main topics in the transcript.

    To keep the prompt small we don't send the full transcript.
    Instead we build a 'preview': the first 80 words of every 2nd segment.
    This gives Claude a representative sample spread across the whole episode.

    LIMITATION: Topic discovery is based on a sampled preview, not the full text.
    Rare topics that appear only in unsampled segments may be missed entirely.

    Returns a list of dicts, each with 'name' and 'description'.
    """

    # --- Build a condensed preview ---
    # Take every 2nd segment (indices 0, 2, 4, …) and grab the first 80 words.
    preview_lines = []
    for i, seg in enumerate(segments):
        if i % 2 == 0:
            first_80 = ' '.join(seg.text.split()[:80])
            preview_lines.append(f"[Excerpt {i}]: {first_80}")

    preview_text = '\n'.join(preview_lines)

    # --- Build the prompt ---
    prompt = (
        f"You are analyzing a podcast transcript. "
        f"Based on these excerpts, identify exactly {num_topics} main topics discussed. "
        f"For each topic return a JSON array with objects containing "
        f"'name' (short label, 2-5 words) and 'description' (one sentence explaining the topic). "
        f"Return ONLY valid JSON, no other text.\n\n"
        f"{preview_text}"
    )

    # --- Call the API ---
    try:
        response = client.messages.create(
            model=BULK_MODEL,
            max_tokens=1024,
            temperature=0,   # deterministic topic discovery
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed in discover_topics: {e}") from e

    # --- Parse the JSON response ---
    # Claude is instructed to return only JSON, but strip any accidental markdown fences.
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        topics = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse topic JSON from Claude response:\n{raw}\nError: {e}") from e

    return topics


# ---------------------------------------------------------------------------
# 2. classify_segments
# ---------------------------------------------------------------------------

def classify_segments(
    segments: list[TranscriptSegment],
    topics: list[dict],
) -> list[Topic]:
    """
    Ask Claude to classify each segment into one of the discovered topics.

    Segments are processed in batches of 10 to keep each prompt a manageable size.
    Claude returns a JSON array mapping segment index → topic name.
    We then assemble Topic objects with their assigned segment indices and proportions.
    """

    # Build a compact topic list string to include in every batch prompt.
    topic_names = ', '.join(t['name'] for t in topics)

    # --- Process segments in batches of 10 ---
    BATCH_SIZE = 10
    # classifications will map segment index (within the full list) → topic name
    classifications: dict[int, str] = {}

    for batch_start in range(0, len(segments), BATCH_SIZE):
        batch = segments[batch_start: batch_start + BATCH_SIZE]

        # Number each segment within the batch using its position in the full list
        # so Claude's response refers to indices we can look up directly.
        numbered_segments = '\n'.join(
            f"{batch_start + j}. {seg.text[:300]}"   # truncate very long segments for prompt size
            for j, seg in enumerate(batch)
        )

        prompt = (
            f"Here are the topics: {topic_names}.\n\n"
            f"For each of the following numbered text segments, classify which topic it belongs to. "
            f"Return ONLY a JSON array of objects with 'segment_index' (int) and 'topic_name' (str). "
            f"No other text.\n\n"
            f"{numbered_segments}"
        )

        # --- Call the API ---
        try:
            response = client.messages.create(
                model=BULK_MODEL,
                max_tokens=1024,
                temperature=0,   # deterministic classification
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(
                f"API call failed in classify_segments (batch starting at {batch_start}): {e}"
            ) from e

        # --- Parse the JSON response ---
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            batch_results = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse classification JSON for batch {batch_start}:\n{raw}\nError: {e}"
            ) from e

        # Store each result — Claude returns segment_index and topic_name for each segment.
        for item in batch_results:
            classifications[item['segment_index']] = item['topic_name']

    # --- Build Topic objects ---
    total_words = sum(seg.word_count for seg in segments)

    # Map topic name → the Topic object we're building up.
    topic_map: dict[str, Topic] = {
        t['name']: Topic(
            name=t['name'],
            description=t['description'],
            proportion=0.0,          # calculated below once all segments are assigned
            segment_indices=[],
        )
        for t in topics
    }

    # Assign each segment to its topic.
    for seg_index, topic_name in classifications.items():
        if topic_name in topic_map:
            topic_map[topic_name].segment_indices.append(seg_index)

    # Calculate each topic's proportion: what fraction of all words does it cover?
    for topic in topic_map.values():
        words_in_topic = sum(segments[i].word_count for i in topic.segment_indices)
        topic.proportion = words_in_topic / total_words if total_words > 0 else 0.0

    return list(topic_map.values())


# ---------------------------------------------------------------------------
# 3. segment_transcript  (main orchestrator, with disk cache)
# ---------------------------------------------------------------------------

# Segmentation (topic discovery + classification) is the most expensive API
# stage and its output is deterministic for a given transcript, so we cache it
# to disk keyed by a hash of the transcript content + num_topics. `cache/` is
# gitignored. Pass force_refresh=True to bypass the cache and recompute.
CACHE_DIR = "cache"


def _segmentation_key(segments: list[TranscriptSegment], num_topics: int) -> str:
    """Stable hash of the transcript text + num_topics — the cache key."""
    h = hashlib.sha256()
    for seg in segments:
        h.update(seg.text.encode("utf-8"))
        h.update(b"\x00")
    h.update(str(num_topics).encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"segmentation_{key}.json")


def _load_cached_topics(path: str) -> list[Topic]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Topic(**t) for t in data]


def _save_topics(path: str, topics: list[Topic]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in topics], f)


def segment_transcript(
    segments: list[TranscriptSegment],
    num_topics: int = NUM_TOPICS,
    force_refresh: bool = False,
) -> list[Topic]:
    """
    Orchestrates the full topic-discovery and classification pipeline.

    Results are cached to disk keyed by transcript hash. On a cache hit the
    cached topics are returned without any API calls. Pass force_refresh=True to
    ignore the cache and recompute (then overwrite it).

    1. Calls discover_topics to get topic names + descriptions from Claude.
    2. Calls classify_segments to assign every segment to a topic.
    3. Prints a summary and returns the list of Topic objects.
    """
    key = _segmentation_key(segments, num_topics)
    path = _cache_path(key)

    if not force_refresh and os.path.exists(path):
        topics = _load_cached_topics(path)
        print(f"Loaded cached segmentation ({len(topics)} topics) from {path}")
        return topics

    print(f"Discovering {num_topics} topics from {len(segments)} segments...")
    topic_dicts = discover_topics(segments, num_topics)
    print(f"Topics found: {[t['name'] for t in topic_dicts]}\n")

    print("Classifying segments into topics (in batches of 10)...")
    topics = classify_segments(segments, topic_dicts)

    # Print a summary of what was found.
    print("\n--- Topic breakdown ---")
    for topic in topics:
        print(
            f"  {topic.name}: {topic.proportion:.1%} of transcript "
            f"({len(topic.segment_indices)} segments)"
        )

    _save_topics(path, topics)
    print(f"Cached segmentation to {path}")
    return topics


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.loader import load_transcript

    segments = load_transcript("data/transcripts/episode1.txt")
    topics = segment_transcript(segments)
