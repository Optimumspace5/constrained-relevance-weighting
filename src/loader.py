import re
from src.models import TranscriptSegment


# --- Regex patterns ---

# YouTube auto-generated timestamps appear at the start of each line in two parts:
#   Part 1 (M:SS):  e.g. "1:03"
#   Part 2 (spoken time glued to next word): e.g. "1 minute, 3 secondsBut"
# Combined they look like: "1:031 minute, 3 secondsBut"
# This regex matches the entire prefix up to and including "second(s)".
_TIMESTAMP_RE = re.compile(
    r'^\d+:\d+'           # M:SS or H:MM:SS clock stamp (e.g. "1:03", "1:00:06")
    r'(?::\d+)?'          # optional extra :SS for H:MM:SS format
    r'\d+'                # display number stuck on with no space (e.g. "1" → "1 hour/minute")
    r'(?:\s+hours?'       # optional " hour(s)" block for >=1-hour marks
    r'(?:,\s*\d+)?'       #   optional ", NN" (the minutes number after hours)
    r')?'                 # close the hours group
    r'(?:\s+minutes?'     # optional " minute(s)" block for >=1-minute marks
    r'(?:,\s*\d+)?'       #   optional ", NN" (the seconds number after minutes)
    r')?'                 # close the minutes group
    r'(?:\s+seconds?)?'   # optional " second(s)" — absent on exact-minute marks like 15:00
)

# Sound artifacts inserted by YouTube's auto-captioner: [music], [snorts], [applause], etc.
_ARTIFACT_RE = re.compile(r'\[.*?\]')

# Collapse runs of whitespace (including newlines) to a single space.
_WHITESPACE_RE = re.compile(r'\s+')

# Extract just the clock portion from the start of a timestamped line.
# Handles both M:SS (e.g. "1:03") and H:MM:SS (e.g. "1:00:06").
_CLOCK_RE = re.compile(r'^(\d+:\d{2}(?::\d{2})?)')

# Sentence boundary: period, exclamation, or question mark followed by a space or end-of-string.
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

# .sub means substitute, in this case substitute with space
def _clean_line(line: str) -> str:
    """Strip timestamp prefix and sound artifacts from a single transcript line."""
    line = _TIMESTAMP_RE.sub('', line)      # remove timestamp prefix
    line = _ARTIFACT_RE.sub('', line)       # remove [music], [snorts], etc.
    line = _WHITESPACE_RE.sub(' ', line)    # collapse whitespace
    return line.strip()


def _split_into_chunks(text: str, target_words: int = 250) -> list[str]:
    """
    Split a continuous text into chunks of roughly `target_words` words.
    Splits are made at sentence boundaries where possible; if a single sentence
    exceeds the target it becomes its own chunk.
    """
    sentences = _SENTENCE_END_RE.split(text)

    chunks = []
    current_words: list[str] = []
    current_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = sentence.split()
        word_count = len(sentence_words)

        # If adding this sentence would overshoot the target, flush first.
        if current_count > 0 and current_count + word_count > target_words:
            chunks.append(' '.join(current_words))
            current_words = []
            current_count = 0

        # If a single sentence exceeds the target (common in unpunctuated transcripts),
        # force-split it into target_words-sized pieces.
        if word_count > target_words and current_count == 0:
            for j in range(0, word_count, target_words):
                piece = sentence_words[j:j + target_words]
                if current_words:
                    chunks.append(' '.join(current_words))
                    current_words = []
                    current_count = 0
                current_words = piece
                current_count = len(piece)
        else:
            current_words.extend(sentence_words)
            current_count += word_count

    # Flush whatever is left.
    if current_words:
        chunks.append(' '.join(current_words))

    return chunks


def load_transcript(file_path: str) -> list[TranscriptSegment]:
    """
    Load a YouTube auto-generated transcript and return a list of TranscriptSegment objects.

    Steps:
      1. Read the file line by line.
      2. Skip lines without a timestamp (chapter headings, blank lines).
      3. Strip the timestamp prefix and sound artifacts from each line.
      4. Join all cleaned lines into one continuous text.
      5. Split that text into 200-300-word chunks at sentence boundaries.
      6. Build TranscriptSegment objects with character-position indices into the full text.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    # Step 1-3: clean each timestamped line; skip chapter headings and blanks.
    # Also extract the M:SS clock timestamp from each line so we can assign
    # timestamps to the chunks later.
    cleaned_lines = []
    line_timestamps = []   # parallel list: clock timestamp for each cleaned line
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Lines without a leading timestamp (e.g. "Chapter 2: …") are not speech.
        if not re.match(r'^\d+:\d+', line):
            continue
        # Extract the M:SS clock before cleaning strips it.
        clock_match = _CLOCK_RE.match(line)
        timestamp = clock_match.group(1) if clock_match else ""
        cleaned = _clean_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)
            line_timestamps.append(timestamp)

    # Step 4: join into one continuous text.
    full_text = ' '.join(cleaned_lines)

    # Build a mapping from character position in full_text → timestamp.
    # Each cleaned line's starting character position maps to its timestamp.
    char_to_timestamp: list[tuple[int, str]] = []
    pos = 0
    for i, cl in enumerate(cleaned_lines):
        char_to_timestamp.append((pos, line_timestamps[i]))
        pos += len(cl) + 1   # +1 for the space joining them

    # Step 5: split into chunks.
    chunks = _split_into_chunks(full_text, target_words=250)

    # Step 6: build TranscriptSegment objects.
    # Track character positions inside full_text so segment_indices remain meaningful.
    # Look up the timestamp of the first transcript line that falls within each chunk.
    segments: list[TranscriptSegment] = []
    search_start = 0
    for chunk in chunks:
        start = full_text.find(chunk, search_start)
        end = start + len(chunk)

        # Find the timestamp for this chunk: the last entry in char_to_timestamp
        # whose position is <= start (i.e. the transcript line that starts at or before this chunk).
        timestamp = ""
        for char_pos, ts in char_to_timestamp:
            if char_pos <= start:
                timestamp = ts
            else:
                break

        segments.append(TranscriptSegment(
            text=chunk,
            start_index=start,
            end_index=end,
            word_count=len(chunk.split()),
            timestamp=timestamp,
        ))
        search_start = end

    # Print stats.
    total_words = len(full_text.split())
    print(f"Total words: {total_words}  |  Segments created: {len(segments)}")

    return segments


if __name__ == "__main__":
    segments = load_transcript("data/transcripts/episode1.txt")
    for i, seg in enumerate(segments[:2]):
        print(f"\n--- Segment {i} ({seg.word_count} words, chars {seg.start_index}-{seg.end_index}) ---")
        print(seg.text)
