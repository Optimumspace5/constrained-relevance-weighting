# Constrained Relevance Weighting
### Personalized Podcast Summarization with Bounded Topic Proportions

A system that generates three variants of a podcast summary — generic, unconstrained, and constrained — and evaluates them against each other. The core idea is that users have different interests, and a good personalized summary should reflect those interests while remaining anchored to what the episode actually spent time on.

---

## Theoretical Foundation

Standard extractive and abstractive summarization treats all content as equally important. User-preference systems (like those explored in query-focused summarization literature) allow users to weight topics but impose no bound on how far the output can deviate from the source distribution. This creates a failure mode: a minor topic can completely dominate a summary just because a user rated it highly.

**Constrained Relevance Weighting (CRW)** addresses this by introducing a mathematical bound `delta` that limits how far any topic's proportion can shift from its baseline — regardless of the user's expressed preference. The result is a summary that is *personalised but honest*: it reflects user interest while remaining proportionally faithful to what the episode actually discussed.

---

## How the Pipeline Works

```
.txt transcript file
        │
        ▼
1. load_transcript        — strip timestamps & artifacts, chunk into ~250-word segments
        │
        ▼
2. segment_transcript     — discover 8 topics via Claude, classify every segment into a topic
        │
        ▼
3. get_preferences        — user rates each topic: high (1.5) / medium (1.0) / low (0.5)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
4a. generate_generic_summary        4b. generate_unconstrained_summary
    (mirrors episode proportions)        (preferences fully override proportions)
        │                                      │
        └──────────────┬───────────────────────┘
                       │
                       ▼
              4c. generate_constrained_summary
                   (preferences bounded by ±delta)
                       │
                       ▼
5. link_evidence      — classify each paragraph by topic, link to source segments
                       │
                       ▼
6. evaluate           — faithfulness (LLM-scored), coverage, relevance
```

---

## The Constraint Math

`calculate_constrained_proportions` in `src/summarizers.py` applies the following four-step formula:

**Step 1 — Raw target:**
```
raw_target = base_proportion × weight
```
Where `weight` is 1.5 (high), 1.0 (medium), or 0.5 (low).

**Step 2 — Clamp to delta bounds:**
```
clamped = max(base − delta, min(raw_target, base + delta))
```

**Step 3 — Enforce minimum floor:**
```
clamped = max(clamped, 0.01)
```
Every topic receives at least 1% coverage regardless of preference.

**Step 4 — Normalize to sum to 1.0:**
```
constrained_proportion[topic] = clamped[topic] / Σ clamped[all topics]
```

**Concrete example** (delta = 0.15):
| Topic | Base | Weight | Raw target | Clamped | After normalization |
|---|---|---|---|---|---|
| Topic A | 0.40 | 1.5 (high) | 0.60 | 0.55 | higher |
| Topic B | 0.10 | 0.5 (low) | 0.05 | 0.01 (floor) | lower |

Three delta values are defined in config for experimentation: `[0.10, 0.15, 0.20]`. The default is `0.15`.

---

## Project Structure

```
constrained-relevance-weighting/
├── app.py                   # Streamlit demo app
├── requirements.txt         # anthropic, streamlit, python-dotenv
├── .env.example             # ANTHROPIC_API_KEY=your_api_key_here
├── data/
│   └── episode1.txt         # Sample YouTube auto-generated transcript
└── src/
    ├── models.py            # Dataclasses: TranscriptSegment, Topic, UserPreference, SummarySegment, Summary
    ├── config.py            # Constants: NUM_TOPICS, CONSTRAINT_DELTAS, DEFAULT_DELTA, PREFERENCE_WEIGHTS, LLM_MODEL, MAX_SUMMARY_WORDS
    ├── loader.py            # Transcript loading and preprocessing
    ├── segmenter.py         # Topic discovery and segment classification via Claude API
    ├── preferences.py       # User preference collection (CLI and programmatic)
    ├── summarizers.py       # Three summary generators + calculate_constrained_proportions
    ├── evidence.py          # Paragraph-level evidence linking and report formatting
    └── evaluator.py         # Faithfulness, coverage, and relevance evaluation
```

---

## Data Models (`src/models.py`)

```python
@dataclass
class TranscriptSegment:
    text: str               # cleaned text of this ~250-word chunk
    start_index: int        # character position in the full transcript
    end_index: int          # character position end
    word_count: int

@dataclass
class Topic:
    name: str               # short label, e.g. "Childhood Trauma"
    description: str        # one-sentence explanation
    proportion: float       # fraction of transcript words (0.0–1.0)
    segment_indices: list[int]  # which TranscriptSegments belong to this topic

@dataclass
class UserPreference:
    topic_name: str
    weight: float           # 1.5=high, 1.0=medium, 0.5=low

@dataclass
class SummarySegment:
    text: str               # the summary text (paragraph or full)
    source_segment_indices: list[int]  # traceable back to TranscriptSegments
    topic_name: str

@dataclass
class Summary:
    segments: list[SummarySegment]
    summary_type: str       # "generic", "unconstrained", or "constrained"
    metadata: dict          # word_count, num_topics, delta, preferences, proportions
```

---

## Configuration (`src/config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `NUM_TOPICS` | 8 | Number of topics Claude extracts |
| `CONSTRAINT_DELTAS` | [0.10, 0.15, 0.20] | Experimental delta conditions |
| `DEFAULT_DELTA` | 0.15 | Default constraint bound |
| `PREFERENCE_WEIGHTS` | high=1.5, medium=1.0, low=0.5 | Numerical weights |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model used for all API calls |
| `MAX_SUMMARY_WORDS` | 800 | Target summary length passed to Claude |

---

## Setup

**1. Clone and install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Add your Anthropic API key:**
```bash
cp .env.example .env
# edit .env and replace with your real key:
# ANTHROPIC_API_KEY=sk-ant-...
```

**3. Add transcript files:**

Place YouTube auto-generated `.txt` transcripts in `data/`. The loader handles the YouTube timestamp format:
```
0:000 secondsI'm trying not to cry.
1:031 minute, 3 secondsBut it became a very useful distinction.
2:582 minutes, 58 secondsTony. I was shocked.
```
Chapter heading lines (e.g. `Chapter 2: A Stranger Changed My Life`) are automatically skipped.

---

## Usage

### Streamlit app (recommended)
```bash
streamlit run app.py
```

**App flow:**
1. Upload a `.txt` transcript in the sidebar
2. Click **Discover Topics** — calls the Claude API to identify 8 topics and classify all segments
3. Set your preference (high / medium / low) for each topic using the selectboxes
4. Adjust the **Constraint bound (delta)** slider (0.05–0.25, default 0.15)
5. Click **Generate Summaries** — generates all three variants in sequence
6. View the proportion comparison table and read each summary in the three tabs
7. Expand **Evaluation** to run faithfulness, coverage, and relevance scoring
8. Expand **Evidence report** to see each paragraph linked to its source transcript excerpts

### CLI — individual modules

Run any module directly to test its stage in isolation:
```bash
# Load and preview transcript
python -m src.loader

# Discover topics and classify segments
python -m src.segmenter

# Generate and compare all three summaries
python -m src.summarizers

# Generate evidence report for constrained summary
python -m src.evidence

# Run full evaluation with comparison table
python -m src.evaluator
```
Each module's `__main__` block loads `data/transcripts/episode1.txt` and uses hardcoded test preferences (first 2 topics high, next 3 medium, remaining low) to avoid interactive input.

---

## What Each File Actually Does

### `src/loader.py`
Reads a YouTube auto-generated transcript line by line. Strips timestamps using a compiled regex that matches the YouTube format (`M:SS{display_number} [minutes?, ]seconds`), removes sound artifacts (`[music]`, `[snorts]`, etc.), joins all lines into one continuous string, then splits into chunks of ~250 words at sentence boundaries. Returns `list[TranscriptSegment]` with character-position indices into the full text.

### `src/segmenter.py`
Two-stage Claude pipeline. `discover_topics` samples every other segment (first 80 words each) and asks Claude to identify exactly `NUM_TOPICS` topics as JSON. `classify_segments` processes segments in batches of 10, asking Claude to assign each segment a topic name; proportions are then computed as the fraction of total word count belonging to each topic. Returns `list[Topic]`.

### `src/preferences.py`
Two modes: `get_preferences_cli` prompts the user interactively in the terminal (with input validation and an Enter=medium default); `create_preferences` accepts a pre-built `dict[topic_name, rating_string]` for programmatic use (Streamlit passes ratings this way). Both return `list[UserPreference]`.

### `src/summarizers.py`
Three summary generators, each making one Claude API call with `max_tokens=2048`:
- **Generic**: shows Claude each topic's actual episode proportion; instructs proportional coverage with no user input.
- **Unconstrained**: shows Claude HIGH/MEDIUM/LOW labels; instructs preferences to fully override proportions. High-rated topics receive up to 5 sample segments; low-rated receive 1.
- **Constrained**: runs `calculate_constrained_proportions` first, then shows Claude both the original proportion and the bounded target; instructs coverage according to the target. Sample count per topic scales with constrained proportion (≥20% → 5 samples, ≥10% → 3, below → 1).

All three store the same full summary text across one `SummarySegment` per topic at generation time; fine-grained paragraph-level attribution is added later by `link_evidence`.

### `src/evidence.py`
`link_evidence` splits the summary on double newlines to get paragraphs, asks Claude in one API call to classify each numbered paragraph by topic, then rebuilds the `Summary` with one `SummarySegment` per paragraph — each carrying the topic's `segment_indices`. `format_evidence_report` renders a fixed-width text report showing each paragraph followed by up to 2 source excerpts (first 100 words each).

### `src/evaluator.py`
Three metrics:
- **Faithfulness** (`evaluate_faithfulness`): one Claude API call per paragraph. Provides up to 5 source transcript segments spread at first/25%/middle/75%/last positions across the topic (200 words each). Claude scores support 1–5 and describes any unsupported claims.
- **Coverage** (`evaluate_coverage`): counts how many of the discovered topics appear as `topic_name` in at least one `SummarySegment`. No API calls.
- **Relevance** (`evaluate_relevance`): `score = Σ(paragraph_count_for_topic × user_weight) / total_paragraphs`. Also checks per-topic alignment: whether high-preference topics received above-average paragraph counts and low-preference topics received below-average. No API calls.

---

## Known Limitations

**Topic assignment is single-label.** Each segment is assigned to exactly one topic. Segments that span multiple topics are attributed to whichever Claude chose, which can distort proportions.

**Proportion math is word-count-based.** Topic proportions reflect how many words were assigned to a topic, not how semantically central it was to the episode. A topic that appears briefly but intensely may be underweighted.

**Evidence linking is coarse.** `source_segment_indices` on a `SummarySegment` contains all segments for the matched topic — not just the segments that actually informed that specific paragraph. The link is a topic-level attribution, not a sentence-level one.

**Faithfulness evaluation uses the same model that generated the summary.** `claude-sonnet-4-20250514` evaluating its own output may exhibit self-consistency bias, scoring its own summaries more favourably than an independent judge would.

**The 800-word target is a soft instruction.** Claude treats `MAX_SUMMARY_WORDS` as a target, not a hard limit. Actual word counts vary.

**No speaker diarization.** The loader strips all timestamps but cannot distinguish between host and guest speech. Topic proportions reflect total transcript content, not per-speaker content.

**Transcript format is YouTube-specific.** The timestamp regex is designed for YouTube's auto-generated caption format. Other transcript formats (e.g. Descript, Rev, AssemblyAI) will not parse correctly without modification.

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-20250514`) via `anthropic` Python SDK |
| Web app | `streamlit` |
| Environment | `python-dotenv` |
| Data structures | Python `dataclasses` (stdlib) |
| Text processing | `re` (stdlib), `json` (stdlib) |

---

## Team

- Shyuan Rui
- Clarence Lee
- Selwyn Ray Oesjadi
- Benjamin Loo

---

## References

- Pirolli, P., & Card, S. (1999). Information foraging. *Psychological Review*, 106(4), 643–675.
- Maynez, J., et al. (2020). On faithfulness and factuality in abstractive summarization. *ACL 2020*.
- Wang, A., Cho, K., & Lewis, M. (2020). QAGS: Question answering for evaluating factual consistency. *ACL 2020*.
- Li, X., et al. (2024). PersonalSum: A dataset for personalized summarization. *NeurIPS 2024*.
- Zhang, Y., et al. (2024). MACSum: Attribute-controllable summarization dataset.
- Asimiyu, T. (2025). Bias in personalized summarization.
- Bjelvér, C., & Melander, P. (2025). AI-generated summaries and decision quality in microcontent contexts. Lund University.
