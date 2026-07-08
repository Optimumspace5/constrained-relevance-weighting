# Constrained Relevance Weighting
### Personalized Podcast Summarization with Bounded Topic Proportions and an Independent Faithfulness Judge

A system that generates several variants of a podcast summary — a naive extractive baseline plus generic, unconstrained, and constrained LLM summaries — and evaluates them against each other. The core idea: users have different interests, and a good personalized summary should reflect those interests **while remaining anchored to what the episode actually spent time on**. Faithfulness is scored by an **independent, local NLI model** — not by the model that wrote the summary.

> **Status:** the system and evaluation harness are complete and tested; the evaluation *matrix has not yet been run*. Every quantitative result in this document is therefore marked `[RESULT PENDING]`. No numbers here are invented.

---

## Theoretical Foundation

Standard summarization treats all content as equally important. User-preference systems (as in query-focused summarization) let users weight topics but impose no bound on how far the output can deviate from the source distribution. This creates a failure mode: a minor topic can dominate a summary just because a user rated it highly.

**Constrained Relevance Weighting (CRW)** introduces a mathematical bound `delta` that limits how far any topic's proportion can shift from its baseline — regardless of the user's expressed preference. The result is a summary that is *personalized but honest*: it reflects user interest while remaining proportionally faithful to what the episode actually discussed.

---

## The Independent NLI Faithfulness Judge

A personalized summarizer is only useful if its summaries are *faithful* — every claim traceable to the source. The weak point of LLM-as-judge faithfulness scoring is **self-consistency bias**: asking the same model family that wrote the summary to grade it tends to inflate the score.

This project verifies faithfulness with a **separate, local Natural Language Inference (NLI) model** (`src/nli_judge.py`), decoupled from the generator:

- **Claim extraction (LLM, deterministic).** Atomic factual claims are extracted from each summary paragraph via the API at `temperature=0`.
- **Claim verification (local NLI).** Each claim (the *hypothesis*) is checked against the source segments already linked to its paragraph (the *premise*) using a cross-encoder NLI model (`cross-encoder/nli-deberta-v3-base`). A claim is **supported** only if the premise *entails* it with probability ≥ a chosen threshold.

Why this is stronger than LLM-as-judge:

- **Independent** — a different model family from the generator, so it is not grading its own phrasing.
- **Deterministic & reproducible** — no sampling temperature, no API drift.
- **Semantic, not lexical** — recognizes paraphrases that ROUGE misses, and rejects contradictions that word overlap would accept.

The operating **entailment threshold is calibrated by a human against hand labels** using the validation harness (`experiments/validate_judge.py`), which reports agreement / false-positive / false-negative rates across a threshold sweep. The chosen threshold is `[PENDING VALIDATION]`; the code default is a placeholder.

The two legacy LLM-based faithfulness scorers are retained **as comparison baselines only** (clearly marked deprecated): `evaluate_faithfulness` (1–5 LLM-as-judge) and `evaluate_faithfulness_qa_llm` (same claims + same evidence, but an LLM verifier). The latter exists so the NLI judge can be compared head-to-head against an LLM verifier where the *verifier type is the only variable* — the direct test of whether independence changes the score.

---

## How the Pipeline Works

```
.txt transcript file
        │
        ▼
1. load_transcript        — strip timestamps & artifacts, chunk into ~250-word segments
        │
        ▼
2. segment_transcript     — discover topics + classify segments via the API (deterministic,
                            temperature=0); result cached to disk by transcript hash
        │
        ▼
3. create_preferences     — user rates each topic: high (1.5) / medium (1.0) / low (0.5)
        │
        ├───────────────┬───────────────────────┬───────────────────────┐
        ▼               ▼                       ▼                       ▼
4a. baseline     4b. generic           4c. unconstrained        4d. constrained
   (extractive,     (mirrors episode      (preferences fully       (preferences bounded
    no LLM)          proportions)          override proportions)    by ±delta)     ← the 3
        │               │                       │                       │            API
        └───────────────┴───────────┬───────────┴───────────────────────┘        generators
                                     │                                          run concurrently
                                     ▼
5. link_evidence          — attribute each paragraph to a topic and to its source segments
   (LLM or TF-IDF top-k)     (this linked set is the NLI judge's premise)
                                     │
                                     ▼
6. evaluate               — NLI faithfulness (primary) + LLM baselines, ROUGE,
                            extractive overlap, coverage, relevance, matched-topic precision
```

---

## The Constraint Math

`calculate_constrained_proportions` in `src/summarizers.py` applies:

**Step 1 — Raw target:** `raw_target = base_proportion × weight` (weight = 1.5 / 1.0 / 0.5).

**Step 2 — Clamp to delta bounds:** `clamped = max(base − delta, min(raw_target, base + delta))`.

**Step 3 — Enforce floor:** `clamped = max(clamped, 0.01)` — every topic keeps at least 1%.

**Step 4 — Normalize to 1.0** via iterative projection that re-clamps after each rescale, so normalization cannot push any topic outside its `±delta` bound (verified by `tests/test_proportions.py`).

Config deltas: `CONSTRAINT_DELTAS = [0.10, 0.15, 0.20]`, default `0.15`. The delta sweep experiment scans `[0.05, 0.10, 0.15, 0.20, 0.25]`.

---

## Evaluation Metrics

| Metric | What it measures | LLM? |
|---|---|---|
| **NLI faithfulness** (primary) | Fraction of extracted claims entailed by their linked source; per-topic precision, `matched_topic_precision` (restricted to topics all variants cover), and the list of unsupported claims | Extraction only; verification is local |
| LLM-verify baseline (optional) | Same claims + evidence, LLM verdict — for the independence comparison | Yes (pinned to the generation model) |
| LLM-as-judge 1–5 (deprecated) | Legacy subjective faithfulness score | Yes (pinned to the generation model) |
| ROUGE-1 / 2 / L | n-gram / LCS overlap with the source | No |
| Extractive overlap | Fraction of summary unigrams/bigrams found in the source | No |
| Coverage | How many discovered topics appear with substantive content | No |
| Relevance | Preference-weighted topic-proportion score + proportion MAE vs. target | No |

---

## Results

Not yet produced — the evaluation matrix has not been run.

| Summary variant | NLI faithfulness | Matched-topic precision | ROUGE-L | Extractive overlap | Coverage | Relevance | Proportion MAE |
|---|---|---|---|---|---|---|---|
| baseline | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` |
| generic | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` |
| unconstrained | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` |
| constrained | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` | `[RESULT PENDING]` |

- NLI judge ↔ human-label agreement: `[RESULT PENDING]` · chosen entailment threshold: `[PENDING VALIDATION]`
- NLI vs. LLM verifier (independence check): `[RESULT PENDING]`
- Delta sweep (mean ± std across runs): `[RESULT PENDING]`

---

## Project Structure

```
constrained-relevance-weighting/
├── app.py                       # Streamlit demo app
├── requirements.txt
├── .env                         # ANTHROPIC_API_KEY=... (gitignored)
├── cache/                       # disk cache for segmentation (gitignored)
├── data/transcripts/
│   ├── episode1.txt             # YouTube auto-generated transcripts (3 episodes)
│   ├── episode2.txt
│   └── episode3.txt
├── src/
│   ├── models.py                # Dataclasses: TranscriptSegment, Topic, UserPreference, SummarySegment, Summary
│   ├── config.py                # GENERATION_MODEL / BULK_MODEL (env), API_CONCURRENCY, deltas, weights, word band
│   ├── loader.py                # Transcript loading, timestamp stripping, chunking
│   ├── segmenter.py             # Topic discovery + segment classification (cached, deterministic)
│   ├── preferences.py           # Preference collection (CLI + programmatic)
│   ├── summarizers.py           # Baseline + 3 generators + calculate_constrained_proportions + generate_all_summaries
│   ├── evidence.py              # Evidence linking: link_evidence (LLM) and link_evidence_tfidf (deterministic)
│   ├── profiles.py              # Preset preference patterns for experiments
│   ├── evaluator.py             # NLI faithfulness + LLM baselines + ROUGE + overlap + coverage + relevance
│   └── nli_judge.py             # Independent local NLI faithfulness judge
├── experiments/
│   ├── delta_sweep.py           # Delta sweep experiment (--runs N → per-config mean ± std)
│   └── validate_judge.py        # Human-in-the-loop judge validation harness (build → label → score)
└── tests/
    ├── test_proportions.py      # Constraint-math unit tests (no API)
    └── test_nli_judge.py        # NLI judge unit + semantic tests
```

---

## Data Models (`src/models.py`)

```python
@dataclass
class TranscriptSegment:
    text: str
    start_index: int          # character position in the full transcript
    end_index: int
    word_count: int
    timestamp: str = ""       # source clock timestamp, if available

@dataclass
class Topic:
    name: str
    description: str
    proportion: float         # fraction of transcript words (0.0–1.0)
    segment_indices: list[int]

@dataclass
class UserPreference:
    topic_name: str
    weight: float             # 1.5=high, 1.0=medium, 0.5=low

@dataclass
class SummarySegment:
    text: str
    source_segment_indices: list[int]   # traceable back to TranscriptSegments (the judge premise)
    topic_name: str

@dataclass
class Summary:
    segments: list[SummarySegment]
    summary_type: str         # "baseline" | "generic" | "unconstrained" | "constrained"
    metadata: dict
```

---

## Configuration (`src/config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `NUM_TOPICS` | 8 | Topics extracted per episode |
| `CONSTRAINT_DELTAS` / `DEFAULT_DELTA` | [0.10, 0.15, 0.20] / 0.15 | Constraint bounds |
| `PREFERENCE_WEIGHTS` | high=1.5, medium=1.0, low=0.5 | Numerical weights |
| `GENERATION_MODEL` | env, default `claude-sonnet-4-6` | Creative summary generation + LLM judge baselines |
| `BULK_MODEL` | env, default `claude-sonnet-4-6` | Mechanical calls: discovery, classification, linking, claim extraction |
| `API_CONCURRENCY` | env, default 4 | Max concurrent API calls when parallelized |
| `MIN` / `MAX` / `CEIL` summary words | 800 / 900 / 1000 | Target length band (all variants aim for the 800–1000 word band for a fair comparison) |

The generation/bulk split lets mechanical calls run on a cheaper model (e.g. `BULK_MODEL=claude-haiku-4-5`) while generation and the judge baselines stay on the stronger model. Both default to `claude-sonnet-4-6`, which accepts the `temperature=0` used throughout for determinism — note that Claude Sonnet 5 / Opus 4.7+ reject `temperature`, so moving to them requires dropping those arguments first.

The NLI judge (`src/nli_judge.py`) defaults to `cross-encoder/nli-deberta-v3-base`; `DEFAULT_NLI_MODEL`, the entailment threshold, and the premise word cap are defined there.

---

## Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```
This includes `transformers`, `torch`, and `sentencepiece` for the local NLI judge. The NLI model weights (~a few hundred MB) download automatically on first use and are cached thereafter.

**2. Add your Anthropic API key:**
```bash
# create .env with:
# ANTHROPIC_API_KEY=sk-ant-...
# optional overrides: GENERATION_MODEL=..., BULK_MODEL=..., API_CONCURRENCY=4
```

**3. Add transcripts:** place YouTube auto-generated `.txt` transcripts in `data/transcripts/`. The loader handles the YouTube timestamp format and skips chapter-heading lines.

---

## Usage

### Streamlit app
```bash
streamlit run app.py
```
Upload a transcript → **Discover Topics** (cached; tick "Force re-discovery" to bypass) → set preferences → **Generate Summaries** (the three API variants run concurrently) → view proportions, summaries, evaluation, and the evidence report.

### Delta sweep experiment
```bash
python -m experiments.delta_sweep --all-episodes --qags --runs 3
```
Repeats each configuration `--runs` times, tags rows with `run_id`, and writes a raw CSV plus a `_agg.csv` of per-config mean ± std. Flags: `--transcript`, `--profile`, `--faithfulness`, `--qags`, `--output-dir`.

### Judge validation harness
```bash
# 1. build an UNLABELED sheet of ~40 claims (with their linked premise segments)
python -m experiments.validate_judge build --n 40 --out experiments/validation/claims.csv
#    ...fill the my_label column with 'supported' / 'unsupported' by hand...
# 2. score your labels with the judge and report agreement + threshold sweep
python -m experiments.validate_judge score --in experiments/validation/claims.csv
```
The harness never assigns labels and never picks a threshold — it reports the sweep; you choose. `--link {tfidf,llm}` selects the premise linker (default `tfidf` for reproducible sheets).

### Individual modules
```bash
python -m src.loader        # load and preview a transcript
python -m src.segmenter     # discover topics + classify
python -m src.summarizers   # generate and compare summaries
python -m src.evidence      # evidence report for the constrained summary
python -m src.evaluator     # full evaluation comparison table
```

### Tests
```bash
pytest tests/                       # full suite
pytest tests/test_nli_judge.py -m "not nli"   # skip the model-download semantic tests
```

---

## What Each File Actually Does

- **`src/loader.py`** — parses YouTube transcripts (regex timestamp/artifact stripping), joins lines, chunks into ~250-word segments at sentence boundaries; returns `list[TranscriptSegment]`.
- **`src/segmenter.py`** — `discover_topics` + `classify_segments` (both `BULK_MODEL`, `temperature=0`); `segment_transcript` caches the result to `cache/` keyed by transcript hash, with a `force_refresh` bypass.
- **`src/preferences.py`** — interactive CLI and programmatic preference collection.
- **`src/summarizers.py`** — the extractive baseline, three API generators (`GENERATION_MODEL`), `calculate_constrained_proportions`, and `generate_all_summaries` (runs the three generators concurrently).
- **`src/evidence.py`** — `link_evidence` (LLM paragraph→segment linking) and `link_evidence_tfidf` (deterministic TF-IDF top-k, no API); `format_evidence_report` for human inspection.
- **`src/profiles.py`** — preset preference patterns (`skewed_high`, `skewed_low`, `balanced`, `alternating`, `one_dominant`, `inverse`).
- **`src/nli_judge.py`** — the independent NLI judge: premise construction, entailment scoring (softmax over the model's `id2label`), threshold decision, and per-claim verification. Fully unit-tested.
- **`src/evaluator.py`** — `evaluate_faithfulness_qa` (claim extraction + NLI verification, the primary metric), the two deprecated LLM baselines, ROUGE, extractive overlap, coverage, relevance, and `run_full_evaluation` (which adds `matched_topic_precision`).
- **`experiments/delta_sweep.py`** — the delta sweep with multi-run mean/std aggregation.
- **`experiments/validate_judge.py`** — the human-in-the-loop judge validation harness.

---

## Known Limitations

- **Faithfulness verification is independent, but extraction is not.** Claim *extraction* still uses the LLM; only *verification* is independent. Extraction is deterministic (`temperature=0`) but shares the generator's model family.
- **Narrow premise.** The judge verifies each claim against only the segments linked to its paragraph. This is the correct regime for a cross-encoder NLI model (long premises dilute the entailment signal and risk silent truncation), but it can raise false negatives when a claim's support is spread across unlinked segments. Since false negatives are the safe error for a faithfulness metric, this is an acceptable, defensible trade — but it should be spot-checked during validation.
- **Validation vs. production linking.** The validation harness defaults to deterministic TF-IDF linking for reproducible ground truth, while the app/experiment path links via the LLM. A threshold calibrated on TF-IDF premises transfers to the LLM path only insofar as the two linkers surface similar evidence — a cheap spot-check is recommended before trusting transfer.
- **Single-label topic assignment; word-count-based proportions.** Each segment is assigned to one topic, and proportions reflect word counts, not semantic centrality.
- **Soft word band.** The 800–1000 word band is a prompt instruction, not a hard limit; actual counts vary.
- **YouTube-specific transcript format** and **no speaker diarization.**

---

## Tech Stack

| Component | Library |
|---|---|
| Generation & LLM-judge baselines | Anthropic Claude via `anthropic` |
| Faithfulness judge | `transformers` + `torch` (`cross-encoder/nli-deberta-v3-base`), `sentencepiece` tokenizer |
| Non-LLM metrics | `rouge-score`, `scikit-learn` (TF-IDF) |
| Web app | `streamlit` |
| Environment / tests | `python-dotenv`, `pytest` |
| Data structures | Python `dataclasses` (stdlib) |

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
