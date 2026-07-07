"""
Local NLI-based faithfulness judge.

This module replaces the *verification* half of the QAGS-style faithfulness
check in src/evaluator.py. Claim *extraction* still happens via the Claude API;
here we take each extracted claim and decide whether the source transcript
supports it — but instead of asking an LLM to judge "supported / unsupported",
we run a local Natural Language Inference (NLI) model.

NLI framing
-----------
An NLI model takes a (premise, hypothesis) pair and predicts one of three
relationships: ENTAILMENT, NEUTRAL, or CONTRADICTION.

We map faithfulness onto that framing:
    premise    = the source transcript evidence (what the podcast actually said)
    hypothesis = the atomic claim taken from the summary

A claim is "supported" when the premise *entails* the hypothesis with high
enough probability. This is stronger than lexical overlap (ROUGE) because the
model can recognise paraphrases and reject contradictions.

Why local?
----------
- Deterministic and reproducible (no sampling temperature, no API drift).
- No self-consistency bias: the judge is a different model family from the
  summariser, so it is not grading its own phrasing.
- Free and offline once the weights are cached.

IMPLEMENTATION STATUS: skeleton only. Every function below raises
NotImplementedError — the bodies are YOURS to write. The signatures here and
the tests in tests/test_nli_judge.py define the contract you must satisfy.
Do not import torch / transformers at module top level; load them inside
load_nli_model so this file imports cleanly before you've written that body.
"""

from __future__ import annotations

from src.models import TranscriptSegment


# ---------------------------------------------------------------------------
# Configuration (you own these values — they are decisions you must defend)
# ---------------------------------------------------------------------------

# Candidate models. Start with whichever you can justify on the accuracy/speed
# tradeoff; you can swap this without touching the rest of the module.
#   - "cross-encoder/nli-deberta-v3-base"                         (smaller, faster)
#   - "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"  (larger, stronger)
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# PLACEHOLDER — the value here is NOT an answer. Choosing and justifying this
# number is part of your task (see the checkpoint question your tutor asked).
DEFAULT_ENTAILMENT_THRESHOLD = 0.5

# Cap on how many words of premise you feed the model. NLI encoders have a hard
# max sequence length (often 512 tokens); an over-long premise gets silently
# truncated by the tokenizer, which can drop the very evidence you need.
DEFAULT_MAX_PREMISE_WORDS = 400


def load_nli_model(model_name: str = DEFAULT_NLI_MODEL):
    """
    Load the NLI model and return whatever object(s) score_entailment needs.

    You decide the return shape — e.g. a (tokenizer, model) tuple, a Hugging
    Face `pipeline("text-classification", ...)`, or a small wrapper object.
    Whatever you return here is passed straight back into score_entailment and
    verify_claim as their `model` argument, so keep the two consistent.

    Load this ONCE (it is expensive) and reuse it across every claim.

    Args:
        model_name: Hugging Face model id to load.

    Returns:
        An opaque handle consumed by score_entailment.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model
    raise NotImplementedError


def build_premise(
    segments: list[TranscriptSegment],
    source_indices: list[int],
    max_words: int = DEFAULT_MAX_PREMISE_WORDS,
) -> str:
    """
    Assemble the NLI *premise* from the linked source transcript segments.

    Contract (enforced by tests):
        - Concatenate the text of each segment referenced by `source_indices`,
          in the order given.
        - Silently skip indices that are out of range for `segments`.
        - If no valid segments remain, return "" (the empty string).
        - The returned premise must contain at most `max_words` words.

    Args:
        segments: the full list of transcript segments.
        source_indices: indices into `segments` that were linked to the claim.
        max_words: hard cap on premise length (see DEFAULT_MAX_PREMISE_WORDS).

    Returns:
        A single premise string ready to pass to the model.
    """
    parts = []
    for index in source_indices:
        if 0 <= index < len(segments):
            parts.append(segments[index].text)
    premise = ""
    for strings in parts:
        premise += strings + "\n"
    words = premise.split()
    capped = words[:max_words]
    premise = " ".join(capped)
    return premise


def score_entailment(premise: str, hypothesis: str, model) -> float:
    """
    Return P(premise entails hypothesis) as a float in [0.0, 1.0].

    This is the single forward pass through the NLI model: encode the
    (premise, hypothesis) pair, read the logits, softmax them, and return the
    probability mass on the ENTAILMENT class specifically.

    NOTE (a real trap): different checkpoints order their labels differently —
    some are [contradiction, neutral, entailment], others the reverse. Do NOT
    hard-code index 0. Read the model's `config.id2label` and select the
    entailment column by name. Getting this wrong silently inverts your judge.

    Args:
        premise: the source evidence (from build_premise).
        hypothesis: the atomic claim under test.
        model: the handle returned by load_nli_model.

    Returns:
        Entailment probability in [0.0, 1.0].
    """
    tokenizer, model = model
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=-1)
    entail_idx = None
    for i, label in model.config.id2label.items():
        if label.lower() == "entailment":
            entail_idx = i
    return probs[0][entail_idx].item()


    raise NotImplementedError


def is_supported(
    entailment_prob: float,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> bool:
    """
    Turn an entailment probability into a binary supported / unsupported verdict.

    Contract (enforced by tests): return True when `entailment_prob` is greater
    than or equal to `threshold`, else False.

    Args:
        entailment_prob: output of score_entailment.
        threshold: decision boundary (a value you must justify).

    Returns:
        True if the claim counts as supported.
    """
    return entailment_prob >= threshold
    raise NotImplementedError


def verify_claim(
    claim: str,
    segments: list[TranscriptSegment],
    source_indices: list[int],
    model,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> dict:
    """
    Verify a single claim end to end.

    Compose the pieces above: build the premise from `source_indices`, score
    entailment of the claim against it, and apply the threshold.

    Returns a dict with at least:
        {
            "claim": str,
            "entailment_prob": float,
            "supported": bool,
        }

    (You will need to decide what to do when there is no usable premise — that
    is a design question, not a given. Your tutor will ask about it in review.)
    """
    premise = build_premise(segments, source_indices)
    if not premise:
        return {"claim": claim, "entailment_prob": 0.0, "supported": False}
    entailment_prob = score_entailment(premise, claim, model)
    supported = is_supported(entailment_prob, threshold)
    return {
        "claim": claim,
        "entailment_prob": entailment_prob,
        "supported": supported,
    }
    raise NotImplementedError


def verify_claims(
    claims: list[dict],
    segments: list[TranscriptSegment],
    model,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> dict:
    """
    Verify many claims and aggregate the results (the per-claim loop is YOURS).

    Args:
        claims: list of dicts, each shaped like
            {"claim": str, "source_indices": list[int]}
            (add "topic_name" if you want per-topic aggregation later).
        segments: full transcript segments.
        model: handle from load_nli_model.
        threshold: decision boundary passed through to each claim.

    Returns a dict with at least:
        {
            "total_claims": int,
            "supported_claims": int,
            "precision": float,          # supported / total, 0.0 if no claims
            "results": list[dict],       # one verify_claim dict per claim
        }
    """
    results = []
    for c in claims:
        result = verify_claim(c["claim"], segments, c["source_indices"], model, threshold)
        results.append(result)
    total_claims = len(results)
    supported_claims = sum(1 for r in results if r["supported"])
    precision = supported_claims / total_claims if total_claims > 0 else 0.0
    return {
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "precision": precision,
        "results": results,
    }


    raise NotImplementedError
