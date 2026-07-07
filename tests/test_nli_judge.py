"""
Contract tests for the local NLI faithfulness judge (src/nli_judge.py).

Two tiers:
  1. Pure-logic tests (build_premise, is_supported) — fast, no model download.
     These fail with NotImplementedError until you write those two functions.
  2. Semantic tests (verify_claim) — require torch + transformers and will
     download the NLI weights on first run. These are the tests that actually
     prove your judge distinguishes a supported claim, a contradiction, and a
     paraphrase. They are marked `nli`.

Run everything:            pytest tests/test_nli_judge.py
Skip the slow model tests: pytest tests/test_nli_judge.py -m "not nli"
Run only the model tests:  pytest tests/test_nli_judge.py -m nli
"""

import pytest

from src.models import TranscriptSegment
from src import nli_judge


def seg(text: str) -> TranscriptSegment:
    """Minimal TranscriptSegment for tests — only .text matters to the judge."""
    words = text.split()
    return TranscriptSegment(
        text=text, start_index=0, end_index=len(text), word_count=len(words)
    )


# ---------------------------------------------------------------------------
# Tier 1: pure-logic tests (no model needed)
# ---------------------------------------------------------------------------

class TestBuildPremise:
    def test_empty_indices_returns_empty_string(self):
        segments = [seg("anything at all")]
        assert nli_judge.build_premise(segments, []) == ""

    def test_includes_text_from_each_referenced_segment(self):
        segments = [seg("alpha one"), seg("beta two"), seg("gamma three")]
        premise = nli_judge.build_premise(segments, [0, 2])
        assert "alpha" in premise
        assert "gamma" in premise
        assert "beta" not in premise   # index 1 was not requested

    def test_out_of_range_indices_are_skipped(self):
        segments = [seg("only segment")]
        # Index 5 does not exist; it must be skipped, not raise.
        premise = nli_judge.build_premise(segments, [0, 5])
        assert "only segment" in premise

    def test_respects_max_words(self):
        long_text = " ".join(f"w{i}" for i in range(1000))
        segments = [seg(long_text)]
        premise = nli_judge.build_premise(segments, [0], max_words=10)
        assert len(premise.split()) <= 10


class TestIsSupported:
    def test_above_threshold_is_supported(self):
        assert nli_judge.is_supported(0.92, threshold=0.5) is True

    def test_below_threshold_is_unsupported(self):
        assert nli_judge.is_supported(0.20, threshold=0.5) is False

    def test_exactly_at_threshold_is_supported(self):
        # Boundary: >= threshold counts as supported.
        assert nli_judge.is_supported(0.5, threshold=0.5) is True


# ---------------------------------------------------------------------------
# Tier 2: semantic tests (real NLI model; slow, marked `nli`)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    """Load the NLI model once for the whole test session."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return nli_judge.load_nli_model()


# Premise reused across the supported / unsupported cases.
_FACTS = (
    "The host explained that the startup was founded in 2010 in Berlin "
    "and today employs more than five hundred people."
)


@pytest.mark.nli
class TestSemantics:
    def test_obviously_supported_claim(self, model):
        segments = [seg(_FACTS)]
        result = nli_judge.verify_claim(
            claim="The startup was founded in 2010.",
            segments=segments,
            source_indices=[0],
            model=model,
        )
        assert 0.0 <= result["entailment_prob"] <= 1.0
        assert result["supported"] is True

    def test_obviously_unsupported_claim(self, model):
        segments = [seg(_FACTS)]
        result = nli_judge.verify_claim(
            claim="The startup is headquartered in Tokyo.",
            segments=segments,
            source_indices=[0],
            model=model,
        )
        assert result["supported"] is False

    def test_paraphrase_is_supported(self, model):
        # Low lexical overlap, high semantic entailment — the case that
        # ROUGE / word-overlap gets wrong and NLI should get right.
        segments = [seg(
            "She said sales roughly doubled last year compared with the year before."
        )]
        result = nli_judge.verify_claim(
            claim="The company's revenue grew substantially year over year.",
            segments=segments,
            source_indices=[0],
            model=model,
        )
        assert result["supported"] is True
