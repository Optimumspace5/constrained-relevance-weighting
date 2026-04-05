"""
Unit tests for calculate_constrained_proportions — the mathematical core of CRW.

These tests verify clamping, floor enforcement, normalization, and edge cases
without any API calls. All tests are deterministic and fast.
"""

import pytest
from src.models import Topic, UserPreference
from src.summarizers import calculate_constrained_proportions


# ---------------------------------------------------------------------------
# Helper: build Topic and UserPreference objects for testing
# ---------------------------------------------------------------------------

def make_topics(proportions: dict[str, float]) -> list[Topic]:
    """Create Topic objects from a {name: proportion} dict."""
    return [
        Topic(name=name, description=f"About {name}", proportion=prop, segment_indices=[])
        for name, prop in proportions.items()
    ]


def make_preferences(weights: dict[str, float]) -> list[UserPreference]:
    """Create UserPreference objects from a {name: weight} dict."""
    return [
        UserPreference(topic_name=name, weight=weight)
        for name, weight in weights.items()
    ]


# ---------------------------------------------------------------------------
# 1. Output always sums to 1.0
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_all_medium_sums_to_one(self):
        topics = make_topics({"A": 0.4, "B": 0.35, "C": 0.25})
        prefs = make_preferences({"A": 1.0, "B": 1.0, "C": 1.0})
        result = calculate_constrained_proportions(topics, prefs, delta=0.15)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_mixed_preferences_sums_to_one(self):
        topics = make_topics({"A": 0.4, "B": 0.35, "C": 0.25})
        prefs = make_preferences({"A": 1.5, "B": 1.0, "C": 0.5})
        result = calculate_constrained_proportions(topics, prefs, delta=0.15)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_all_high_sums_to_one(self):
        topics = make_topics({"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2})
        prefs = make_preferences({"A": 1.5, "B": 1.5, "C": 1.5, "D": 1.5})
        result = calculate_constrained_proportions(topics, prefs, delta=0.10)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_all_low_sums_to_one(self):
        topics = make_topics({"A": 0.5, "B": 0.3, "C": 0.2})
        prefs = make_preferences({"A": 0.5, "B": 0.5, "C": 0.5})
        result = calculate_constrained_proportions(topics, prefs, delta=0.10)
        assert abs(sum(result.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. Clamping: raw target is bounded to [base - delta, base + delta]
# ---------------------------------------------------------------------------

class TestClamping:
    def test_high_preference_clamped_upward(self):
        """A high weight on a large topic should be clamped at base + delta."""
        topics = make_topics({"A": 0.50, "B": 0.50})
        prefs = make_preferences({"A": 1.5, "B": 1.0})
        delta = 0.10
        result = calculate_constrained_proportions(topics, prefs, delta)
        # A: base=0.50, raw=0.75, clamped to 0.60. B: base=0.50, raw=0.50.
        # After normalization: A=0.60/1.10, B=0.50/1.10 — A should be > B.
        assert result["A"] > result["B"]

    def test_low_preference_clamped_downward(self):
        """A low weight should push the topic down, but not below floor."""
        topics = make_topics({"A": 0.20, "B": 0.80})
        prefs = make_preferences({"A": 0.5, "B": 1.0})
        delta = 0.10
        result = calculate_constrained_proportions(topics, prefs, delta)
        # A: base=0.20, raw=0.10, clamped to max(0.10, 0.01)=0.10. A should be smaller.
        assert result["A"] < result["B"]

    def test_medium_preference_stays_near_base(self):
        """Weight 1.0 means raw = base, so no clamping needed."""
        topics = make_topics({"A": 0.40, "B": 0.60})
        prefs = make_preferences({"A": 1.0, "B": 1.0})
        result = calculate_constrained_proportions(topics, prefs, delta=0.15)
        # With all-medium weights, proportions should stay exactly at base after normalization.
        assert abs(result["A"] - 0.40) < 1e-9
        assert abs(result["B"] - 0.60) < 1e-9


# ---------------------------------------------------------------------------
# 3. Floor: every topic gets at least 0.01 (before normalization)
# ---------------------------------------------------------------------------

class TestFloor:
    def test_small_topic_with_low_pref_gets_floor(self):
        """A tiny topic with low preference should hit the 0.01 floor."""
        topics = make_topics({"A": 0.05, "B": 0.95})
        prefs = make_preferences({"A": 0.5, "B": 1.0})
        delta = 0.10
        result = calculate_constrained_proportions(topics, prefs, delta)
        # A: base=0.05, raw=0.025, clamped to max(0.05-0.10, 0.025)=max(-0.05, 0.025)=0.025
        # then floor: max(0.025, 0.01) = 0.025. So A should be > 0 after normalization.
        assert result["A"] > 0

    def test_all_topics_have_positive_proportion(self):
        """Even the smallest topics should have a non-zero proportion."""
        topics = make_topics({"A": 0.01, "B": 0.02, "C": 0.97})
        prefs = make_preferences({"A": 0.5, "B": 0.5, "C": 1.5})
        result = calculate_constrained_proportions(topics, prefs, delta=0.15)
        for name in ["A", "B", "C"]:
            assert result[name] > 0


# ---------------------------------------------------------------------------
# 4. Delta sensitivity: larger delta allows more deviation
# ---------------------------------------------------------------------------

class TestDeltaSensitivity:
    def test_larger_delta_allows_more_shift(self):
        """With a bigger delta, a high-preference topic should get a larger share."""
        topics = make_topics({"A": 0.30, "B": 0.70})
        prefs = make_preferences({"A": 1.5, "B": 0.5})

        small_delta = calculate_constrained_proportions(topics, prefs, delta=0.05)
        large_delta = calculate_constrained_proportions(topics, prefs, delta=0.25)

        # A should get a bigger share with a larger delta.
        assert large_delta["A"] > small_delta["A"]

    def test_zero_delta_preserves_original(self):
        """With delta=0, clamping forces raw back to base, so proportions stay original."""
        topics = make_topics({"A": 0.40, "B": 0.35, "C": 0.25})
        prefs = make_preferences({"A": 1.5, "B": 1.0, "C": 0.5})
        result = calculate_constrained_proportions(topics, prefs, delta=0.0)
        # With delta=0: clamp range is [base, base], so every topic stays at base.
        # After normalization they should match original proportions.
        assert abs(result["A"] - 0.40) < 1e-9
        assert abs(result["B"] - 0.35) < 1e-9
        assert abs(result["C"] - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_topic(self):
        """A single topic should always get 100% regardless of weight or delta."""
        topics = make_topics({"Only": 1.0})
        prefs = make_preferences({"Only": 0.5})
        result = calculate_constrained_proportions(topics, prefs, delta=0.15)
        assert abs(result["Only"] - 1.0) < 1e-9

    def test_many_topics_equal_proportions(self):
        """8 equal topics with mixed preferences should still sum to 1.0."""
        names = [f"Topic{i}" for i in range(8)]
        topics = make_topics({n: 0.125 for n in names})
        weights = {n: [1.5, 1.0, 0.5][i % 3] for i, n in enumerate(names)}
        prefs = make_preferences(weights)
        result = calculate_constrained_proportions(topics, prefs, delta=0.10)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        # High-weight topics should have higher proportions than low-weight topics.
        high_topics = [n for n, w in weights.items() if w == 1.5]
        low_topics = [n for n, w in weights.items() if w == 0.5]
        for h in high_topics:
            for l in low_topics:
                assert result[h] >= result[l]

    def test_missing_preference_defaults_to_medium(self):
        """Topics not in the preference list should be treated as weight=1.0."""
        topics = make_topics({"A": 0.50, "B": 0.50})
        prefs = make_preferences({"A": 1.5})   # B has no preference
        result = calculate_constrained_proportions(topics, prefs, delta=0.15)
        # A gets boosted, B stays at base → A should be larger.
        assert result["A"] > result["B"]
