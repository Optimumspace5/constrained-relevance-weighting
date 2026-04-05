"""
Predefined preference profiles for testing and experimentation.

Each profile is a function that takes a list of Topic objects and returns
a ratings dict (topic_name → "high"/"medium"/"low") suitable for passing
to create_preferences().

These profiles let us systematically test how CRW behaves under different
user preference patterns without manually configuring ratings each time.
"""

from src.models import Topic


# ---------------------------------------------------------------------------
# Profile functions
# ---------------------------------------------------------------------------

def profile_skewed_high(topics: list[Topic]) -> dict[str, str]:
    """First 2 topics high, rest low — maximum personalization pressure."""
    return {t.name: "high" if i < 2 else "low" for i, t in enumerate(topics)}


def profile_skewed_low(topics: list[Topic]) -> dict[str, str]:
    """Last 2 topics high, rest low — tests whether position matters."""
    n = len(topics)
    return {t.name: "high" if i >= n - 2 else "low" for i, t in enumerate(topics)}


def profile_balanced(topics: list[Topic]) -> dict[str, str]:
    """All medium — should preserve original proportions exactly."""
    return {t.name: "medium" for t in topics}


def profile_alternating(topics: list[Topic]) -> dict[str, str]:
    """Alternates high/low — stress-tests the clamping across many topics."""
    return {t.name: "high" if i % 2 == 0 else "low" for i, t in enumerate(topics)}


def profile_one_dominant(topics: list[Topic]) -> dict[str, str]:
    """Only the largest topic is high, everything else low — extreme focus."""
    largest = max(topics, key=lambda t: t.proportion)
    return {t.name: "high" if t.name == largest.name else "low" for t in topics}


def profile_inverse(topics: list[Topic]) -> dict[str, str]:
    """
    Invert the episode's natural balance: small topics get high, large get low.
    Tests CRW's ability to resist extreme rebalancing via the delta constraint.
    """
    sorted_by_prop = sorted(topics, key=lambda t: t.proportion)
    n = len(sorted_by_prop)
    third = max(n // 3, 1)
    small_names = {t.name for t in sorted_by_prop[:third]}
    large_names = {t.name for t in sorted_by_prop[-third:]}
    return {
        t.name: "high" if t.name in small_names else "low" if t.name in large_names else "medium"
        for t in topics
    }


# ---------------------------------------------------------------------------
# Registry — maps profile name to function for easy lookup
# ---------------------------------------------------------------------------

PROFILES: dict[str, callable] = {
    "skewed_high": profile_skewed_high,
    "skewed_low": profile_skewed_low,
    "balanced": profile_balanced,
    "alternating": profile_alternating,
    "one_dominant": profile_one_dominant,
    "inverse": profile_inverse,
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "skewed_high": "First 2 topics high, rest low",
    "skewed_low": "Last 2 topics high, rest low",
    "balanced": "All topics medium (baseline control)",
    "alternating": "Alternates high/low across topics",
    "one_dominant": "Only the largest topic is high",
    "inverse": "Small topics high, large topics low",
}
