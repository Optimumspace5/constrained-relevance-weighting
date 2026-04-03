from src.models import UserPreference, Topic
from src.config import PREFERENCE_WEIGHTS


# PREFERENCE_WEIGHTS from config maps rating strings to numerical weights:
# {"high": 1.5, "medium": 1.0, "low": 0.5}
# A weight of 1.0 means "keep as-is". Above 1.0 = expand. Below 1.0 = compress.

VALID_RATINGS = set(PREFERENCE_WEIGHTS.keys())  # {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# 1. get_preferences_cli
# ---------------------------------------------------------------------------

def get_preferences_cli(topics: list[Topic]) -> list[UserPreference]:
    """
    Interactive terminal prompt — asks the user to rate each topic.

    Shows each topic's name, description, and how much of the transcript it
    covers, then asks for a high/medium/low rating. Loops until valid input
    is received. Defaults to "medium" on empty Enter.

    Returns a list of UserPreference objects with numerical weights attached.
    """
    print("\n--- Rate each topic ---")
    print("How much detail do you want on each topic in your summary?")
    print("  high   = more detail  (weight 1.5)")
    print("  medium = keep as-is  (weight 1.0)  [default]")
    print("  low    = less detail  (weight 0.5)\n")

    preferences = []

    for topic in topics:
        # Show the user what this topic is about before asking.
        print(f"Topic:       {topic.name}")
        print(f"Description: {topic.description}")
        print(f"Coverage:    {topic.proportion:.1%} of transcript")

        # Keep asking until we get a valid rating.
        while True:
            raw = input("Your rating [high / medium / low] (Enter = medium): ").strip().lower()

            # Empty input → default to medium.
            if raw == "":
                rating = "medium"
                break

            if raw in VALID_RATINGS:
                rating = raw
                break

            # Invalid input — tell the user and loop again.
            print(f"  Invalid input '{raw}'. Please type high, medium, or low.")

        weight = PREFERENCE_WEIGHTS[rating]
        preferences.append(UserPreference(topic_name=topic.name, weight=weight))
        print(f"  Saved: {topic.name} → {rating} ({weight})\n")

    return preferences


# ---------------------------------------------------------------------------
# 2. create_preferences
# ---------------------------------------------------------------------------

def create_preferences(topics: list[Topic], ratings: dict[str, str]) -> list[UserPreference]:
    """
    Programmatic version of preference collection — no terminal interaction.

    Used by Streamlit (or any other caller) that already has the ratings as a
    dict. Any topic not present in the dict defaults to "medium".

    Args:
        topics:  the list of Topic objects from segmenter
        ratings: dict mapping topic name → rating string, e.g.
                 {"Childhood Trauma": "high", "Gratitude": "low"}

    Returns a list of UserPreference objects.
    """
    preferences = []

    for topic in topics:
        # Look up the rating for this topic; fall back to "medium" if not provided.
        rating = ratings.get(topic.name, "medium")

        # Guard against invalid values coming in from the caller.
        if rating not in VALID_RATINGS:
            print(f"  Warning: invalid rating '{rating}' for '{topic.name}', defaulting to medium.")
            rating = "medium"

        weight = PREFERENCE_WEIGHTS[rating]
        preferences.append(UserPreference(topic_name=topic.name, weight=weight))

    return preferences


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.loader import load_transcript
    from src.segmenter import segment_transcript

    # Load and segment the transcript.
    segments = load_transcript("data/transcripts/episode1.txt")
    topics = segment_transcript(segments)

    # Ask the user to rate each topic interactively.
    preferences = get_preferences_cli(topics)

    # Print the final preferences.
    print("\n--- Final preferences ---")
    for pref in preferences:
        print(f"  {pref.topic_name}: weight {pref.weight}")
