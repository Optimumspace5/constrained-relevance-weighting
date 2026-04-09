# how many topics the system should extract
NUM_TOPICS = 8

# the 3 experimental conditions for how much expansion a preferred topic is allowed
CONSTRAINT_DELTAS = [0.10, 0.15, 0.20]
# the default constraint bound
DEFAULT_DELTA = 0.15
#your user preference scale
PREFERENCE_WEIGHTS = {
    "high": 1.5,
    "medium": 1.0,
    "low": 0.5,
}
# which claude model to use for generation
LLM_MODEL = "claude-sonnet-4-20250514"
# target summary length — all summaries must land in this range for fair comparison
MAX_SUMMARY_WORDS = 900          # target midpoint
MIN_SUMMARY_WORDS = 800          # hard floor
MAX_SUMMARY_WORDS_CEIL = 1000    # hard ceiling
