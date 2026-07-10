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
import os

# Model routing. Generation (creative summarization) can warrant a stronger model;
# bulk/mechanical calls (topic discovery, segment & paragraph classification,
# evidence linking, claim extraction & verification) can use a cheaper/faster one.
# Both are overridable via env.
#
# Defaults are Claude Sonnet 4.6, which still accepts temperature=0 — used
# throughout for deterministic, reproducible calls. NOTE: Sonnet 5 / Opus 4.7+
# REJECT the temperature parameter with a 400, so switching to those models
# requires removing every temperature=0 argument first. Set
# BULK_MODEL=claude-haiku-4-5 in .env to run mechanical calls on a cheaper model
# (Haiku 4.5 also accepts temperature=0).
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-sonnet-4-6")
BULK_MODEL = os.getenv("BULK_MODEL", "claude-sonnet-4-6")

# Backwards-compatible alias — existing `from src.config import LLM_MODEL`
# imports keep working and resolve to the generation model.
LLM_MODEL = GENERATION_MODEL

# Max concurrent Claude API calls when work is parallelized (summary generation,
# per-paragraph claim extraction). Overridable via env.
API_CONCURRENCY = int(os.getenv("API_CONCURRENCY", "4"))
# target summary length — all summaries must land in this range for fair comparison
MAX_SUMMARY_WORDS = 900          # target midpoint
MIN_SUMMARY_WORDS = 800          # hard floor
MAX_SUMMARY_WORDS_CEIL = 1000    # hard ceiling
