from dataclasses import dataclass, field
# dataclass = auto-generates __init__, __repr__, __eq__ for classes that hold data
# instead of writing self.name = name, self.description = description... manually,
# we just list the fields and their types, and @dataclass builds it for us

# field(default_factory=list) = gives each new instance its own fresh empty list
# without this, all instances would share the SAME list (changes to one affect all)
# only needed for mutable defaults like list and dict, not for str/int/float

# represents one chunk of the original transcript
# Why it matters: when we later link a summary sentence back to the source, we need to know
# exactly where in the transcript it came from. These indices make that possible.
@dataclass
class TranscriptSegment:
    # the actual words in this chunk
    text: str
    # where this chunk starts in the full transcript (character position)
    start_index: int
    # where it ends
    end_index: int
    # how many words are in this chunk
    word_count: int
    # the original timestamp from the transcript (e.g. "3:09"), if available.
    # This is the timestamp of the first line that contributed to this chunk.
    timestamp: str = ""

# represents one discovered theme in the podcast
@dataclass
class Topic:
    # human-readable label like "Mental Health" or "Career Advice"
    name: str
    # a short explanation of what this topic covers.
    description: str
    # what fraction of the transcript this topic occupies (e.g., 0.35 means 35%). This is the baseline that user preference modulates against
    proportion: float
    # which TranscriptSegment belong to this topic (by their position in the list)
    segment_indices: list[int] = field(default_factory=list)

#Represents how much the user cares about a specific topic.
@dataclass
class UserPreference:
    # which topic this preference is for
    topic_name: str
    # the numerical weight. High = 1.5, Medium = 1.0, Low = 0.5. A weight of 1.0 means keep it as is. Above 1.0 means give me more detail on this.
    # Below 1.0 means compress this.
    weight: float  # High=1.5, Medium=1.0, Low=0.5


# one piece of a generated summary, with traceability built in.
@dataclass
class SummarySegment:
    # the actual summary sentence or paragraph
    text: str
    # which transcript segments this summary text was derived from. This is your evidence linking - a reader can trace any summary claim back to the original transcript
    source_segment_indices: list[int]
    # which topic this summary segment belongs to
    topic_name: str

# the complete summary output.
@dataclass
class Summary:
    # a list of SummarySegment objects that together form the full summary
    segments: list[SummarySegment]
    #one of "generic", "unconstrained", or "constrained". This is metadata that downstream code can use to understand how this summary was generated and how to treat it.
    summary_type: str  # "generic", "unconstrained", or "constrained"
    # a flexible dictionary for extra info (e.g total word count, delta used, generation timestamp)
    metadata: dict = field(default_factory=dict)
