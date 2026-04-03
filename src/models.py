from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    text: str
    start_index: int
    end_index: int
    word_count: int


@dataclass
class Topic:
    name: str
    description: str
    proportion: float
    segment_indices: list[int] = field(default_factory=list)


@dataclass
class UserPreference:
    topic_name: str
    weight: float  # High=1.5, Medium=1.0, Low=0.5


@dataclass
class SummarySegment:
    text: str
    source_segment_indices: list[int]
    topic_name: str


@dataclass
class Summary:
    segments: list[SummarySegment]
    summary_type: str  # "generic", "unconstrained", or "constrained"
    metadata: dict = field(default_factory=dict)
