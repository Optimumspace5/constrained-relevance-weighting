import os
import tempfile
import streamlit as st

from src.loader import load_transcript
from src.segmenter import segment_transcript
from src.preferences import create_preferences
from src.summarizers import (
    generate_generic_summary,
    generate_unconstrained_summary,
    generate_constrained_summary,
    calculate_constrained_proportions,
)
from src.evidence import link_evidence, format_evidence_report
from src.evaluator import run_full_evaluation
from src.config import DEFAULT_DELTA, CONSTRAINT_DELTAS, PREFERENCE_WEIGHTS


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Podcast Summarizer — CRW",
    layout="wide",
)

st.title("Podcast Summarizer — Constrained Relevance Weighting")
st.caption("Compare generic, unconstrained, and constrained personalized summaries")


# ---------------------------------------------------------------------------
# Sidebar: file upload → topic discovery → preference collection → generate
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    # --- Step 1: file upload ---
    uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])

    if uploaded_file is not None:
        # Only reload the transcript if a new file was uploaded.
        if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            # Save the uploaded bytes to a temp file so load_transcript can open it normally.
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Loading transcript..."):
                    st.session_state.segments = load_transcript(tmp_path)
                st.session_state.uploaded_filename = uploaded_file.name
                # Clear downstream state so the user must re-discover topics.
                for key in ("topics", "summaries", "linked_summaries"):
                    st.session_state.pop(key, None)
            except Exception as e:
                st.error(f"Failed to load transcript: {e}")
            finally:
                os.unlink(tmp_path)   # clean up the temp file

        st.success(f"Loaded: {uploaded_file.name} ({len(st.session_state.segments)} segments)")

    # --- Step 2: discover topics ---
    if "segments" in st.session_state:
        st.divider()
        if st.button("Discover Topics", use_container_width=True):
            try:
                with st.spinner("Discovering topics... (this calls the API)"):
                    st.session_state.topics = segment_transcript(st.session_state.segments)
                # Clear downstream state when topics are refreshed.
                for key in ("summaries", "linked_summaries"):
                    st.session_state.pop(key, None)
            except Exception as e:
                st.error(f"Topic discovery failed: {e}")

    # --- Step 3: preference sliders + delta ---
    if "topics" in st.session_state:
        st.divider()
        st.subheader("Your preferences")
        st.caption("How much detail do you want on each topic?")

        # Collect a selectbox per topic; store ratings in a local dict.
        rating_options = list(PREFERENCE_WEIGHTS.keys())   # ["high", "medium", "low"]
        ratings: dict[str, str] = {}
        for topic in st.session_state.topics:
            ratings[topic.name] = st.selectbox(
                label=f"{topic.name} ({topic.proportion:.0%})",
                options=rating_options,
                index=rating_options.index("medium"),   # default to medium
                key=f"rating_{topic.name}",
                help=topic.description,
            )

        st.divider()
        delta = st.slider(
            "Constraint bound (delta)",
            min_value=0.05,
            max_value=0.25,
            value=DEFAULT_DELTA,
            step=0.05,
            help="Maximum allowed deviation from a topic's original proportion. "
                 "Smaller = more faithful to episode structure; larger = more personalized.",
        )

        # --- Step 4: generate summaries ---
        st.divider()
        if st.button("Generate Summaries", type="primary", use_container_width=True):
            try:
                preferences = create_preferences(st.session_state.topics, ratings)
                segments = st.session_state.segments
                topics = st.session_state.topics

                with st.spinner("Generating generic summary..."):
                    generic = generate_generic_summary(segments, topics)

                with st.spinner("Generating unconstrained summary..."):
                    unconstrained = generate_unconstrained_summary(segments, topics, preferences)

                with st.spinner("Generating constrained summary..."):
                    constrained = generate_constrained_summary(segments, topics, preferences, delta=delta)

                st.session_state.summaries = {
                    "generic": generic,
                    "unconstrained": unconstrained,
                    "constrained": constrained,
                }
                st.session_state.preferences = preferences
                st.session_state.delta = delta
                # Clear linked summaries so they're re-derived from the new summaries.
                st.session_state.pop("linked_summaries", None)
                st.success("Summaries generated!")
            except Exception as e:
                st.error(f"Summary generation failed: {e}")


# ---------------------------------------------------------------------------
# Main area: shown only once summaries exist
# ---------------------------------------------------------------------------

if "summaries" not in st.session_state:
    st.info("Upload a transcript, discover topics, set your preferences, then click Generate Summaries.")
    st.stop()

summaries    = st.session_state.summaries
topics       = st.session_state.topics
segments     = st.session_state.segments
preferences  = st.session_state.preferences
delta        = st.session_state.delta

# ---------------------------------------------------------------------------
# Proportion comparison table
# ---------------------------------------------------------------------------

st.subheader("Topic proportion breakdown")
st.caption(f"Constraint bound (delta): ±{delta:.0%}")

constrained_proportions = calculate_constrained_proportions(topics, preferences, delta)
pref_lookup = {p.topic_name: p.weight for p in preferences}
weight_to_label = {v: k for k, v in PREFERENCE_WEIGHTS.items()}   # {1.5: "high", ...}

# Build table rows as a list of dicts — st.dataframe renders these cleanly.
table_rows = []
for topic in topics:
    weight = pref_lookup.get(topic.name, 1.0)
    table_rows.append({
        "Topic": topic.name,
        "Original %": f"{topic.proportion:.1%}",
        "Preference": weight_to_label.get(weight, "medium"),
        "Constrained Target %": f"{constrained_proportions[topic.name]:.1%}",
    })

st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Three summary tabs
# ---------------------------------------------------------------------------

st.subheader("Summaries")
tab_generic, tab_unconstrained, tab_constrained = st.tabs(["Generic", "Unconstrained", "Constrained"])

with tab_generic:
    summary_text = summaries["generic"].segments[0].text
    st.markdown(summary_text)
    st.caption(f"Word count: {summaries['generic'].metadata.get('word_count', '—')}")

with tab_unconstrained:
    summary_text = summaries["unconstrained"].segments[0].text
    st.markdown(summary_text)
    st.caption(f"Word count: {summaries['unconstrained'].metadata.get('word_count', '—')}")

with tab_constrained:
    summary_text = summaries["constrained"].segments[0].text
    st.markdown(summary_text)
    st.caption(f"Word count: {summaries['constrained'].metadata.get('word_count', '—')}")

st.divider()

# ---------------------------------------------------------------------------
# Evidence linking (lazy — only runs when the expander is opened)
# ---------------------------------------------------------------------------

# We cache linked summaries in session_state to avoid re-running API calls
# every time Streamlit reruns (e.g. on widget interaction).
def get_linked_summaries():
    if "linked_summaries" not in st.session_state:
        with st.spinner("Linking evidence to paragraphs..."):
            st.session_state.linked_summaries = {
                name: link_evidence(summary, segments, topics)
                for name, summary in summaries.items()
            }
    return st.session_state.linked_summaries


# ---------------------------------------------------------------------------
# Evaluation expander
# ---------------------------------------------------------------------------

with st.expander("Evaluation — compare all three summaries"):
    st.caption("Runs faithfulness (per-paragraph API calls), coverage, and relevance checks.")

    if st.button("Run Evaluation"):
        try:
            linked = get_linked_summaries()
            with st.spinner("Evaluating... (multiple API calls)"):
                results = run_full_evaluation(
                    linked["generic"],
                    linked["unconstrained"],
                    linked["constrained"],
                    segments,
                    topics,
                    preferences,
                )

            # Display results as a tidy table.
            eval_rows = []
            for name, r in results.items():
                eval_rows.append({
                    "Summary": name,
                    "Faithfulness (avg/5)": r["faithfulness"]["average_score"],
                    "Coverage": f"{r['coverage']['topics_covered']}/{r['coverage']['total_topics']} topics",
                    "Relevance score": r["relevance"]["relevance_score"],
                    "Word count": r["word_count"],
                })
            st.dataframe(eval_rows, use_container_width=True, hide_index=True)

            # Show any faithfulness issues flagged for the constrained summary.
            constrained_issues = results["constrained"]["faithfulness"]["issues"]
            if constrained_issues:
                st.warning("Faithfulness issues in constrained summary:")
                for issue in constrained_issues:
                    st.markdown(f"- {issue}")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# ---------------------------------------------------------------------------
# Evidence report expander (constrained summary only)
# ---------------------------------------------------------------------------

with st.expander("Evidence report — constrained summary"):
    st.caption("Shows each summary paragraph alongside the source transcript excerpts it came from.")

    if st.button("Generate Evidence Report"):
        try:
            linked = get_linked_summaries()
            report = format_evidence_report(linked["constrained"], segments)
            st.text(report)   # st.text preserves the fixed-width formatting from format_evidence_report
        except Exception as e:
            st.error(f"Evidence report failed: {e}")
