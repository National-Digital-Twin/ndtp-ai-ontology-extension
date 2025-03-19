import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


def final_ontology_section():
    """Final ontology section."""
    final_snippet = st.text_area(
        "Final Ontology (editable)",
        value=st.session_state.generated_snippet,
        height=400,
    )

    if final_snippet != st.session_state.generated_snippet:
        st.session_state.generated_snippet = final_snippet
        if st.session_state.iteration_history:
            st.session_state.iteration_history[-1]["snippet"] = final_snippet


def export_section():
    """Export section."""
    st.download_button(
        label="Download as TTL",
        data=st.session_state.generated_snippet,
        file_name=st.session_state.output_filename,
        mime="text/turtle",
    )


def basic_statistics():
    """Basic statistics."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Iterations", st.session_state.current_iteration + 1)

    with col2:
        st.metric("Model Used", st.session_state.model)

    with col3:
        # Calculate time elapsed from first to last iteration
        if len(st.session_state.iteration_history) >= 2:
            first_time = datetime.strptime(
                st.session_state.iteration_history[0]["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            last_time = datetime.strptime(
                st.session_state.iteration_history[-1]["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            time_elapsed = last_time - first_time
            st.metric("Time Elapsed", f"{time_elapsed}")
        else:
            st.metric("Time Elapsed", "N/A")


def iteration_history_visualisation():
    """Iteration history visualisation."""
    history_data = []
    for item in st.session_state.iteration_history:
        snippet_lines = len(item["snippet"].split("\n"))
        error_count = len(item.get("errors", []))
        feedback_words = (
            len(item.get("feedback", "").split()) if item.get("feedback") else 0
        )

        history_data.append(
            {
                "Iteration": item["iteration"],
                "Timestamp": item["timestamp"],
                "Snippet Lines": snippet_lines,
                "Error Count": error_count,
                "Feedback Words": feedback_words,
            }
        )

    history_df = pd.DataFrame(history_data)

    st.dataframe(history_df)

    tab1, tab2, tab3 = st.tabs(["Snippet Size", "Errors", "Feedback"])
    with tab1:
        fig = px.line(
            history_df,
            x="Iteration",
            y="Snippet Lines",
            title="Ontology Size by Iteration",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.line(
            history_df,
            x="Iteration",
            y="Error Count",
            title="Errors by Iteration",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.line(
            history_df,
            x="Iteration",
            y="Feedback Words",
            title="Feedback Size by Iteration",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)


def detailed_history():
    """Detailed history."""
    for i, iteration in enumerate(st.session_state.iteration_history):
        with st.expander(
            f"Iteration {iteration['iteration']} - {iteration['timestamp']}"
        ):
            st.code(iteration["snippet"], language="turtle")
            if iteration["feedback"]:
                st.write("Feedback:")
                st.info(iteration["feedback"])
            if iteration.get("errors"):
                st.write("Errors:")
                for error in iteration["errors"]:
                    st.error(error)


def summary_section():
    """Summary section."""
    st.subheader("Statistics")
    basic_statistics()

    if st.session_state.iteration_history:
        st.subheader("Iteration History")
        iteration_history_visualisation()

    st.subheader("Detailed History")
    detailed_history()


def show():
    """Display the results page."""
    st.title("Generation Results")

    if not st.session_state.generated_snippet:
        st.warning(
            "No ontology has been generated yet. Please go to the Generation page first."
        )
        if st.button("Go to Generation Page"):
            st.session_state.page = "generation"
            st.rerun()
        return

    st.header("Final Ontology")
    final_ontology_section()

    st.subheader("Export Options")
    export_section()

    st.header("Generation Summary")
    summary_section()

    st.header("Start New Generation")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Generation with Same Settings"):
            st.session_state.current_iteration = 0
            st.session_state.generated_snippet = None
            st.session_state.iteration_history = []
            st.session_state.human_feedback = ""
            st.session_state.page = "generation"
            st.rerun()

    with col2:
        if st.button("Back to Setup"):
            st.session_state.current_iteration = 0
            st.session_state.generated_snippet = None
            st.session_state.iteration_history = []
            st.session_state.human_feedback = ""
            st.session_state.page = "setup"
            st.rerun()
