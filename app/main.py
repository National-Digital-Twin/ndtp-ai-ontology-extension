import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.views import setup, generation, results
from app.utils.logging import log


def initialise_state():
    """Initialise session state variables with default values."""
    if "page" not in st.session_state:
        st.session_state.page = "setup"
        log("Initialized session state: page = setup")

    if "common_snippet" not in st.session_state:
        st.session_state.common_snippet = None
        log("Initialized session state: common_snippet = None")

    if "csv_data" not in st.session_state:
        st.session_state.csv_data = None
        log("Initialized session state: csv_data = None")

    if "reference_snippet" not in st.session_state:
        st.session_state.reference_snippet = None
        log("Initialized session state: reference_snippet = None")

    if "instructions" not in st.session_state:
        st.session_state.instructions = None
        log("Initialized session state: instructions = None")

    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"
        log("Initialized session state: model = gpt-4o-mini")

    if "output_filename" not in st.session_state:
        st.session_state.output_filename = "ontology.ttl"
        log("Initialized session state: output_filename = ontology.ttl")

    if "current_iteration" not in st.session_state:
        st.session_state.current_iteration = 0
        log("Initialized session state: current_iteration = 0")

    if "generated_snippet" not in st.session_state:
        st.session_state.generated_snippet = None
        log("Initialized session state: generated_snippet = None")

    if "iteration_history" not in st.session_state:
        st.session_state.iteration_history = []
        log("Initialized session state: iteration_history = []")

    if "human_feedback" not in st.session_state:
        st.session_state.human_feedback = ""
        log('Initialized session state: human_feedback = ""')


def is_csv_data_valid():
    """Check if CSV data is loaded and not empty."""
    is_valid = st.session_state.csv_data is not None and not (
        isinstance(st.session_state.csv_data, pd.DataFrame)
        and st.session_state.csv_data.empty
    )
    log(f"CSV data validation check: {is_valid}")
    return is_valid


def navigate_to(page):
    """Change the current page in the application."""
    log(f"Navigating to page: {page}")
    st.session_state.page = page


def sidebar():
    """Render the application sidebar with navigation buttons."""
    st.sidebar.title("Ontology Generator")
    st.sidebar.markdown("---")

    if st.sidebar.button("Setup", key="nav_setup"):
        log("User clicked Setup navigation button")
        navigate_to("setup")

    generation_disabled = not (
        st.session_state.common_snippet is not None
        and is_csv_data_valid()
        and st.session_state.instructions is not None
    )

    if st.sidebar.button(
        "Generation", key="nav_generation", disabled=generation_disabled
    ):
        log("User clicked Generation navigation button")
        navigate_to("generation")

    if st.sidebar.button(
        "Results", key="nav_results", disabled=not st.session_state.generated_snippet
    ):
        log("User clicked Results navigation button")
        navigate_to("results")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application helps you generate ontologies using AI. "
        "Start by configuring your inputs in the Setup page."
    )


def main():
    """Application entry point. Sets up configuration and renders the current view."""
    log("Starting Ontology Generator application")

    st.set_page_config(
        page_title="Ontology Generator",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    log("Streamlit page configuration set")

    initialise_state()
    sidebar()

    log(f"Displaying page: {st.session_state.page}")
    if st.session_state.page == "setup":
        setup.show()
    elif st.session_state.page == "generation":
        generation.show()
    elif st.session_state.page == "results":
        results.show()

    log("Page rendering complete")


if __name__ == "__main__":
    main()
