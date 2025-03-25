import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.components.cache import Cache
from app.utils.logging import log
from app.state import AppState
from app.views import data_input, processing, generation, evaluation


def show():
    """Display the wizard interface for ontology generation."""
    st.title("Ontology Generator")

    # Create sidebar navigation
    with st.sidebar:
        st.markdown(
            """
        ## Ontology Generator
                    
        The Ontology Generator helps you create, process, 
        and validate ontologies in a step-by-step workflow.
        """
        )

        st.markdown(
            """
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
            <p style='margin: 5px 0;'><strong>Steps</strong></p>
            <p style='margin: 5px 0;'><a href='#data-input'>1. Data Input</a></p>
            <p style='margin: 5px 0;'><a href='#data-processing'>2. Data Processing</a></p>
            <p style='margin: 5px 0;'><a href='#ontology-generation'>3. Extension Generation</a></p>
            <p style='margin: 5px 0;'><a href='#validation'>4. Validation</a></p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Initialize state if needed
    state = AppState.get()

    # Data Input Section
    st.header("1. Data Input", anchor="data-input")
    with st.container():
        data_input.show()

    st.divider()

    # Processing Section
    st.header("2. Data Processing", anchor="data-processing")
    with st.container():
        processing.show()

    st.divider()

    # Generation Section
    st.header("3. Extension Generation", anchor="ontology-generation")
    with st.container():
        generation.show()

    st.divider()

    # Evaluation Section
    st.header("4. Validation", anchor="validation")
    with st.container():
        evaluation.show()


def main():
    """Application entry point. Sets up configuration and renders the current view."""
    log("Starting Ontology Generator application")

    st.set_page_config(
        page_title="Ontology Generator",
        page_icon="app/images/logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    log("Streamlit page configuration set")

    # Initialise state and cache
    AppState.initialize()
    use_cache = st.get_option("app.use_cache")
    cache_path = st.get_option("app.cache_path")
    st.session_state.cache = Cache(use_cache=use_cache, cache_file=cache_path)
    if use_cache:
        log(f"Cache initialised at: {cache_path}")
    else:
        log("Cache disabled")

    log(f"Showing page...")
    show()
    log("Page rendering complete")


if __name__ == "__main__":
    main()
