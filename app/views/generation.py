import streamlit as st
from app.state.app_state import AppState
from app.components.config import ConfigHandler
from app.components.processors import ProcessingHandler


def show():
    """Handle Tab 3: Instructions and iteration"""
    state = AppState.get()

    # Check if data is uploaded and processing results are complete
    if state.csv_data is None or getattr(state.csv_data, "empty", True):
        st.warning("Please upload data first")
        state.status["ontology_generation"] = False
        return

    processing_keys_required = [
        "analysis",
        "triplets",
        "concepts",
        "usage",
        "classification",
    ]
    for key in processing_keys_required:
        if state.processing_results.get(key, None) is None:
            st.warning(
                f"Please complete data processing steps in the previous section first"
            )
            state.status["ontology_generation"] = False
            return

    # Instructions Section
    st.subheader("Instructions")
    cols_inst = st.columns(3)

    with cols_inst[0]:
        st.subheader("AI Profile")
        background = ConfigHandler.handle_instructions("background_prompt.txt")
    with cols_inst[1]:
        st.subheader("Main Instructions")
        prompt = ConfigHandler.handle_instructions("instructions.txt")
    with cols_inst[2]:
        st.subheader("Additional Instructions")
        guidelines = ConfigHandler.handle_instructions("custom_instructions.txt")

    # Iteration Controls
    st.subheader("Iteration Controls")
    col_control = st.columns(2)
    with col_control[0]:
        state.chunk_start = st.number_input(
            "Chunk Start", min_value=0, value=state.chunk_start, step=1
        )
    with col_control[1]:
        state.chunk_size = st.number_input(
            "Chunk Size", min_value=1, value=state.chunk_size, step=1
        )

    # Generation Actions
    if st.button("Generate Ontology", use_container_width=True, type="primary"):
        with st.spinner("Generating ontology..."):
            result = ProcessingHandler.generate_ontology(
                df=state.csv_data,
                model=state.model,
                background=background,
                prompt=prompt,
                guidelines=guidelines,
            )
        state.new_output = result

    if state.new_output:
        with st.expander("Generated Ontology", expanded=True):
            st.code(state.new_output, language="turtle")
        state.status["ontology_generation"] = True
