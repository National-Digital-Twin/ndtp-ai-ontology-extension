import streamlit as st
from app.state import AppState
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
        "triples",
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
        background = ConfigHandler.handle_instructions(
            "background_prompt.txt", label="AI Profile"
        )
    with cols_inst[1]:
        prompt = ConfigHandler.handle_instructions(
            "instructions.txt", label="Main Instructions"
        )
    with cols_inst[2]:
        guidelines = ConfigHandler.handle_instructions(
            "custom_instructions.txt", label="Guidelines"
        )

    ontology_feedback = st.text_area("Ontology Feedback", value="", height=100)

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
    if st.button("Generate Extension", use_container_width=True, type="primary"):
        with st.spinner("Generating ontology extension..."):
            result = ProcessingHandler.generate_ontology(
                df=state.csv_data,
                model=state.model,
                background=background,
                prompt=prompt,
                guidelines=guidelines,
                ontology_feedback=ontology_feedback,
            )
        state.new_output = result
        state.iteration_history.append(
            {
                "background": background,
                "prompt": prompt,
                "guidelines": guidelines,
                "feedback": ontology_feedback,
                "result": result,
            }
        )
        state.current_iteration += 1

    if state.new_output:
        suffix = (
            f" (Iteration {state.current_iteration})"
            if len(state.iteration_history) > 1
            else ""
        )
        with st.expander(f"Generated Ontology Extension{suffix}", expanded=True):
            st.code(state.new_output, language="turtle")
        state.status["ontology_generation"] = True

    if len(state.iteration_history) > 1:
        st.subheader("Iteration History")
        for i in range(len(state.iteration_history) - 2, -1, -1):
            with st.expander(f"Iteration {i+1}", expanded=False):
                st.code(state.iteration_history[i]["result"], language="turtle")
