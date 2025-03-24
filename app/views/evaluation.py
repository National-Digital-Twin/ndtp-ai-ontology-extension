import json
import streamlit as st
from app.state.app_state import AppState
from app.components.evaluators import EvaluationHandler
from app.components.config import ConfigHandler
from app.components.file_handlers import FileHandler


def show():
    """Handle Tab 4: Validation and results"""
    state = AppState.get()

    if not state.new_output:
        st.warning("Please generate ontology in the previous tab first")
        state.status["evaluation"] = False
        return

    st.subheader("4.1 Validate Ontology")
    if st.button("Run Validation", use_container_width=True, type="primary"):
        with st.spinner("Running Validation..."):
            clean_ontology, error_log = EvaluationHandler.validate_ontology(
                state.new_output, state.model
            )

        if clean_ontology:
            st.success("Validation Passed!")
            state.clean_ontology = clean_ontology
            state.status["evaluation"] = True
        else:
            st.error("Validation Failed!")
            state.clean_ontology = None
            state.status["evaluation"] = False
        state.error_log = error_log

        with st.expander("Error Log"):
            st.code("\n".join(error_log), language="text")

    if state.clean_ontology:
        with st.expander("Validated Ontology", expanded=True):
            st.code(state.clean_ontology, language="turtle")

    st.subheader("4.2 ABM Simulation")
    if not state.clean_ontology:
        st.warning(
            "Validated ontology not available. Please complete validation successfully first."
        )
    else:
        st.markdown("Build Personas & Scenario")
        col_abm = st.columns(2)
        with col_abm[0]:
            personas = ConfigHandler.handle_instructions("personas.json")
        with col_abm[1]:
            scene = ConfigHandler.handle_instructions("scenario_personas.txt")
        if st.button(
            "Start Multi Agent Discussion", use_container_width=True, type="primary"
        ):
            with st.spinner("Running ABM Simulation..."):
                state.discussion_result = EvaluationHandler.run_abm_simulation(
                    model=state.model,
                    clean_ontology=state.clean_ontology,
                    scene=scene,
                    personas=personas,
                )

        if state.discussion_result:
            discussion_dict = json.loads(state.discussion_result)
            for role, feedback in discussion_dict.items():
                with st.expander(f"💬 **{role}**", expanded=True):
                    st.write(feedback)

            st.download_button(
                label="Download Discussion Result as JSON",
                data=state.discussion_result,
                file_name="discussion_result.json",
                mime="application/json",
            )

    st.subheader("4.3 Compare Against Reference")

    if not state.clean_ontology:
        st.warning(
            "Validated ontology not available. Please complete validation successfully first."
        )
    else:
        FileHandler.handle_reference_snippet()
        reference = state.reference_snippet
        if reference and st.button(
            "Compare with Reference", use_container_width=True, type="primary"
        ):
            with st.spinner("Comparing with Reference..."):
                state.comparison_result = EvaluationHandler.compare_with_reference(
                    model=state.model,
                    synthetic_extension=state.clean_ontology,
                    reference=reference,
                )

        if state.comparison_result:
            with st.container(border=True):
                st.subheader("Comparison Report")
                st.write(state.comparison_result)
