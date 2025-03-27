# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

import json
import streamlit as st

from app.state import AppState
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

    st.subheader("4.1 Validate Extension")
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

    if state.error_log:
        with st.expander("Error Log"):
            st.code("\n".join(state.error_log), language="text")

    if state.clean_ontology:
        with st.expander("Validated Ontology Extension", expanded=True):
            st.code(state.clean_ontology, language="turtle")

        st.download_button(
            label="Download Validated Ontology Extension",
            data=state.clean_ontology,
            file_name="validated_ontology.ttl",
            mime="text/turtle",
        )

    st.subheader("4.2 ABM Simulation")
    if not state.clean_ontology:
        st.warning(
            "Validated ontology extension not available. Please complete validation successfully first."
        )
    else:
        st.markdown("Build Personas & Scenario")
        col_abm = st.columns(2)
        with col_abm[0]:
            # Load default personas
            default_personas = json.loads(
                ConfigHandler.load_default_instructions("personas.json")
            )
            if "personas_list" not in st.session_state:
                st.session_state.personas_list = default_personas["personas"]

            st.subheader("Personas")

            # Add new persona button
            if st.button("Add New Persona"):
                st.session_state.personas_list.append(
                    {"name": "", "description": "", "prompt": ""}
                )

            # Create form for each persona
            for i, persona in enumerate(st.session_state.personas_list):
                with st.container(border=True):
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.markdown(f"#### Persona {i+1}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{i}"):
                            st.session_state.personas_list.pop(i)
                            st.rerun()

                    persona["name"] = st.text_input(
                        "Name", persona["name"], key=f"name_{i}"
                    )
                    persona["description"] = st.text_input(
                        "Description", persona["description"], key=f"desc_{i}"
                    )
                    persona["prompt"] = st.text_area(
                        "Profile", persona["prompt"], key=f"prompt_{i}", height=100
                    )

            # Convert back to required format
            personas = {"personas": st.session_state.personas_list}

        with col_abm[1]:
            scene = ConfigHandler.handle_instructions(
                "scenario_personas.txt", label="Scenario"
            )
        if st.button(
            "Start Multi Agent Discussion", use_container_width=True, type="primary"
        ):
            with st.spinner("Running multi-agent discussion simulation..."):
                (
                    state.discussion_result,
                    state.discussion_error_message,
                ) = EvaluationHandler.run_abm_simulation(
                    model=state.model,
                    clean_ontology=state.clean_ontology,
                    scene=scene,
                    personas=personas,
                )

        if state.discussion_result:
            try:
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
            except Exception as e:
                st.error(str(e))

        if state.discussion_error_message:
            st.error(state.discussion_error_message)

    st.subheader("4.3 Compare Against Reference")

    if not state.clean_ontology:
        st.warning(
            "Validated ontology extension not available. Please complete validation successfully first."
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
                st.text(state.comparison_result)
