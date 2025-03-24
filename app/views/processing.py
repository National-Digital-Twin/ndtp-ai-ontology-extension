import streamlit as st
from app.state.app_state import AppState
from app.components.processors import ProcessingHandler


def show():
    """Handle Tab 2: Analysis and processing"""
    state = AppState.get()
    df = state.csv_data
    model = state.model

    if df is None:
        st.warning("Please upload CSV data in the previous tab first")
        state.status["data_processing"] = False
        return

    # Extraction Dashboard
    with st.container():
        if st.button("Run Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing CSV..."):
                analysis_result = ProcessingHandler.analyze_csv(df, model)
                if analysis_result:
                    st.success("Analysis complete!")
                    state.processing_results["analysis"] = analysis_result
        if state.processing_results.get("analysis") is not None:
            with st.expander("Analysis Results", expanded=True):
                st.json(state.processing_results["analysis"])

        if st.button("Extract Triplets", use_container_width=True, type="primary"):
            with st.spinner("Extracting triplets..."):
                triplets_result = ProcessingHandler.extract_triplets(df, model)
                if triplets_result:
                    st.success("Triplets extracted!")
                    state.processing_results["triplets"] = triplets_result
        if state.processing_results.get("triplets") is not None:
            with st.expander("Triplet Results", expanded=True):
                st.json(state.processing_results["triplets"])

        if st.button("Extract Concepts", use_container_width=True, type="primary"):
            with st.spinner("Extracting concepts..."):
                concepts_result = ProcessingHandler.extract_concepts(df, model)
                if concepts_result:
                    st.success("Concepts extracted!")
                    state.processing_results["concepts"] = concepts_result
        if state.processing_results.get("concepts") is not None:
            with st.expander("Concept Results", expanded=True):
                st.json(state.processing_results["concepts"])

        if st.button("Gather Usage Patterns", use_container_width=True, type="primary"):
            with st.spinner("Gathering usage patterns..."):
                usage_result = ProcessingHandler.gather_usage_patterns(model)
                if usage_result:
                    st.success("Usage patterns gathered.")
                    state.processing_results["patterns"] = usage_result
        if state.processing_results.get("patterns") is not None:
            with st.expander("Pattern Results", expanded=True):
                st.json(state.processing_results["patterns"])

        if st.button("Classify Extensions", use_container_width=True, type="primary"):
            with st.spinner("Classifying extensions..."):
                classification_result = ProcessingHandler.classify_extensions(model)
                if classification_result:
                    st.success("Extensions classified.")
                    state.processing_results["classification"] = classification_result
        if state.processing_results.get("classification") is not None:
            with st.expander("Classification Results", expanded=True):
                st.json(state.processing_results["classification"])

    # Mark tab as valid if we have at least some processing results
    state.status["data_processing"] = bool(state.processing_results)
