import os
import sys
from datetime import datetime
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from app.utils.ontology import generate_ontology_iteration


def action_buttons(feedback):
    """Action buttons section."""
    col1, col2, col3 = st.columns(3)

    with col1:
        # Next iteration button
        next_button = st.button(
            "Generate Next Iteration",
            help="Generate the next iteration based on the current snippet and feedback",
        )

        if next_button:
            with st.spinner("Generating next iteration..."):
                try:
                    if st.session_state.iteration_history:
                        st.session_state.iteration_history[-1]["feedback"] = feedback

                    result = generate_ontology_iteration(
                        common_snippet=st.session_state.common_snippet,
                        csv_data=st.session_state.csv_data,
                        instructions=st.session_state.instructions,
                        model=st.session_state.model,
                        human_feedback=feedback,
                        previous_snippet=st.session_state.generated_snippet,
                    )

                    # Update session state
                    st.session_state.current_iteration += 1
                    st.session_state.generated_snippet = result["generated_snippet"]
                    st.session_state.human_feedback = ""

                    # Add to history
                    st.session_state.iteration_history.append(
                        {
                            "iteration": st.session_state.current_iteration + 1,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "snippet": result["generated_snippet"],
                            "feedback": "",
                            "errors": result.get("errors", []),
                        }
                    )

                    st.success(
                        f"Iteration {st.session_state.current_iteration + 1} generated successfully!"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating next iteration: {e}")

    with col2:
        if st.button("Finish & View Results"):
            st.session_state.page = "results"
            st.rerun()

    with col3:
        st.download_button(
            label="Export Current Snippet",
            data=st.session_state.generated_snippet,
            file_name=st.session_state.output_filename,
            mime="text/turtle",
        )


def feedback_section():
    """Feedback section."""
    feedback = st.text_area(
        "Enter feedback for the next iteration",
        value=st.session_state.human_feedback,
        height=150,
    )
    st.session_state.human_feedback = feedback
    return feedback


def initialise_generation():
    """Initialise generation."""
    with st.spinner("Generating initial ontology..."):
        try:
            # Generate the first iteration
            result = generate_ontology_iteration(
                common_snippet=st.session_state.common_snippet,
                csv_data=st.session_state.csv_data,
                instructions=st.session_state.instructions,
                model=st.session_state.model,
                human_feedback="",
            )

            st.session_state.generated_snippet = result["generated_snippet"]
            st.session_state.iteration_history.append(
                {
                    "iteration": st.session_state.current_iteration + 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "snippet": result["generated_snippet"],
                    "feedback": "",
                    "errors": result.get("errors", []),
                }
            )

            st.success("Initial ontology generated successfully!")
        except Exception as e:
            st.error(f"Error generating initial ontology: {e}")
            return


def generated_section():
    """Generated ontology section."""
    generated_snippet = st.text_area(
        "Edit the generated ontology if needed",
        value=st.session_state.generated_snippet,
        height=400,
    )

    if generated_snippet != st.session_state.generated_snippet:
        st.session_state.generated_snippet = generated_snippet
        if st.session_state.iteration_history:
            st.session_state.iteration_history[-1]["snippet"] = generated_snippet

    if st.session_state.reference_snippet:
        with st.expander("Compare with Reference Snippet"):
            st.code(st.session_state.reference_snippet, language="turtle")


def errors_section():
    """Errors section."""
    errors = st.session_state.iteration_history[-1]["errors"]
    if errors:
        with st.expander("Validation Errors"):
            for error in errors:
                st.error(error)


def iteration_history_section():
    """Iteration history section."""
    for i, iteration in enumerate(st.session_state.iteration_history):
        with st.expander(
            f"Iteration {iteration['iteration']} - {iteration['timestamp']}"
        ):
            st.code(iteration["snippet"], language="turtle")
            if iteration["feedback"]:
                st.write("Feedback:")
                st.info(iteration["feedback"])


def show():
    """Display the generation page."""
    st.title("Ontology Generation")

    st.header(f"Iteration {st.session_state.current_iteration + 1}")

    if (
        st.session_state.current_iteration == 0
        and st.session_state.generated_snippet is None
    ):
        initialise_generation()

    st.subheader("Generated Ontology")
    generated_section()

    if (
        st.session_state.iteration_history
        and "errors" in st.session_state.iteration_history[-1]
    ):
        errors_section()

    st.subheader("Provide Feedback")
    feedback = feedback_section()

    action_buttons(feedback)

    if st.session_state.iteration_history:
        st.subheader("Iteration History")
        iteration_history_section()
