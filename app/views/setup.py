import os
import pandas as pd
import streamlit as st


def load_default_instructions():
    """Load the default instructions from the template file."""
    try:
        instructions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "app",
            "templates",
            "instructions.txt",
        )

        if os.path.exists(instructions_path):
            with open(instructions_path, "r") as f:
                return f.read()
        else:
            raise FileNotFoundError(
                f"Instructions template file not found at: {instructions_path}"
            )

    except Exception as e:
        st.error(f"Error loading instructions: {e}")
        return "# Enter your instructions here"


def common_section():
    """Common snippet section."""
    common_snippet_file = st.file_uploader(
        "Upload Common Snippet (TTL template)", type=["ttl", "txt"]
    )
    if common_snippet_file is not None:
        common_snippet = common_snippet_file.getvalue().decode("utf-8")
        st.session_state.common_snippet = common_snippet
        st.success("Common snippet uploaded successfully!")
        with st.expander("Preview Common Snippet"):
            st.code(common_snippet, language="turtle")


def csv_data_upload_section():
    """Upload CSV data section."""
    csv_file = st.file_uploader("Upload CSV Data", type=["csv"])
    if csv_file is not None:
        try:
            # Only load the first 25 rows of the CSV file
            csv_data = pd.read_csv(csv_file, nrows=25, low_memory=False)

            st.session_state.csv_data = csv_data
            st.success(f"CSV data uploaded successfully! (Limited to first 25 rows)")

            display_data = csv_data.copy()

            # Convert mixed-type columns to string
            object_columns = display_data.select_dtypes(include=["object"]).columns
            for col in object_columns:
                display_data[col] = display_data[col].astype(str)

            with st.expander("Preview CSV Data"):
                try:
                    st.dataframe(display_data)
                    st.write(f"Columns: {', '.join(display_data.columns.tolist())}")
                    st.write(
                        f"Shape: {display_data.shape[0]} rows × {display_data.shape[1]} columns"
                    )
                except Exception as e:
                    st.warning(
                        f"Unable to display preview due to data format issues. Showing simplified view."
                    )

                    # Fallback to displaying just the first few rows as text
                    st.write(f"First 5 rows (simplified view):")
                    st.write(display_data.head(5).to_dict())
                    st.write(f"Columns: {', '.join(display_data.columns.tolist())}")
                    st.write(
                        f"Shape: {display_data.shape[0]} rows × {display_data.shape[1]} columns"
                    )
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.info(
                "Try uploading a CSV file with consistent data types or simpler structure."
            )


def reference_section():
    """Reference ontology section."""
    st.write(
        "Optional: Upload a reference snippet to compare with the generated output"
    )
    reference_snippet_file = st.file_uploader(
        "Upload Reference Snippet (optional)", type=["ttl", "txt"]
    )
    if reference_snippet_file is not None:
        reference_snippet = reference_snippet_file.getvalue().decode("utf-8")
        st.session_state.reference_snippet = reference_snippet
        st.success("Reference snippet uploaded successfully!")
        with st.expander("Preview Reference Snippet"):
            st.code(reference_snippet, language="turtle")


def instructions_section():
    """LLM instructions section."""
    if st.session_state.instructions is None:
        st.session_state.instructions = load_default_instructions()

    instructions = st.text_area(
        "Edit Instructions", value=st.session_state.instructions, height=300
    )

    if instructions != st.session_state.instructions:
        st.session_state.instructions = instructions

    if st.button("Reset to Default Instructions"):
        st.session_state.instructions = load_default_instructions()
        st.rerun()


def config_section():
    model_options = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=model_options.index(st.session_state.model)
        if st.session_state.model in model_options
        else 1,
    )
    st.session_state.model = selected_model


def show():
    """Display the setup page."""
    st.title("Setup")
    st.write("Configure the inputs for ontology generation.")

    st.header("Upload Files")
    common_section()
    csv_data_upload_section()
    reference_section()

    st.header("Instructions")
    instructions_section()

    st.header("Configuration")
    config_section()

    st.markdown("---")

    required_inputs_ready = (
        st.session_state.common_snippet is not None
        and st.session_state.csv_data is not None
        and st.session_state.instructions is not None
    )

    if st.button("Proceed to Generation", disabled=not required_inputs_ready):
        st.session_state.page = "generation"
        st.rerun()

    if not required_inputs_ready:
        st.warning(
            "Please provide all required inputs (Common Snippet, CSV Data, and Instructions) to proceed."
        )
