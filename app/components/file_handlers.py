import streamlit as st
import pandas as pd
from typing import Optional, List

from app.utils.logging import log
from app.state import AppState


class FileHandler:
    @staticmethod
    def handle_common_snippet() -> None:
        """Handle common snippet upload."""
        common_snippet_file = st.file_uploader(
            "Upload Base Ontology (TTL template)", type=["ttl", "txt"]
        )
        if common_snippet_file is not None:
            content = common_snippet_file.getvalue().decode("utf-8")
            AppState.get().common_snippet = content
            st.success("Base ontology file uploaded successfully!")
            with st.expander("Preview Base Ontology File"):
                st.code(content, language="turtle")
            log("Base ontology file uploaded and stored")

    @staticmethod
    def handle_csv_upload() -> Optional[pd.DataFrame]:
        """Handle CSV file upload and processing."""
        csv_files = st.file_uploader(
            "Upload CSV Data", type=["csv"], accept_multiple_files=True
        )
        if not csv_files:
            return None

        return FileHandler._process_csv_files(csv_files)

    @staticmethod
    def handle_reference_snippet() -> None:
        """Handle reference snippet upload."""
        st.write("Upload a reference extension to compare with the generated output")
        reference_file = st.file_uploader(
            "Upload Reference Ontology Extension", type=["ttl", "txt"]
        )
        if reference_file is not None:
            content = reference_file.getvalue().decode("utf-8")
            AppState.get().reference_snippet = content
            st.success("Reference ontology extension uploaded successfully!")
            with st.expander("Preview Reference Extension"):
                st.code(content, language="turtle")
            log("Reference ontology extension uploaded and stored")

    @staticmethod
    def _process_csv_files(files: List) -> Optional[pd.DataFrame]:
        """Process and optionally merge CSV files."""
        dfs = []
        file_names = []
        for file in files:
            try:
                df = pd.read_csv(file, nrows=25, low_memory=False)
                dfs.append(df)
                file_names.append(file.name)  # Save file name
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                log(f"Error reading CSV file: {e}", "ERROR")

        if not dfs:
            st.error("No CSV files could be read.")
            return None

        # Preview each file
        FileHandler._preview_dataframes(dfs, file_names)

        # Handle merging
        return FileHandler._merge_dataframes(dfs)

    @staticmethod
    def _preview_dataframes(dfs: List[pd.DataFrame], file_names: List[str]) -> None:
        """Preview uploaded dataframes."""
        for df, name in zip(dfs, file_names):
            with st.expander(f"📄 {name} Info"):
                st.markdown(f"**Columns:** {', '.join(df.columns)}")
                st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    @staticmethod
    def _merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple dataframes if needed."""
        if len(dfs) == 1:
            st.info("Only one CSV uploaded. Using it as the data source.")
            with st.expander("📄 Preview Combined Data"):
                st.dataframe(dfs[0].head(5))
            return dfs[0]

        # Find common columns
        column_count = {}
        for df in dfs:
            for col in df.columns:
                column_count[col] = column_count.get(col, 0) + 1
        common_columns = [col for col, count in column_count.items() if count >= 2]

        if not common_columns:
            st.warning("No common columns found. Using first CSV as fallback.")
            return dfs[0]

        # Merge based on user selection
        st.info(f"Common columns found: {', '.join(common_columns)}")
        chosen_column = st.selectbox(
            "Select the column to merge on", options=common_columns
        )

        merge_dfs = [df for df in dfs if chosen_column in df.columns]
        if not merge_dfs:
            st.warning(
                "Selected column not found in all files. Using first CSV as fallback."
            )
            return dfs[0]

        # Perform merge
        result = merge_dfs[0]
        for df in merge_dfs[1:]:
            result = pd.merge(result, df, on=chosen_column, how="outer")
        st.success(f"CSV files merged on column: {chosen_column}.")
        with st.expander("📄 Preview Combined Data"):
            st.dataframe(result.head(5))
        return result
