import streamlit as st
from app.state.app_state import AppState
from app.components.file_handlers import FileHandler


def show():
    """Handle Tab 1: File uploads and configuration"""
    state = AppState.get()

    st.subheader("Upload Your Files")
    FileHandler.handle_common_snippet()
    df = FileHandler.handle_csv_upload()
    if df is not None:
        state.csv_data = df

    state.status["data_input"] = (
        not state.csv_data.empty if state.csv_data is not None else False
    )
