import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from app.utils.logging import log


@dataclass
class AppState:
    """Centralized state management for the application."""

    # LLM state
    client: Optional[OpenAI] = None

    # File-related state
    common_snippet: Optional[str] = None
    csv_data: Optional[pd.DataFrame] = None
    reference_snippet: Optional[str] = None

    # Processing state
    processing_results: Dict = field(default_factory=dict)

    # Generation state
    generated_snippet: Optional[str] = None
    new_output: Optional[str] = None
    current_iteration: int = 0
    iteration_history: List = field(default_factory=list)

    # Evaluation state
    clean_ontology: Optional[str] = None
    error_log: Optional[str] = None
    discussion_result: Optional[str] = None
    discussion_error_message: Optional[str] = None
    comparison_result: Optional[str] = None

    # Configuration
    model: str = "o3-mini"
    output_filename: str = "ontology.ttl"
    chunk_start: int = 0
    chunk_size: int = 25
    chunk_columns: List[str] = field(default_factory=list)
    running: bool = False
    extension_saved: bool = False

    # Tab state
    status = {
        "data_input": False,
        "data_processing": False,
        "ontology_generation": False,
        "evaluation": False,
    }

    @classmethod
    def initialize(cls) -> None:
        """Initialize all required session state variables."""
        if "app_state" not in st.session_state:
            st.session_state.app_state = cls()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.app_state.client = OpenAI(api_key=api_key)
            log("Initialized application state")

    @staticmethod
    def get() -> "AppState":
        """Get the current application state."""
        if "app_state" not in st.session_state:
            AppState.initialize()
        return st.session_state.app_state
