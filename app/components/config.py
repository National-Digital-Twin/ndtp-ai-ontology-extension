import os
import streamlit as st
from dataclasses import dataclass

from app.utils.logging import log


@dataclass
class ModelConfig:
    name: str
    description: str
    max_tokens: int
    temperature: float


class ConfigHandler:
    @staticmethod
    def load_default_instructions(name: str) -> str:
        """Load the default instructions from the template file."""
        try:
            instructions_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "app",
                "templates",
                name,
            )
            if os.path.exists(instructions_path):
                with open(instructions_path, "r") as f:
                    return f.read()
            else:
                log(f"Instructions template not found: {instructions_path}", "WARNING")
                return ""
        except Exception as e:
            log(f"Error loading instructions: {e}", "ERROR")
            return ""

    @staticmethod
    def handle_instructions(name: str, label: str = "Edit Instructions") -> str:
        """Handle instruction loading and editing."""
        state_key = f"instructions_{name}"
        if state_key not in st.session_state:
            st.session_state[state_key] = ConfigHandler.load_default_instructions(name)

        return st.text_area(label, key=state_key, height=300)
