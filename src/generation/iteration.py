# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Iteration Management Module

This module provides functions for managing the iterative workflow of ontology
generation, including checkpoint management for resuming iterations.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_checkpoint(file_path: Path, default_instructions: str = "") -> dict:
    """
    Load the iteration checkpoint from a JSON file if it exists.

    Args:
        file_path: Path to the checkpoint file
        default_instructions: Default instructions to use if no checkpoint exists

    Returns:
        The most recent checkpoint state as a dictionary
    """
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            try:
                checkpoints = json.load(f)
                if isinstance(checkpoints, list) and checkpoints:
                    return checkpoints[-1]
            except json.JSONDecodeError:
                pass

    # File doesn't exist or contains no valid checkpoints; create a default checkpoint
    default_state = {
        "current_instructions": default_instructions,
        "iteration": 0,
        "iteration_feedback": "",
        "last_generated_snippet": "",
    }
    save_checkpoint(file_path, default_state)
    return default_state


def save_checkpoint(file_path: Path, state: dict) -> None:
    """
    Append the current iteration checkpoint to a JSON file.

    Args:
        file_path: Path to the checkpoint file
        state: Current iteration state to save
    """
    checkpoints: List[Dict[str, Any]] = []

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            try:
                checkpoints = json.load(f)
                if not isinstance(checkpoints, list):
                    checkpoints = []
            except json.JSONDecodeError:
                checkpoints = []

    checkpoints.append(state)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2)


def get_iteration_history(file_path: Path) -> List[dict]:
    """
    Get the complete history of iterations from the checkpoint file.

    Args:
        file_path: Path to the checkpoint file

    Returns:
        A list of all checkpoint states
    """
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as f:
        try:
            checkpoints = json.load(f)
            if isinstance(checkpoints, list):
                return checkpoints
            return []
        except json.JSONDecodeError:
            return []
