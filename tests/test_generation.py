"""
Tests for the generation module components.

This module contains tests for the LLM interface, comparison, and iteration
management functionality in the generation module.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.generation.llm_interface import (
    build_prompt_for_generation,
    generate_ttl_snippet,
    refine_instructions,
    compare_snippets,
    interpret_comparison_result,
    build_error_summary,
)
from src.generation.iteration import (
    load_checkpoint,
    save_checkpoint,
    get_iteration_history,
)


# ===== LLM Interface Tests =====


def test_build_prompt_for_generation():
    """Test that the prompt builder correctly formats the prompt."""
    # Test with basic inputs
    main_instructions = "Create an ontology."
    common_snippet = "# Common snippet content"
    csv_data = "col1,col2\nval1,val2"

    prompt = build_prompt_for_generation(main_instructions, common_snippet, csv_data)

    # Check that all components are included
    assert main_instructions in prompt
    assert common_snippet in prompt
    assert csv_data in prompt
    assert "Additional feedback" not in prompt

    # Test with feedback
    feedback = "Fix these issues."
    prompt_with_feedback = build_prompt_for_generation(
        main_instructions, common_snippet, csv_data, feedback
    )

    assert feedback in prompt_with_feedback
    assert "Additional feedback" in prompt_with_feedback


@patch("src.generation.llm_interface.client")
def test_generate_ttl_snippet(mock_client):
    """Test TTL snippet generation with mocked OpenAI client."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Generated TTL content"
    mock_client.chat.completions.create.return_value = mock_response

    # Call function
    result = generate_ttl_snippet("Test prompt")

    # Verify
    assert result == "Generated TTL content"
    mock_client.chat.completions.create.assert_called_once()

    # No need to test API key missing error since that's now handled at module level


@patch("src.generation.llm_interface.client")
def test_refine_instructions(mock_client):
    """Test instruction refinement with mocked OpenAI client."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Refined instructions"
    mock_client.chat.completions.create.return_value = mock_response

    # Test with error summary
    result = refine_instructions("Missing entities", "Original instructions")
    assert result == "Refined instructions"
    mock_client.chat.completions.create.assert_called_once()

    # Test with "No errors" - should return original instructions
    mock_client.chat.completions.create.reset_mock()
    result = refine_instructions("No errors.", "Original instructions")
    assert result == "Original instructions"
    mock_client.chat.completions.create.assert_not_called()


# ===== Comparison Tests =====


@patch("src.generation.llm_interface.client")
def test_compare_snippets(mock_client):
    """Test snippet comparison with mocked OpenAI client."""
    # Setup mock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "entities_missing_in_A": ["Entity1"],
            "entities_missing_in_B": [],
            "states_missing_in_A": ["State1"],
            "states_missing_in_B": [],
        }
    )
    mock_client.chat.completions.create.return_value = mock_response

    # Call function
    result = compare_snippets("Generated TTL", "Reference TTL")

    # Verify
    assert "entities_missing_in_A" in result
    assert result["entities_missing_in_A"] == ["Entity1"]
    assert "states_missing_in_A" in result
    assert result["states_missing_in_A"] == ["State1"]
    mock_client.chat.completions.create.assert_called_once()

    # Test invalid JSON response
    mock_response.choices[0].message.content = "Not valid JSON"
    result = compare_snippets("Generated TTL", "Reference TTL")
    assert "error" in result
    assert "raw_output" in result


def test_interpret_comparison_result():
    """Test interpretation of comparison results."""
    # Test with normal input
    raw_result = {
        "entities_missing_in_A": ["Entity1", "Entity2"],
        "entities_missing_in_B": ["Entity3"],
        "states_missing_in_A": ["State1"],
        "states_missing_in_B": ["State2", "State3"],
    }

    result = interpret_comparison_result(raw_result)

    assert result["entities_missing"] == ["Entity1", "Entity2"]
    assert result["entities_extra"] == ["Entity3"]
    assert result["states_missing"] == ["State1"]
    assert result["states_extra"] == ["State2", "State3"]

    # Test with error input
    error_result = {"error": "Test error", "raw_output": "Error output"}
    result = interpret_comparison_result(error_result)
    assert result["error"] == "Test error"
    assert result["raw_output"] == "Error output"


def test_build_error_summary():
    """Test building error summary from comparison results."""
    # Test with all types of errors
    comparison_result = {
        "entities_missing": ["Entity1", "Entity2"],
        "entities_extra": ["Entity3"],
        "states_missing": ["State1"],
        "states_extra": ["State2", "State3"],
    }

    summary = build_error_summary(comparison_result)

    assert "Missing Entities: Entity1, Entity2" in summary
    assert "Extra Entities: Entity3" in summary
    assert "Missing States: State1" in summary
    assert "Extra States: State2, State3" in summary

    # Test with no errors
    empty_result = {
        "entities_missing": [],
        "entities_extra": [],
        "states_missing": [],
        "states_extra": [],
    }

    summary = build_error_summary(empty_result)
    assert summary == "No errors."

    # Test with partial errors
    partial_result = {
        "entities_missing": ["Entity1"],
        "entities_extra": [],
        "states_missing": [],
        "states_extra": [],
    }

    summary = build_error_summary(partial_result)
    assert summary == "Missing Entities: Entity1"


# ===== Iteration Tests =====


def test_load_checkpoint(tmp_path):
    """Test loading checkpoint from file."""
    checkpoint_file = tmp_path / "checkpoint.json"

    # Test with non-existent file
    state = load_checkpoint(checkpoint_file, "Default instructions")
    assert state["current_instructions"] == "Default instructions"
    assert state["iteration"] == 0
    assert state["iteration_feedback"] == ""
    assert state["last_generated_snippet"] == ""

    # Test with existing file
    test_state = {
        "current_instructions": "Test instructions",
        "iteration": 2,
        "iteration_feedback": "Test feedback",
        "last_generated_snippet": "Test snippet",
    }

    with checkpoint_file.open("w") as f:
        json.dump([test_state], f)

    loaded_state = load_checkpoint(checkpoint_file)
    assert loaded_state["current_instructions"] == "Test instructions"
    assert loaded_state["iteration"] == 2
    assert loaded_state["iteration_feedback"] == "Test feedback"
    assert loaded_state["last_generated_snippet"] == "Test snippet"

    # Test with invalid JSON
    with checkpoint_file.open("w") as f:
        f.write("Not valid JSON")

    state = load_checkpoint(checkpoint_file, "Default instructions")
    assert state["current_instructions"] == "Default instructions"


def test_save_checkpoint(tmp_path):
    """Test saving checkpoint to file."""
    checkpoint_file = tmp_path / "checkpoint.json"

    # Test saving to new file
    test_state = {
        "current_instructions": "Test instructions",
        "iteration": 1,
        "iteration_feedback": "Test feedback",
        "last_generated_snippet": "Test snippet",
    }

    save_checkpoint(checkpoint_file, test_state)

    with checkpoint_file.open("r") as f:
        loaded = json.load(f)

    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert loaded[0] == test_state

    # Test appending to existing file
    test_state2 = {
        "current_instructions": "Updated instructions",
        "iteration": 2,
        "iteration_feedback": "Updated feedback",
        "last_generated_snippet": "Updated snippet",
    }

    save_checkpoint(checkpoint_file, test_state2)

    with checkpoint_file.open("r") as f:
        loaded = json.load(f)

    assert len(loaded) == 2
    assert loaded[0] == test_state
    assert loaded[1] == test_state2

    # Test with nested directory that doesn't exist
    nested_file = tmp_path / "nested" / "checkpoint.json"
    save_checkpoint(nested_file, test_state)
    assert nested_file.exists()


def test_get_iteration_history(tmp_path):
    """Test retrieving iteration history."""
    checkpoint_file = tmp_path / "checkpoint.json"

    # Test with non-existent file
    history = get_iteration_history(checkpoint_file)
    assert history == []

    # Test with valid file
    test_states = [
        {
            "current_instructions": "Initial instructions",
            "iteration": 1,
            "iteration_feedback": "",
            "last_generated_snippet": "Initial snippet",
        },
        {
            "current_instructions": "Updated instructions",
            "iteration": 2,
            "iteration_feedback": "Feedback",
            "last_generated_snippet": "Updated snippet",
        },
    ]

    with checkpoint_file.open("w") as f:
        json.dump(test_states, f)

    history = get_iteration_history(checkpoint_file)
    assert len(history) == 2
    assert history == test_states

    # Test with invalid JSON
    with checkpoint_file.open("w") as f:
        f.write("Not valid JSON")

    history = get_iteration_history(checkpoint_file)
    assert history == []
