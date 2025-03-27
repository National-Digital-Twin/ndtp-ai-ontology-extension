# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
LLM Interface Module for Ontology Generation

This module provides functions for interacting with LLMs to generate, refine,
and compare ontology snippets in Turtle format.
"""

import json
from openai import OpenAI


def build_prompt_for_generation(
    instructions: str,
    common_snippet: str,
    csv_data: str,
    iteration_feedback: str = "",
) -> str:
    """
    Combine the main instructions, common snippet, CSV data, and iteration feedback
    into a single prompt for generating a Turtle snippet.

    Args:
        instructions: The primary instructions for ontology creation
        common_snippet: The common ontology template (snippet A)
        csv_data: Sample data from the CSV file
        iteration_feedback: Feedback from previous iterations (if any)

    Returns:
        A formatted prompt string for the LLM
    """
    prompt = (
        f"{instructions}\n\n"
        "### Here is snippet A (common.ttl) content:\n"
        f"{common_snippet}\n\n"
        "### Here is the CSV data:\n"
        f"{csv_data}\n\n"
    )
    if iteration_feedback.strip():
        prompt += (
            "### Additional feedback from previous iteration(s):\n"
            f"{iteration_feedback}\n\n"
            "Please refine or correct the snippet accordingly.\n"
        )
    return prompt


def generate_ttl_snippet(
    client: OpenAI, prompt_text: str, model: str = "o3-mini"
) -> str:
    """
    Calls an LLM to produce a Turtle snippet based on the provided prompt.

    Args:
        client: The OpenAI client
        prompt_text: The formatted prompt text
        model: The OpenAI model to use (default: "o3-mini")

    Returns:
        The generated Turtle snippet as a string
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI.",
            },
            {"role": "user", "content": prompt_text},
        ],
    )
    return response.choices[0].message.content


def refine_instructions(
    client: OpenAI,
    error_summary: str,
    original_instructions: str,
    model: str = "o3-mini",
) -> str:
    """
    Refine ontology creation instructions based on error feedback.

    Args:
        client: The OpenAI client
        error_summary: A summary of errors or issues to address
        original_instructions: The current instructions to refine
        model: The OpenAI model to use (default: "o3-mini")

    Returns:
        Refined instructions as a string
    """
    if error_summary.strip() == "No errors.":
        return original_instructions

    refinement_prompt = f"""
Below are the current ontology-creation instructions, followed by a summary of issues discovered:

CURRENT INSTRUCTIONS:
{original_instructions}

ISSUES DISCOVERED:
{error_summary}

Please revise or refine these instructions to address the missing/extra entities/states,
without revealing or copying from any external snippet. The revised instructions
must still create a standalone Turtle snippet that follows the 4D pattern rules.
Return only the refined instructions, with no additional commentary.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI.",
            },
            {"role": "user", "content": refinement_prompt},
        ],
    )
    return response.choices[0].message.content


def compare_snippets(
    client: OpenAI, generated_ttl: str, reference_ttl: str, model: str = "o3-mini"
) -> dict:
    """
    Use an LLM to compare generated TTL with a reference TTL, focusing on
    entity classes and state classes.

    Args:
        client: The OpenAI client
        generated_ttl: The generated Turtle snippet
        reference_ttl: The reference Turtle snippet to compare against
        model: The OpenAI model to use (default: "o3-mini")

    Returns:
        A dictionary containing comparison results or an error message
    """
    compare_prompt = f"""
You are given two Turtle (TTL) snippets, snippet_A and snippet_B.
Identify entity classes (subclasses of ies:Asset, ies:Element, or ies:Location)
and state classes (subclasses of ies:State) in each snippet.

Return a JSON object with:
- "entities_missing_in_A": array of entity classes in snippet_B but not in snippet_A
- "entities_missing_in_B": array of entity classes in snippet_A but not in snippet_B
- "states_missing_in_A": array of state classes in snippet_B but not in snippet_A
- "states_missing_in_B": array of state classes in snippet_A but not in snippet_B

No extra commentary. Only valid JSON.

snippet_A:
{generated_ttl}

snippet_B:
{reference_ttl}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise JSON generator."},
            {"role": "user", "content": compare_prompt},
        ],
    )
    raw_output = response.choices[0].message.content

    try:
        parsed = json.loads(raw_output)
        return parsed
    except json.JSONDecodeError:
        return {"error": "LLM did not produce valid JSON", "raw_output": raw_output}


def interpret_comparison_result(comparison_result: dict) -> dict:
    """
    Convert the LLM comparison result into a standardized form.

    Args:
        comparison_result: The raw comparison result from compare_snippets

    Returns:
        A standardized dictionary with missing and extra entities/states
    """
    if "error" in comparison_result:
        return comparison_result  # pass through the error as-is

    return {
        "entities_missing": comparison_result.get("entities_missing_in_A", []),
        "entities_extra": comparison_result.get("entities_missing_in_B", []),
        "states_missing": comparison_result.get("states_missing_in_A", []),
        "states_extra": comparison_result.get("states_missing_in_B", []),
    }


def build_error_summary(comparison_result: dict) -> str:
    """
    Build a short text describing missing/extra entities/states.

    Args:
        comparison_result: The standardized comparison result

    Returns:
        A string summarizing the errors found
    """
    lines = []
    if comparison_result.get("entities_missing"):
        lines.append(
            "Missing Entities: " + ", ".join(comparison_result["entities_missing"])
        )
    if comparison_result.get("entities_extra"):
        lines.append(
            "Extra Entities: " + ", ".join(comparison_result["entities_extra"])
        )
    if comparison_result.get("states_missing"):
        lines.append(
            "Missing States: " + ", ".join(comparison_result["states_missing"])
        )
    if comparison_result.get("states_extra"):
        lines.append("Extra States: " + ", ".join(comparison_result["states_extra"]))

    if not lines:
        return "No errors."
    return "; ".join(lines)
