import os
import re
from datetime import datetime
import streamlit as st
import sys
from openai import OpenAI

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.generation.llm_interface import (
    build_prompt_for_generation,
    generate_ttl_snippet,
    compare_snippets,
)
from app.utils.logging import log


def get_openai_client():
    """Get OpenAI client with API key from Streamlit secrets or environment variable."""
    log("Initializing OpenAI client")
    # Try to get API key from Streamlit secrets
    api_key = None

    # Check if we're running in Streamlit
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        log("Using API key from Streamlit secrets")
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        # Fall back to environment variable
        log("Attempting to use API key from environment variable")
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        log("ERROR: OpenAI API key not found", level="ERROR")
        st.error(
            "OpenAI API key not found. Please add it to .streamlit/secrets.toml or set the OPENAI_API_KEY environment variable."
        )
        st.stop()

    log("OpenAI client initialized successfully")
    return OpenAI(api_key=api_key)


client = get_openai_client()


def validate_ttl(ttl_snippet):
    """
    Basic validation of a Turtle snippet.
    Returns a list of errors, or an empty list if no errors are found.

    Note: This is a simplified validation. For production use,
    consider using a proper RDF library like rdflib.
    """
    errors = []

    # Check for basic syntax issues
    if not ttl_snippet.strip():
        errors.append("Empty TTL snippet")
        return errors

    # Check for unclosed brackets or parentheses
    if ttl_snippet.count("(") != ttl_snippet.count(")"):
        errors.append("Mismatched parentheses")

    if ttl_snippet.count("[") != ttl_snippet.count("]"):
        errors.append("Mismatched square brackets")

    if ttl_snippet.count("{") != ttl_snippet.count("}"):
        errors.append("Mismatched curly braces")

    # Check for missing semicolons or periods at the end of statements
    lines = ttl_snippet.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if (
            line
            and not line.startswith("#")
            and not line.endswith(";")
            and not line.endswith(".")
            and not line.endswith("[")
            and not line.endswith("{")
        ):
            if i < len(lines) - 1 and not (
                lines[i + 1].strip().startswith("]")
                or lines[i + 1].strip().startswith("}")
            ):
                errors.append(
                    f"Line {i+1} may be missing a semicolon or period: {line}"
                )

    # Check for common prefix declarations
    if not re.search(r"@prefix\s+:\s+<.*>", ttl_snippet) and not re.search(
        r"@base\s+<.*>", ttl_snippet
    ):
        errors.append("Missing base prefix declaration")

    return errors


def extract_ttl_from_response(response):
    """
    Extract the TTL snippet from the model's response.
    Handles cases where the model might include markdown formatting.
    """
    # Try to extract code from markdown code blocks
    code_block_pattern = r"```(?:ttl|turtle)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, response)

    if matches:
        return matches[0].strip()

    # If no code blocks found, return the entire response
    return response.strip()


def generate_ontology_iteration(
    common_snippet,
    csv_data,
    instructions,
    model="gpt-4o-mini",
    human_feedback="",
    previous_snippet=None,
):
    """
    Generate a single iteration of the ontology.

    Args:
        common_snippet (str): The common snippet template
        csv_data (pd.DataFrame): The CSV data
        instructions (str): The instructions for generation
        model (str): The OpenAI model to use
        human_feedback (str): Optional feedback from the user
        previous_snippet (str): Optional previous iteration of the ontology

    Returns:
        dict: A dictionary containing the generated snippet and any errors
    """
    log(f"Starting ontology generation iteration with model: {model}")

    # Convert DataFrame to CSV string
    log("Converting DataFrame to CSV string")
    csv_str = csv_data.to_csv(index=False)
    log(f"CSV data converted: {len(csv_str)} characters")

    # Use the existing build_prompt_for_generation function
    log("Building prompt for generation")
    prompt = build_prompt_for_generation(
        instructions=instructions,
        common_snippet=common_snippet,
        csv_data=csv_str,
        iteration_feedback=human_feedback if human_feedback else "",
    )
    log(f"Prompt built: {len(prompt)} characters")

    # Generate the snippet using the existing function
    log(f"Generating TTL snippet with {model}")
    client = get_openai_client()
    response = generate_ttl_snippet(client, prompt, model=model)
    log("TTL snippet generated successfully")

    # Extract the TTL snippet
    log("Extracting TTL from response")
    generated_snippet = extract_ttl_from_response(response)
    log(f"TTL extracted: {len(generated_snippet)} characters")

    # Validate the generated snippet
    log("Validating generated TTL")
    errors = validate_ttl(generated_snippet)
    if errors:
        log(f"Found {len(errors)} validation errors", level="WARNING")
    else:
        log("TTL validation passed with no errors")

    # Compare with reference snippet if provided
    comparison_result = None
    if previous_snippet:
        log("Comparing with previous snippet")
        try:
            comparison_result = compare_snippets(
                client, generated_snippet, previous_snippet, model=model
            )
            log("Comparison completed successfully")
        except Exception as e:
            log(f"Error comparing snippets: {str(e)}", level="ERROR")
            st.warning(f"Could not compare snippets: {e}")

    log("Ontology generation iteration completed")
    return {
        "generated_snippet": generated_snippet,
        "errors": errors,
        "comparison": comparison_result,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
