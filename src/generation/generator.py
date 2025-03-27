# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

from openai import OpenAI
import pandas as pd
from typing import Optional


def generate_ontology_prompt(
    partial_csv_preview: pd.DataFrame,
    existing_analysis: str,
    existing_triples: str,
    extracted_concepts: str,
    usage: str,
    classified_extensions: str,
    base_ontology: str,
    extra_context: str,
    prompt: str,
    prompt2: str,
    previous_iteration: any,
    ontologist_feedback: str,
) -> str:
    """
    Generate a prompt for the ontology generator.

    Args:
        partial_csv_preview (pd.DataFrame): The partial CSV preview to use for the ontology extension.
        existing_analysis (str): The existing analysis of the data.
        existing_triples (str): The existing triples of the data.
        extracted_concepts (str): The extracted concepts of the data.
        usage (str): The usage of the data.
        classified_extensions (str): The classified extensions of the data.
        base_ontology (str): The base ontology to use for the ontology extension.
        extra_context (str): The extra context to use for the ontology extension.
        prompt (str): The prompt to use for the ontology extension.
        prompt2 (str): The prompt to use for the ontology extension.
        previous_iteration (any): The previous iteration to use for the ontology extension.
        ontologist_feedback (str): The feedback to use for the ontology extension.

    Returns:
        str: The generated prompt.
    """
    return f"""
    ### Role
    You are an expert ontology engineer specializing in BORO (Business Object Reference Ontology) principles.

    ### Task
    {prompt} {prompt2}

    ### Additional Context
    {extra_context}

    ### Data Sample
    ```
    {partial_csv_preview}
    ```

    ### Previously Extracted Insights
    - Analysis: {existing_analysis}
    - Triples: {existing_triples}
    - Concepts: {extracted_concepts}
    - Usage: {usage}
    - Classified Extensions: {classified_extensions}

    ### Base Ontology
    ```
    {base_ontology}
    ```

    ### Previous Iteration and Feedback
    - Previous Iteration Output:
    ```
    {"None" if previous_iteration is None else previous_iteration}
    ```
    - Ontologist Feedback:
    ```
    {"None" if ontologist_feedback is None else ontologist_feedback}
    ```
    
    ### Instructions for Current Iteration
    Please refine your previous output based on the feedback provided. If there is no feedback, provide an improved version of your last output. Ensure clarity, precision, and adherence to BORO principles.
    
    ### Return a TTL file and nothing else.
    """


def ontology_generator(
    client: OpenAI,
    df: pd.DataFrame,
    model: str,
    existing_analysis: str,
    existing_triples: str,
    extracted_concepts: str,
    usage: str,
    classified_extensions: str,
    base_ontology: str,
    extra_context: str,
    prompt: str,
    prompt2: str,
    ontologist_feedback: Optional[str] = None,
    previous_iteration: Optional[str] = None,
) -> str:
    """
    Generate an ontology extension using the provided prompt and feedback.

    Args:
        client (OpenAI): The OpenAI client to use for the ontology extension.
        df (pd.DataFrame): The data to use for the ontology extension.
        model (str): The model to use for the ontology extension.
        existing_analysis (str): The existing analysis of the data.
        existing_triples (str): The existing triples of the data.
        extracted_concepts (str): The extracted concepts of the data.
        usage (str): The usage of the data.
        classified_extensions (str): The classified extensions of the data.
        base_ontology (str): The base ontology to use for the ontology extension.
        extra_context (str): The extra context to use for the ontology extension.
        prompt (str): The prompt to use for the ontology extension.
        prompt2 (str): The prompt to use for the ontology extension.
        ontologist_feedback (str): The feedback to use for the ontology extension.
        previous_iteration (str): The previous iteration to use for the ontology extension.

    Returns:
        str: The generated ontology extension.
    """
    ontologist_feedback = ontologist_feedback or ""

    full_prompt = generate_ontology_prompt(
        partial_csv_preview=df,
        existing_analysis=existing_analysis,
        existing_triples=existing_triples,
        extracted_concepts=extracted_concepts,
        usage=usage,
        classified_extensions=classified_extensions,
        base_ontology=base_ontology,
        extra_context=extra_context,
        prompt=prompt,
        prompt2=prompt2,
        previous_iteration=previous_iteration,
        ontologist_feedback=ontologist_feedback,
    )

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": full_prompt}]
    )

    new_result = response.choices[0].message.content.strip()
    return new_result
