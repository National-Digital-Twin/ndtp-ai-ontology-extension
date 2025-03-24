"""
Processing Module for Data Analysis

This module provides functions for ontology analysis and concept extraction using ChatGPT.
It implements several key functionalities:

1) CSV Analysis:
   - Analyzes CSV data to identify dataset themes, metrics, and characteristics
   - Extracts BORO (Business Objects Reference Ontology) triplets from CSV data

2) Concept Extraction:
   - Performs pseudo-NER (Named Entity Recognition) using ChatGPT to extract domain-relevant terms
   - Gathers usage patterns for extracted concepts
   - Classifies concepts according to BORO principles (entities vs states)

3) Extension Classification:
   - Classifies ontology extensions as subclass, relationship, state, datatype property, 
     or disposition
   - Provides explanations for classifications based on BORO principles

Main functions:
- analyze_csv_with_chatgpt(): Analyzes CSV data for themes and characteristics
- pseudo_ner_phrase_extraction(): Extracts domain-relevant terms
- extract_boro_triplets(): Analyzes CSV data using BORO principles
- gather_usage_patterns_and_subtypes(): Identifies usage patterns and subtypes
- classify_extension_type(): Classifies concepts according to ontology extension types

Usage:
    from extract import analyze_step, analyze_tri, extract_concepts_step
    import pandas as pd
    from openai import OpenAI

    client = OpenAI(api_key="YOUR_API_KEY")
    df = pd.read_csv("some_dataset.csv")
    
    # Analyze CSV data
    domain_theme = analyze_step(df, "gpt-3.5-turbo")
    
    # Extract BORO triplets
    triplets = analyze_tri(df, "gpt-3.5-turbo")
    
    # Extract concepts
    concepts = extract_concepts_step(df, "gpt-3.5-turbo")
"""

import json
import pandas as pd
from typing import List, Dict, Any
from openai import OpenAI


def analyze_csv_with_chatgpt(client: OpenAI, df: pd.DataFrame, model: str) -> str:
    """
    Analyze a DataFrame to identify dataset theme, domain-specific metrics, etc.
    Returns a dict with { "theme": ..., "metrics": ..., "characteristics": ..., "summary": ... }.
    """

    # Convert a small sample of the CSV to a string for the prompt
    partial_csv_preview = df.head(10)

    prompt = f"""
    You are an ontology expert. Below is a sample of a dataset in CSV format:
    {partial_csv_preview}

    1. Identify the overall theme or domain of this dataset.
    2. Suggest domain-specific metrics or characteristics.
    3. Provide a short summary of the dataset's possible uses or significance.

    Return JSON with keys: "theme", "metrics", "characteristics", "summary".
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def pseudo_ner_phrase_extraction(client: OpenAI, text: str, model: str) -> str:
    """
    Use ChatGPT to extract domain-relevant terms or phrases from the given text (pseudo-NER).
    Returns a list of unique terms/phrases.
    """

    prompt = f"""
    Perform a domain-focused NER or key-phrase extraction on the following df:
    {text}

    Return a list array of unique, domain-relevant terms. Return list only nothing else.
    """

    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content.strip()


def extract_boro_triplets(client: OpenAI, df: pd.DataFrame, model: str) -> str:
    """
    Use BORO (Business Objects Reference Ontology) principles to analyze the CSV data.
    Extract (head, relation, tail) triples clearly distinguishing:
        - Enduring entities (objects/entities persisting through time)
        - States or bounding states (temporal or condition-based entities)
        - Relationships or dispositions connecting entities

    Returns a JSON containing extracted triplets along with BORO-aligned reasoning.
    """

    partial_csv_preview = df.head(10)

    prompt = f"""
    You are an expert in ontology creation using BORO (Business Objects Reference Ontology).

    Here's a sample of data in CSV format:
    {partial_csv_preview}

    Perform the following steps:

    1. Identify enduring entities (entities persisting over time).
    2. Identify states or bounding states (temporary conditions or states of entities).
    3. Identify relevant relationships or dispositions connecting these entities.

    Based on your analysis, generate a list of meaningful (head, relation, tail) triples adhering to BORO principles.

    Return your analysis as a JSON with the following structure:
    {{
        "triplets": [{{"head": "...", "relation": "...", "tail": "..."}}],
        "boro_reasoning": "Explain clearly how each triplet aligns with BORO's classification of enduring entities, states, or relationships."
    }}
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def gather_usage_patterns_and_subtypes(
    client: OpenAI, concepts: List[str], domain_theme: str, model: str
) -> List[Dict[str, Any]]:
    """
    For each concept in 'concepts', gather usage patterns and propose specialized
    subtypes or dispositions with ChatGPT. References domain_theme for context.
    Returns a list of dicts: { "concept": c, "usagePatterns": [...], "proposedSubtypes": [...] }.
    """
    results = []
    for c in concepts:
        prompt = f"""
        You are an expert in the domain: {domain_theme}.
        The concept is: '{c}'.
        1. Describe typical usage or references in this domain.
        2. Suggest potential specialized subtypes or dispositions.

        Return JSON with keys: "usagePatterns" and "proposedSubtypes".
        """

        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}

        usage = parsed.get("usagePatterns", [])
        subs = parsed.get("proposedSubtypes", [])

        results.append({"concept": c, "usagePatterns": usage, "proposedSubtypes": subs})
    return results


def classify_extension_type(client: OpenAI, concept: str, model: str) -> str:
    """
    Ask ChatGPT to classify 'concept' as one of:
    'subclass', 'relationship', 'state', 'datatype property', or 'disposition'.
    Returns a string with that classification.
    """
    prompt = f"""
       You are an expert in BORO (Business Objects Reference Ontology). 
       Analyze the concept: "{concept}" to determine if it is:
        - An "enduring entity," which persists through time with a consistent identity,
        - Or a "state" (or "bounding state"), which refers to a temporary condition 
          or time-bound phase.
        
        After classifying the concept, propose the most appropriate ontology 
        extension type (for example, subclass, relationship, state, datatype property, 
        disposition, or another relevant category) that fits the BORO approach. 
        Briefly justify your classification and extension choice.
        
        **Return your answer as valid JSON** with the following keys:
        - "concept": (the concept name),
        - "classification": either "entity" or "state",
        - "extensionType": your chosen ontology extension type,
        - "explanation": a concise explanation (one or two sentences) summarizing why 
          you made this determination.

        **No additional text**—just the JSON object. For instance:

        
        "concept": "ExampleConcept",
        "classification": "entity",
        "extensionType": "subclass",
        "explanation": "Short rationale here."""

    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content.strip()


def analyze_step(client: OpenAI, csv_df: pd.DataFrame, model: Any) -> str:
    if csv_df is not None and not csv_df.empty:
        extract_boro_triplets(client=client, df=csv_df, model=model)
        domain_theme = analyze_csv_with_chatgpt(client=client, df=csv_df, model=model)
        return domain_theme
    return "UnknownTheme"


def analyze_tri(client: OpenAI, csv_df: pd.DataFrame, model: Any) -> str:
    if csv_df is not None and not csv_df.empty:
        tri = extract_boro_triplets(client=client, df=csv_df, model=model)
        return tri
    return "UnknownTheme"


def extract_concepts_step(client: OpenAI, csv_df: pd.DataFrame, model: Any) -> str:
    concepts = pseudo_ner_phrase_extraction(client=client, text=csv_df, model=model)
    return concepts


def gather_usage_step(
    client: OpenAI, concepts: List[str], domain_theme: str, model: Any
) -> List[Dict]:
    return gather_usage_patterns_and_subtypes(
        client=client, concepts=concepts, domain_theme=domain_theme, model=model
    )


def classify_extensions(
    client: OpenAI, usage_info: List[Dict], model: Any
) -> List[Dict]:
    blocks = []

    for item in usage_info:
        concept = item["concept"]
        extension_type = classify_extension_type(
            client=client, concept=concept, model=model
        )
        blocks.append(extension_type)

    return blocks
