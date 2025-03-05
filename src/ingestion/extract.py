"""
Entity Extraction Module for Ontology Development

This module provides functionality to extract candidate entities from tabular data
for ontology development. It uses multiple extraction methods including:

1. spaCy Named Entity Recognition (NER) for identifying entities in text
2. ChatGPT API for intelligent entity extraction and fuzzy matching

The module supports processing CSV and JSON files, extracting entities from textual
columns, and optionally matching these entities against existing ontology constraints
using fuzzy matching algorithms.

Key functions:
- extract_entities_spacy: Extract entities using spaCy NER
- extract_entities_chatgpt: Extract entities using ChatGPT
- fuzzy_matches: Match candidate entities against ontology constraints using RapidFuzz
- fuzzy_matches_chatgpt: Match candidate entities using ChatGPT
- process_data: Main function to process data files and extract entities

Usage:
    results = process_data(
        file_path='data.csv',
        output_path='entities.json',
        method='both',
        ontology_constraints_path='ontology.json',
        fuzzy_threshold=80,
        fuzzy_method='rapidfuzz'
    )
"""

import os
import json
import pandas as pd
import spacy
import openai
from rapidfuzz import fuzz

from .helpers import read_data


# Set up OpenAI API key from the environment (non-fatal if missing)
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    print("Warning: OPENAI_API_KEY is not set. ChatGPT queries will be skipped.")

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")


def extract_entities_spacy(text):
    """
    Extract candidate entities from text using spaCy NER.
    Returns a sorted list of unique entity strings.
    """
    doc = nlp(text)
    entities = sorted({ent.text for ent in doc.ents})
    return entities


def extract_entities_chatgpt(column_name, sample_values):
    """
    Build a prompt for ChatGPT based on the column name and sample values.
    Queries ChatGPT (GPT-4o-mini) to identify candidate ontology entities.
    Returns a sorted list of unique entity names.
    """
    if not openai.api_key:
        print(
            f"Warning: ChatGPT extraction skipped for column '{column_name}' because OPENAI_API_KEY is not set."
        )
        return []

    prompt = (
        f"Given the following sample values for the column '{column_name}': {sample_values}. "
        "Identify potential entities or concepts that could be represented in an ontology. "
        'Return a JSON array of candidate entity names. For example: ["Entity1", "Entity2"].'
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an ontology expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )
    except Exception as e:
        print("Error calling OpenAI API:", e)
        return []

    try:
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error accessing response fields:", e)
        return []

    try:
        entities = json.loads(answer)
        if not isinstance(entities, list):
            entities = []
    except Exception as e:
        print("Error parsing JSON from ChatGPT response:", e)
        entities = [line.strip() for line in answer.splitlines() if line.strip()]

    return sorted(set(entities))


def fuzzy_matches(candidate_list, constraints_list, threshold=80):
    """
    Return a sorted list of ontology terms from constraints_list that have a fuzzy
    match (using rapidfuzz.token_sort_ratio) to any of the candidate_list entries.
    """
    matches = set()
    for candidate in candidate_list:
        for constraint in constraints_list:
            if fuzz.token_sort_ratio(candidate, constraint) >= threshold:
                matches.add(constraint)
                break
    return sorted(matches)


def fuzzy_matches_chatgpt(candidate_list, constraints_list, threshold=80):
    """
    Use ChatGPT to perform fuzzy matching.
    Constructs a prompt with the candidate entities and ontology constraints and
    asks ChatGPT to return a JSON array of matching ontology terms.
    """
    if not openai.api_key:
        print(
            "Warning: OPENAI_API_KEY is not set. Cannot perform fuzzy matching via ChatGPT."
        )
        return []

    prompt = (
        f"Here are candidate ontology entities: {candidate_list}.\n"
        f"Here are known ontology terms: {constraints_list}.\n"
        f"Using a similarity threshold of {threshold} (0 to 100), "
        "please return a JSON array of ontology terms from the known list that are similar to the candidate entities."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in ontology matching.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )
    except Exception as e:
        print("Error calling OpenAI API for fuzzy matching:", e)
        return []

    try:
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error accessing response fields for fuzzy matching:", e)
        return []

    try:
        matches = json.loads(answer)
        if not isinstance(matches, list):
            matches = []
    except Exception as e:
        print("Error parsing JSON from ChatGPT fuzzy matching response:", e)
        matches = [line.strip() for line in answer.splitlines() if line.strip()]

    return sorted(set(matches))


def process_data(
    file_path,
    output_path=None,
    method="both",
    ontology_constraints_path=None,
    fuzzy_threshold=80,
    fuzzy_method="rapidfuzz",
):
    """
    Process a tabular data file (CSV or JSON) to extract candidate ontology entities
    from each textual column.

    For each column, the output format is:
      {
        "sample_values": [ ... ],
        "entities": {
          "spacy": [ ... ],
          "chatgpt": [ ... ],
          "matches": {
             "spacy": [ ... ],
             "chatgpt": [ ... ]
          }
        }
      }

    If an ontology_constraints_path is provided (a JSON file with an "entities" field
    containing keys "classes" and "properties"), the function will use fuzzy matching to
    compare each candidate entity against these ontology constraints.

    Parameters:
      file_path: Path to the data file.
      output_path: Optional path to save the extraction results as JSON.
      method: Extraction method to use. Acceptable values are:
              - "spacy": Use only spaCy extraction.
              - "chatgpt": Use only ChatGPT extraction.
              - "both": Use both methods.
      ontology_constraints_path: Optional path to a JSON file that exports ontology entities.
             The JSON should have an "entities" field with keys "classes" and "properties".
      fuzzy_threshold: Similarity threshold (0–100) for fuzzy matching. Default is 80.
      fuzzy_method: Method for fuzzy matching. Acceptable values:
                    - "rapidfuzz" (default)
                    - "chatgpt"

    Returns:
      A dictionary with the extraction results.
    """
    data = read_data(file_path)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "Candidate entity extraction for tabular data is supported only for CSV or JSON files."
        )

    # Load ontology constraints if provided.
    constraints_set = None
    if ontology_constraints_path:
        with open(ontology_constraints_path, "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
        constraints_set = set(ontology_data.get("entities", {}).get("classes", []))
        constraints_set.update(ontology_data.get("entities", {}).get("properties", []))

    extraction_results = {}

    for col in data.columns:
        if data[col].dtype == object:  # Process only textual columns
            unique_vals = data[col].dropna().unique()
            sample_vals = list(unique_vals[:5])  # Use up to 5 sample values
            sample_text = " ".join(map(str, sample_vals))

            result = {}
            if method in ["both", "spacy"]:
                spacy_result = extract_entities_spacy(sample_text)
                result["spacy"] = spacy_result
            else:
                result["spacy"] = []

            if method in ["both", "chatgpt"]:
                chatgpt_result = extract_entities_chatgpt(col, sample_vals)
                result["chatgpt"] = chatgpt_result
            else:
                result["chatgpt"] = []

            # Compute fuzzy matches if ontology constraints are provided.
            matches = {"spacy": [], "chatgpt": []}
            if constraints_set is not None:
                constraints_list = sorted(constraints_set)
                if fuzzy_method == "rapidfuzz":
                    matches["spacy"] = fuzzy_matches(
                        result["spacy"], constraints_list, threshold=fuzzy_threshold
                    )
                    matches["chatgpt"] = fuzzy_matches(
                        result["chatgpt"], constraints_list, threshold=fuzzy_threshold
                    )
                elif fuzzy_method == "chatgpt":
                    matches["spacy"] = fuzzy_matches_chatgpt(
                        result["spacy"], constraints_list, threshold=fuzzy_threshold
                    )
                    matches["chatgpt"] = fuzzy_matches_chatgpt(
                        result["chatgpt"], constraints_list, threshold=fuzzy_threshold
                    )
            result["matches"] = matches

            extraction_results[col] = {"sample_values": sample_vals, "entities": result}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extraction_results, f, indent=2)

    return extraction_results
