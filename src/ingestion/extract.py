# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Entity Extraction Module for Ontology Development

This module provides functionality to extract candidate ontology terms from tabular data
for ontology development. It supports classifying candidate terms as either Entities or States.
If classification is activated (classify_candidates=True), the ChatGPT prompts are adjusted
to instruct the model to return a JSON array of objects, where each object includes a "term"
and a "classification" (either "entity" or "state").

It uses multiple extraction methods including:
1. spaCy Named Entity Recognition (NER) for identifying terms in text.
2. OpenAI ChatGPT API for intelligent extraction and fuzzy (semantic) matching.
   For ChatGPT-based fuzzy matching, a HYBRID approach is used:
      - Local pre-filtering with RapidFuzz to select the top-N likely matches.
      - Chunking the pre-filtered terms to keep prompts small, then performing semantic matching via ChatGPT.

A configuration parameter, `chatgpt_model`, is available to choose the ChatGPT API model
(e.g., "gpt-4o-mini" by default). The `candidate_type` parameter (default "entity")
controls the basic candidate term wording, and the `classify_candidates` flag adds classification
instructions to the prompt and output.

Additionally, a secondary verification method is included. After extraction, the output is
checked to ensure that it exactly follows the structure:
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
If not, a query is sent to ChatGPT to fix the JSON structure.

Usage Example:
    results = process_data(
        file_path='data/raw/mydata.csv',
        output_path='results/candidate_entities.json',
        method='both',
        ontology_constraints_path='data/ontologies/ontology_entities.json',
        fuzzy_threshold=80,
        fuzzy_method='chatgpt',
        chatgpt_model='gpt-4o-mini',
        candidate_type='entity',       # or "state"
        classify_candidates=True,        # Activate classification in the output
        chunk_size=100,
        top_n=20,
        verify_structure=True           # Activate secondary verification/fixing step
    )
"""

import os
import json
import pandas as pd
import spacy
from rapidfuzz import fuzz
from openai import OpenAI

from .helpers import read_data

# Attempt to load the spaCy model; if unavailable, skip spaCy extraction.
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(
        "Warning: spaCy model 'en_core_web_sm' not found. spaCy-based extraction will be skipped."
    )
    nlp = None


def extract_entities_spacy(text):
    """
    Extract candidate terms from text using spaCy NER.
    Returns a sorted list of unique term strings.
    If spaCy is not available, returns an empty list.
    """
    if nlp is None:
        print("Warning: spaCy model not loaded; skipping spaCy extraction.")
        return []
    doc = nlp(text)
    entities = sorted({ent.text for ent in doc.ents})
    return entities


def extract_entities_chatgpt(
    client: OpenAI,
    column_name: str,
    sample_values: list[str],
    candidate_type: str = "entity",
    chatgpt_model: str = "gpt-4o-mini",
    classify_candidates: bool = False,
):
    """
    Build a prompt for ChatGPT based on the column name and sample values.
    Queries ChatGPT (using the specified model) to identify candidate terms.

    If classify_candidates is True, the prompt instructs ChatGPT to classify each candidate
    as either an "entity" or a "state", and to return a JSON array of objects with keys "term"
    and "classification". Otherwise, a simple JSON array of term strings is expected.

    Returns a sorted list of candidate terms (or objects if classification is active).

    Args:
        client: OpenAI client instance
        column_name: Name of the column to extract entities from
        sample_values: List of sample values from the column
        candidate_type: Type of candidate term to extract (default "entity")
        chatgpt_model: ChatGPT model to use (default "gpt-4o-mini")
        classify_candidates: Whether to classify candidates (default False)

    Returns:
        Sorted list of candidate terms or objects with classification
    """
    if not client or not client.api_key:
        print(
            f"Warning: ChatGPT extraction skipped for column '{column_name}' because OPENAI_API_KEY is not set."
        )
        return []

    if classify_candidates:
        prompt = (
            f"Given the following sample values for the column '{column_name}': {sample_values}. "
            f"Identify potential {candidate_type}s or concepts that could be represented in an ontology. "
            "For each candidate, classify it as either 'entity' or 'state'. "
            'Return a JSON array of objects, each with keys "term" and "classification". '
            "Return only the JSON array without any markdown formatting or extra text."
        )
    else:
        prompt = (
            f"Given the following sample values for the column '{column_name}': {sample_values}. "
            f"Identify potential {candidate_type}s or concepts that could be represented in an ontology. "
            "Return a JSON array of candidate names. "
            "Return only the JSON array without any markdown formatting or extra text."
        )

    try:
        response = client.chat.completions.create(
            model=chatgpt_model,
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
        terms = json.loads(answer)
    except Exception as e:
        print("Error parsing JSON from ChatGPT response:", e)
        terms = [line.strip() for line in answer.splitlines() if line.strip()]

    if classify_candidates:
        try:
            return sorted(terms, key=lambda x: x.get("term", ""))
        except Exception:
            return terms
    else:
        return sorted(set(terms))


def fuzzy_matches(candidate_list, constraints_list, threshold=80):
    """
    Return a sorted list of ontology terms from constraints_list that have a fuzzy
    match (using rapidfuzz.token_sort_ratio) to any of the candidate_list entries.
    Uses string-based matching.
    """
    matches = set()
    for candidate in candidate_list:
        candidate_str = (
            candidate.get("term", "") if isinstance(candidate, dict) else str(candidate)
        )
        for constraint in constraints_list:
            if fuzz.token_sort_ratio(candidate_str, str(constraint)) >= threshold:
                matches.add(str(constraint))
                break
    return sorted(matches)


def get_top_n_rapidfuzz(candidate, constraints_list, top_n=20):
    """
    For a single candidate term, returns the top N constraints by RapidFuzz similarity.
    If the candidate is a dictionary, uses its 'term' key.
    Sorted by descending similarity.
    """
    candidate_str = (
        candidate.get("term", "") if isinstance(candidate, dict) else str(candidate)
    )
    scored = [
        (str(constraint), fuzz.token_sort_ratio(candidate_str, str(constraint)))
        for constraint in constraints_list
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [constraint for constraint, score in scored[:top_n]]


def fuzzy_matches_chatgpt(
    client: OpenAI,
    candidate_list: list[str],
    constraints_list: list[str],
    threshold: int = 80,
    chunk_size: int = 200,
    top_n: int = 20,
    candidate_type: str = "entity",
    chatgpt_model: str = "gpt-4o-mini",
    classify_candidates: bool = False,
):
    """
    Use a HYBRID approach to ChatGPT fuzzy matching, combining RapidFuzz pre-filtering
    with final semantic verification by ChatGPT. This avoids large prompts when matching
    candidate terms against a big ontology.

    1) For each candidate term, select the top_n constraints with the highest string similarity via RapidFuzz.
    2) Combine these top matches into a unique set (prefiltered).
    3) Split the prefiltered set into chunks of size 'chunk_size'.
    4) For each chunk, call ChatGPT for semantic matching.
       If classify_candidates is True, instruct ChatGPT to return a JSON array of objects
       with "term" and "classification"; otherwise, return a JSON array of strings.
    5) Return a sorted, unique list of final matched ontology terms.

    Args:
        client: OpenAI client instance
        candidate_list: List of candidate terms
        constraints_list: List of ontology terms
        threshold: RapidFuzz similarity threshold (default 80)
        chunk_size: Number of terms per ChatGPT prompt (default 200)
        top_n: Number of RapidFuzz matches to consider (default 20)
        candidate_type: Type of candidate term to extract (default "entity")
        chatgpt_model: ChatGPT model to use (default "gpt-4o-mini")
        classify_candidates: Whether to classify candidates (default False)

    Returns:
        Sorted list of final matched ontology terms
    """
    if not client or not client.api_key:
        print(
            "Warning: OPENAI_API_KEY is not set. Cannot perform fuzzy matching via ChatGPT."
        )
        return []

    prefiltered = set()
    for candidate in candidate_list:
        top_matches = get_top_n_rapidfuzz(candidate, constraints_list, top_n=top_n)
        prefiltered.update(top_matches)

    if not prefiltered:
        return []

    prefiltered_list = sorted(prefiltered)
    candidate_type_text = "entities" if candidate_type.lower() == "entity" else "states"
    all_matches = set()

    for i in range(0, len(prefiltered_list), chunk_size):
        chunk = prefiltered_list[i : i + chunk_size]
        if classify_candidates:
            prompt = (
                f"Here are candidate ontology {candidate_type_text}: {candidate_list}.\n"
                f"Here are known ontology terms: {chunk}.\n"
                f"Using a similarity threshold of {threshold} (0 to 100), please return a JSON array of objects. "
                "Each object should have keys 'term' (an ontology term from the known list that is similar) and "
                "'classification' (either 'entity' or 'state'). "
                "Return only the JSON array without any markdown formatting or additional text."
            )
        else:
            prompt = (
                f"Here are candidate ontology {candidate_type_text}: {candidate_list}.\n"
                f"Here are known ontology terms: {chunk}.\n"
                f"Using a similarity threshold of {threshold} (0 to 100), please return a JSON array of ontology terms "
                f"from the known list that are similar to the candidate {candidate_type_text}. "
                "Return only the JSON array without any markdown formatting or additional text."
            )

        try:
            response = client.chat.completions.create(
                model=chatgpt_model,
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
            continue

        try:
            answer = response.choices[0].message.content.strip()
            if not answer:
                continue  # Skip empty responses
        except Exception as e:
            print("Error accessing response fields for fuzzy matching:", e)
            continue

        try:
            matches_chunk = json.loads(answer)
            if not isinstance(matches_chunk, list):
                matches_chunk = [
                    line.strip() for line in answer.splitlines() if line.strip()
                ]
        except Exception as e:
            print("Error parsing JSON from ChatGPT fuzzy matching response:", e)
            matches_chunk = [
                line.strip() for line in answer.splitlines() if line.strip()
            ]

        for m in matches_chunk:
            if classify_candidates:
                if isinstance(m, dict):
                    if m.get("term"):
                        all_matches.add(json.dumps(m))
                else:
                    try:
                        m_obj = json.loads(m)
                        if m_obj.get("term"):
                            all_matches.add(json.dumps(m_obj))
                    except Exception:
                        pass
            else:
                all_matches.add(m)

    if classify_candidates:
        all_matches = [json.loads(x) for x in all_matches]
        return sorted(all_matches, key=lambda x: x.get("term", ""))
    else:
        return sorted(all_matches)


def verify_and_fix_column_structure(
    client: OpenAI,
    column_data: dict,
    chatgpt_model: str = "gpt-4o-mini",
):
    """
    Verify that a column's output exactly matches the expected structure:

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

    If the structure is not exactly as expected, this function sends a query to ChatGPT
    instructing it to transform the given JSON to match the required structure.

    Args:
        client: OpenAI client instance
        column_data: Dictionary containing the column's output
        chatgpt_model: ChatGPT model to use (default "gpt-4o-mini")

    Returns:
        Fixed JSON object
    """
    expected_template = (
        '{"sample_values": [ ... ], "entities": {"spacy": [ ... ], "chatgpt": [ ... ], '
        '"matches": {"spacy": [ ... ], "chatgpt": [ ... ]}}}'
    )

    needs_fix = False
    if not isinstance(column_data, dict):
        needs_fix = True
    else:
        if "sample_values" not in column_data or "entities" not in column_data:
            needs_fix = True
        else:
            entities = column_data.get("entities", {})
            if not isinstance(entities, dict) or not all(
                k in entities for k in ["spacy", "chatgpt", "matches"]
            ):
                needs_fix = True
            else:
                matches = entities.get("matches", {})
                if not isinstance(matches, dict) or not all(
                    k in matches for k in ["spacy", "chatgpt"]
                ):
                    needs_fix = True

    if not needs_fix:
        return column_data

    prompt = (
        "The following JSON object does not exactly match the required structure. "
        "Please transform it so that it exactly matches the following structure:\n\n"
        f"{expected_template}\n\n"
        "Return only the fixed JSON object without any markdown formatting or extra text.\n\n"
        "Input JSON:\n" + json.dumps(column_data)
    )

    try:
        response = client.chat.completions.create(
            model=chatgpt_model,
            messages=[
                {"role": "system", "content": "You are an expert in JSON formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        fixed = response.choices[0].message.content.strip()
        fixed_json = json.loads(fixed)
        return fixed_json
    except Exception as e:
        print("Error in verification/fixing process:", e)
        return column_data


def process_data(
    client: OpenAI,
    file_path: str,
    output_path: str | None = None,
    method: str = "both",
    ontology_constraints_path: str | None = None,
    fuzzy_threshold: int = 80,
    fuzzy_method: str = "rapidfuzz",
    chunk_size: int = 200,
    top_n: int = 20,
    chatgpt_model: str = "gpt-4o-mini",
    candidate_type: str = "entity",
    classify_candidates: bool = False,
    verify_structure: bool = False,
):
    """
    Process a tabular data file (CSV or JSON) to extract candidate ontology terms
    from each textual column, optionally matching them against ontology constraints.
    Additionally, if classify_candidates is True, each candidate term is classified as
    either an Entity or a State.

    The output for each column is structured as:
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

    If verify_structure is True, each column's output is verified and, if necessary,
    fixed via a secondary ChatGPT query to ensure the structure matches exactly.

    Args:
        client: OpenAI client instance
        file_path: Path to the input data file
        output_path: Path to save the output JSON file (optional)
        method: Extraction method ("spacy", "chatgpt", or "both")
        ontology_constraints_path: Path to the ontology constraints file (optional)
        fuzzy_threshold: RapidFuzz similarity threshold (default 80)
        fuzzy_method: Matching method ("rapidfuzz" or "chatgpt")
        chunk_size: Number of terms per ChatGPT prompt (default 200)
        top_n: Number of RapidFuzz matches to consider (default 20)
        chatgpt_model: ChatGPT model to use (default "gpt-4o-mini")
        candidate_type: Type of candidate term to extract (default "entity")
        classify_candidates: Whether to classify candidates (default False)
        verify_structure: Whether to verify and fix the column structure (default False)

    Returns:
        Dictionary with the extraction results
    """
    data = read_data(file_path)
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "Candidate extraction is supported only for CSV or JSON files."
        )

    constraints_set = None
    if ontology_constraints_path:
        with open(ontology_constraints_path, "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
        constraints_set = set(ontology_data.get("entities", {}).get("classes", []))
        constraints_set.update(ontology_data.get("entities", {}).get("properties", []))

    extraction_results = {}

    for col in data.columns:
        if data[col].dtype == object:
            unique_vals = data[col].dropna().unique()
            sample_vals = list(unique_vals[:5])
            result = {}
            if method in ["both", "spacy"]:
                spacy_result = extract_entities_spacy(" ".join(map(str, sample_vals)))
                result["spacy"] = spacy_result
            else:
                result["spacy"] = []

            if method in ["both", "chatgpt"]:
                chatgpt_result = extract_entities_chatgpt(
                    client,
                    col,
                    sample_vals,
                    candidate_type=candidate_type,
                    chatgpt_model=chatgpt_model,
                    classify_candidates=classify_candidates,
                )
                result["chatgpt"] = chatgpt_result
            else:
                result["chatgpt"] = []

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
                        client,
                        result["spacy"],
                        constraints_list,
                        threshold=fuzzy_threshold,
                        chunk_size=chunk_size,
                        top_n=top_n,
                        candidate_type=candidate_type,
                        chatgpt_model=chatgpt_model,
                        classify_candidates=classify_candidates,
                    )
                    matches["chatgpt"] = fuzzy_matches_chatgpt(
                        client,
                        result["chatgpt"],
                        constraints_list,
                        threshold=fuzzy_threshold,
                        chunk_size=chunk_size,
                        top_n=top_n,
                        candidate_type=candidate_type,
                        chatgpt_model=chatgpt_model,
                        classify_candidates=classify_candidates,
                    )
            result["matches"] = matches
            column_output = {"sample_values": sample_vals, "entities": result}
            if verify_structure:
                column_output = verify_and_fix_column_structure(
                    client,
                    column_output,
                    chatgpt_model=chatgpt_model,
                )
            extraction_results[col] = column_output

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extraction_results, f, indent=2)

    return extraction_results
