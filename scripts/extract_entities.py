#!/usr/bin/env python3
"""
Entity Extraction Script

This script demonstrates the extraction of candidate entities from tabular data
using different extraction methods. It serves as a command-line utility for
processing data files and identifying potential entities for ontology development.

The script uses the process_data() function from the extract_entities module to:
1. Process a CSV file containing address data
2. Extract entities using different methods (spaCy NER, ChatGPT, or both)
3. Optionally match extracted entities against ontology constraints
4. Save the results to a JSON file
5. Print a summary of the extraction results

The script demonstrates two different extraction approaches:
1. Basic extraction using only spaCy NER
2. Advanced extraction using both spaCy and ChatGPT with fuzzy matching against
   existing ontology constraints

Usage (from the root directory):
    ./scripts/extract_entities.py

The input and output paths are currently hardcoded in the script.
To modify these paths or extraction parameters, edit the script directly.
"""

from src.ingestion.extract import process_data

results = process_data(
    "data/raw/address_base_plus_john_2023-10-06_122302.csv",
    output_path="results/candidate_entities.json",
    method="spacy",
)
print(results)


results = process_data(
    "data/raw/address_base_plus_john_2023-10-06_122302.csv",
    output_path="results/candidate_entities.json",
    method="both",
    ontology_constraints_path="data/ontologies/ontology_entities.json",
    fuzzy_threshold=80,
    fuzzy_method="chatgpt",  # or "rapidfuzz"
)
print(results)
