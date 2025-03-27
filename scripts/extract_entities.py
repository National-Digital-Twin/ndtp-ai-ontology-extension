#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Entity Extraction Script using Only ChatGPT (Three-Step Process with Absolute Paths)

This script demonstrates a three-step workflow:
1. Reference Ontology Extraction:
   - Extracts a reference ontology from a Turtle (.ttl) file and saves it as JSON.
2. Candidate Entity Extraction:
   - Extracts candidate entities from a CSV file using only ChatGPT.
   - Fuzzy matching is performed against the reference ontology using ChatGPT-based semantic matching.
3. Candidate Entity Classification:
   - Extracts candidate entities from the CSV file using ChatGPT with classification activated.
   - The prompt instructs ChatGPT to return a JSON array of objects with "term" and "classification".
   - The classified results are saved as JSON.

Usage (from any directory):
    ./scripts/extract_entities.py
"""

import os
from openai import OpenAI
from src.ingestion.ontology import process_ttl
from src.ingestion.extract import process_data

BASE_ONTOLOGY_PATH = "data/ontologies/ies-common.ttl"
TARGET_ENTITIES_JSON = "data/ontologies/ontology_entities.json"
CSV_DATA = "data/raw/address_base_plus_john_2023-10-06_122302.csv"

CANDIDATE_ENTITIES_JSON = "results/candidate_entities_with_reference.json"  # Step 2
CANDIDATE_ENTITIES_CLASSIFIED_JSON = (
    "results/candidate_entities_classified.json"  # Step 3
)


# Initialise OpenAI client (make sure API key is set in environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

ttl_path = os.path.join(project_root, BASE_ONTOLOGY_PATH)
ontology_json_path = os.path.join(project_root, TARGET_ENTITIES_JSON)
csv_path = os.path.join(project_root, CSV_DATA)

output_path_step2 = os.path.join(project_root, CANDIDATE_ENTITIES_JSON)
output_path_step3 = os.path.join(project_root, CANDIDATE_ENTITIES_CLASSIFIED_JSON)

# Step 1: Extract the reference ontology from the TTL file.
ontology_result = process_ttl(
    ttl_path, output_path=ontology_json_path, namespace_prefix=""
)
print("Step 1 - Reference Ontology Extraction Result:")
print(ontology_result)

# Step 2: Extract candidate entities from the CSV file using only ChatGPT.
# results_step2 = process_data(
#     client=client,
#     file_path=csv_path,
#     output_path=output_path_step2,
#     method="chatgpt",                      # Use only ChatGPT for extraction.
#     ontology_constraints_path=ontology_json_path,
#     fuzzy_threshold=80,                    # Similarity threshold for fuzzy matching.
#     fuzzy_method="chatgpt",                # Use ChatGPT-based fuzzy matching.
#     candidate_type="entity",               # Candidate type wording.
#     classify_candidates=False              # No classification in this step.
# )
# print("Step 2 - Candidate Entity Extraction Result:")
# print(results_step2)

# Step 3: Extract candidate entities with classification from the CSV file using ChatGPT.
results_step3 = process_data(
    client=client,
    file_path=csv_path,
    output_path=output_path_step3,
    method="chatgpt",  # Use only ChatGPT for extraction.
    ontology_constraints_path=ontology_json_path,
    fuzzy_threshold=80,  # Similarity threshold for fuzzy matching.
    fuzzy_method="chatgpt",  # Use ChatGPT-based fuzzy matching.
    candidate_type="entity",  # Candidate type wording.
    classify_candidates=True,  # Activate classification in the output.
)
print("Step 3 - Candidate Entity Classification Result:")
print(results_step3)
