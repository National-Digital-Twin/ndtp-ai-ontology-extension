#!/usr/bin/env python3
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
from src.ingestion.ontology import process_ttl
from src.ingestion.extract import process_data

# Construct absolute paths based on the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Absolute path for the TTL file (reference ontology)
ttl_path = os.path.join(project_root, "data", "ontologies", "ies-common.ttl")
# Absolute path where the extracted ontology JSON will be saved
ontology_json_path = os.path.join(
    project_root, "data", "ontologies", "ontology_entities.json"
)
# Absolute path for the CSV file with candidate data
csv_path = os.path.join(
    project_root, "data", "raw", "address_base_plus_john_2023-10-06_122302.csv"
)
# Absolute path for the output candidate entities JSON file (step 2)
output_path_step2 = os.path.join(
    project_root, "results", "candidate_entities_with_reference.json"
)
# Absolute path for the output candidate entities classification JSON file (step 3)
output_path_step3 = os.path.join(
    project_root, "results", "candidate_entities_classified.json"
)

# Step 1: Extract the reference ontology from the TTL file.
ontology_result = process_ttl(
    ttl_path, output_path=ontology_json_path, namespace_prefix=""
)
print("Step 1 - Reference Ontology Extraction Result:")
print(ontology_result)

# Step 2: Extract candidate entities from the CSV file using only ChatGPT.
# results_step2 = process_data(
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
