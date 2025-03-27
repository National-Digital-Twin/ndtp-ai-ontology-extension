#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
TTL Entity Extraction Script

This script extracts ontology entities (classes and properties) from a specified 
RDF/Turtle file and saves the results to a JSON file. It serves as a command-line 
utility for quickly processing ontology files.

The script uses the process_ttl() function from the ttl_ingestion module to:
1. Parse the specified TTL file
2. Extract classes and properties with their human-readable labels
3. Save the extracted entities to a JSON file
4. Print a summary of the extraction results

The output JSON file contains two main sections:
- "classes": A list of extracted class names/labels
- "properties": A list of extracted property names/labels

Usage (from the root directory):
    ./scripts/extract_entities_ttl.py

The input and output paths are currently hardcoded in the script:
- Input: "data/ontologies/ies-building1.ttl"
- Output: "results/ontology_entities.json"

To modify these paths, edit the script directly.
"""

from src.ingestion.ontology import process_ttl

ontology_config = process_ttl(
    "data/ontologies/ies-building1.ttl", output_path="results/ontology_entities.json"
)
print("Extracted Ontology Entities:", ontology_config)
