#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Ontology Analysis Script

This script demonstrates the analysis of ontology data for ontology development. It serves as a command-line utility for
processing ontology data and identifying potential entities for ontology development.

The script uses the analyze_namespace_similarities() function from the analysis module to:
1. Analyze similarities between entities across different namespaces
2. Analyze similarities within the same namespace
3. Create t-SNE visualizations for namespaces
4. Create t-SNE visualizations for column mappings
5. Create t-SNE visualizations for value mappings
6. Print a summary of the analysis results

The script demonstrates two different analysis approaches:
1. Basic analysis using only spaCy NER
2. Advanced analysis using both spaCy and ChatGPT with fuzzy matching against
   existing ontology constraints

Usage (from the root directory):
    ./scripts/analyse_ontology.py \
        --files data/ontologies/ies-common.ttl data/ontologies/ies-building1.ttl \
        --output_dir data/analysis

The input and output paths are currently hardcoded in the script.
To modify these paths or analysis parameters, edit the script directly.
"""

import argparse
import json
import os
import re
import pandas as pd
from collections import defaultdict

from src.analysis.representations import extract_entities_from_ttl
from src.analysis.namespaces import (
    compare_namespaces,
    find_similar_entities_within_namespace,
    visualize_namespace_embeddings_tsne,
)
from src.analysis.columns import (
    generate_column_descriptions,
    find_ontology_column_mappings,
    visualize_column_mappings_tsne,
)
from src.analysis.values import (
    generate_value_descriptions,
    find_ontology_value_mappings,
    visualize_value_mappings_tsne,
)
from src.ingestion.embeddings import (
    initialize_vector_store,
    add_to_vector_store,
    save_vector_store,
    embed_texts,
    analyze_vector_store,
)


def load_ontology_entities(file_path="data/analysis/ontology_entities.json"):
    """Load ontology entities from JSON file"""
    with open(file_path, "r") as f:
        entities = json.load(f)

    print(f"Loaded {len(entities)} ontology entities")
    return entities


def extract_namespaces(entities):
    """Extract and group entities by namespace"""
    namespace_pattern = r"http://([^/]+)/ontology/([^#]+)"

    # Group entities by namespace
    namespace_entities = defaultdict(list)

    for entity in entities:
        uri = entity.get("uri", "")
        if not uri:
            continue

        match = re.search(namespace_pattern, uri)
        if match:
            domain = match.group(1)
            ontology = match.group(2)
            namespace = f"{domain}/{ontology}"

            # Add namespace to entity for reference
            entity["namespace"] = namespace
            namespace_entities[namespace].append(entity)

    # Print namespace statistics
    print(f"Found {len(namespace_entities)} distinct namespaces:")
    for namespace, ns_entities in namespace_entities.items():
        print(f"  - {namespace}: {len(ns_entities)} entities")

    return namespace_entities


def load_data_for_embedding_analysis():
    """Load the CSV data for embedding analysis"""
    try:
        # Load CSV files
        address_df = pd.read_csv(
            "data/raw/address_base_plus_john_2023-10-06_122302.csv"
        )
        energy_df = pd.read_csv(
            "data/raw/address_profiling_john_2023-10-06_123003.csv", low_memory=False
        )

        print(f"Loaded address data: {len(address_df)} rows")
        print(f"Loaded energy data: {len(energy_df)} rows")

        # Check if UPRN exists in both dataframes
        if "UPRN" not in address_df.columns:
            print(
                f"WARNING: 'UPRN' column not found in address data. Available columns: {address_df.columns.tolist()}"
            )
            return address_df

        if "UPRN" not in energy_df.columns:
            print(
                f"WARNING: 'UPRN' column not found in energy data. Available columns: {energy_df.columns.tolist()}"
            )
            return address_df

        # Convert UPRN to string in both dataframes to ensure consistent matching
        address_df["UPRN"] = address_df["UPRN"].astype(str)
        energy_df["UPRN"] = energy_df["UPRN"].astype(str)

        # Merge dataframes on UPRN
        merged_df = pd.merge(address_df, energy_df, on="UPRN", how="inner")

        print(f"After merge: {len(merged_df)} rows")

        # If merge resulted in 0 rows, use address dataframe as fallback
        if len(merged_df) == 0:
            print("No matching UPRNs found. Using address dataframe as fallback.")
            merged_df = address_df

        return merged_df
    except FileNotFoundError:
        print("Data files not found. Skipping data loading for embedding analysis.")
        return None


def build_complete_vector_store(entities, df, model, embedding_dimension):
    """Build a complete vector store with all entities, column descriptions, and value descriptions"""
    # Initialize vector store
    index, metadata = initialize_vector_store(embedding_dimension)
    print(f"Initialized vector store with dimension {embedding_dimension}")

    # Create a dictionary mapping entity URIs to their embeddings
    uri_to_embedding = {}

    # Step 1: Add ontology entities
    print("Generating embeddings for all ontology entities...")
    texts = [entity.get("string_representation", "") for entity in entities]
    embeddings = embed_texts(texts, model)

    # Store URI to embedding mapping
    for i, entity in enumerate(entities):
        uri = entity.get("uri", "")
        if uri:
            uri_to_embedding[uri] = embeddings[i]

    # Create metadata for each entity
    entity_meta = []
    for i, entity in enumerate(entities):
        meta = {
            "type": "ontology_entity",
            "uri": entity.get("uri", ""),
            "id": entity.get("id", ""),
            "label": entity.get("label", ""),
            "namespace": entity.get("namespace", ""),
            "description": entity.get("description", ""),
            "text": entity.get("string_representation", ""),
        }
        entity_meta.append(meta)

    # Add to vector store
    index, metadata = add_to_vector_store(index, metadata, embeddings, entity_meta)
    print(f"Added {len(embeddings)} ontology entity embeddings to vector store")

    # Step 2: Add column descriptions if data is available
    column_descriptions = {}
    if df is not None:
        print("Generating column descriptions...")
        column_descriptions = generate_column_descriptions(df)

        column_texts = list(column_descriptions.values())
        column_names = list(column_descriptions.keys())

        print(f"Generating embeddings for {len(column_texts)} column descriptions...")
        column_embeddings = embed_texts(column_texts, model)

        # Create metadata for column descriptions
        column_meta = []
        for i, (col_name, description) in enumerate(column_descriptions.items()):
            meta = {
                "type": "column_description",
                "column_name": col_name,
                "description": description,
                "text": description,
            }
            column_meta.append(meta)

        # Add to vector store
        index, metadata = add_to_vector_store(
            index, metadata, column_embeddings, column_meta
        )
        print(
            f"Added {len(column_embeddings)} column description embeddings to vector store"
        )

    # Step 3: Add value descriptions if data is available
    value_descriptions = {}
    if df is not None:
        print("Generating value descriptions...")
        value_descriptions = generate_value_descriptions(df)

        value_texts = list(value_descriptions.values())
        value_keys = list(value_descriptions.keys())

        print(f"Generating embeddings for {len(value_texts)} value descriptions...")
        value_embeddings = embed_texts(value_texts, model)

        # Create metadata for value descriptions
        value_meta = []
        for i, (value_key, description) in enumerate(value_descriptions.items()):
            col, value = value_key.split("::", 1)
            meta = {
                "type": "column_value",
                "column_name": col,
                "value": value,
                "description": description,
                "text": description,
            }
            value_meta.append(meta)

        # Add to vector store
        index, metadata = add_to_vector_store(
            index, metadata, value_embeddings, value_meta
        )
        print(
            f"Added {len(value_embeddings)} value description embeddings to vector store"
        )

    vector_store = (index, metadata)
    return vector_store, uri_to_embedding, column_descriptions, value_descriptions


def run_namespace_analysis(
    namespace_entities, uri_to_embedding, vector_store, output_dir
):
    """Run the namespace analysis"""
    print("\n===== RUNNING NAMESPACE ANALYSIS =====")

    # Create analysis directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Analyze similarities between namespaces
    print("\nAnalyzing similarities between different namespaces...")
    cross_namespace_pairs, namespace_similarity_counts = analyze_namespace_similarities(
        namespace_entities, uri_to_embedding, threshold=0.6, vector_store=vector_store
    )

    # Analyze similarities within namespaces
    print("\nAnalyzing similarities within the same namespace...")
    within_namespace_pairs = analyze_within_namespace_similarities(
        namespace_entities, uri_to_embedding, threshold=0.9, vector_store=vector_store
    )

    # Create t-SNE visualization for all namespaces
    print("\nCreating t-SNE visualization for all namespaces...")
    all_namespaces = list(namespace_entities.keys())
    tsne_output_path = os.path.join(output_dir, "namespaces_tsne.html")
    visualize_namespace_embeddings_tsne(
        all_namespaces, uri_to_embedding, vector_store, tsne_output_path
    )

    # Save results to files
    print("\nSaving namespace analysis results...")
    with open(
        os.path.join(output_dir, "cross_namespace_similar_entities.json"), "w"
    ) as f:
        json.dump(cross_namespace_pairs, f, indent=2)

    with open(
        os.path.join(output_dir, "within_namespace_similar_entities.json"), "w"
    ) as f:
        json.dump(within_namespace_pairs, f, indent=2)

    # Print summary
    print("\nNamespace analysis complete!")
    print(
        f"Found {len(cross_namespace_pairs)} similar entity pairs across different namespaces"
    )
    print(
        f"Found {len(within_namespace_pairs)} similar entity pairs within the same namespace"
    )

    # Print top similar pairs
    if cross_namespace_pairs:
        print("\nTop 5 most similar cross-namespace entity pairs:")
        for i, pair in enumerate(
            sorted(
                cross_namespace_pairs, key=lambda x: x["similarity_score"], reverse=True
            )[:5]
        ):
            print(
                f"{i+1}. {pair['entity1_label']} ({pair['namespace1']}) ↔ {pair['entity2_label']} ({pair['namespace2']}) - Score: {pair['similarity_score']:.3f}"
            )

    if within_namespace_pairs:
        print("\nTop 5 most similar within-namespace entity pairs:")
        for i, pair in enumerate(
            sorted(
                within_namespace_pairs,
                key=lambda x: x["similarity_score"],
                reverse=True,
            )[:5]
        ):
            print(
                f"{i+1}. {pair['entity1_label']} ↔ {pair['entity2_label']} ({pair['namespace']}) - Score: {pair['similarity_score']:.3f}"
            )


def run_embedding_analysis(
    df,
    model,
    vector_store,
    uri_to_embedding,
    column_descriptions,
    value_descriptions,
    output_dir,
):
    """Run the embedding analysis"""
    print("\n===== RUNNING EMBEDDING ANALYSIS =====")

    if df is None:
        print("No data available for embedding analysis. Skipping.")
        return

    # 1. Column Mappings
    print("Finding column mappings...")
    column_mappings, _ = find_ontology_column_mappings(
        column_descriptions, model, vector_store
    )

    # Create t-SNE visualization for column mappings
    print("Creating t-SNE visualization for column mappings...")
    visualize_column_mappings_tsne(
        column_descriptions,
        vector_store,
        uri_to_embedding,
        os.path.join(output_dir, "column_mappings_tsne.html"),
    )

    # 2. Value Mappings
    print("Finding value mappings...")
    value_mappings, _ = find_ontology_value_mappings(
        value_descriptions, model, vector_store
    )

    # Create t-SNE visualization for value mappings
    print("Creating t-SNE visualization for value mappings...")
    visualize_value_mappings_tsne(
        value_descriptions,
        vector_store,
        uri_to_embedding,
        os.path.join(output_dir, "value_mappings_tsne.html"),
    )

    # Combine all mappings
    all_mappings = column_mappings + value_mappings

    # Sort mappings by similarity score
    all_mappings.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Output mappings
    with open(os.path.join(output_dir, "ontology_data_mappings.json"), "w") as f:
        json.dump(all_mappings, f, indent=2)

    print(f"\nGenerated {len(all_mappings)} total mappings:")
    print(f"  - {len(column_mappings)} column mappings")
    print(f"  - {len(value_mappings)} value mappings")

    # Print top mappings of each type
    print("\nTop column mappings:")
    for i, mapping in enumerate(
        sorted(column_mappings, key=lambda x: x["similarity_score"], reverse=True)[:5]
    ):
        print(
            f"{i+1}. {mapping['ontology_entity_label']} → {mapping['data_column']} (score: {mapping['similarity_score']:.3f})"
        )

    print("\nTop value mappings:")
    for i, mapping in enumerate(
        sorted(value_mappings, key=lambda x: x["similarity_score"], reverse=True)[:5]
    ):
        print(
            f"{i+1}. {mapping['ontology_entity_label']} → {mapping['data_column']}::{mapping['data_value']} (score: {mapping['similarity_score']:.3f})"
        )


def analyze_namespace_similarities(
    namespace_entities, uri_to_embedding, threshold=0.8, vector_store=None
):
    """Analyze similarities between entities across different namespaces"""
    all_similar_pairs = []
    namespace_similarity_counts = defaultdict(int)

    namespaces = list(namespace_entities.keys())

    # Compare each pair of namespaces
    for i in range(len(namespaces)):
        for j in range(i + 1, len(namespaces)):
            namespace1 = namespaces[i]
            namespace2 = namespaces[j]

            entities1 = namespace_entities[namespace1]
            entities2 = namespace_entities[namespace2]

            print(
                f"Comparing {namespace1} ({len(entities1)} entities) with {namespace2} ({len(entities2)} entities)"
            )

            similar_pairs = compare_namespaces(
                namespace1, namespace2, uri_to_embedding, threshold, vector_store
            )

            all_similar_pairs.extend(similar_pairs)
            namespace_similarity_counts[(namespace1, namespace2)] = len(similar_pairs)

            print(
                f"  Found {len(similar_pairs)} similar entity pairs with similarity >= {threshold}"
            )

    return all_similar_pairs, namespace_similarity_counts


def analyze_within_namespace_similarities(
    namespace_entities, uri_to_embedding, threshold=0.9, vector_store=None
):
    """Analyze similarities between entities within the same namespace"""
    all_similar_pairs = []

    for namespace, entities in namespace_entities.items():
        print(f"Analyzing similarities within {namespace} ({len(entities)} entities)")

        similar_pairs = find_similar_entities_within_namespace(
            namespace, uri_to_embedding, threshold, vector_store
        )

        all_similar_pairs.extend(similar_pairs)
        print(
            f"  Found {len(similar_pairs)} similar entity pairs with similarity >= {threshold}"
        )

    return all_similar_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from TTL files for embedding"
    )
    parser.add_argument(
        "--files", nargs="+", required=True, help="List of TTL files to process"
    )
    parser.add_argument(
        "--output_dir",
        default="data/analysis",
        help="Output directory (default: data/analysis)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        default=1536,
        help="Embedding dimension (default: 1536)",
    )

    args = parser.parse_args()

    all_entities = []

    # Process each TTL file
    for ttl_file in args.files:
        if os.path.exists(ttl_file):
            entities = extract_entities_from_ttl(ttl_file)
            all_entities.extend(entities)
            print(f"Extracted {len(entities)} entities from {ttl_file}")
        else:
            print(f"Warning: File {ttl_file} does not exist, skipping.")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Save the results to a JSON file
    entities_file = os.path.join(args.output_dir, "ontology_entities.json")
    with open(entities_file, "w") as f:
        json.dump(all_entities, f, indent=2)

    print(f"Total entities extracted: {len(all_entities)}")
    print(f"Results saved to {entities_file}")

    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vector_store"), exist_ok=True)

    # Load the embedding model
    print("Loading embedding model...")
    model = args.model
    embedding_dimension = args.embedding_dimension

    # Extract namespaces
    namespace_entities = extract_namespaces(all_entities)

    # Load data for embedding analysis
    print("\nLoading data for embedding analysis...")
    df = load_data_for_embedding_analysis()

    # Build complete vector store with all entities, column descriptions, and value descriptions
    print("\nBuilding complete vector store...")
    (
        vector_store,
        uri_to_embedding,
        column_descriptions,
        value_descriptions,
    ) = build_complete_vector_store(all_entities, df, model, embedding_dimension)

    # Save vector store before running analyses
    print("\nSaving vector store...")
    save_vector_store(
        vector_store[0], vector_store[1], "unified_ontology_analysis", args.output_dir
    )

    # Run namespace analysis
    run_namespace_analysis(
        namespace_entities, uri_to_embedding, vector_store, args.output_dir
    )

    # Run embedding analysis
    run_embedding_analysis(
        df,
        model,
        vector_store,
        uri_to_embedding,
        column_descriptions,
        value_descriptions,
        args.output_dir,
    )

    # Analyze vector store
    print("\nAnalyzing vector store...")
    analysis_results = analyze_vector_store(vector_store, model)

    # Save vector store analysis results
    with open(os.path.join(args.output_dir, "vector_store_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)

    print("\nAll analyses complete!")


if __name__ == "__main__":
    main()
