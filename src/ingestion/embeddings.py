"""
Embedding Module for Ontology Development

This module provides functionality to embed text data for ontology development. It uses 
OpenAI's API to generate embeddings.

Key functions:
- embed_texts: Generate embeddings for a list of texts using OpenAI
- embed_texts_openai: Generate embeddings using OpenAI's API
- get_embedding_model: Get a model name for generating embeddings with OpenAI
- analyze_vector_store: Perform analysis on the vector store

Usage:
    results = embed_texts(
        texts=['text1', 'text2', 'text3'],
        model='text-embedding-3-small'
    )
"""

import numpy as np
import faiss
import pickle
import os
from openai import OpenAI


# Set up OpenAI API key from the environment (non-fatal if missing)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    print("Warning: OPENAI_API_KEY is not set. Embedding generation will fail.")


def initialize_vector_store(dimension):
    """Initialize a FAISS vector store with the given embedding dimension"""
    index = faiss.IndexFlatL2(dimension)
    metadata = []
    return index, metadata


def add_to_vector_store(index, metadata, embeddings, meta_info):
    """Add embeddings to the vector store with metadata"""
    if len(embeddings) == 0:
        return index, metadata

    embeddings_np = np.array(embeddings).astype("float32")
    index.add(embeddings_np)

    for i, meta in enumerate(meta_info):
        metadata.append({"index": index.ntotal - len(meta_info) + i, **meta})

    return index, metadata


def search_vector_store(index, metadata, query_vector, k=10, filter_func=None):
    """Search the vector store for similar vectors with optional filtering"""
    query_np = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_np, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # Valid index
            meta = next((m for m in metadata if m["index"] == idx), None)
            if meta:
                result = {
                    "distance": distances[0][i],
                    "similarity": 1 / (1 + distances[0][i]),
                    **meta,
                }

                if filter_func is None or filter_func(result):
                    results.append(result)

    return results


def save_vector_store(index, metadata, filename_prefix, directory="data/vector_store"):
    """Save the vector store and metadata to disk"""
    os.makedirs(directory, exist_ok=True)
    faiss.write_index(index, f"{directory}/{filename_prefix}_index.faiss")

    with open(f"{directory}/{filename_prefix}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Vector store saved to {directory}/{filename_prefix}_index.faiss")
    print(f"Metadata saved to {directory}/{filename_prefix}_metadata.pkl")


def load_vector_store(filename_prefix, directory="data/vector_store"):
    """Load the vector store and metadata from disk"""
    index = faiss.read_index(f"{directory}/{filename_prefix}_index.faiss")

    with open(f"{directory}/{filename_prefix}_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded vector store with {index.ntotal} vectors")
    return index, metadata


def embed_texts(texts, model="text-embedding-3-small"):
    """Generate embeddings for a list of texts using OpenAI

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model name

    Returns:
        List of embeddings as numpy arrays
    """
    return embed_texts_openai(texts, model)


def embed_texts_openai(texts, model_name="text-embedding-3-small"):
    """Generate embeddings using OpenAI's API

    Args:
        texts: List of text strings to embed
        model_name: OpenAI embedding model name

    Returns:
        List of embeddings as numpy arrays
    """
    if not openai_api_key:
        print("Warning: Cannot generate embeddings because OPENAI_API_KEY is not set.")
        return np.array([])

    client = OpenAI(api_key=openai_api_key)

    # Process in batches if needed (OpenAI has input limits)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(model=model_name, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error calling OpenAI API for embeddings: {e}")
            return np.array([])

    return np.array(all_embeddings)


def analyze_vector_store(vector_store, model=None):
    """Perform analysis on the vector store"""
    index, metadata = vector_store

    # Count items by type
    type_counts = {}
    for item in metadata:
        item_type = item.get("type", "unknown")
        type_counts[item_type] = type_counts.get(item_type, 0) + 1

    print("\nVector store contents:")
    for item_type, count in type_counts.items():
        print(f"  - {item_type}: {count} items")

    # Find similar items across different types
    print("\nCross-type similarity examples:")

    # Find similar items for a random ontology entity
    ontology_items = [m for m in metadata if m.get("type") == "ontology_entity"]
    if ontology_items and model:
        sample_entity = ontology_items[0]
        entity_text = sample_entity.get("text", "")
        entity_embedding = embed_texts([entity_text], model)[0]

        # Search for similar items
        results = search_vector_store(index, metadata, entity_embedding, k=5)

        print(
            f"\nItems similar to ontology entity '{sample_entity.get('entity_label')}':"
        )
        for i, result in enumerate(results):
            print(
                f"{i+1}. [{result.get('type')}] {result.get('text', '')[:50]}... (similarity: {result.get('similarity'):.3f})"
            )

    return type_counts
