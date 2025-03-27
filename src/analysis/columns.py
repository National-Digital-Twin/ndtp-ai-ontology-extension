# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

import pandas as pd
import re
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

from ..ingestion.embeddings import search_vector_store


def generate_column_descriptions(df):
    """Generate rich descriptions for each column based on name and values"""
    descriptions = {}

    for col in df.columns:
        col_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", col)  # CamelCase to spaces
        col_name = col_name.replace("_", " ").title()
        dtype = df[col].dtype
        unique_count = df[col].nunique()

        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            if dtype == "object" or dtype == "string":
                value_counts = df[col].value_counts(normalize=True)
                top_values = value_counts.head(5).index.tolist()
                sample_str = f"Common values: {', '.join(str(v) for v in top_values)}"
            elif pd.api.types.is_numeric_dtype(dtype):
                sample_str = f"Range: {non_null_values.min()} to {non_null_values.max()}, Mean: {non_null_values.mean():.2f}"
            else:
                sample_str = f"Sample: {', '.join(str(v) for v in non_null_values.sample(min(3, len(non_null_values))))}"
        else:
            sample_str = "No non-null values"

        descriptions[
            col
        ] = f"{col_name}: {sample_str}. {unique_count} unique values. Type: {dtype}"

    return descriptions


def find_ontology_column_mappings(
    column_descriptions, model, vector_store, threshold=0.5
):
    """Maps ontology entities to data columns using vector similarity"""
    index, metadata = vector_store

    mappings = []
    ontology_entities = [
        item for item in metadata if item.get("type") == "ontology_entity"
    ]

    for col_name, description in column_descriptions.items():
        column_item = next(
            (
                item
                for item in metadata
                if item.get("type") == "column_description"
                and item.get("column_name") == col_name
            ),
            None,
        )

        if column_item is None:
            continue

        idx = column_item.get("index")
        if idx is None:
            continue

        embedding = index.reconstruct(int(idx))

        filter_func = lambda x: x.get("type") == "ontology_entity"
        results = search_vector_store(
            index, metadata, embedding, k=50, filter_func=filter_func
        )

        for result in results:
            sim_score = result.get("similarity", 0)
            if sim_score >= threshold:
                mappings.append(
                    {
                        "ontology_entity_id": result.get("id", ""),
                        "ontology_entity_label": result.get("label", ""),
                        "data_column": col_name,
                        "similarity_score": float(sim_score),
                        "column_description": description,
                        "mapping_type": "column_mapping",
                    }
                )

    return mappings, vector_store


def visualize_column_mappings_tsne(
    column_descriptions, vector_store, uri_to_embedding, output_path, max_entities=1000
):
    """Creates t-SNE visualization of column descriptions and ontology entities"""
    if not output_path.endswith(".html"):
        output_path = f"{output_path.rsplit('.', 1)[0]}.html"

    index, metadata = vector_store

    vis_column_embeddings = []
    vis_column_names = []
    vis_column_texts = []

    for item in metadata:
        if item.get("type") == "column_description":
            col_name = item.get("column_name")
            if col_name in column_descriptions:
                idx = item.get("index")
                if idx is not None:
                    embedding = index.reconstruct(int(idx))
                    vis_column_embeddings.append(embedding)
                    vis_column_names.append(col_name)
                    vis_column_texts.append(item.get("description", ""))

    print(f"Using {len(vis_column_embeddings)} column embeddings for visualization")

    ontology_embeddings = []
    ontology_labels = []
    ontology_uris = []
    ontology_namespaces = []

    filter_func = lambda x: x.get("type") == "ontology_entity"
    ontology_entities = [item for item in metadata if filter_func(item)]

    entity_relevance = {}
    for entity in ontology_entities:
        uri = entity.get("uri", "")
        if uri not in uri_to_embedding:
            continue

        entity_embedding = uri_to_embedding[uri]
        max_sim = 0

        for col_embedding in vis_column_embeddings:
            sim = np.dot(entity_embedding, col_embedding) / (
                np.linalg.norm(entity_embedding) * np.linalg.norm(col_embedding)
            )
            max_sim = max(max_sim, sim)

        entity_relevance[uri] = max_sim

    top_entities = sorted(
        [(uri, score) for uri, score in entity_relevance.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:max_entities]

    for uri, _ in top_entities:
        for entity in ontology_entities:
            if entity.get("uri") == uri:
                ontology_embeddings.append(uri_to_embedding[uri])
                ontology_labels.append(entity.get("label", ""))
                ontology_uris.append(uri)
                ontology_namespaces.append(entity.get("namespace", ""))
                break

    all_embeddings = vis_column_embeddings + ontology_embeddings

    if len(all_embeddings) < 3:
        print("Not enough embeddings to create t-SNE visualization")
        return None

    embeddings_array = np.array(all_embeddings)

    print(f"Applying t-SNE to {len(all_embeddings)} embeddings...")
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    data = []

    for i, col_name in enumerate(vis_column_names):
        data.append(
            {
                "x": embeddings_2d[i, 0],
                "y": embeddings_2d[i, 1],
                "label": col_name,
                "type": "Column",
                "description": vis_column_texts[i],
                "namespace": "Data Columns",
            }
        )

    for i, label in enumerate(ontology_labels):
        idx = i + len(vis_column_names)
        data.append(
            {
                "x": embeddings_2d[idx, 0],
                "y": embeddings_2d[idx, 1],
                "label": label,
                "type": "Ontology Entity",
                "description": label,
                "namespace": ontology_namespaces[i],
                "uri": ontology_uris[i],
            }
        )

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="type",
        hover_data=["label", "description", "namespace", "uri"],
        labels={"type": "Type"},
        title="t-SNE Projection of Column Mappings to Ontology Entities",
        color_discrete_sequence=[
            "#636EFA",
            "#EF553B",
        ],  # Blue for columns, red for entities
    )

    fig.update_layout(
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=1000,
        height=800,
    )

    fig.write_html(output_path)
    print(f"Interactive t-SNE visualization for column mappings saved to {output_path}")

    return output_path
