import pandas as pd
import re
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

from ..ingestion.embeddings import (
    search_vector_store,
)


def generate_value_descriptions(df, categorical_threshold=20):
    """Generate descriptions for categorical column values"""
    value_descriptions = {}

    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count > categorical_threshold or unique_count <= 1:
            continue

        value_counts = df[col].value_counts(normalize=True)

        for value, frequency in value_counts.items():
            if pd.isna(value):
                continue

            value_key = f"{col}::{value}"

            col_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", col)
            col_name = col_name.replace("_", " ").title()

            description = f"Value '{value}' for {col_name}. Appears in {frequency:.1%} of records."

            sample_records = df[df[col] == value].sample(
                min(3, (df[col] == value).sum())
            )
            if not sample_records.empty:
                id_cols = [
                    c
                    for c in ["UPRN", "UDPRN", "Address", "Postcode"]
                    if c in df.columns
                ]
                if id_cols:
                    sample_ids = sample_records[id_cols[0]].astype(str).tolist()
                    description += f" Examples: {', '.join(sample_ids)}"

            value_descriptions[value_key] = description

    return value_descriptions


def find_ontology_value_mappings(
    value_descriptions, model, vector_store, threshold=0.1
):
    """Find mappings between ontology entities and column values"""
    index, metadata = vector_store

    value_meta = []
    for value_key, description in value_descriptions.items():
        col, value = value_key.split("::", 1)
        value_meta.append(
            {
                "type": "column_value",
                "column_name": col,
                "value": value,
                "description": description,
                "text": description,
            }
        )

    mappings = []
    ontology_entities = [
        item for item in metadata if item.get("type") == "ontology_entity"
    ]

    for value_key, description in value_descriptions.items():
        col, value = value_key.split("::", 1)

        value_item = next(
            (
                item
                for item in metadata
                if item.get("type") == "column_value"
                and item.get("column_name") == col
                and item.get("value") == value
            ),
            None,
        )

        if value_item is None:
            continue

        idx = value_item.get("index")
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
                        "data_column": col,
                        "data_value": value,
                        "similarity_score": float(sim_score),
                        "value_description": description,
                        "mapping_type": "value_mapping",
                    }
                )

    return mappings, vector_store


def visualize_value_mappings_tsne(
    value_descriptions,
    vector_store,
    uri_to_embedding,
    output_path,
    max_entities=1000,
    max_values=1000,
):
    """
    Create t-SNE visualization comparing column values with ontology entities
    """
    if not output_path.endswith(".html"):
        output_path = f"{output_path.rsplit('.', 1)[0]}.html"

    index, metadata = vector_store

    value_embeddings = []
    value_keys = []
    value_texts = []
    value_columns = []
    value_values = []

    for item in metadata:
        if item.get("type") == "column_value":
            col_name = item.get("column_name")
            value = item.get("value")
            value_key = f"{col_name}::{value}"

            if value_key in value_descriptions:
                idx = item.get("index")
                if idx is not None:
                    embedding = index.reconstruct(int(idx))
                    value_keys.append(value_key)
                    value_texts.append(item.get("description", ""))
                    value_columns.append(col_name)
                    value_values.append(value)
                    value_embeddings.append(embedding)

    if len(value_keys) > max_values:
        indices = sorted(range(len(value_columns)), key=lambda i: value_columns[i])[
            :max_values
        ]
        value_keys = [value_keys[i] for i in indices]
        value_texts = [value_texts[i] for i in indices]
        value_columns = [value_columns[i] for i in indices]
        value_values = [value_values[i] for i in indices]
        value_embeddings = [value_embeddings[i] for i in indices]

    print(f"Using {len(value_embeddings)} value embeddings for visualization")

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

        for val_embedding in value_embeddings:
            sim = np.dot(entity_embedding, val_embedding) / (
                np.linalg.norm(entity_embedding) * np.linalg.norm(val_embedding)
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

    all_embeddings = list(value_embeddings) + ontology_embeddings

    if len(all_embeddings) < 3:
        print("Not enough embeddings to create t-SNE visualization")
        return None

    embeddings_array = np.array(all_embeddings)

    print(f"Applying t-SNE to {len(all_embeddings)} embeddings...")
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    data = []

    for i, key in enumerate(value_keys):
        data.append(
            {
                "x": embeddings_2d[i, 0],
                "y": embeddings_2d[i, 1],
                "label": f"{value_columns[i]}: {value_values[i]}",
                "type": "Column Value",
                "description": value_texts[i],
                "column": value_columns[i],
                "value": value_values[i],
                "namespace": "Data Values",
            }
        )

    for i, label in enumerate(ontology_labels):
        idx = i + len(value_keys)
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
        hover_data=[
            "label",
            "description",
            "namespace",
            "uri",
            "column",
            "value",
        ],
        labels={"type": "Type"},
        title="t-SNE Projection of Value Mappings to Ontology Entities",
        color_discrete_sequence=[
            "#00CC96",
            "#EF553B",
        ],
    )

    fig.update_layout(
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=1000,
        height=800,
    )

    fig.write_html(output_path)
    print(f"Interactive t-SNE visualization for value mappings saved to {output_path}")

    return output_path
