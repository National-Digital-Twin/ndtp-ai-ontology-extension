# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from ..ingestion.embeddings import search_vector_store


def compare_namespaces(
    namespace1, namespace2, uri_to_embedding, threshold=0.8, vector_store=None
):
    """Compare entities between two namespaces and find similar pairs"""
    similar_pairs = []

    if vector_store:
        index, metadata = vector_store

        filter_func_ns1 = (
            lambda x: x.get("namespace") == namespace1
            and x.get("type") == "ontology_entity"
        )
        entities1 = [item for item in metadata if filter_func_ns1(item)]

        for entity1 in entities1:
            uri1 = entity1.get("uri", "")
            if uri1 not in uri_to_embedding:
                continue

            embedding1 = uri_to_embedding[uri1]

            filter_func = (
                lambda x: x.get("namespace") == namespace2
                and x.get("type") == "ontology_entity"
            )
            results = search_vector_store(
                index, metadata, embedding1, k=50, filter_func=filter_func
            )

            for result in results:
                sim_score = result.get("similarity", 0)
                if sim_score >= threshold:
                    similar_pairs.append(
                        {
                            "namespace1": namespace1,
                            "entity1_uri": uri1,
                            "entity1_label": entity1.get("label", ""),
                            "namespace2": namespace2,
                            "entity2_uri": result.get("uri", ""),
                            "entity2_label": result.get("label", ""),
                            "similarity_score": float(sim_score),
                        }
                    )

    return similar_pairs


def find_similar_entities_within_namespace(
    namespace, uri_to_embedding, threshold=0.9, vector_store=None
):
    """Find highly similar entities within the same namespace"""
    similar_pairs = []

    if vector_store:
        index, metadata = vector_store

        filter_func = (
            lambda x: x.get("namespace") == namespace
            and x.get("type") == "ontology_entity"
        )
        entities = [item for item in metadata if filter_func(item)]

        for entity in entities:
            uri = entity.get("uri", "")
            if uri not in uri_to_embedding:
                continue

            embedding = uri_to_embedding[uri]

            filter_func = (
                lambda x: x.get("namespace") == namespace
                and x.get("uri") != uri
                and x.get("type") == "ontology_entity"
            )
            results = search_vector_store(
                index, metadata, embedding, k=50, filter_func=filter_func
            )

            for result in results:
                sim_score = result.get("similarity", 0)
                if sim_score >= threshold:
                    similar_pairs.append(
                        {
                            "namespace": namespace,
                            "entity1_uri": uri,
                            "entity1_label": entity.get("label", ""),
                            "entity2_uri": result.get("uri", ""),
                            "entity2_label": result.get("label", ""),
                            "similarity_score": float(sim_score),
                        }
                    )

    return similar_pairs


def visualize_namespace_embeddings_tsne(
    namespaces,
    uri_to_embedding,
    vector_store,
    output_path,
    interactive=True,
):
    """
    Create t-SNE visualization comparing entities from multiple namespaces.

    Args:
        namespaces: List of namespaces to visualize
        uri_to_embedding: Dictionary mapping URIs to embeddings
        vector_store: Tuple of (index, metadata)
        output_path: Path to save the visualization
        interactive: Whether to create interactive Plotly visualization (default: True)
    """
    if interactive and not output_path.endswith(".html"):
        output_path = f"{output_path.rsplit('.', 1)[0]}.html"
    elif not interactive and not output_path.endswith(".png"):
        output_path = f"{output_path.rsplit('.', 1)[0]}.png"

    _, metadata = vector_store

    all_embeddings = []
    all_labels = []
    all_uris = []
    all_namespace_indices = []
    all_namespaces = []

    if interactive:
        colors = px.colors.qualitative.Plotly
        if len(namespaces) > len(colors):
            colors = px.colors.qualitative.Alphabet
    else:
        colors = list(mcolors.TABLEAU_COLORS)
        if len(namespaces) > len(colors):
            cmap = plt.cm.get_cmap("tab20", len(namespaces))
            colors = [cmap(i) for i in range(len(namespaces))]

    for ns_idx, namespace in enumerate(namespaces):
        entities = [
            item
            for item in metadata
            if item.get("namespace") == namespace
            and item.get("type") == "ontology_entity"
        ]

        print(f"Found {len(entities)} entities in {namespace}")

        for entity in entities:
            uri = entity.get("uri", "")
            if uri in uri_to_embedding:
                all_embeddings.append(uri_to_embedding[uri])
                all_labels.append(entity.get("label", ""))
                all_uris.append(uri)
                all_namespace_indices.append(ns_idx)
                all_namespaces.append(namespace)

    if len(all_embeddings) < 3:
        print(f"Not enough entities with embeddings to create t-SNE visualization")
        return None

    embeddings_array = np.array(all_embeddings)

    print(f"Applying t-SNE to {len(all_embeddings)} embeddings...")
    perplexity = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    if interactive:
        df = pd.DataFrame(
            {
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "label": all_labels,
                "uri": all_uris,
                "namespace": all_namespaces,
                "namespace_idx": all_namespace_indices,
            }
        )

        namespace_short_names = {}
        for ns in namespaces:
            ns_parts = ns.split("/")
            if len(ns_parts) >= 2:
                short_name = ns_parts[-1]
            else:
                short_name = ns
            namespace_short_names[ns] = short_name

        df["namespace_short"] = df["namespace"].map(namespace_short_names)

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="namespace_short",
            hover_data=["label", "uri", "namespace"],
            labels={"namespace_short": "Namespace"},
            title=f"t-SNE Projection of Namespace Entity Embeddings",
            color_discrete_sequence=colors[: len(namespaces)],
        )

        fig.update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            width=1000,
            height=800,
        )

        fig.write_html(output_path)
        print(f"Interactive t-SNE visualization saved to {output_path}")

    else:
        plt.figure(figsize=(14, 12))

        for ns_idx, namespace in enumerate(namespaces):
            indices = [
                i for i, idx in enumerate(all_namespace_indices) if idx == ns_idx
            ]

            if not indices:
                continue

            ns_parts = namespace.split("/")
            if len(ns_parts) >= 2:
                short_name = f"{ns_parts[0].split('.')[0]}/../{ns_parts[-1]}"
            else:
                short_name = namespace

            plt.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=[colors[ns_idx]],
                alpha=0.7,
                label=short_name,
                s=50,
            )

        plt.title(f"t-SNE Visualization of Entities from {len(namespaces)} Namespaces")
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"t-SNE visualization saved to {output_path}")

    return output_path
