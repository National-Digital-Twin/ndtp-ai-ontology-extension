# Ontology Embeddings Analysis

Embedding-based analysis allows us to explore the relationships between ontology entities and real-world data. The `scripts/analyse_ontology.py` script gives a practical demonstration of how we can generate and visualise embeddings for ontology data, and use them to identify relationships between different ontological elements.

## What the Script Does

1. **Extracts ontology data**: Reads TTL (Turtle) files and extracts classes, properties, and instances
2. **Groups by namespace**: Organises entities by their namespace for comparison
3. **Analyzes similarities**: Identifies similar concepts within and across namespaces
4. **Maps to data**: Connects ontology concepts to real-world data columns and values
5. **Creates visualisations**: Generates interactive plots to explore relationships

## Understanding Embeddings

Embeddings are numerical representations of text or concepts in a high-dimensional space. In this case, we are using OpenAI's text-embedding-3-small embedding model, which has 1536 dimensions. Here's how embeddings work:

- Each ontology entity is converted to a text description using the `create_entity_string` function
- This text is processed by an AI model that converts it to a vector of numbers
- Similar concepts end up with similar vectors (closer together in the vector space)
- This allows measuring similarity between concepts mathematically

For ontologists, embeddings offer a way to:

1. Discover relationships that might not be explicitly defined in the ontology
2. Find potential duplicates or overlapping concepts
3. Map ontology concepts to real-world data
4. Visualise the conceptual structure of your ontology

## Key Outputs

The script produces several files in the output directory:

### Data Files

- JSON files containing extracted entities and similarity analyses
- Vector store files that store the embeddings for future use

### Interactive Visualisations

Three HTML files containing interactive t-SNE plots:

1. **`namespaces_tsne.html`**: Shows how entities from different namespaces relate to each other. This helps identify where ontologies overlap or diverge conceptually.

2. **`column_mappings_tsne.html`**: Shows relationships between data columns and ontology entities. This helps identify which parts of your ontology match real-world data structures.

3. **`value_mappings_tsne.html`**: Shows relationships between specific data values and ontology entities. This helps identify which ontology concepts match specific data values.

## What t-SNE Does

t-SNE (t-Distributed Stochastic Neighbour Embedding) is a technique that reduces high-dimensional data (like our 1536-dimension embeddings) to 2 dimensions for visualisation. It tries to preserve the relationships between points, so similar entities appear close together in the visualisation.

## Practical Findings

From the script output:

- 1,011 entities were extracted from two ontology files
- 271 similar entity pairs were found across different namespaces
- 268 similar entity pairs were found within the same namespace
- 587 mappings were found between ontology entities and data elements

The script identified potential duplicates like "pluriverse" and "Pluriverse" across namespaces, and found meaningful connections between ontology concepts and data columns/values.

## How to Use the Visualisations

To explore the relationships:

1. Open the HTML files in any web browser
2. Hover over points to see details about each entity
3. Look for clusters of points, which indicate related concepts
4. Pay attention to where different namespaces or types overlap

These visualisations can help guide ontology refinement, identify gaps, and improve alignment with real-world data.
