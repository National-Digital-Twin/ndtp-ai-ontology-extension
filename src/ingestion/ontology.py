"""
RDF/Turtle Ontology Entity Extraction

This file provides functionality for extracting ontology entities (classes and properties) 
from RDF/Turtle files. It focuses on extracting human-readable labels for entities within 
a specified namespace.

The main function, process_ttl(), parses a Turtle file and extracts:
- Classes: entities with rdf:type rdfs:Class or owl:Class
- Properties: entities with rdf:type rdf:Property, owl:ObjectProperty, or owl:DatatypeProperty

By default, it filters entities to only include those whose URIs start with the 
specified namespace_prefix. Instead of returning full URIs, it extracts human-readable 
labels from rdfs:label. If a label is missing, it falls back to the local name 
(the part after '#' or the last '/').

The output is a structured dictionary containing the extracted classes and properties,
which can optionally be saved as a JSON file.

Usage:
    from scripts.ingestion.ttl_ingestion import process_ttl

    result = process_ttl(
        "data/ontologies/ies-building1.ttl", 
        output_path="results/ontology_entities.json"
    )
    print(result)
"""

import os
import json
import rdflib
from rdflib.namespace import RDF, RDFS, OWL


def get_label_or_localname(uri, graph):
    """
    Given an RDFLib URIRef and the graph, try to fetch its rdfs:label.
    If not found, fallback to extracting the local name from the URI.
    """
    label = graph.value(uri, RDFS.label)
    if label:
        return str(label)
    else:
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        else:
            return uri_str.rstrip("/").split("/")[-1]


def process_ttl(
    file_path,
    output_path=None,
    namespace_prefix="http://ies.data.gov.uk/ontology/ies-building1#",
):
    """
    Process an RDF/Turtle ontology file to extract candidate ontology entities.

    This function reads the Turtle file and extracts:
      - Classes: entities with rdf:type rdfs:Class or owl:Class.
      - Properties: entities with rdf:type rdf:Property, owl:ObjectProperty, or owl:DatatypeProperty.

    It then filters the extracted entities so that only those whose URI starts with
    the given namespace_prefix are included, and converts each entity to its human-readable
    label (or local name as a fallback).

    The output format is:
      {
         "ontology_file": <file_path>,
         "entities": {
             "classes": [ ... ],
             "properties": [ ... ]
         }
      }

    Parameters:
      file_path: Path to the Turtle file.
      output_path: Optional path to save the extraction results as JSON.
      namespace_prefix: The base URI to filter entities by (default is the ies-building1 namespace).

    Returns:
      A dictionary with the extraction results.
    """
    graph = rdflib.Graph()
    try:
        graph.parse(file_path, format="turtle")
    except Exception as e:
        raise ValueError(f"Error parsing Turtle file '{file_path}': {e}")

    # Extract all classes from the graph
    classes = set(graph.subjects(RDF.type, RDFS.Class))
    classes.update(graph.subjects(RDF.type, OWL.Class))

    # Extract all properties from the graph
    properties = set(graph.subjects(RDF.type, RDF.Property))
    properties.update(graph.subjects(RDF.type, OWL.ObjectProperty))
    properties.update(graph.subjects(RDF.type, OWL.DatatypeProperty))

    # Filter entities by the provided namespace_prefix
    classes = {c for c in classes if str(c).startswith(namespace_prefix)}
    properties = {p for p in properties if str(p).startswith(namespace_prefix)}

    # Convert each entity to its human-readable label (or local name if no label exists)
    classes_list = sorted(get_label_or_localname(c, graph) for c in classes)
    properties_list = sorted(get_label_or_localname(p, graph) for p in properties)

    result = {
        "ontology_file": file_path,
        "entities": {"classes": classes_list, "properties": properties_list},
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
