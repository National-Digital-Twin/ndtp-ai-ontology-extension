# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Data Ingestion Helper Functions

This file provides utility functions for reading and processing data from various file formats
commonly used in data analysis and knowledge graph development. It supports multiple data formats
including CSV, JSON, RDF/Turtle, and SQLite databases.

These functions abstract away the specifics of reading different file formats, providing a unified
interface through the read_data() function that automatically detects the file type based on
its extension and applies the appropriate reading method.

Supported file formats:
- CSV (.csv): Loaded as pandas DataFrames
- JSON (.json): Loaded as pandas DataFrames
- RDF/Turtle (.ttl, .rdf): Loaded as rdflib Graph objects
- SQLite databases (.db, .sqlite, .sqlite3): Requires SQL queries

Also included are specialized functions for working with semantic data, such as
executing SPARQL queries against RDF graphs.

Usage:
    # Read data automatically based on file extension
    data = read_data('data.csv')
    
    # Execute SQL query on a database
    results = read_sql('database.sqlite', 'SELECT * FROM table')
    
    # Execute SPARQL query on an ontology
    results = execute_sparql_query('ontology.ttl', 'SELECT ?s ?p ?o WHERE {?s ?p ?o}')
"""

import os
import pandas as pd
import json
import rdflib
import sqlite3


def read_csv(file_path):
    """Read a CSV file and return a pandas DataFrame."""
    return pd.read_csv(file_path)


def read_json(file_path):
    """Read a JSON file and return a pandas DataFrame."""
    return pd.read_json(file_path)


def read_ttl(file_path):
    """
    Read an RDF Turtle file using rdflib.
    Returns an rdflib.Graph object.
    """
    g = rdflib.Graph()
    g.parse(file_path, format="turtle")
    return g


def read_sql(file_path, query):
    """
    Read data from a SQLite database file by executing the provided SQL query.
    Returns a pandas DataFrame with the result set.

    Parameters:
      file_path: Path to the SQLite database file (e.g., .db, .sqlite, .sqlite3)
      query: SQL query to execute.
    """
    conn = sqlite3.connect(file_path)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


def read_data(file_path):
    """
    Read data from a file based on its extension.

    Supported formats:
      - .csv           → returns a pandas DataFrame
      - .json          → returns a pandas DataFrame
      - .ttl or .rdf   → returns an rdflib.Graph
      - .db, .sqlite, .sqlite3 → requires an SQL query. Use the read_sql() function.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return read_csv(file_path)
    elif ext == ".json":
        return read_json(file_path)
    elif ext in [".ttl", ".rdf"]:
        return read_ttl(file_path)
    elif ext in [".db", ".sqlite", ".sqlite3"]:
        raise ValueError(
            "SQL database files require a query. Please use the read_sql() function with an SQL query."
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def execute_sparql_query(file_path, query):
    """
    Load an ontology from a Turtle file and execute a SPARQL query on it.

    Parameters:
      file_path: Path to the RDF Turtle file containing your ontology.
      query: SPARQL query to execute.

    Returns:
      The result of the SPARQL query.
    """
    graph = read_ttl(file_path)
    results = graph.query(query)
    return results
