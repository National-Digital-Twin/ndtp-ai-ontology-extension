# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Unit tests for the ingestion module components.

This file contains unit tests for:
- helpers.py: Data reading utilities
- extract.py: Entity extraction functionality
- ontology.py: RDF/Turtle ontology processing
- embeddings.py: Embedding utilities
"""

import json
import numpy as np
import faiss
import tempfile
import pickle
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open
import rdflib
from rdflib.namespace import RDF, RDFS, OWL

from src.ingestion.helpers import (
    read_csv,
    read_json,
    read_ttl,
    read_sql,
    read_data,
    execute_sparql_query,
)
from src.ingestion.extract import (
    extract_entities_spacy,
    extract_entities_chatgpt,
    fuzzy_matches,
    fuzzy_matches_chatgpt,
    process_data,
    get_top_n_rapidfuzz,
    verify_and_fix_column_structure,
)
from src.ingestion.ontology import get_label_or_localname, process_ttl
from src.ingestion.embeddings import (
    initialize_vector_store,
    add_to_vector_store,
    search_vector_store,
    save_vector_store,
    load_vector_store,
    embed_texts,
    embed_texts_openai,
    analyze_vector_store,
)
from src.ingestion.processing import (
    analyze_csv_with_chatgpt,
    pseudo_ner_phrase_extraction,
    extract_boro_triples,
    gather_usage_patterns_and_subtypes,
    classify_extension_type,
    analyze_step,
    analyze_tri,
    extract_concepts_step,
    gather_usage_step,
    classify_extensions,
)


# ===== Fixtures =====


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text_column": [
                "Apple Inc. is based in California.",
                "Microsoft was founded by Bill Gates.",
            ],
            "numeric_column": [1, 2],
        }
    )


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    return json.dumps(
        {
            "text_column": [
                "Apple Inc. is based in California.",
                "Microsoft was founded by Bill Gates.",
            ],
            "numeric_column": [1, 2],
        }
    )


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return "text_column,numeric_column\nApple Inc. is based in California.,1\nMicrosoft was founded by Bill Gates.,2"


@pytest.fixture
def sample_ontology_constraints():
    """Create sample ontology constraints for testing."""
    return {
        "entities": {
            "classes": ["Person", "Organization", "Location"],
            "properties": ["foundedBy", "basedIn"],
        }
    }


@pytest.fixture
def sample_rdf_graph():
    """Create a sample RDF graph for testing."""
    g = rdflib.Graph()
    ns = rdflib.Namespace("http://example.org/ontology#")

    # Add classes
    g.add((ns.Person, RDF.type, RDFS.Class))
    g.add((ns.Person, RDFS.label, rdflib.Literal("Person")))

    g.add((ns.Organization, RDF.type, OWL.Class))
    # No label for Organization to test fallback

    # Add properties
    g.add((ns.worksFor, RDF.type, OWL.ObjectProperty))
    g.add((ns.worksFor, RDFS.label, rdflib.Literal("works for")))

    return g


# ===== Tests for helpers.py =====


class TestHelpers:
    @patch("pandas.read_csv")
    def test_read_csv(self, mock_read_csv, sample_dataframe):
        """Test reading CSV files."""
        mock_read_csv.return_value = sample_dataframe
        result = read_csv("dummy.csv")
        mock_read_csv.assert_called_once_with("dummy.csv")
        assert isinstance(result, pd.DataFrame)

    @patch("pandas.read_json")
    def test_read_json(self, mock_read_json, sample_dataframe):
        """Test reading JSON files."""
        mock_read_json.return_value = sample_dataframe
        result = read_json("dummy.json")
        mock_read_json.assert_called_once_with("dummy.json")
        assert isinstance(result, pd.DataFrame)

    @patch("rdflib.Graph")
    def test_read_ttl(self, mock_graph):
        """Test reading Turtle files."""
        mock_instance = mock_graph.return_value
        result = read_ttl("dummy.ttl")
        mock_instance.parse.assert_called_once_with("dummy.ttl", format="turtle")
        assert result == mock_instance

    @patch("sqlite3.connect")
    @patch("pandas.read_sql_query")
    def test_read_sql(self, mock_read_sql_query, mock_connect, sample_dataframe):
        """Test reading from SQL databases."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_read_sql_query.return_value = sample_dataframe

        query = "SELECT * FROM table"
        result = read_sql("dummy.db", query)

        mock_connect.assert_called_once_with("dummy.db")
        mock_read_sql_query.assert_called_once_with(query, mock_conn)
        mock_conn.close.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch("os.path.splitext")
    @patch("src.ingestion.helpers.read_csv")
    def test_read_data_csv(self, mock_read_csv, mock_splitext, sample_dataframe):
        """Test read_data with CSV files."""
        mock_splitext.return_value = ("dummy", ".csv")
        mock_read_csv.return_value = sample_dataframe

        result = read_data("dummy.csv")

        mock_read_csv.assert_called_once_with("dummy.csv")
        assert isinstance(result, pd.DataFrame)

    @patch("os.path.splitext")
    @patch("src.ingestion.helpers.read_json")
    def test_read_data_json(self, mock_read_json, mock_splitext, sample_dataframe):
        """Test read_data with JSON files."""
        mock_splitext.return_value = ("dummy", ".json")
        mock_read_json.return_value = sample_dataframe

        result = read_data("dummy.json")

        mock_read_json.assert_called_once_with("dummy.json")
        assert isinstance(result, pd.DataFrame)

    @patch("os.path.splitext")
    @patch("src.ingestion.helpers.read_ttl")
    def test_read_data_ttl(self, mock_read_ttl, mock_splitext):
        """Test read_data with Turtle files."""
        mock_splitext.return_value = ("dummy", ".ttl")
        mock_graph = MagicMock()
        mock_read_ttl.return_value = mock_graph

        result = read_data("dummy.ttl")

        mock_read_ttl.assert_called_once_with("dummy.ttl")
        assert result == mock_graph

    @patch("os.path.splitext")
    def test_read_data_db_error(self, mock_splitext):
        """Test read_data with database files raises appropriate error."""
        mock_splitext.return_value = ("dummy", ".db")

        with pytest.raises(ValueError, match="SQL database files require a query"):
            read_data("dummy.db")

    @patch("os.path.splitext")
    def test_read_data_unsupported_format(self, mock_splitext):
        """Test read_data with unsupported file format."""
        mock_splitext.return_value = ("dummy", ".xyz")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            read_data("dummy.xyz")

    @patch("src.ingestion.helpers.read_ttl")
    def test_execute_sparql_query(self, mock_read_ttl):
        """Test executing SPARQL queries."""
        mock_graph = MagicMock()
        mock_results = MagicMock()
        mock_read_ttl.return_value = mock_graph
        mock_graph.query.return_value = mock_results

        query = "SELECT ?s ?p ?o WHERE {?s ?p ?o}"
        result = execute_sparql_query("dummy.ttl", query)

        mock_read_ttl.assert_called_once_with("dummy.ttl")
        mock_graph.query.assert_called_once_with(query)
        assert result == mock_results


# ===== Tests for extract.py =====


class TestExtract:
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create mock OpenAI client
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = MagicMock()

    def test_extract_entities_spacy(self):
        """Test entity extraction using spaCy."""
        text = "Apple Inc. is based in California and was founded by Steve Jobs."
        entities = extract_entities_spacy(text)

        # Check that we got some entities (exact results may vary by spaCy model)
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert all(isinstance(e, str) for e in entities)

        # Common entities that should be detected
        common_entities = ["Apple Inc.", "California", "Steve Jobs"]
        assert any(entity in common_entities for entity in entities)

    def test_extract_entities_chatgpt(self):
        """Test entity extraction using ChatGPT."""
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = '["Apple", "California", "Steve Jobs"]'
        self.mock_client.chat.completions.create.return_value = mock_response

        column_name = "Company"
        sample_values = ["Apple Inc.", "Microsoft"]

        entities = extract_entities_chatgpt(
            self.mock_client, column_name, sample_values
        )

        assert entities == ["Apple", "California", "Steve Jobs"]
        self.mock_client.chat.completions.create.assert_called_once()

    def test_extract_entities_chatgpt_no_api_key(self):
        """Test ChatGPT extraction when API key is missing."""
        # Set client.api_key to None
        self.mock_client.api_key = None
        entities = extract_entities_chatgpt(self.mock_client, "Company", ["Apple Inc."])

        assert entities == []
        self.mock_client.chat.completions.create.assert_not_called()

    def test_extract_entities_chatgpt_api_error(self):
        """Test ChatGPT extraction when API call fails."""
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        entities = extract_entities_chatgpt(self.mock_client, "Company", ["Apple Inc."])

        assert entities == []

    def test_fuzzy_matches(self):
        """Test fuzzy matching using RapidFuzz."""
        candidates = ["Appl", "Microsft", "Califrnia"]
        constraints = ["Apple", "Microsoft", "Google", "California"]
        threshold = 80

        matches = fuzzy_matches(candidates, constraints, threshold)

        assert isinstance(matches, list)
        assert set(matches) == {"Apple", "Microsoft", "California"}

    def test_fuzzy_matches_no_matches(self):
        """Test fuzzy matching when no matches are found."""
        candidates = ["XYZ", "ABC"]
        constraints = ["Apple", "Microsoft"]
        threshold = 80

        matches = fuzzy_matches(candidates, constraints, threshold)

        assert matches == []

    def test_fuzzy_matches_chatgpt(self):
        """Test fuzzy matching using ChatGPT."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = '["Apple"]'
        self.mock_client.chat.completions.create.return_value = mock_response

        matches = fuzzy_matches_chatgpt(self.mock_client, ["Appl"], ["Apple"], 80)

        assert matches == ["Apple"]
        self.mock_client.chat.completions.create.assert_called_once()

    def test_fuzzy_matches_chatgpt_no_api_key(self):
        """Test ChatGPT fuzzy matching when API key is missing."""
        self.mock_client.api_key = None
        matches = fuzzy_matches_chatgpt(self.mock_client, ["Appl"], ["Apple"], 80)

        assert matches == []
        self.mock_client.chat.completions.create.assert_not_called()

    @patch("src.ingestion.extract.read_data")
    @patch("src.ingestion.extract.extract_entities_spacy")
    @patch("src.ingestion.extract.extract_entities_chatgpt")
    def test_process_data(
        self, mock_chatgpt, mock_spacy, mock_read_data, sample_dataframe
    ):
        """Test the main process_data function."""
        mock_read_data.return_value = sample_dataframe
        mock_spacy.return_value = ["Apple", "California"]
        mock_chatgpt.return_value = ["Apple Inc.", "California"]

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "json.dump"
        ) as mock_json_dump, patch("os.makedirs") as mock_makedirs:
            result = process_data(
                client=self.mock_client,
                file_path="dummy.csv",
                output_path="output.json",
                method="both",
            )

        # Check that the function processed the text column
        assert "text_column" in result
        assert "sample_values" in result["text_column"]
        assert "entities" in result["text_column"]

        # Check that both extraction methods were used
        entities = result["text_column"]["entities"]
        assert "spacy" in entities
        assert "chatgpt" in entities
        assert entities["spacy"] == ["Apple", "California"]
        assert entities["chatgpt"] == ["Apple Inc.", "California"]

        # Check that the output was saved
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once_with("output.json", "w", encoding="utf-8")
        mock_json_dump.assert_called_once()

    def test_verify_and_fix_column_structure(self):
        """Test verifying and fixing column structure."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = (
            '{"sample_values": ["test"], "entities": {"spacy": [], "chatgpt": [], '
            '"matches": {"spacy": [], "chatgpt": []}}}'
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        invalid_structure = {"sample_values": ["test"], "entities": {"spacy": []}}
        result = verify_and_fix_column_structure(self.mock_client, invalid_structure)

        assert "sample_values" in result
        assert "entities" in result
        assert all(k in result["entities"] for k in ["spacy", "chatgpt", "matches"])
        assert all(k in result["entities"]["matches"] for k in ["spacy", "chatgpt"])
        self.mock_client.chat.completions.create.assert_called_once()


# ===== Additional Tests for extract.py =====


class TestExtractAdditional:
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create mock OpenAI client
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = MagicMock()

    def test_get_top_n_rapidfuzz(self):
        """Test the get_top_n_rapidfuzz function for ranking constraints by similarity."""
        # Test with string candidate
        candidate = "Appl"
        constraints = ["Apple", "Microsoft", "Google", "Application"]

        top_matches = get_top_n_rapidfuzz(candidate, constraints, top_n=2)

        # Should return the top 2 matches: "Apple" and "Application"
        assert len(top_matches) == 2
        assert "Apple" in top_matches
        assert "Application" in top_matches

        # Test with dictionary candidate
        candidate_dict = {"term": "Appl", "classification": "entity"}
        top_matches_dict = get_top_n_rapidfuzz(candidate_dict, constraints, top_n=2)

        # Should return the same results as with string candidate
        assert top_matches_dict == top_matches

        # Test with top_n larger than constraints list
        top_matches_all = get_top_n_rapidfuzz(candidate, constraints, top_n=10)
        assert len(top_matches_all) == len(constraints)

    def test_extract_entities_chatgpt_with_classification(self):
        """Test entity extraction using ChatGPT with classification enabled."""
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = '[{"term": "Apple", "classification": "entity"}, {"term": "California", "classification": "entity"}, {"term": "Founded", "classification": "state"}]'
        self.mock_client.chat.completions.create.return_value = mock_response

        column_name = "Company"
        sample_values = ["Apple Inc.", "Microsoft"]

        entities = extract_entities_chatgpt(
            self.mock_client, column_name, sample_values, classify_candidates=True
        )

        # Check that we got the expected classified entities
        assert len(entities) == 3
        assert entities[0]["term"] == "Apple"
        assert entities[0]["classification"] == "entity"
        assert entities[2]["term"] == "Founded"
        assert entities[2]["classification"] == "state"

        # Verify the prompt included classification instructions
        call_args = self.mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        user_message = [m for m in messages if m["role"] == "user"][0]["content"]
        assert "classify it as either 'entity' or 'state'" in user_message
        assert (
            'Return a JSON array of objects, each with keys "term" and "classification"'
            in user_message
        )

    def test_extract_entities_chatgpt_invalid_classification_json(self):
        """Test ChatGPT extraction with classification when response is not valid JSON."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = "Apple: entity\nCalifornia: entity\nFounded: state"
        self.mock_client.chat.completions.create.return_value = mock_response

        entities = extract_entities_chatgpt(
            self.mock_client, "Company", ["Apple Inc."], classify_candidates=True
        )

        # Should extract lines from the non-JSON response
        assert entities == ["Apple: entity", "California: entity", "Founded: state"]

    def test_fuzzy_matches_chatgpt_with_classification(self):
        """Test fuzzy matching using ChatGPT with classification enabled."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = '[{"term": "Apple", "classification": "entity"}, {"term": "Microsoft", "classification": "entity"}]'
        self.mock_client.chat.completions.create.return_value = mock_response

        candidates = [
            {"term": "Appl", "classification": "entity"},
            {"term": "Microsft", "classification": "entity"},
        ]
        constraints = ["Apple", "Microsoft", "Google", "California"]

        matches = fuzzy_matches_chatgpt(
            self.mock_client,
            candidates,
            constraints,
            threshold=80,
            classify_candidates=True,
        )

        assert len(matches) == 2
        assert matches[0]["term"] == "Apple"
        assert matches[0]["classification"] == "entity"
        assert matches[1]["term"] == "Microsoft"
        assert matches[1]["classification"] == "entity"

        # Verify the prompt included classification instructions
        call_args = self.mock_client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        user_message = [m for m in messages if m["role"] == "user"][0]["content"]
        assert "return a JSON array of objects" in user_message
        assert "classification" in user_message

    def test_fuzzy_matches_chatgpt_chunking(self):
        """Test the chunking functionality in fuzzy_matches_chatgpt."""
        # Create a large constraints list that will require chunking
        large_constraints = [f"Term{i}" for i in range(300)]
        candidates = ["TermX", "TermY"]

        # Set up mock to return different responses for each chunk
        chunk1_response = MagicMock()
        chunk1_response.choices = [MagicMock()]
        chunk1_response.choices[0].message = MagicMock()
        chunk1_response.choices[0].message.content = '["Term1", "Term99"]'

        chunk2_response = MagicMock()
        chunk2_response.choices = [MagicMock()]
        chunk2_response.choices[0].message = MagicMock()
        chunk2_response.choices[0].message.content = '["Term200", "Term299"]'

        # Configure mock to return different responses for each call
        self.mock_client.chat.completions.create.side_effect = [
            chunk1_response,
            chunk2_response,
        ]

        matches = fuzzy_matches_chatgpt(
            self.mock_client,
            candidates,
            large_constraints,
            threshold=80,
            chunk_size=150,  # This will create 2 chunks (150 terms each)
        )

        # Check that results from both chunks were returned
        assert "Term1" in matches
        assert "Term99" in matches

    def test_verify_and_fix_column_structure_valid(self):
        """Test verify_and_fix_column_structure with already valid structure."""
        valid_structure = {
            "sample_values": ["Apple Inc.", "Microsoft"],
            "entities": {
                "spacy": ["Apple", "Microsoft"],
                "chatgpt": ["Apple Inc.", "Microsoft Corp"],
                "matches": {
                    "spacy": ["Organization"],
                    "chatgpt": ["Organization", "Company"],
                },
            },
        }

        # Function should return the structure unchanged
        result = verify_and_fix_column_structure(self.mock_client, valid_structure)
        assert result == valid_structure

        # API should not be called
        self.mock_client.chat.completions.create.assert_not_called()

    def test_verify_and_fix_column_structure_invalid(self):
        """Test verify_and_fix_column_structure with invalid structure."""
        invalid_structure = {
            "sample_values": ["Apple Inc.", "Microsoft"],
            "entities": {
                "spacy": ["Apple", "Microsoft"],
                "chatgpt": ["Apple Inc.", "Microsoft Corp"]
                # Missing "matches" key
            },
        }

        # Mock the API response with a fixed structure
        fixed_structure = {
            "sample_values": ["Apple Inc.", "Microsoft"],
            "entities": {
                "spacy": ["Apple", "Microsoft"],
                "chatgpt": ["Apple Inc.", "Microsoft Corp"],
                "matches": {"spacy": [], "chatgpt": []},
            },
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps(fixed_structure)
        self.mock_client.chat.completions.create.return_value = mock_response

        result = verify_and_fix_column_structure(self.mock_client, invalid_structure)

        # Should return the fixed structure
        assert result == fixed_structure

        # API should be called once
        self.mock_client.chat.completions.create.assert_called_once()

    def test_verify_and_fix_column_structure_api_error(self):
        """Test verify_and_fix_column_structure when API call fails."""
        invalid_structure = {
            "sample_values": ["Apple Inc."],
            # Missing "entities" key
        }

        # Mock API to raise an exception
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Function should return the original structure when API fails
        result = verify_and_fix_column_structure(self.mock_client, invalid_structure)
        assert result == invalid_structure

    def test_process_data_with_verify_structure(self):
        """Test process_data with verify_structure=True."""
        sample_dataframe = pd.DataFrame(
            {"text_column": ["Apple Inc. is based in California."]}
        )

        # Mock dependencies
        with patch("src.ingestion.extract.read_data") as mock_read_data, patch(
            "src.ingestion.extract.extract_entities_spacy"
        ) as mock_spacy, patch(
            "src.ingestion.extract.extract_entities_chatgpt"
        ) as mock_chatgpt, patch(
            "src.ingestion.extract.verify_and_fix_column_structure"
        ) as mock_verify:
            mock_read_data.return_value = sample_dataframe
            mock_spacy.return_value = ["Apple", "California"]
            mock_chatgpt.return_value = ["Apple Inc.", "California"]

            # Set up mock for verify_and_fix_column_structure to return a modified structure
            def verify_side_effect(client, column_data, **kwargs):
                # Add a marker to verify this function was called
                column_data["verified"] = True
                return column_data

            mock_verify.side_effect = verify_side_effect

            result = process_data(
                client=self.mock_client,
                file_path="dummy.csv",
                method="both",
                verify_structure=True,
            )

        # Check that verify_and_fix_column_structure was called
        assert "text_column" in result
        assert result["text_column"].get("verified") is True
        mock_verify.assert_called_once()

    def test_process_data_with_classification(self):
        """Test process_data with classify_candidates=True."""
        sample_dataframe = pd.DataFrame(
            {"text_column": ["Apple Inc. is based in California."]}
        )

        # Mock dependencies
        with patch("src.ingestion.extract.read_data") as mock_read_data, patch(
            "src.ingestion.extract.extract_entities_chatgpt"
        ) as mock_chatgpt:
            mock_read_data.return_value = sample_dataframe
            mock_chatgpt.return_value = [
                {"term": "Apple", "classification": "entity"},
                {"term": "California", "classification": "entity"},
            ]

            result = process_data(
                client=self.mock_client,
                file_path="dummy.csv",
                method="chatgpt",
                classify_candidates=True,
            )

        # Check that classification data was preserved
        entities = result["text_column"]["entities"]
        assert len(entities["chatgpt"]) == 2
        assert entities["chatgpt"][0]["term"] == "Apple"
        assert entities["chatgpt"][0]["classification"] == "entity"

        # Verify extract_entities_chatgpt was called with classify_candidates=True
        mock_chatgpt.assert_called_once()
        assert mock_chatgpt.call_args[1]["classify_candidates"] is True


# ===== Tests for ontology.py =====


class TestOntology:
    def test_get_label_or_localname_with_label(self, sample_rdf_graph):
        """Test getting label when it exists."""
        ns = rdflib.Namespace("http://example.org/ontology#")
        label = get_label_or_localname(ns.Person, sample_rdf_graph)
        assert label == "Person"

    def test_get_label_or_localname_without_label(self, sample_rdf_graph):
        """Test falling back to local name when label doesn't exist."""
        ns = rdflib.Namespace("http://example.org/ontology#")
        label = get_label_or_localname(ns.Organization, sample_rdf_graph)
        assert label == "Organization"

    def test_get_label_or_localname_with_slash_uri(self):
        """Test extracting local name from URI with slashes."""
        uri = rdflib.URIRef("http://example.org/ontology/Organization")
        graph = rdflib.Graph()
        label = get_label_or_localname(uri, graph)
        assert label == "Organization"

    @patch("rdflib.Graph")
    def test_process_ttl(self, mock_graph_class, sample_rdf_graph):
        """Test processing a Turtle file."""
        mock_graph_instance = MagicMock()
        mock_graph_class.return_value = mock_graph_instance
        mock_graph_instance.subjects.side_effect = [
            # Classes
            {rdflib.URIRef("http://ies.data.gov.uk/ontology/ies-building1#Person")},
            {
                rdflib.URIRef(
                    "http://ies.data.gov.uk/ontology/ies-building1#Organization"
                )
            },
            # Properties
            {rdflib.URIRef("http://ies.data.gov.uk/ontology/ies-building1#worksFor")},
            set(),  # No ObjectProperties
            set(),  # No DatatypeProperties
        ]

        # Mock label lookup
        def mock_value(uri, predicate):
            if uri == rdflib.URIRef(
                "http://ies.data.gov.uk/ontology/ies-building1#Person"
            ):
                return rdflib.Literal("Person")
            return None

        mock_graph_instance.value = mock_value

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "json.dump"
        ) as mock_json_dump, patch("os.makedirs") as mock_makedirs:
            result = process_ttl(file_path="ontology.ttl", output_path="output.json")

        assert result["ontology_file"] == "ontology.ttl"
        assert "entities" in result
        assert "classes" in result["entities"]
        assert "properties" in result["entities"]

        # Check extracted entities
        assert set(result["entities"]["classes"]) == {"Person", "Organization"}
        assert set(result["entities"]["properties"]) == {"worksFor"}

        # Check that output was saved
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once_with("output.json", "w", encoding="utf-8")
        mock_json_dump.assert_called_once()

    @patch("rdflib.Graph")
    def test_process_ttl_parse_error(self, mock_graph_class):
        """Test handling of parsing errors in process_ttl."""
        mock_graph_instance = mock_graph_class.return_value
        mock_graph_instance.parse.side_effect = Exception("Parsing error")

        with pytest.raises(ValueError, match="Error parsing Turtle file"):
            process_ttl(file_path="invalid.ttl")

    @patch("rdflib.Graph")
    def test_process_ttl_custom_namespace(self, mock_graph_class):
        """Test process_ttl with a custom namespace."""
        mock_graph_instance = MagicMock()
        mock_graph_class.return_value = mock_graph_instance

        # Set up mock to return entities with different namespaces
        mock_graph_instance.subjects.side_effect = [
            # Classes
            {
                rdflib.URIRef("http://example.org/ontology#Person"),
                rdflib.URIRef("http://ies.data.gov.uk/ontology/ies-building1#Building"),
            },
            set(),  # No owl:Class
            # Properties
            {
                rdflib.URIRef("http://example.org/ontology#name"),
                rdflib.URIRef("http://ies.data.gov.uk/ontology/ies-building1#hasFloor"),
            },
            set(),  # No ObjectProperties
            set(),  # No DatatypeProperties
        ]

        # Mock label lookup to return None (forcing local name extraction)
        mock_graph_instance.value.return_value = None

        result = process_ttl(
            file_path="ontology.ttl", namespace_prefix="http://example.org/ontology#"
        )

        # Only entities with the custom namespace should be included
        assert set(result["entities"]["classes"]) == {"Person"}
        assert set(result["entities"]["properties"]) == {"name"}


# ===== Tests for embeddings.py =====


class TestEmbeddings:
    def setup_method(self):
        """Set up test environment before each test method."""
        # Patch the OpenAI API key check to prevent initialization errors
        self.openai_patcher = patch(
            "src.ingestion.embeddings.openai_api_key", "dummy_key"
        )
        self.openai_patcher.start()

        # Patch the OpenAI client
        self.openai_client_patcher = patch("src.ingestion.embeddings.OpenAI")
        self.mock_openai_client = self.openai_client_patcher.start()

        # Set up mock client instance
        self.mock_client_instance = MagicMock()
        self.mock_openai_client.return_value = self.mock_client_instance

        # Set up mock embeddings response
        self.mock_embeddings = MagicMock()
        self.mock_client_instance.embeddings = self.mock_embeddings

        # Sample texts for testing
        self.sample_texts = [
            "Apple Inc. is a technology company.",
            "Microsoft was founded by Bill Gates.",
            "Google is known for its search engine.",
        ]

        # Sample embeddings (simplified 4D vectors for testing)
        self.sample_embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6],
        ]

    def teardown_method(self):
        """Clean up after each test method."""
        self.openai_patcher.stop()
        self.openai_client_patcher.stop()

    def test_initialize_vector_store(self):
        """Test initializing a vector store."""
        dimension = 4
        index, metadata = initialize_vector_store(dimension)

        assert isinstance(index, faiss.IndexFlatL2)
        assert index.d == dimension
        assert metadata == []

    def test_add_to_vector_store(self):
        """Test adding embeddings to a vector store."""
        # Initialize a vector store
        dimension = 4
        index, metadata = initialize_vector_store(dimension)

        # Create sample embeddings and metadata
        embeddings = np.array(self.sample_embeddings).astype("float32")
        meta_info = [
            {"text": self.sample_texts[0], "type": "company"},
            {"text": self.sample_texts[1], "type": "company"},
            {"text": self.sample_texts[2], "type": "company"},
        ]

        # Add to vector store
        index, metadata = add_to_vector_store(index, metadata, embeddings, meta_info)

        # Check results
        assert index.ntotal == 3
        assert len(metadata) == 3
        assert metadata[0]["text"] == self.sample_texts[0]
        assert metadata[0]["type"] == "company"
        assert metadata[0]["index"] == 0

    def test_add_to_vector_store_empty(self):
        """Test adding empty embeddings to a vector store."""
        dimension = 4
        index, metadata = initialize_vector_store(dimension)

        # Add empty embeddings
        index, metadata = add_to_vector_store(index, metadata, [], [])

        # Check that nothing changed
        assert index.ntotal == 0
        assert metadata == []

    def test_search_vector_store(self):
        """Test searching a vector store."""
        # Initialize and populate vector store
        dimension = 4
        index, metadata = initialize_vector_store(dimension)

        embeddings = np.array(self.sample_embeddings).astype("float32")
        meta_info = [
            {"text": self.sample_texts[0], "type": "company"},
            {"text": self.sample_texts[1], "type": "company"},
            {"text": self.sample_texts[2], "type": "company"},
        ]

        index, metadata = add_to_vector_store(index, metadata, embeddings, meta_info)

        # Search with a query vector similar to the first embedding
        query_vector = np.array([0.15, 0.25, 0.35, 0.45]).astype("float32")
        results = search_vector_store(index, metadata, query_vector, k=2)

        # Check results
        assert len(results) == 2
        assert results[0]["text"] == self.sample_texts[0]
        assert "distance" in results[0]
        assert "similarity" in results[0]

    def test_search_vector_store_with_filter(self):
        """Test searching a vector store with filtering."""
        # Initialize and populate vector store
        dimension = 4
        index, metadata = initialize_vector_store(dimension)

        embeddings = np.array(self.sample_embeddings).astype("float32")
        meta_info = [
            {"text": self.sample_texts[0], "type": "company", "name": "Apple"},
            {"text": self.sample_texts[1], "type": "company", "name": "Microsoft"},
            {"text": self.sample_texts[2], "type": "company", "name": "Google"},
        ]

        index, metadata = add_to_vector_store(index, metadata, embeddings, meta_info)

        # Search with a filter for Microsoft
        filter_func = lambda x: x.get("name") == "Microsoft"
        query_vector = np.array([0.5, 0.5, 0.5, 0.5]).astype("float32")
        results = search_vector_store(
            index, metadata, query_vector, k=3, filter_func=filter_func
        )

        # Check results
        assert len(results) == 1
        assert results[0]["name"] == "Microsoft"

    def test_save_and_load_vector_store(self):
        """Test saving and loading a vector store."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize and populate vector store
            dimension = 4
            index, metadata = initialize_vector_store(dimension)

            embeddings = np.array(self.sample_embeddings).astype("float32")
            meta_info = [
                {"text": self.sample_texts[0], "type": "company"},
                {"text": self.sample_texts[1], "type": "company"},
                {"text": self.sample_texts[2], "type": "company"},
            ]

            index, metadata = add_to_vector_store(
                index, metadata, embeddings, meta_info
            )

            # Save vector store
            with patch("os.makedirs") as mock_makedirs:
                with patch("faiss.write_index") as mock_write_index:
                    with patch("builtins.open", mock_open()) as mock_file:
                        with patch("pickle.dump") as mock_pickle_dump:
                            save_vector_store(
                                index, metadata, "test", directory=temp_dir
                            )

            # Check that the right functions were called
            mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True)
            mock_write_index.assert_called_once()
            mock_file.assert_called_once()
            mock_pickle_dump.assert_called_once()

            # Load vector store
            with patch("faiss.read_index") as mock_read_index:
                mock_read_index.return_value = faiss.IndexFlatL2(dimension)

                with patch(
                    "builtins.open", mock_open(read_data=pickle.dumps(metadata))
                ) as mock_file:
                    with patch("pickle.load") as mock_pickle_load:
                        mock_pickle_load.return_value = metadata
                        loaded_index, loaded_metadata = load_vector_store(
                            "test", directory=temp_dir
                        )

            # Check that the right functions were called
            mock_read_index.assert_called_once()
            mock_file.assert_called_once()
            mock_pickle_load.assert_called_once()

    def test_embed_texts_openai(self):
        """Test embedding texts with OpenAI."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=self.sample_embeddings[0]),
            MagicMock(embedding=self.sample_embeddings[1]),
            MagicMock(embedding=self.sample_embeddings[2]),
        ]
        self.mock_embeddings.create.return_value = mock_response

        # Call the function
        result = embed_texts_openai(self.sample_texts)

        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 4)  # 3 texts, 4D embeddings
        assert np.array_equal(result[0], self.sample_embeddings[0])

        # Check that the API was called correctly
        self.mock_embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=self.sample_texts
        )

    def test_embed_texts(self):
        """Test the main embed_texts function."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=self.sample_embeddings[0]),
            MagicMock(embedding=self.sample_embeddings[1]),
            MagicMock(embedding=self.sample_embeddings[2]),
        ]
        self.mock_embeddings.create.return_value = mock_response

        # Call the function
        result = embed_texts(self.sample_texts, model="text-embedding-3-small")

        # Check results
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 4)

        # Check that the API was called correctly
        self.mock_embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=self.sample_texts
        )

    def test_embed_texts_no_api_key(self):
        """Test embedding when API key is missing."""
        with patch("src.ingestion.embeddings.openai_api_key", None):
            result = embed_texts(self.sample_texts)

        # Should return empty array
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        self.mock_embeddings.create.assert_not_called()

    def test_embed_texts_api_error(self):
        """Test embedding when API call fails."""
        self.mock_embeddings.create.side_effect = Exception("API Error")

        result = embed_texts(self.sample_texts)

        # Should return empty array
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_analyze_vector_store(self):
        """Test analyzing a vector store."""
        # Initialize and populate vector store
        dimension = 4
        index, metadata = initialize_vector_store(dimension)

        # Add mixed types of data
        meta_info = [
            {"text": "Apple Inc.", "type": "company", "entity_label": "Apple"},
            {"text": "Microsoft Corp", "type": "company", "entity_label": "Microsoft"},
            {"text": "Person", "type": "ontology_entity", "entity_label": "Person"},
            {
                "text": "Organization",
                "type": "ontology_entity",
                "entity_label": "Organization",
            },
        ]

        # Mock embeddings for these items
        embeddings = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 0.8, 0.7, 0.6],
                [0.4, 0.3, 0.2, 0.1],
            ]
        ).astype("float32")

        index, metadata = add_to_vector_store(index, metadata, embeddings, meta_info)

        # Set up mock for embed_texts
        with patch("src.ingestion.embeddings.embed_texts") as mock_embed_texts:
            mock_embed_texts.return_value = np.array(
                [[0.9, 0.8, 0.7, 0.6]]
            )  # Same as "Person"

            # Capture print output
            with patch("builtins.print") as mock_print:
                type_counts = analyze_vector_store(
                    (index, metadata), model="text-embedding-3-small"
                )

        # Check results
        assert type_counts == {"company": 2, "ontology_entity": 2}
        mock_print.assert_called()  # Just verify it printed something
        mock_embed_texts.assert_called_once()


# ===== Tests for processing.py =====


class TestProcessing:
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create mock OpenAI client
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = MagicMock()

        # Sample DataFrame for testing
        self.sample_df = pd.DataFrame(
            {"column1": ["Apple Inc.", "Microsoft"], "column2": [100, 200]}
        )

    def test_analyze_csv_with_chatgpt(self):
        """Test CSV analysis with ChatGPT."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "theme": "Technology Companies",
                "metrics": ["Revenue", "Employees"],
                "characteristics": ["Global presence"],
                "summary": "Dataset about tech companies",
            }
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        result = analyze_csv_with_chatgpt(
            self.mock_client, self.sample_df, "gpt-4o-mini"
        )

        assert isinstance(result, str)
        parsed_result = json.loads(result)
        assert "theme" in parsed_result
        assert "metrics" in parsed_result
        self.mock_client.chat.completions.create.assert_called_once()

    def test_pseudo_ner_phrase_extraction(self):
        """Test domain-relevant term extraction."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["Apple", "Microsoft"]'
        self.mock_client.chat.completions.create.return_value = mock_response

        result = pseudo_ner_phrase_extraction(
            self.mock_client, str(self.sample_df), "gpt-4o-mini"
        )

        assert isinstance(result, str)
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, list)
        assert "Apple" in parsed_result
        self.mock_client.chat.completions.create.assert_called_once()

    def test_extract_boro_triples(self):
        """Test BORO triple extraction."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "triples": [{"head": "Apple", "relation": "isA", "tail": "Company"}],
                "boro_reasoning": "Apple is an enduring entity",
            }
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        result = extract_boro_triples(self.mock_client, self.sample_df, "gpt-4o-mini")

        assert isinstance(result, str)
        parsed_result = json.loads(result)
        assert "triples" in parsed_result
        assert "boro_reasoning" in parsed_result
        self.mock_client.chat.completions.create.assert_called_once()

    def test_gather_usage_patterns_and_subtypes(self):
        """Test gathering usage patterns and subtypes."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"usagePatterns": ["Pattern1"], "proposedSubtypes": ["Subtype1"]}
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        concepts = ["Apple"]
        domain_theme = "Technology"
        result = gather_usage_patterns_and_subtypes(
            self.mock_client, concepts, domain_theme, "gpt-4o-mini"
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "concept" in result[0]
        assert "usagePatterns" in result[0]
        assert "proposedSubtypes" in result[0]
        self.mock_client.chat.completions.create.assert_called_once()

    def test_classify_extension_type(self):
        """Test extension type classification."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "concept": "Apple",
                "classification": "entity",
                "extensionType": "subclass",
                "explanation": "Test explanation",
            }
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        result = classify_extension_type(self.mock_client, "Apple", "gpt-4o-mini")

        assert isinstance(result, str)
        parsed_result = json.loads(result)
        assert "concept" in parsed_result
        assert "classification" in parsed_result
        assert "extensionType" in parsed_result
        self.mock_client.chat.completions.create.assert_called_once()

    def test_analyze_step(self):
        """Test analyze step function."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Technology"
        self.mock_client.chat.completions.create.return_value = mock_response

        result = analyze_step(self.mock_client, self.sample_df, "gpt-4o-mini")

        assert isinstance(result, str)
        self.mock_client.chat.completions.create.assert_called()

    def test_analyze_step_empty_df(self):
        """Test analyze step with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = analyze_step(self.mock_client, empty_df, "gpt-4o-mini")

        assert result == "UnknownTheme"
        self.mock_client.chat.completions.create.assert_not_called()

    def test_analyze_tri(self):
        """Test analyze tri function."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"triples": [{"head": "Apple", "relation": "isA", "tail": "Company"}]}
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        result = analyze_tri(self.mock_client, self.sample_df, "gpt-4o-mini")

        assert isinstance(result, str)
        self.mock_client.chat.completions.create.assert_called_once()

    def test_analyze_tri_empty_df(self):
        """Test analyze tri with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = analyze_tri(self.mock_client, empty_df, "gpt-4o-mini")

        assert result == "UnknownTheme"
        self.mock_client.chat.completions.create.assert_not_called()

    def test_extract_concepts_step(self):
        """Test concept extraction step."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["Apple", "Microsoft"]'
        self.mock_client.chat.completions.create.return_value = mock_response

        result = extract_concepts_step(self.mock_client, self.sample_df, "gpt-4o-mini")

        assert isinstance(result, str)
        self.mock_client.chat.completions.create.assert_called_once()

    def test_gather_usage_step(self):
        """Test usage pattern gathering step."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"usagePatterns": ["Pattern1"], "proposedSubtypes": ["Subtype1"]}
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        concepts = ["Apple"]
        domain_theme = "Technology"
        result = gather_usage_step(
            self.mock_client, concepts, domain_theme, "gpt-4o-mini"
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "concept" in result[0]
        self.mock_client.chat.completions.create.assert_called_once()

    def test_classify_extensions(self):
        """Test extension classification."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "concept": "Apple",
                "classification": "entity",
                "extensionType": "subclass",
            }
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        usage_info = [{"concept": "Apple"}]
        result = classify_extensions(self.mock_client, usage_info, "gpt-4o-mini")

        assert isinstance(result, list)
        assert len(result) == 1
        self.mock_client.chat.completions.create.assert_called_once()
