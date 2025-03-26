"""
Unit tests for the validation module components.

This file contains unit tests for:
- validator.py: Ontology validation and conversion utilities
- comparison.py: Ontology comparison utilities
- abm.py: Agent-based modeling utilities
"""

import pytest
from unittest.mock import MagicMock
from rdflib import Graph
from openai import OpenAI
import json
from unittest.mock import patch, mock_open

from src.validation.validator import (
    convert_ttl_to_owl,
    convert_owl_to_ttl,
    validate_with_pyshacl,
    query_chatgpt_for_format_issue,
    extract_turtle_code_blocks,
    iterative_chatgpt_fix_ttl,
    generate_shacl_shapes,
    OntologyValidator,
)
from src.validation.comparison import compare
from src.validation.abm import (
    load_personas,
    chunk_text,
    rank_chunks_for_persona,
    build_multi_agent_prompt,
    simulate_multi_agent_discussion,
)

# ===== Fixtures =====


@pytest.fixture
def sample_ttl_content():
    """Create sample Turtle content for testing."""
    return """
    @prefix : <http://example.org/ontology#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    :Person rdf:type owl:Class .
    :name rdf:type owl:DatatypeProperty .
    """


@pytest.fixture
def sample_owl_content():
    """Create sample OWL/XML content for testing."""
    return """<?xml version="1.0"?>
    <rdf:RDF xmlns="http://example.org/ontology#"
         xml:base="http://example.org/ontology"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <owl:Class rdf:about="http://example.org/ontology#Person"/>
    </rdf:RDF>"""


@pytest.fixture
def sample_shacl_shapes():
    """Create sample SHACL shapes for testing."""
    return """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix : <http://example.org/ontology#> .

    :PersonShape
        a sh:NodeShape ;
        sh:targetClass :Person ;
        sh:property [
            sh:path :name ;
            sh:datatype xsd:string ;
            sh:minCount 1 ;
        ] .
    """


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock(spec=OpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client


# ===== Tests for validator.py =====


class TestValidator:
    def test_convert_ttl_to_owl(self, sample_ttl_content, tmp_path):
        """Test converting Turtle to OWL/XML format."""
        input_path = tmp_path / "input.ttl"
        output_path = tmp_path / "output.owl"

        with open(input_path, "w") as f:
            f.write(sample_ttl_content)

        convert_ttl_to_owl(str(input_path), str(output_path))

        assert output_path.exists()
        g = Graph()
        g.parse(str(output_path), format="xml")
        assert len(g) > 0

    def test_convert_owl_to_ttl(self, sample_owl_content, tmp_path):
        """Test converting OWL/XML to Turtle format."""
        input_path = tmp_path / "input.owl"
        output_path = tmp_path / "output.ttl"

        with open(input_path, "w") as f:
            f.write(sample_owl_content)

        convert_owl_to_ttl(str(input_path), str(output_path))

        assert output_path.exists()
        g = Graph()
        g.parse(str(output_path), format="turtle")
        assert len(g) > 0

    def test_validate_with_pyshacl(
        self, sample_ttl_content, sample_shacl_shapes, tmp_path
    ):
        """Test SHACL validation."""
        data_path = tmp_path / "data.ttl"
        shapes_path = tmp_path / "shapes.ttl"

        with open(data_path, "w") as f:
            f.write(sample_ttl_content)
        with open(shapes_path, "w") as f:
            f.write(sample_shacl_shapes)

        result = validate_with_pyshacl(str(data_path), str(shapes_path))
        assert isinstance(result, bool)

    def test_query_chatgpt_for_format_issue(self, mock_openai_client):
        """Test querying ChatGPT for format issues."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```\nFixed turtle content\n```"
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = query_chatgpt_for_format_issue(
            mock_openai_client, "gpt-4o-mini", "Syntax error", "Invalid turtle content"
        )

        assert isinstance(result, str)
        assert "Fixed turtle content" in result
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_extract_turtle_code_blocks(self):
        """Test extracting code blocks from ChatGPT response."""
        response = "Here's the fix:\n```turtle\nFixed content\n```\nEnd of fix."
        blocks = extract_turtle_code_blocks(response)

        assert len(blocks) == 1
        assert blocks[0] == "Fixed content"

    def test_iterative_chatgpt_fix_ttl(
        self, mock_openai_client, sample_ttl_content, tmp_path
    ):
        """Test iterative fixing of Turtle files."""
        input_path = tmp_path / "input.ttl"
        with open(input_path, "w") as f:
            f.write(sample_ttl_content)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = f"```\n{sample_ttl_content}\n```"
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = iterative_chatgpt_fix_ttl(
            mock_openai_client,
            "gpt-4o-mini",
            str(input_path),
            "Syntax error",
            max_iterations=2,
        )

        assert isinstance(result, bool)
        mock_openai_client.chat.completions.create.assert_called()

    def test_generate_shacl_shapes(self, mock_openai_client, sample_ttl_content):
        """Test generating SHACL shapes using LLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```\nGenerated SHACL shapes\n```"
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = generate_shacl_shapes(
            mock_openai_client, "gpt-4o-mini", sample_ttl_content
        )

        assert isinstance(result, str)
        assert "Generated SHACL shapes" in result
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_convert_ttl_to_owl_error(self, tmp_path):
        """Test error handling in TTL to OWL conversion."""
        input_path = tmp_path / "invalid.ttl"
        output_path = tmp_path / "output.owl"

        with open(input_path, "w") as f:
            f.write("Invalid Turtle content")

        with pytest.raises(Exception):
            convert_ttl_to_owl(str(input_path), str(output_path))

    def test_convert_owl_to_ttl_error(self, tmp_path):
        """Test error handling in OWL to TTL conversion."""
        input_path = tmp_path / "invalid.owl"
        output_path = tmp_path / "output.ttl"

        with open(input_path, "w") as f:
            f.write("Invalid OWL content")

        with pytest.raises(Exception):
            convert_owl_to_ttl(str(input_path), str(output_path))

    def test_validate_with_pyshacl_error(self, tmp_path):
        """Test error handling in SHACL validation."""
        data_path = tmp_path / "invalid.ttl"
        shapes_path = tmp_path / "invalid_shapes.ttl"

        with open(data_path, "w") as f:
            f.write("Invalid Turtle")
        with open(shapes_path, "w") as f:
            f.write("Invalid SHACL")

        result = validate_with_pyshacl(str(data_path), str(shapes_path))
        assert result is False

    def test_query_chatgpt_for_format_issue_error(self, mock_openai_client):
        """Test error handling in ChatGPT querying."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        result = query_chatgpt_for_format_issue(
            mock_openai_client, "gpt-4o-mini", "Error", "Content"
        )

        assert "Error querying ChatGPT" in result

    def test_extract_turtle_code_blocks_no_blocks(self):
        """Test code block extraction with no blocks present."""
        response = "No code blocks here"
        blocks = extract_turtle_code_blocks(response)

        assert len(blocks) == 0

    def test_generate_shacl_shapes_error(self, mock_openai_client):
        """Test error handling in SHACL shape generation."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        result = generate_shacl_shapes(
            mock_openai_client, "gpt-4o-mini", "ontology content"
        )

        assert result is None

    def test_ontology_validator_init(self, mock_openai_client):
        """Test OntologyValidator initialization."""
        validator = OntologyValidator(
            client=mock_openai_client, model="gpt-4o-mini", max_chatgpt_fixes=2
        )
        assert validator.client == mock_openai_client
        assert validator.model == "gpt-4o-mini"
        assert validator.max_chatgpt_fixes == 2
        assert validator.error_log == []
        assert validator.input_ttl_path is None
        assert validator.owl_output_path is None

    def test_ontology_validator_validate(
        self, mock_openai_client, sample_ttl_content, tmp_path
    ):
        """Test the main validation method of OntologyValidator."""
        input_path = tmp_path / "input.ttl"
        with open(input_path, "w") as f:
            f.write(sample_ttl_content)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```\nGenerated SHACL shapes\n```"
        mock_openai_client.chat.completions.create.return_value = mock_response

        validator = OntologyValidator(
            client=mock_openai_client, model="gpt-4o-mini", max_chatgpt_fixes=1
        )

        cleaned_ontology, error_log = validator.validate(str(input_path))

        assert isinstance(cleaned_ontology, str)
        assert isinstance(error_log, list)
        assert len(error_log) > 0

    def test_ontology_validator_handle_input_ttl(
        self, mock_openai_client, sample_ttl_content
    ):
        """Test handling of input TTL content."""
        validator = OntologyValidator(client=mock_openai_client, model="gpt-4o-mini")

        result = validator._handle_input_ttl(sample_ttl_content)

        assert result is True
        assert validator.input_ttl_path is not None
        assert validator.owl_output_path is not None

    def test_ontology_validator_error_handling(self, mock_openai_client):
        """Test error handling in OntologyValidator."""
        validator = OntologyValidator(client=mock_openai_client, model="gpt-4o-mini")

        cleaned_ontology, error_log = validator.validate("invalid content")

        assert cleaned_ontology is None
        assert isinstance(error_log, list)
        assert len(error_log) > 0

    def test_ontology_validator_shacl_validation(
        self, mock_openai_client, sample_ttl_content, sample_shacl_shapes, tmp_path
    ):
        """Test SHACL validation in OntologyValidator."""
        input_path = tmp_path / "input.ttl"
        shapes_path = tmp_path / "shapes.ttl"

        with open(input_path, "w") as f:
            f.write(sample_ttl_content)
        with open(shapes_path, "w") as f:
            f.write(sample_shacl_shapes)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "```\nGenerated SHACL shapes\n```"
        mock_openai_client.chat.completions.create.return_value = mock_response

        validator = OntologyValidator(client=mock_openai_client, model="gpt-4o-mini")

        cleaned_ontology, error_log = validator.validate(
            str(input_path), shape_file_path=str(shapes_path)
        )

        assert isinstance(cleaned_ontology, str)
        assert isinstance(error_log, list)
        assert len(error_log) > 0


# ===== Tests for comparison.py =====


class TestComparison:
    def setup_method(self):
        """Set up test environment before each test method."""
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = MagicMock()

    def test_compare_success(self):
        """Test successful comparison between synthetic and reference ontologies."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Comparison analysis result"
        self.mock_client.chat.completions.create.return_value = mock_response

        result = compare(
            self.mock_client,
            "gpt-4o",
            "synthetic ontology content",
            "reference ontology content",
        )

        assert result == "Comparison analysis result"
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["role"] == "user"

    def test_compare_api_error(self):
        """Test handling of API errors in comparison."""
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = compare(
            self.mock_client,
            "gpt-4o",
            "synthetic ontology content",
            "reference ontology content",
        )

        assert "Error querying ChatGPT" in result
        self.mock_client.chat.completions.create.assert_called_once()


# ===== Tests for abm.py =====


class TestABM:
    def setup_method(self):
        """Set up test environment before each test method."""
        self.mock_client = MagicMock()
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = MagicMock()

        # Sample personas configuration
        self.sample_personas = {
            "personas": [
                {
                    "name": "Domain Expert",
                    "description": "A domain expert.",
                    "prompt": "You are a domain expert.",
                },
                {
                    "name": "Ontologist",
                    "description": "An ontology expert.",
                    "prompt": "You are an ontology expert.",
                },
            ]
        }

    def test_load_personas_from_dict(self):
        """Test loading personas from a dictionary."""
        personas = load_personas(self.sample_personas)

        assert len(personas) == 2
        assert personas[0]["name"] == "Domain Expert"
        assert personas[1]["name"] == "Ontologist"

    def test_load_personas_from_json_string(self):
        """Test loading personas from a JSON string."""
        json_string = json.dumps(self.sample_personas)
        personas = load_personas(json_string)

        assert len(personas) == 2
        assert personas[0]["name"] == "Domain Expert"

    def test_load_personas_from_file(self, tmp_path):
        """Test loading personas from a file."""
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.sample_personas, f)

        personas = load_personas(str(config_file))

        assert len(personas) == 2
        assert personas[0]["name"] == "Domain Expert"

    def test_load_personas_invalid_json(self):
        """Test handling invalid JSON string."""
        with pytest.raises(ValueError):
            load_personas("invalid json")

    def test_load_personas_invalid_type(self):
        """Test handling invalid input type."""
        with pytest.raises(TypeError):
            load_personas(123)

    def test_chunk_text(self):
        """Test text chunking functionality."""
        text = "This is a test text that needs to be chunked into smaller pieces."
        chunk_size = 10
        overlap = 3

        chunks = chunk_text(text, chunk_size, overlap)

        assert len(chunks) > 1
        assert all(len(chunk) <= chunk_size for chunk in chunks)
        # Check overlap
        for i in range(len(chunks) - 1):
            assert chunks[i][-overlap:] == chunks[i + 1][:overlap]

    def test_rank_chunks_for_persona(self):
        """Test ranking chunks for a persona."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "Relevance: 8\nExplanation: Very relevant"
        self.mock_client.chat.completions.create.return_value = mock_response

        chunks = ["chunk1", "chunk2", "chunk3"]
        result = rank_chunks_for_persona(
            self.mock_client,
            "gpt-4o",
            "Domain Expert",
            "You are a domain expert",
            chunks,
            top_n=2,
        )

        assert isinstance(result, list)
        assert len(result) <= 2
        self.mock_client.chat.completions.create.assert_called()

    def test_build_multi_agent_prompt(self):
        """Test building multi-agent prompt."""
        personas = self.sample_personas["personas"]
        relevant_texts = {
            "Domain Expert": "relevant text 1",
            "Ontologist": "relevant text 2",
        }
        ontology_description = "Test ontology"

        messages = build_multi_agent_prompt(
            ontology_description, personas, relevant_texts
        )

        assert isinstance(messages, list)
        assert len(messages) == 3  # system message + 2 persona messages
        assert messages[0]["role"] == "system"
        assert all(msg["role"] == "user" for msg in messages[1:])
        assert all("name" in msg for msg in messages[1:])

    def test_simulate_multi_agent_discussion(self):
        """Test multi-agent discussion simulation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"Domain Expert": "Expert opinion", "Ontologist": "Ontology perspective"}
        )
        self.mock_client.chat.completions.create.return_value = mock_response

        result, error_message = simulate_multi_agent_discussion(
            self.mock_client,
            "gpt-4o",
            "ontology text",
            "ontology description",
            self.sample_personas,
        )
        assert isinstance(result, str)
        parsed_result = json.loads(result)
        assert "Domain Expert" in parsed_result
        assert "Ontologist" in parsed_result
        self.mock_client.chat.completions.create.assert_called()

    def test_simulate_multi_agent_discussion_error(self):
        """Test error handling in multi-agent discussion."""
        result, error_message = simulate_multi_agent_discussion(
            self.mock_client,
            "gpt-4o",
            "ontology text",
            "ontology description",
            "nonexistent_config.json",
        )

        assert "Error loading personas" in error_message
