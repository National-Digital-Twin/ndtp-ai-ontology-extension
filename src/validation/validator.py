"""
Ontology validation and conversion utilities for:
- Converting between Turtle (.ttl) and OWL/XML formats
- Validating ontologies using Owlready2 reasoning
- Automatically fixing broken Turtle syntax using LLM suggestions
- Validating against SHACL constraints

Key Features:
- Bidirectional format conversion between Turtle and OWL/XML
- Ontology reasoning and consistency checking with Owlready2
- Automated repair of syntax issues using LLM suggestions
- Optional SHACL validation with auto-generated constraints
- Detailed error logging and progress tracking

Usage:
    from src.validation.validator import OntologyValidator
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    validator = OntologyValidator(
        client=client,
        model="o3-mini",
        max_chatgpt_fixes=2
    )
    
    cleaned_ontology, error_log = validator.validate(
        input_ttl_path="ies-building1.ttl",
        shape_file_path="shapes.ttl"  # Optional
    )
"""

import os
import re
from openai import OpenAI
from rdflib import Graph

from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate


# Regular expression to extract code blocks from LLM responses.
CODE_BLOCK_REGEX = re.compile(
    r"```(?:turtle|ttl)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE
)


def convert_ttl_to_owl(input_ttl_path: str, output_owl_path: str):
    """
    Convert a Turtle (.ttl) ontology file into OWL/XML format using RDFLib.

    Args:
        input_ttl_path (str): The path to the input Turtle (.ttl) file.
        output_owl_path (str): The path to the output OWL/XML file.
    """
    try:
        g = Graph()
        g.parse(input_ttl_path, format="turtle")
        g.serialize(destination=output_owl_path, format="xml")
        print(f"Conversion complete: {input_ttl_path} -> {output_owl_path}")
    except Exception as e:
        print(f"Error during conversion from TTL to OWL/XML: {e}")
        raise


def convert_owl_to_ttl(input_owl_path: str, output_ttl_path: str):
    """
    Convert an OWL/XML file back to Turtle format using RDFLib.

    Args:
        input_owl_path (str): The path to the input OWL/XML file.
        output_ttl_path (str): The path to the output Turtle (.ttl) file.
    """
    try:
        g = Graph()
        g.parse(input_owl_path, format="xml")
        g.serialize(destination=output_ttl_path, format="turtle")
        print(f"Conversion complete: {input_owl_path} -> {output_ttl_path}")
    except Exception as e:
        print(f"Error during conversion from OWL/XML to TTL: {e}")
        raise


def validate_with_pyshacl(data_path: str, shapes_path: str) -> bool:
    """
    Validate the given data graph against SHACL shapes using pySHACL.

    Args:
        data_path (str): The path to the data graph file (Turtle format).
        shapes_path (str): The path to the SHACL shapes file (Turtle format).

    Returns:
        bool: True if the validation passes, False otherwise.
    """
    try:
        data_g = Graph()
        data_g.parse(data_path, format="turtle")

        shapes_g = Graph()
        shapes_g.parse(shapes_path, format="turtle")

        conforms, results_graph, results_text = validate(
            data_graph=data_g,
            shacl_graph=shapes_g,
            ont_graph=None,
            inference="rdfs",
            abort_on_first=False,
            meta_shacl=False,
            debug=False,
            serialize_report_graph=True,
        )
        print("\n--- pySHACL Validation Results ---")
        if conforms:
            print("SHACL conformance: PASSED")
        else:
            print("SHACL conformance: FAILED")
        print("Detailed Report:")
        print(results_text)
        return bool(conforms)
    except Exception as e:
        print(f"Error during SHACL validation: {e}")
        return False


def query_chatgpt_for_format_issue(
    client: OpenAI, model: str, error_message: str, ontology_content: str
) -> str:
    """
    Query ChatGPT for suggestions when the ontology fails to load.

    Args:
        client: OpenAI client
        model: OpenAI model
        error_message: Error message from the ontology loading
        ontology_content: Current Turtle content of the ontology

    Returns:
        str: Corrected Turtle content
    """
    prompt = (
        f"The ontology file failed to load due to this error:\n\n{error_message}\n\n"
        f"Here is the current Turtle content:\n\n{ontology_content}\n\n"
        "Please provide a CORRECTED Turtle file. Put the corrected code inside "
        "triple backticks so that I can parse it automatically. If there are no "
        "errors, or you can't fix them, please say so."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in RDF, OWL, and Turtle syntax.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying ChatGPT: {e}"


def extract_turtle_code_blocks(chatgpt_response: str) -> list:
    """
    Extract code blocks (the content within triple backticks) from a ChatGPT response.

    Args:
        chatgpt_response (str): The response from ChatGPT

    Returns:
        list: List of code blocks found in the response
    """
    matches = CODE_BLOCK_REGEX.findall(chatgpt_response)
    return [m.strip() for m in matches if m.strip()]


def iterative_chatgpt_fix_ttl(
    client: OpenAI,
    model: str,
    ttl_file_path: str,
    original_error: str,
    max_iterations: int = 3,
) -> bool:
    """
    Attempt iterative fixes of a Turtle file by querying ChatGPT for a corrected file,
    rewriting the local .ttl, and testing with RDFLib.

    Args:
        client: OpenAI client
        model: OpenAI model
        ttl_file_path: Path to the local Turtle file
        original_error: Error message from the initial attempt
        max_iterations: Maximum number of iterations to attempt

    Returns:
        bool: True if the Turtle file was fixed, False otherwise
    """
    for attempt in range(1, max_iterations + 1):
        print(f"\n[ChatGPT Fix Attempt {attempt}/{max_iterations}]")
        try:
            with open(ttl_file_path, "r", encoding="utf-8") as f:
                current_content = f.read()
        except Exception as e:
            print(f"Could not read TTL file: {e}")
            return False

        chatgpt_response = query_chatgpt_for_format_issue(
            client, model, original_error, current_content
        )
        code_blocks = extract_turtle_code_blocks(chatgpt_response)
        if not code_blocks:
            print(
                "No triple-backtick code blocks found in ChatGPT response. Cannot fix automatically."
            )
            print("Response was:\n", chatgpt_response)
            return False

        new_ttl_content = code_blocks[0]
        backup_path = ttl_file_path + f".backup_{attempt}"
        with open(backup_path, "w", encoding="utf-8") as bf:
            bf.write(current_content)
        print(f"Backed up current TTL to: {backup_path}")

        with open(ttl_file_path, "w", encoding="utf-8") as wf:
            wf.write(new_ttl_content)
        print(f"Rewrote {ttl_file_path} with ChatGPT's suggested fix.")

        test_graph = Graph()
        try:
            test_graph.parse(ttl_file_path, format="turtle")
            print("Success: The new Turtle now parses without error.")
            return True
        except Exception as parse_err:
            print(f"Still failing to parse: {parse_err}")
            original_error = str(parse_err)

    print("Max iterations reached. Could not fix the Turtle file automatically.")
    return False


def generate_shacl_shapes(client: OpenAI, model: str, ontology_content: str) -> str:
    """
    Generate SHACL shapes constraints for the given ontology using an LLM.

    Args:
        client: OpenAI client
        model: OpenAI model
        ontology_content: Current Turtle content of the ontology

    Returns:
        str: Generated SHACL shapes in Turtle format
    """
    prompt = (
        "You are an expert in RDF and SHACL. Given the following ontology in Turtle format, "
        "please generate a SHACL shapes file in Turtle format that defines constraints to validate "
        "the ontology. Include constraints for classes, properties, and relationships as appropriate. "
        "Ontology:\n\n"
        + ontology_content
        + "\n\nPlease provide the SHACL shapes inside triple backticks."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in RDF, OWL, and SHACL.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        reply = response.choices[0].message.content
        matches = CODE_BLOCK_REGEX.findall(reply)
        if matches:
            shacl_shapes = matches[0].strip()
        else:
            shacl_shapes = reply.strip()
        return shacl_shapes
    except Exception as e:
        print(f"Error generating SHACL shapes: {e}")
        return None


class OntologyValidator:
    """
    Validates and converts ontologies between formats with support for automated fixes.

    Features:
    - TTL to OWL/XML conversion
    - Ontology reasoning with Owlready2
    - Automated fixes using LLM suggestions
    - SHACL validation with optional constraint generation
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_chatgpt_fixes: int = 0,
        temp_dir: str = "data/temp",
    ):
        """
        Initialize the validator.

        Args:
            client: OpenAI client for LLM-based fixes
            model: OpenAI model to use
            max_chatgpt_fixes: Maximum number of fix attempts (0 to disable)
            temp_dir: Temporary directory for storing files (default: None)
        """
        self.client = client
        self.model = model
        self.max_chatgpt_fixes = max_chatgpt_fixes
        self.error_log = []
        self.input_ttl_path = None
        self.owl_output_path = None
        self.temp_dir = temp_dir
        if self.temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)

    def log(self, message: str) -> None:
        """Add a message to the error log and print it."""
        print(message)
        self.error_log.append(message)

    def _handle_input_ttl(self, input_ttl_path: str) -> bool:
        """Handle initial TTL file processing."""
        if not os.path.isfile(input_ttl_path):
            temp_file = os.path.join(self.temp_dir, "temp_ontology.ttl")
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(input_ttl_path)
                self.log(
                    f"Input provided as Turtle content. Written to temporary file: {temp_file}"
                )
                self.input_ttl_path = temp_file
            except Exception as e:
                self.log(f"Error writing temporary Turtle file: {e}")
                return False
        else:
            self.input_ttl_path = input_ttl_path

        self.owl_output_path = self.input_ttl_path.replace(".ttl", ".owl")
        return True

    def _convert_ttl_to_owl_with_fixes(self) -> bool:
        """Handle TTL to OWL conversion with optional fixes."""
        try:
            convert_ttl_to_owl(self.input_ttl_path, self.owl_output_path)
            self.log("Step 1: TTL→OWL conversion succeeded.")
            return True
        except Exception as e:
            self.log(f"Step 1: Initial TTL→OWL conversion failed. Error: {e}")

            if self.max_chatgpt_fixes > 0:
                if iterative_chatgpt_fix_ttl(
                    self.client,
                    self.model,
                    self.input_ttl_path,
                    str(e),
                    max_iterations=self.max_chatgpt_fixes,
                ):
                    try:
                        convert_ttl_to_owl(self.input_ttl_path, self.owl_output_path)
                        self.log(
                            "Step 1: TTL→OWL conversion succeeded after ChatGPT fix."
                        )
                        return True
                    except Exception as e2:
                        self.log(
                            f"Step 1: Even after ChatGPT fixes, conversion still fails: {e2}"
                        )
                else:
                    self.log("Step 1: No successful fix found. Stopping.")
            else:
                self.log("Step 1: No ChatGPT fix attempts enabled. Stopping.")
            return False

    def _handle_owl_reasoning(self) -> bool:
        """Handle OWL loading and reasoning with optional fixes."""
        try:
            owl_graph = Graph()
            owl_graph.parse(self.owl_output_path, format="xml")
            DeductiveClosure(OWLRL_Semantics).expand(owl_graph)
            self.log(
                "Step 2: OWL/XML loaded and OWL RL reasoning executed successfully."
            )
            return True
        except Exception as e:
            self.log(f"Step 2: Could not load or reason over the OWL file. Error: {e}")

            if self.max_chatgpt_fixes > 0:
                if iterative_chatgpt_fix_ttl(
                    self.client,
                    self.model,
                    self.input_ttl_path,
                    str(e),
                    max_iterations=self.max_chatgpt_fixes,
                ):
                    try:
                        convert_ttl_to_owl(self.input_ttl_path, self.owl_output_path)
                        owl_graph = Graph()
                        owl_graph.parse(self.owl_output_path, format="xml")
                        DeductiveClosure(OWLRL_Semantics).expand(owl_graph)
                        self.log(
                            "Step 2: OWL/XML loaded and reasoned successfully after ChatGPT fix."
                        )
                        return True
                    except Exception as e2:
                        self.log(
                            f"Step 2: Still could not load or reason after ChatGPT fix: {e2}"
                        )
                else:
                    self.log(
                        "Step 2: No successful fix found after multiple attempts. Stopping."
                    )
            else:
                self.log("Step 2: No ChatGPT fix attempts enabled. Stopping.")
            return False

    def _handle_shacl_validation(self, shape_file_path: str = None) -> None:
        """Handle SHACL validation with optional shape generation."""
        if shape_file_path and os.path.isfile(shape_file_path):
            self.log(
                f"Step 3: Running pySHACL validation with shapes file: {shape_file_path}"
            )
            _ = validate_with_pyshacl(self.input_ttl_path, shape_file_path)
            return

        self.log(
            "Step 3: No SHACL shapes file found. Attempting to generate SHACL shapes using LLM."
        )

        with open(self.input_ttl_path, "r", encoding="utf-8") as f:
            ontology_content = f.read()
        generated_shapes = generate_shacl_shapes(
            self.client, self.model, ontology_content
        )

        if generated_shapes:
            temp_shapes_path = os.path.join(self.temp_dir, "temp_shapes.ttl")
            with open(temp_shapes_path, "w", encoding="utf-8") as f:
                f.write(generated_shapes)
            self.log(
                f"Step 3: Generated SHACL shapes saved to temporary file: {temp_shapes_path}"
            )
            _ = validate_with_pyshacl(self.input_ttl_path, temp_shapes_path)
        else:
            self.log(
                "Step 3: Failed to generate SHACL shapes. Skipping SHACL validation."
            )

    def validate(
        self, input_ttl_path: str, shape_file_path: str = None
    ) -> tuple[str | None, list]:
        """
        Validate and convert the ontology through multiple steps.

        Args:
            input_ttl_path: Path to input TTL file or TTL content string
            shape_file_path: Optional path to SHACL shapes file

        Returns:
            tuple: (cleaned_ontology, error_log) where cleaned_ontology is None if validation fails
        """
        # Reset state for new validation
        self.error_log = []

        # Step 1: Handle input TTL
        if not self._handle_input_ttl(input_ttl_path):
            return None, self.error_log

        # Step 2: Convert TTL to OWL
        if not self._convert_ttl_to_owl_with_fixes():
            return None, self.error_log

        # Step 3: Load and reason over OWL
        if not self._handle_owl_reasoning():
            return None, self.error_log

        # Step 4: SHACL Validation
        self._handle_shacl_validation(shape_file_path)

        # Step 5: Return final ontology
        try:
            with open(self.input_ttl_path, "r", encoding="utf-8") as f:
                cleaned_ontology = f.read()
            self.log("Clean ontology successfully read from file.")
            return cleaned_ontology, self.error_log
        except Exception as e:
            self.log(f"Error reading cleaned ontology: {e}")
            return None, self.error_log


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # The main Turtle file that you want to test & possibly fix
    example_ttl_file = "ies-building1.ttl"

    # Optional SHACL shapes file (Turtle) for validation (or set to None)
    example_shacl_shapes = (
        "path/to/building_shapes.ttl"  # Replace with an actual file path or None
    )

    # We'll try up to 2 iterative fixes with ChatGPT if something breaks
    max_fixes = 2

    validator = OntologyValidator(
        client=client, model="o3-mini", max_chatgpt_fixes=max_fixes
    )

    cleaned_ontology, error_log = validator.validate(
        input_ttl_path=example_ttl_file, shape_file_path=example_shacl_shapes  # or None
    )
