"""
================================================================================
Extended Python Script for:
 - Bidirectional Ontology Conversion (Turtle <-> OWL/XML)
 - Loading & Reasoning (Owlready2)
 - Iterative ChatGPT Fixes for Broken Turtle
 - Optional pySHACL Validation
================================================================================

Usage Summary:
1. Convert .ttl to .owl (OWL/XML) or vice versa as needed.
2. Load the ontology in Owlready2, run reasoner checks, and print:
   - Inconsistent classes
   - Inferred superclasses
3. If loading fails due to syntax/format issues in the Turtle file, optionally
   iterate with ChatGPT suggestions:
   a) Ask ChatGPT for a corrected Turtle.
   b) Overwrite the original .ttl with the corrected content.
   c) Attempt to load again, up to max_iterations times.
4. Optionally run pySHACL to validate the loaded ontology graph against a set
   of SHACL shapes.

Dependencies:
    - rdflib
    - owlready2
    - openai
    - pyshacl
    - Java (for Owlready2 reasoner)


================================================================================
"""

import os
import re
import openai
from rdflib import Graph

from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate

# Regular expression to extract code blocks from LLM responses.
CODE_BLOCK_REGEX = re.compile(
    r"```(?:turtle|ttl)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE
)


# ------------------------------------------------------------------------------
# Conversion Functions using rdflib
# ------------------------------------------------------------------------------
def convert_ttl_to_owl(input_ttl_path: str, output_owl_path: str):
    """
    Convert a Turtle (.ttl) ontology file into OWL/XML format using RDFLib.
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
    """
    try:
        g = Graph()
        g.parse(input_owl_path, format="xml")
        g.serialize(destination=output_ttl_path, format="turtle")
        print(f"Conversion complete: {input_owl_path} -> {output_ttl_path}")
    except Exception as e:
        print(f"Error during conversion from OWL/XML to TTL: {e}")
        raise


# ------------------------------------------------------------------------------
# pySHACL Validation
# ------------------------------------------------------------------------------
def validate_with_pyshacl(data_path: str, shapes_path: str) -> bool:
    """
    Validate the given data graph against SHACL shapes using pySHACL.
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


# ------------------------------------------------------------------------------
# ChatGPT Fix Functionality
# ------------------------------------------------------------------------------
def query_chatgpt_for_format_issue(
    model: any, error_message: str, ontology_content: str
) -> str:
    """
    Query ChatGPT for suggestions when the ontology fails to load.
    """
    prompt = (
        f"The ontology file failed to load due to this error:\n\n{error_message}\n\n"
        f"Here is the current Turtle content:\n\n{ontology_content}\n\n"
        "Please provide a CORRECTED Turtle file. Put the corrected code inside "
        "triple backticks so that I can parse it automatically. If there are no "
        "errors, or you can't fix them, please say so."
    )
    try:
        response = openai.chat.completions.create(
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
    """
    matches = CODE_BLOCK_REGEX.findall(chatgpt_response)
    return [m.strip() for m in matches if m.strip()]


def iterative_chatgpt_fix_ttl(
    model: str, ttl_file_path: str, original_error: str, max_iterations: int = 3
) -> bool:
    """
    Attempt iterative fixes of a Turtle file by querying ChatGPT for a corrected file,
    rewriting the local .ttl, and testing with RDFLib.
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
            model, original_error, current_content
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


# ------------------------------------------------------------------------------
# SHACL Shapes Generation via LLM
# ------------------------------------------------------------------------------
def generate_shacl_shapes(model: str, ontology_content: str) -> str:
    """
    Generate SHACL shapes constraints for the given ontology using an LLM.
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
        response = openai.chat.completions.create(
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


# ------------------------------------------------------------------------------
# End-to-End Example: Convert TTL → OWL, Reason, Possibly Fix, SHACL, etc.
# ------------------------------------------------------------------------------
def convert_and_check_ttl_ontology(
    model: str,
    input_ttl_path: str,
    shape_file_path: str = None,
    max_chatgpt_fixes: int = 0,
):
    """
    1. Convert a Turtle (.ttl) file to OWL/XML using rdflib.
    2. Load the OWL/XML file into an rdflib Graph and run OWL RL reasoning using owlrl.
    3. If the load fails, optionally try iterative ChatGPT fixes on the .ttl file.
    4. If a SHACL shapes file is not provided or found, generate constraints using an LLM,
       save them to a temporary file, and run pySHACL validation.

    Returns:
      (cleaned_ontology, error_log)
    """
    error_log = []

    # If input_ttl_path is not an existing file, assume it's Turtle content and write it to a temporary file.
    if not os.path.isfile(input_ttl_path):
        temp_file = "temp_ontology.ttl"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(input_ttl_path)
            msg = f"Input provided as Turtle content. Written to temporary file: {temp_file}"
            print(msg)
            error_log.append(msg)
            input_ttl_path = temp_file
        except Exception as e:
            err_msg = f"Error writing temporary Turtle file: {e}"
            print(err_msg)
            error_log.append(err_msg)
            return None, error_log

    # Step 1: Convert from .ttl to .owl
    owl_output_path = input_ttl_path.replace(".ttl", ".owl")
    try:
        convert_ttl_to_owl(input_ttl_path, owl_output_path)
        msg = "Step 1: TTL→OWL conversion succeeded."
        print(msg)
        error_log.append(msg)
    except Exception as e:
        err_msg = f"Step 1: Initial TTL→OWL conversion failed. Error: {e}"
        print(err_msg)
        error_log.append(err_msg)
        if max_chatgpt_fixes > 0:
            fix_result = iterative_chatgpt_fix_ttl(
                model, input_ttl_path, str(e), max_iterations=max_chatgpt_fixes
            )
            if fix_result:
                try:
                    convert_ttl_to_owl(input_ttl_path, owl_output_path)
                    msg = "Step 1: TTL→OWL conversion succeeded after ChatGPT fix."
                    print(msg)
                    error_log.append(msg)
                except Exception as e2:
                    err_msg = f"Step 1: Even after ChatGPT fixes, conversion still fails: {e2}"
                    print(err_msg)
                    error_log.append(err_msg)
                    return None, error_log
            else:
                err_msg = "Step 1: No successful fix found. Stopping."
                print(err_msg)
                error_log.append(err_msg)
                return None, error_log
        else:
            err_msg = "Step 1: No ChatGPT fix attempts enabled. Stopping."
            print(err_msg)
            error_log.append(err_msg)
            return None, error_log

    # Step 2: Load the OWL/XML file into an rdflib Graph and run OWL RL reasoning.
    try:
        owl_graph = Graph()
        owl_graph.parse(owl_output_path, format="xml")
        DeductiveClosure(OWLRL_Semantics).expand(owl_graph)
        msg = "Step 2: OWL/XML loaded and OWL RL reasoning executed successfully."
        print(msg)
        error_log.append(msg)
    except Exception as e:
        err_msg = f"Step 2: Could not load or reason over the OWL file. Error: {e}"
        print(err_msg)
        error_log.append(err_msg)
        if max_chatgpt_fixes > 0:
            fix_result = iterative_chatgpt_fix_ttl(
                model, input_ttl_path, str(e), max_iterations=max_chatgpt_fixes
            )
            if fix_result:
                try:
                    convert_ttl_to_owl(input_ttl_path, owl_output_path)
                    owl_graph = Graph()
                    owl_graph.parse(owl_output_path, format="xml")
                    DeductiveClosure(OWLRL_Semantics).expand(owl_graph)
                    msg = "Step 2: OWL/XML loaded and reasoned successfully after ChatGPT fix."
                    print(msg)
                    error_log.append(msg)
                except Exception as e2:
                    err_msg = f"Step 2: Still could not load or reason after ChatGPT fix: {e2}"
                    print(err_msg)
                    error_log.append(err_msg)
                    return None, error_log
            else:
                err_msg = (
                    "Step 2: No successful fix found after multiple attempts. Stopping."
                )
                print(err_msg)
                error_log.append(err_msg)
                return None, error_log
        else:
            err_msg = "Step 2: No ChatGPT fix attempts enabled. Stopping."
            print(err_msg)
            error_log.append(err_msg)
            return None, error_log

    # Step 3: SHACL Validation.
    # If a shapes file is provided and exists, use it.
    # Otherwise, attempt to generate SHACL shapes using the LLM.
    if shape_file_path:
        if os.path.isfile(shape_file_path):
            msg = f"Step 3: Running pySHACL validation with shapes file: {shape_file_path}"
            print(msg)
            error_log.append(msg)
            _ = validate_with_pyshacl(input_ttl_path, shape_file_path)
        else:
            msg = f"Step 3: Provided SHACL shapes file not found: {shape_file_path}. Attempting to generate shapes using LLM."
            print(msg)
            error_log.append(msg)
            with open(input_ttl_path, "r", encoding="utf-8") as f:
                ontology_content = f.read()
            generated_shapes = generate_shacl_shapes(model, ontology_content)
            if generated_shapes:
                temp_shapes_path = "temp_shapes.ttl"
                with open(temp_shapes_path, "w", encoding="utf-8") as f:
                    f.write(generated_shapes)
                msg = f"Step 3: Generated SHACL shapes saved to temporary file: {temp_shapes_path}"
                print(msg)
                error_log.append(msg)
                _ = validate_with_pyshacl(input_ttl_path, temp_shapes_path)
            else:
                msg = "Step 3: Failed to generate SHACL shapes. Skipping SHACL validation."
                print(msg)
                error_log.append(msg)
    else:
        msg = "Step 3: No SHACL shapes file provided. Attempting to generate SHACL shapes using LLM."
        print(msg)
        error_log.append(msg)
        with open(input_ttl_path, "r", encoding="utf-8") as f:
            ontology_content = f.read()
        generated_shapes = generate_shacl_shapes(model, ontology_content)
        if generated_shapes:
            temp_shapes_path = "temp_shapes.ttl"
            with open(temp_shapes_path, "w", encoding="utf-8") as f:
                f.write(generated_shapes)
            msg = f"Step 3: Generated SHACL shapes saved to temporary file: {temp_shapes_path}"
            print(msg)
            error_log.append(msg)
            _ = validate_with_pyshacl(input_ttl_path, temp_shapes_path)
        else:
            msg = "Step 3: Failed to generate SHACL shapes. Skipping SHACL validation."
            print(msg)
            error_log.append(msg)

    # Finally, return the cleaned ontology by reading the final TTL file.
    try:
        with open(input_ttl_path, "r", encoding="utf-8") as f:
            cleaned_ontology = f.read()
        msg = "Clean ontology successfully read from file."
        print(msg)
        error_log.append(msg)
        return cleaned_ontology, error_log
    except Exception as e:
        err_msg = f"Error reading cleaned ontology: {e}"
        print(err_msg)
        error_log.append(err_msg)
        return None, error_log


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Example usage with placeholders
    # --------------------------------------------------------------------------

    # The main Turtle file that you want to test & possibly fix
    example_ttl_file = "ies-building1.ttl"

    # Optional SHACL shapes file (Turtle) for validation (or set to None)
    example_shacl_shapes = (
        "path/to/building_shapes.ttl"  # Replace with an actual file path or None
    )

    # We'll try up to 2 iterative fixes with ChatGPT if something breaks
    max_fixes = 2

    convert_and_check_ttl_ontology(
        model="03-mini",
        input_ttl_path=example_ttl_file,
        shape_file_path=example_shacl_shapes,  # or None
        max_chatgpt_fixes=max_fixes,
    )
