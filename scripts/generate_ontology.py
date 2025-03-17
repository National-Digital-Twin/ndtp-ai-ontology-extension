#!/usr/bin/env python3
"""
Ontology Generation Script

This script implements an iterative workflow for generating and refining
ontology snippets in Turtle format using LLMs.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from src
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.generation.llm_interface import (
    build_prompt_for_generation,
    generate_ttl_snippet,
    refine_instructions,
    compare_snippets,
    interpret_comparison_result,
    build_error_summary,
)
from src.generation.iteration import load_checkpoint, save_checkpoint

# Default configuration
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MODEL = "o3-mini"


def load_common_snippet(file_path: Path) -> str:
    """Load the common ontology template."""
    with file_path.open("r", encoding="utf-8") as f:
        return f.read()


def load_csv_data_small_sample(file_path: Path, max_rows=25) -> str:
    """
    Read a small sample of rows from a CSV file.
    Returns a string representation to keep the prompt small.
    """
    rows_text = []
    with file_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
            rows_text.append(row_str)
    return "\n".join(rows_text)


def load_reference_snippet(file_path: Path) -> str:
    """Load the reference ontology snippet."""
    with file_path.open("r", encoding="utf-8") as f:
        return f.read()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and refine ontology snippets using LLMs"
    )

    parser.add_argument(
        "--common-snippet",
        type=str,
        required=True,
        help="Path to the common ontology template (snippet A)",
    )
    parser.add_argument(
        "--csv-file", type=str, required=True, help="Path to the CSV data file"
    )
    parser.add_argument(
        "--reference-snippet",
        type=str,
        required=True,
        help="Path to the reference ontology snippet (snippet B)",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="Path to the initial instructions file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/iteration_checkpoint.json",
        help="Path to the checkpoint file (default: results/iteration_checkpoint.json)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/final_ontology.ttl",
        help="Path to save the final ontology snippet (default: results/final_ontology.ttl)",
    )

    return parser.parse_args()


def main():
    """Main function to run the ontology generation workflow.
    
    Example usage:
    python -m scripts.generate_ontology \
        --common-snippet data/generation/common_template.ttl \
        --csv-file data/generation/data_building_123003.csv \
        --reference-snippet data/generation/building_ontology.ttl \
        --instructions data/generation/initial_prompt.txt \
        --checkpoint results/iteration_checkpoint.json \
        --output results/final_ontology.ttl
    """
    args = parse_arguments()

    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Convert string paths to Path objects
    common_snippet_path = Path(args.common_snippet)
    csv_file_path = Path(args.csv_file)
    reference_snippet_path = Path(args.reference_snippet)
    instructions_path = Path(args.instructions)
    checkpoint_file_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading common snippet from {common_snippet_path}")
    common_snippet = load_common_snippet(common_snippet_path)

    print(f"Loading CSV data from {csv_file_path}")
    csv_data_str = load_csv_data_small_sample(csv_file_path)

    print(f"Loading reference snippet from {reference_snippet_path}")
    reference_snippet = load_reference_snippet(reference_snippet_path)

    print(f"Loading initial instructions from {instructions_path}")
    with open(instructions_path, "r", encoding="utf-8") as f:
        initial_instructions = f.read()

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_file_path}")
    state = load_checkpoint(
        checkpoint_file_path, default_instructions=initial_instructions
    )
    current_instructions = state["current_instructions"]
    iteration = state["iteration"]
    iteration_feedback = state["iteration_feedback"]
    last_generated_snippet = state["last_generated_snippet"]

    print(
        f"Resuming from iteration {iteration}. Press ENTER to continue or type 'stop' to exit:"
    )
    user_input = input().strip().lower()
    if user_input == "stop":
        print("Exiting.")
        sys.exit(0)

    # Iterative refinement loop
    while iteration < args.max_iterations:
        iteration += 1
        print(f"\n=== ITERATION {iteration} / {args.max_iterations} ===")

        # Build prompt and generate snippet
        print("Building prompt and generating TTL snippet...")
        prompt = build_prompt_for_generation(
            instructions=current_instructions,
            common_snippet=common_snippet,
            csv_data=csv_data_str,
            iteration_feedback=iteration_feedback,
        )

        generated_snippet = generate_ttl_snippet(prompt, model=args.model)
        print("\n--- Generated TTL Snippet ---")
        print("(Snippet content hidden for brevity, but it was generated successfully)")

        # Compare with reference snippet
        print("\nComparing generated snippet with reference snippet...")
        comparison_raw = compare_snippets(
            generated_snippet, reference_snippet, model=args.model
        )
        comparison_result = interpret_comparison_result(comparison_raw)

        if "error" in comparison_result:
            print("\nERROR during LLM comparison:", comparison_result["error"])
            print("Raw output:", comparison_result.get("raw_output", "N/A"))
            break

        # Build error summary
        error_summary = build_error_summary(comparison_result)
        print("\n--- Error Summary ---")
        print(error_summary)

        if error_summary == "No errors.":
            print("\nNo missing/extra entities/states. Stopping refinement.")
            last_generated_snippet = generated_snippet

            # Update checkpoint
            state.update(
                {
                    "current_instructions": current_instructions,
                    "iteration": iteration,
                    "iteration_feedback": iteration_feedback,
                    "last_generated_snippet": last_generated_snippet,
                }
            )
            save_checkpoint(checkpoint_file_path, state)

            # Save final snippet
            with output_path.open("w", encoding="utf-8") as f:
                f.write(last_generated_snippet)
            print(f"\nFinal ontology snippet saved to {output_path}")
            break

        # Optional human feedback
        print(
            "\nWould you like to add additional human feedback? (Type it or press ENTER to skip)"
        )
        human_feedback = input(">>> ").strip()
        if human_feedback.lower() == "stop":
            print("Stopping per user request.")
            state.update(
                {
                    "current_instructions": current_instructions,
                    "iteration": iteration,
                    "iteration_feedback": iteration_feedback,
                    "last_generated_snippet": generated_snippet,
                }
            )
            save_checkpoint(checkpoint_file_path, state)

            # Save current snippet
            with output_path.open("w", encoding="utf-8") as f:
                f.write(generated_snippet)
            print(f"\nCurrent ontology snippet saved to {output_path}")
            break
        elif human_feedback:
            error_summary += f"\nUser feedback: {human_feedback}"

        # Refine instructions
        print("\nRefining instructions based on error summary...")
        refined_instructions = refine_instructions(
            error_summary, current_instructions, model=args.model
        )
        print("\n--- Refined Instructions ---")
        print(
            "(Instructions content hidden for brevity, but they were refined successfully)"
        )

        # Update state
        current_instructions = refined_instructions
        iteration_feedback = f"Please address these issues: {error_summary}"
        last_generated_snippet = generated_snippet

        # Save checkpoint
        state.update(
            {
                "current_instructions": current_instructions,
                "iteration": iteration,
                "iteration_feedback": iteration_feedback,
                "last_generated_snippet": last_generated_snippet,
            }
        )
        save_checkpoint(checkpoint_file_path, state)

        # User decision to continue
        print("\nPress ENTER to continue to next iteration or type 'stop' to exit:")
        user_input = input(">>> ").strip().lower()
        if user_input == "stop":
            print("Stopping loop at user's request.")

            # Save current snippet
            with output_path.open("w", encoding="utf-8") as f:
                f.write(last_generated_snippet)
            print(f"\nCurrent ontology snippet saved to {output_path}")
            break

    # Out of iterations
    if iteration >= args.max_iterations:
        print(f"\nReached maximum number of iterations ({args.max_iterations}).")

        # Save final snippet
        with output_path.open("w", encoding="utf-8") as f:
            f.write(last_generated_snippet)
        print(f"\nFinal ontology snippet saved to {output_path}")

    print("\n=== FINAL GENERATED TTL SNIPPET ===")
    print(last_generated_snippet)
    print("===================================")


if __name__ == "__main__":
    main()
