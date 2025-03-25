import streamlit as st
from typing import Tuple, List, Optional

from src.validation.abm import simulate_multi_agent_discussion
from src.validation.comparison import compare
from src.validation.validator import OntologyValidator

from app.utils.logging import log
from app.state import AppState
from app.components.cache import get_cache


class EvaluationHandler:
    @staticmethod
    def validate_ontology(
        ontology_code: str, model: str
    ) -> Tuple[Optional[str], List[str]]:
        """Validate the generated ontology."""
        try:
            validation_result = get_cache().get_from_cache("validation_result")
            if validation_result is None:
                log("No cache found, running validation...")
                client = AppState.get().client

                validator = OntologyValidator(
                    client=client, model=model, max_chatgpt_fixes=5
                )
                clean_ontology, error_log = validator.validate(
                    input_ttl_path=ontology_code
                )

                validation_result = {
                    "clean_ontology": clean_ontology,
                    "error_log": error_log,
                }
                get_cache().save_to_cache("validation_result", validation_result)
            else:
                clean_ontology = validation_result["clean_ontology"]
                error_log = validation_result["error_log"]

            log("Ontology validation completed")
            return clean_ontology, error_log
        except Exception as e:
            log(f"Error validating ontology: {e}", "ERROR")
            st.error(f"Validation failed: {e}")
            return None, [str(e)]

    @staticmethod
    def run_abm_simulation(
        model: str, clean_ontology: str, scene: str, personas: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Run ABM simulation with the validated ontology extension."""
        try:
            client = AppState.get().client
            discussion_result_with_error_message = get_cache().get_from_cache(
                "discussion_result_with_error_message"
            )
            if discussion_result_with_error_message is None:
                log("No cache found, running discussion...")
                discussion_result, error_message = simulate_multi_agent_discussion(
                    client=client,
                    model=model,
                    ontology_text=clean_ontology,
                    ontology_description=scene,
                    config_file=personas,
                    chunk_size=2000,
                    overlap=200,
                    top_n_relevant=2,
                )
                discussion_result_with_error_message = {
                    "result": discussion_result,
                    "error_message": error_message,
                }
                get_cache().save_to_cache(
                    "discussion_result_with_error_message",
                    discussion_result_with_error_message,
                )

            discussion_result = discussion_result_with_error_message["result"]
            error_message = discussion_result_with_error_message["error_message"]

            log("ABM simulation completed successfully")
            return discussion_result, error_message
        except Exception as e:
            log(f"Error in ABM simulation: {e}", "ERROR")
            st.error(f"ABM simulation failed: {e}")
            return None, str(e)

    @staticmethod
    def compare_with_reference(
        model: str, synthetic_extension: str, reference: str
    ) -> Optional[str]:
        """Compare generated ontology with reference."""
        try:
            client = AppState.get().client
            comparison_result = get_cache().get_from_cache("comparison_result")
            if comparison_result is None:
                log("No cache found, running comparison...")
                comparison_result = compare(
                    client=client,
                    model=model,
                    syntetic_extension=synthetic_extension,
                    reference=reference,
                )
                get_cache().save_to_cache("comparison_result", comparison_result)

            log("Ontology comparison completed")
            return comparison_result
        except Exception as e:
            log(f"Error comparing ontologies: {e}", "ERROR")
            st.error(f"Comparison failed: {e}")
            return None
