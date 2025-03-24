import streamlit as st
import pandas as pd
from typing import Dict, Optional
import re
import json

from app.utils.logging import log
from app.state.app_state import AppState
from src.ingestion.extract import (
    analyze_step,
    analyze_tri,
    extract_concepts_step,
    gather_usage_step,
    classify_extensions,
)
from src.generation.generator import ontology_generator
from app.components.cache import get_from_cache, save_to_cache


class ProcessingHandler:
    @staticmethod
    def analyze_csv(df: pd.DataFrame, model: str) -> Dict:
        """Run CSV analysis step."""
        try:
            analysis_result = get_from_cache("analysis_result")
            if analysis_result is None:
                log("No cache found, analyzing CSV...")
                analysis_result = analyze_step(df, model)
                save_to_cache("analysis_result", analysis_result)

            AppState.get().processing_results["analysis"] = analysis_result
            log("CSV analysis completed successfully")
            return analysis_result
        except Exception as e:
            log(f"Error in CSV analysis: {e}", "ERROR")
            st.error(f"Analysis failed: {e}")
            return {}

    @staticmethod
    def extract_triplets(df: pd.DataFrame, model: str) -> Dict:
        """Extract triplets from CSV data."""
        try:
            tri_result = get_from_cache("tri_result")
            if tri_result is None:
                log("No cache found, extracting triplets...")
                tri_result = analyze_tri(df, model)
                save_to_cache("tri_result", tri_result)

            AppState.get().processing_results["triplets"] = tri_result
            log("Triplet extraction completed successfully")
            return tri_result
        except Exception as e:
            log(f"Error in triplet extraction: {e}", "ERROR")
            st.error(f"Triplet extraction failed: {e}")
            return {}

    @staticmethod
    def extract_concepts(df: pd.DataFrame, model: str) -> Dict:
        """Extract concepts from CSV data."""
        try:
            concepts_result = get_from_cache("concepts_result")
            if concepts_result is None:
                log("No cache found, extracting concepts...")
                concepts_result = extract_concepts_step(df, model)
                save_to_cache("concepts_result", concepts_result)

            AppState.get().processing_results["concepts"] = concepts_result
            log("Concept extraction completed successfully")
            return concepts_result
        except Exception as e:
            log(f"Error in concept extraction: {e}", "ERROR")
            st.error(f"Concept extraction failed: {e}")
            return {}

    @staticmethod
    def gather_usage_patterns(model: str) -> Optional[Dict]:
        """Gather usage patterns from analyzed data."""
        state = AppState.get()
        if (
            "analysis" not in state.processing_results
            or "concepts" not in state.processing_results
        ):
            st.warning("Run analysis and concept extraction first.")
            return None

        try:
            raw_str = state.processing_results["concepts"]
            concepts = re.findall(r'"([^"]+)"', raw_str)

            usage_result = get_from_cache("usage_result")
            if usage_result is None:
                log("No cache found, gathering usage patterns...")
                usage_result = gather_usage_step(
                    concepts, state.processing_results["analysis"], model
                )
                save_to_cache("usage_result", usage_result)

            state.processing_results["usage"] = usage_result
            log("Usage patterns gathered successfully")
            return usage_result
        except Exception as e:
            log(f"Error gathering usage patterns: {e}", "ERROR")
            st.error(f"Usage pattern gathering failed: {e}")
            return None

    @staticmethod
    def classify_extensions(model: str) -> Optional[str]:
        """Classify extensions based on usage patterns."""
        state = AppState.get()
        if "usage" not in state.processing_results:
            st.warning("Run 'Gather Usage Patterns' first.")
            return None

        try:
            classification_result = get_from_cache("classification_result")
            if classification_result is None:
                log("No cache found, classifying extensions...")
                classification_result = classify_extensions(
                    usage_info=state.processing_results["usage"], model=model
                )
                save_to_cache("classification_result", classification_result)
            classification_result = [
                json.loads(item) for item in classification_result
            ]

            state.processing_results["classification"] = classification_result
            log("Extensions classified successfully")
            return classification_result
        except Exception as e:
            log(f"Error classifying extensions: {e}", "ERROR")
            st.error(f"Extension classification failed: {e}")
            return None

    @staticmethod
    def generate_ontology(
        df: pd.DataFrame, model: str, background: str, prompt: str, guidelines: str
    ) -> Optional[str]:
        """Generate ontology based on processed data."""
        state = AppState.get()
        try:
            result = get_from_cache("ontology_result")
            if result is None:
                log("No cache found, generating ontology...")
                result = ontology_generator(
                    df=df,
                    model=model,
                    existing_analysis=state.processing_results.get("analysis", ""),
                    existing_triplets=state.processing_results.get("triplets", ""),
                    extracted_concepts=state.processing_results.get("concepts", ""),
                    usage=state.processing_results.get("usage", ""),
                    classified_extensions=state.processing_results.get(
                        "classification", ""
                    ),
                    base_ontology=state.reference_snippet or "",
                    extra_context=background,
                    prompt=prompt,
                    prompt2=guidelines,
                    chunk_start=state.chunk_start,
                    chunk_size=state.chunk_size,
                )
                save_to_cache("ontology_result", result)

            state.new_output = result
            log("Ontology generation completed successfully")
            return result
        except Exception as e:
            log(f"Error generating ontology: {e}", "ERROR")
            st.error(f"Ontology generation failed: {e}")
            return None
