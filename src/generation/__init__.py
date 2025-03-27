# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

"""
Generation Module for Ontology Development

This module provides functionality to generate and refine ontology snippets using LLMs.
It implements an iterative workflow that compares generated snippets with reference
snippets and refines the generation instructions based on the comparison results.

Key components:
- llm_interface: Functions for interacting with LLM to generate, refine, and compare ontology snippets
- iteration: Functions for managing the iterative workflow
- generator: Functions for generating ontology snippets
"""
