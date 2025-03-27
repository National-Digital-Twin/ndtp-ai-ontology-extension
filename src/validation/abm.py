# SPDX-License-Identifier: Apache-2.0
# © Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.

import os
import re
import json
from openai import OpenAI
from typing import Tuple, Optional


def load_personas(config_input) -> list:
    """
    Load persona definitions from a JSON configuration dictionary, a JSON string, or a file path.

    Args:
        config_input: Either a dictionary, a JSON string, or a file path to a JSON file containing persona definitions.

    Returns:
        A list of persona dictionaries, each containing 'name' and 'prompt' keys.
    """
    if isinstance(config_input, dict):
        data = config_input
    elif isinstance(config_input, str):
        # Check if the string is a valid file path
        if os.path.exists(config_input) and os.path.isfile(config_input):
            with open(config_input, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                data = json.loads(config_input)
            except json.JSONDecodeError as e:
                raise ValueError(
                    "Invalid JSON text provided for persona configuration."
                ) from e
    else:
        raise TypeError(
            "config_input must be a dict, a JSON string, or a valid file path"
        )

    return data.get("personas", [])


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to be chunked.
        chunk_size: The size of each chunk.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= (text_length - overlap + 1):
            break
    return chunks


def rank_chunks_for_persona(
    client: OpenAI,
    model: str,
    persona_name: str,
    persona_prompt: str,
    chunks: list,
    top_n: int = 3,
) -> list:
    """
    Ask an LLM to rank each chunk by relevance to a persona and return the top_n most relevant chunks.

    Args:
        client: The OpenAI client.
        model: The model to use.
        persona_name: The name of the persona.
        persona_prompt: The prompt for the persona.
        chunks: The list of chunks to rank.
        top_n: The number of chunks to return.

    Returns:
        A list of the top_n most relevant chunks.
    """
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        relevance_prompt = (
            f"You are analyzing text for persona: '{persona_name}'.\n\n"
            f"Persona's instructions:\n{persona_prompt}\n\n"
            f"Below is a chunk from an ontology:\n\n"
            f"---\n{chunk}\n---\n\n"
            "Rate from 0 (not relevant) to 10 (highly relevant) and provide a brief explanation."
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant deciding relevance.",
                    },
                    {"role": "user", "content": relevance_prompt},
                ],
            )
            content = response.choices[0].message.content
            match = re.search(r"Relevance\s*[:=]\s*(\d+)", content, flags=re.IGNORECASE)
            score = (
                int(match.group(1))
                if match
                else int(
                    re.search(r"\b(\d+)\b", content).group(1)
                    if re.search(r"\b(\d+)\b", content)
                    else 0
                )
            )
            scored_chunks.append((score, chunk, content))
        except Exception as e:
            scored_chunks.append((0, chunk, str(e)))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for (score, chunk, explanation) in scored_chunks[:top_n]]
    return top_chunks


def build_multi_agent_prompt(
    ontology_description: str, personas: list, relevant_texts: dict
) -> list:
    """
    Build a multi-agent prompt including each persona's details and relevant ontology excerpts.

    Args:
        ontology_description: The description of the ontology.
        personas: The list of personas.
        relevant_texts: The dictionary of relevant texts for each persona.

    Returns:
        A list of messages.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are simulating multiple domain experts, each with a specific persona, discussing an ontology. "
                "Each persona message includes relevant excerpts of the ontology. They should point out inconsistencies "
                "or missing elements and finally produce a consolidated response with each persona's perspective."
            ),
        }
    ]
    for persona in personas:
        persona_text = relevant_texts.get(persona["name"], "")
        user_message = (
            f"Persona Name: {persona['name']}\n"
            f"Persona Prompt: {persona['prompt']}\n"
            f"Ontology Excerpts (for {persona['name']}):\n"
            f"---\n{persona_text}\n---\n\n"
            f"User Query / Scenario:\n{ontology_description}\n\n"
            "Please respond from the perspective of your persona."
        )
        # Sanitize the persona name to conform with allowed characters.
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", persona["name"])
        messages.append({"role": "user", "name": safe_name, "content": user_message})
    return messages


def simulate_multi_agent_discussion(
    client: OpenAI,
    model: str,
    ontology_text: str,
    ontology_description: str,
    config_file: str,
    chunk_size: int = 3000,
    overlap: int = 200,
    top_n_relevant: int = 3,
    output_file: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Simulate a multi-agent discussion between domain experts.
      1. Load personas from config.
      2. Chunk the ontology text.
      3. Rank and select top_n relevant chunks for each persona.
      4. Build the multi-agent conversation.
      5. Append an instruction for a JSON-formatted consolidated response.
      6. Call an LLM to simulate the discussion.
      7. Write the JSON result to a file and return the JSON string.

    Args:
        client: The OpenAI client.
        model: The model to use.
        ontology_text: The text of the ontology.
        ontology_description: The description of the ontology.
        config_file: The path to the configuration file.
        chunk_size: The size of each chunk.
        overlap: The number of characters to overlap between chunks.
        top_n_relevant: The number of relevant chunks to return.
        output_file: The path to the output file.

    Returns:
        A tuple containing:
        - A JSON string.
        - An optional error message if an error occurs.
    """
    # 1. Load personas (with error handling)
    try:
        personas = load_personas(config_file)
    except Exception as e:
        return None, f"Error loading personas: {e}"

    if not personas:
        return None, "No personas found in the configuration."

    # 2. Chunk the ontology
    chunks = chunk_text(ontology_text, chunk_size=chunk_size, overlap=overlap)

    # 3. Rank and select relevant chunks for each persona
    relevant_texts = {}
    for persona in personas:
        top_chunks = rank_chunks_for_persona(
            client=client,
            model=model,
            persona_name=persona["name"],
            persona_prompt=persona["prompt"],
            chunks=chunks,
            top_n=top_n_relevant,
        )
        combined = "\n\n".join(top_chunks)
        relevant_texts[persona["name"]] = combined

    # 4. Build the multi-agent prompt
    messages = build_multi_agent_prompt(ontology_description, personas, relevant_texts)

    # 5. Append an instruction to return a consolidated JSON object.
    messages.append(
        {
            "role": "user",
            "content": (
                "Please consolidate the discussion responses from the above agents and return "
                "the responses in a JSON object. Each key should be the agent's name and the value "
                "their response."
            ),
        }
    )

    # 6. Call GPT
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_response_text = response.choices[0].message.content
    except Exception as e:
        final_response_text = f"Error calling GPT: {e}"

    # 7. Parse the response as JSON, write to a file, and return the JSON string.
    try:
        final_json = json.loads(final_response_text)
        json_output = json.dumps(final_json, indent=2)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_output)
        return json_output, None
    except Exception as e:
        return None, f"Error parsing JSON: {e}\nRaw response:\n{final_response_text}"
