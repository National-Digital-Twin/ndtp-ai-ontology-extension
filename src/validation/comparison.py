from openai import OpenAI


def compare(client: OpenAI, model: str, syntetic_extension: str, reference: str) -> str:
    """
    Query ChatGPT for suggestions when the ontology fails to load.

    Args:
        client: OpenAI client
        model: OpenAI model
        syntetic_extension: Generated ontology extension
        reference: Reference ontology extension

    Returns:
        Comparison result
    """
    prompt = (
        f"Given the generated ontology extension:\n\n{syntetic_extension}\n\n"
        f"And the reference ontology extension:\n\n{reference}\n\n"
        "Provide a detailed, structured list highlighting the main differences between the two extensions. "
        "For each difference, clearly indicate which extension performs better and explain why."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Ontology Extensions.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying ChatGPT: {e}"
