# AI Ontology Extension

An AI-assisted ontology development workflow that ingests data from various sources and uses a combination of data profiling, NER, and AI suggestions to extract candidate ontology entities.

## Table of Contents

- [Purpose and Overview](#purpose-and-overview)
- [Installation and Setup](#installation-and-setup)
- [Ontology Extraction and Matching](#ontology-extraction-and-matching)
  - [Ingestion and Entity Extraction](#ingestion-and-entity-extraction)
  - [Fuzzy Matching Options](#fuzzy-matching-options)
  - [Ontology Constraints File](#ontology-constraints-file)
  - [Example Usage](#example-usage)
  - [Output Folder](#output-folder)
  - [Next Steps](#next-steps)
- [Ontology Generation](#ontology-generation)
  - [LLM-Based Generation](#llm-based-generation)
  - [Iterative Refinement](#iterative-refinement)
  - [Generate Ontology Usage](#generate-ontology-usage)
- [Streamlit Application for Generation](#streamlit-application-for-generation)
  - [Features](#features)
  - [Running the App](#running-the-app)
- [Analysis and Visualization](#analysis-and-visualization)
  - [Embedding Analysis](#embedding-analysis)
  - [Namespace Analysis](#namespace-analysis)
- [Folder Structure](#folder-structure)
- [Security](#security)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Purpose and Overview

This project provides an AI-assisted ontology development workflow. It ingests data from various sources 
(e.g., CSV, JSON, RDF/Turtle) and uses a combination of data profiling, spaCy-based Named Entity 
Recognition (NER), and ChatGPT suggestions to extract candidate ontology entities. These candidate 
entities can then be matched against an ontology export using fuzzy matching techniques.

## Installation and Setup

1. **Clone the Repository**  
   Clone the repository to your local machine.

2. **Set Up a Virtual Environment** (recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy Model**  
   **Note:** If you haven't installed spaCy or downloaded its English model, run:
   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

5. **Install rapidfuzz for Fuzzy Matching**  
   This project uses rapidfuzz for fuzzy string comparison. Install it via:
   ```bash
   pip install rapidfuzz
   ```

6. **Set the OpenAI API Key**  
   The extraction process uses the OpenAI API (for ChatGPT queries and fuzzy matching via ChatGPT).  
   Set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```
   (On Windows, use `set OPENAI_API_KEY=your_api_key`)

## Ontology Extraction and Matching

### Ingestion and Entity Extraction

The ingestion process consists of two main parts:

- **Tabular Data Extraction (`extract.py`):**  
  This module processes CSV/JSON files to extract candidate ontology entities from textual columns.
  - For each column, up to five sample values are collected.
  - Entities are extracted using spaCy NER and/or ChatGPT (if the API key is set).
  - The function supports fuzzy matching against an ontology constraints file.
  - Fuzzy matching can be performed either using rapidfuzz or by querying ChatGPT.

- **Ontology Ingestion (`ontology.py`):**  
  This module ingests RDF/Turtle ontology files and extracts candidate ontology entities (classes 
  and properties) as human-readable labels (using rdfs:label, or local names as fallback).

### Fuzzy Matching Options

When extracting candidate entities from tabular data, you can compare them against an ontology 
export (in JSON format) using fuzzy matching. You have two options:

- **Rapidfuzz:** Uses the rapidfuzz library to compute token-sort similarity scores.  
  (This is the default method.)
- **ChatGPT:** Sends a prompt to ChatGPT to compare candidate entities with known ontology terms and return similar matches.

You can select the fuzzy matching method by setting the `fuzzy_method` parameter in the extraction function.

### Ontology Constraints File

If you have a JSON export of your ontology entities (with an "entities" field containing "classes" 
and "properties"), you can provide its path to the extraction function. The function will then perform fuzzy matching
between the extracted candidate entities and the ontology constraints.

Example ontology constraints JSON:

```json
{
  "ontology_file": "data/ontologies/ies-building1.ttl",
  "entities": {
    "classes": [
      "Accredited Energy Assessor",
      "Addressable Location",
      "... other class labels ..."
    ],
    "properties": [
      "Estimated Energy Cost",
      "Lodgement Date",
      "... other property labels ..."
    ]
  }
}
```

### Example Usage

Import and use the extraction functions in your application:

```python
from src.ingestion.extract import process_data

# Extract candidate entities from a CSV file using both spaCy and ChatGPT,
# and perform fuzzy matching against an ontology constraints file using ChatGPT for fuzzy matching.
results = process_data(
    "data/raw/address_base_plus_john_2023-10-06_122302.csv",
    output_path="results/candidate_entities.json",
    method="both",
    ontology_constraints_path="data/ontologies/ontology_export.json",
    fuzzy_threshold=80,
    fuzzy_method="chatgpt"  # or "rapidfuzz"
)
print(results)
```

### Output Folder

It is recommended to store output files in the `results` folder. The extraction functions will 
automatically create this folder if it does not exist.

### Next Steps

Once candidate entities are extracted and matched against your ontology constraints, you can use them to:

- Define or extend your ontology.
- Build mapping configurations to convert raw data into RDF (using transformation scripts).
- Validate the transformed data using validation scripts (e.g., SHACL constraints or SPARQL queries).

## Ontology Generation

The project includes a generation module that uses Large Language Models (LLMs) to generate and refine ontology snippets in Turtle format.

### LLM-Based Generation

The `src/generation` module provides functionality to:

- Generate ontology snippets from instructions, common templates, and sample data
- Compare generated snippets with reference snippets
- Refine generation instructions based on comparison results

Key components include:

- **llm_interface.py**: Functions for interacting with LLMs, including:
  - `generate_ttl_snippet()`: Generates Turtle snippets based on prompts
  - `compare_snippets()`: Compares generated snippets with reference snippets
  - `refine_instructions()`: Refines instructions based on error feedback

### Iterative Refinement

The generation module implements an iterative workflow that:

1. Generates an ontology snippet based on instructions
2. Compares it with a reference snippet
3. Identifies missing or extra entities and states
4. Refines the instructions to address the issues
5. Repeats the process until convergence

The `iteration.py` module provides checkpoint management for resuming iterations:

- `load_checkpoint()`: Loads the most recent iteration state
- `save_checkpoint()`: Saves the current iteration state
- `get_iteration_history()`: Retrieves the complete history of iterations

### Generate Ontology Usage

The project includes a script for generating ontology snippets using LLMs with an iterative refinement approach:

```bash
python -m scripts.generate_ontology \
    --common-snippet data/generation/common_template.ttl \
    --csv-file data/generation/data_building_123003.csv \
    --reference-snippet data/generation/building_ontology.ttl \
    --instructions data/generation/initial_prompt.txt
```

Key parameters:

- `--common-snippet`: Common ontology template
- `--csv-file`: Sample data file
- `--reference-snippet`: Reference ontology snippet
- `--instructions`: Initial instructions file
- `--model`: OpenAI model (default: o3-mini)
- `--output`: Output path (default: results/final_ontology.ttl)

The script iteratively generates and refines ontology snippets by comparing with a reference, stopping when no differences are found or after reaching maximum iterations.

## Streamlit Application for Generation

The project includes a Streamlit web application that provides a user-friendly interface for ontology generation.

### Features

- **Setup Page**: Upload base (common) ontology, CSV data, and (optionally) areference ontology
- **Generation Page**: Iteratively generate and refine ontology with AI assistance
- **Results Page**: View, edit, and export the final ontology with visualisation of the generation process

### Running the App

1. **Install Streamlit**  
   If you haven't installed Streamlit, run:
   ```bash
   pip install streamlit
   ```

2. **Set up OpenAI API Key**  
   The app requires an OpenAI API key, which can be provided in two ways:
   - As an environment variable: `export OPENAI_API_KEY="your_api_key"`
   - In a Streamlit secrets file: Create `.streamlit/secrets.toml` with `OPENAI_API_KEY = "your_api_key"`

3. **Launch the App**  
   ```bash
   streamlit run app/main.py
   ```

4. **Using the App**  
   - Start on the Setup page to configure inputs
   - Proceed to the Generation page to create and refine ontologies
   - View and export results on the Results page

The Streamlit app is currently a work in progress.

## Analysis and Visualization

### Embedding Analysis

The project includes tools for analysing ontology data using embeddings:

- `scripts/analyse_ontology.py`: Analyses similarities between entities across different namespaces
- Visualises relationships using t-SNE projections for:
  - Namespace similarities
  - Column mappings
  - Value mappings

### Namespace Analysis

The analysis module can:

- Extract entities from TTL files and create string representations
- Group entities by namespace
- Find similar entities within and across namespaces
- Generate interactive visualizations to explore relationships

For more details, see the [embedding analysis documentation](docs/embedding_analysis/embedding_analysis.md).

## Folder Structure

The project is organized as follows:

```
ndtp-ai-ontology-extension/
├── .github/                  # GitHub specific files
|   ├── CONTRIBUTING.md
|   └── PULL_REQUEST_TEMPLATE
├── src/                      # Source code directory
│   ├── __init__.py
│   ├── ingestion/            # Ingestion and extraction modules
│   │   ├── __init__.py
│   │   ├── helpers.py        # Functions to read data (CSV, JSON, RDF)
│   │   ├── extract.py        # Extracts candidate ontology entities from tabular data
│   │   └── ontology.py       # Ingests RDF/Turtle ontology files
│   ├── generation/           # Ontology generation modules
│   │   ├── __init__.py
│   │   ├── llm_interface.py  # Functions for interacting with LLMs
│   │   └── iteration.py      # Functions for managing iterative workflow
│   ├── analysis/             # Analysis and visualization modules
│   │   ├── __init__.py
│   │   ├── representations.py # Entity representation functions
│   │   ├── namespaces.py     # Namespace analysis functions
│   │   ├── columns.py        # Column mapping functions
│   │   └── values.py         # Value mapping functions
│   ├── validation/           # Validation modules
│   │   └── __init__.py
├── data/                     # Data directory (no __init__.py needed)
│   ├── raw/                  # Raw data files (e.g., CSV, JSON)
│   ├── processed/            # Data after transformation
│   └── ontologies/           # Ontology definitions and exports
├── tests/                    # Test scripts
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_transformation.py
│   └── test_validation.py
├── results/                  # Output files
├── logs/                     # Log files
├── docs/                     # Documentation
│   └── embedding_analysis/   # Documentation for embedding analysis
├── scripts/                  # Executable scripts
│   ├── extract_entities_ttl.py
│   ├── extract_entities.py
│   └── analyse_ontology.py
├── .gitignore
├── requirements.txt
├── LICENSE.md
└── README.md
```

## Security

If you've found a vulnerability, we would like to know so we can fix it. For full details on how to tell us about vulnerabilities, see our security policy.

## License

Unless stated otherwise, the codebase is released under the MIT License. See [LICENSE](LICENSE.md) for more information.

## Contributing

Contributions and feedback are welcome. Feel free to extend the ingestion helpers to support additional 
data formats or integrate other AI models for improved extraction and matching.

Please see the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for more information on our contribution guidelines.

## Acknowledgements

The development of these works has been made possible with thanks to our [contributors](https://github.com/National-Digital-Twin/ndtp-ai-ontology-extension/graphs/contributors).
