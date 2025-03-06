# AI Ontology Extension

An AI-assisted ontology development workflow that ingests data from various sources and uses a combination of data profiling, NER, and AI suggestions to extract candidate ontology entities.

## Table of Contents

- [Purpose and Overview](#purpose-and-overview)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
  - [Ingestion and Entity Extraction](#ingestion-and-entity-extraction)
  - [Fuzzy Matching Options](#fuzzy-matching-options)
  - [Ontology Constraints File](#ontology-constraints-file)
  - [Example Usage](#example-usage)
  - [Output Folder](#output-folder)
  - [Next Steps](#next-steps)
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

## Usage Instructions

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
│   ├── transformation/       # Data transformation modules
│   │   ├── __init__.py
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
├── scripts/                  # Executable scripts
│   ├── extract_entities_ttl.py
│   └── extract_entities.py
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
