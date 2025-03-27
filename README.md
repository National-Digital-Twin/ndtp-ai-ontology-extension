# AI Ontology Extension Generator

**Repository:** `ndtp-ai-ontology-extension`  
**Description:** An AI-assisted ontology development workflow that enables automated generation and extension of ontologies through an intuitive web interface.  
**SPDX-License-Identifier:** `Apache-2.0 AND OGL-UK-3.0`  

## Overview  
This repository provides an open-source, AI-powered tool for ontology development and extension. It enables the automatic generation and extension of ontologies from various data sources using a human-in-the-loop workflow.

The tool combines data profiling, Named Entity Recognition (NER), and Large Language Models to extract, match, and generate ontology entities through a user-friendly Streamlit interface.

## Prerequisites  
Before using this repository, ensure you have the following dependencies installed:  
- **Required Tooling:** Python 3.8+
- **Pipeline Requirements:** OpenAI API access
- **System Requirements:** 4GB RAM minimum, 8GB recommended

## Quick Start  
Follow these steps to get started with the ontology generator:

### 1. Download and Install Dependencies  
```sh  
git clone https://github.com/National-Digital-Twin/ndtp-ai-ontology-extension.git
cd ndtp-ai-ontology-extension
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```sh
export OPENAI_API_KEY="your_api_key"  # On Windows: set OPENAI_API_KEY=your_api_key
python -m spacy download en_core_web_sm
```

Add OPENAI_API_KEY also into `.streamlit/secrets.toml` as
```toml
OPENAI_API_KEY = "your_api_key"
```

### 3. Launch Application
```sh
streamlit run app/main.py
```

### 4. Uninstallation  
```sh
deactivate  # Exit virtual environment
rm -rf ndtp-ai-ontology-extension  # Remove repository
```

## Features  
- **AI-Powered Ontology Generation:** Automated extraction and generation of ontology entities using LLMs
- **Interactive Web Interface:** Step-by-step wizard for ontology development
- **Multi-Format Support:** Processes CSV, JSON, and RDF/Turtle data sources
- **Validation & Analysis:** Built-in tools for ontology validation and visualization
- **Iterative Refinement:** AI-assisted improvement of generated ontologies

## Usage Guide
The Streamlit application provides a four-step wizard interface:

1. **Data Input**
   - Upload source data (CSV, JSON, RDF)
   - Preview and validate input
   - Configure data processing settings

2. **Data Processing**
   - Extract entities using NER and AI
   - Perform fuzzy matching
   - Review extracted candidates

3. **Ontology Generation**
   - Configure AI generation parameters
   - Generate ontology snippets
   - Iteratively refine results

4. **Validation**
   - Review generated ontology
   - Compare with references
   - Export final results

## Public Funding Acknowledgment  
This repository has been developed with public funding as part of the National Digital Twin Programme (NDTP), a UK Government initiative. NDTP, alongside its partners, has invested in this work to advance open, secure, and reusable digital twin technologies for any organisation, whether from the public or private sector, irrespective of size.  

## License  
This repository contains both source code and documentation, which are covered by different licenses:  
- **Code:** Licensed under the Apache License 2.0.  
- **Documentation:** Licensed under the Open Government Licence v3.0.  

See `LICENSE.md`, `OGL_LICENCE.md`, and `NOTICE.md` for details.  

## Security and Responsible Disclosure  
We take security seriously. If you believe you have found a security vulnerability in this repository, please follow our responsible disclosure process outlined in `SECURITY.md`.  

## Contributing  
We welcome contributions that align with the Programme's objectives. Please read our `CONTRIBUTING.md` guidelines before submitting pull requests.  

## Acknowledgements  
This repository has benefited from collaboration with various organisations. For a list of acknowledgments, see `ACKNOWLEDGEMENTS.md`.  

## Support and Contact  
For questions or support, check our Issues or contact the NDTP team on ndtp@businessandtrade.gov.uk.

**Maintained by the National Digital Twin Programme (NDTP).**  

© Crown Copyright 2025. This work has been developed by the National Digital Twin Programme and is legally attributed to the Department for Business and Trade (UK) as the governing entity.