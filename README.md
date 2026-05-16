# Chest X-Ray Intelligence System
**DSAI 413 - Assignment 2: Multi-Modal Medical Imaging Analysis**

## Overview
This repository contains a dual-mode multi-modal AI system for analyzing Chest X-Rays. It features:
1. **Report Generation Mode:** Automatically generates structured medical reports from Chest X-Ray images using Vision-Language Models.
2. **Clinical QA Mode (RAG):** Allows users to ask clinical questions about an image and generates grounded answers by retrieving similar historical cases from a Knowledge Base.

## Technologies Used
- **MedGemma:** Core generative model for report creation and grounded answer synthesis.
- **ColPali:** Advanced vision-language retrieval model for accurately matching X-ray patches.
- **CLIP:** Baseline contrastive model for semantic image-text similarity.
- **Streamlit:** Interactive web interface.

## Setup & Installation

### 1. Environment Setup
We recommend using a virtual environment or conda:
```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
```

### 2. Install Dependencies
Install all required libraries, including `colpali-engine`:
```bash
pip install -r requirements.txt
pip install colpali-engine
```

### 3. HuggingFace Login
Since `MedGemma` is a gated model, you must have access to it via HuggingFace and be authenticated:
```bash
huggingface-cli login
```

### 4. Data Preparation
To populate the local knowledge base with sample data (dummy X-ray images and synthetic QA pairs), run:
```bash
python prepare_data.py
```
This will create a `data/knowledge_base.json` file used by the RAG system.

### 5. Run the Application
Launch the Streamlit app:
```bash
streamlit run assig.py
```

## Application Modes
1. **Report Generation:** Upload a Chest X-Ray and use MedGemma to generate a structured radiologist report.
2. **Clinical QA:** Upload an image, type a question, and see how the RAG pipeline retrieves similar cases via ColPali/CLIP to answer your question.
3. **Model Comparison:** Compare the execution time and similarity scores of CLIP versus ColPali.
4. **Knowledge Base:** Manage the underlying dataset used for retrieval.

## Deliverables
- `assig.py`: Main application code.
- `prepare_data.py`: Script to initialize the QA dataset.
- `report.md`: Short report detailing architecture and model comparisons.
- `requirements.txt`: Python dependencies.
