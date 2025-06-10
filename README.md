text
# LLM-Powered Fact Checker with Custom Embedding-Based Retrieval

A lightweight system to analyze short news posts or social media statements, extract key claims, and verify them against a vector database of verified facts using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Claim/Entity Extraction:** Uses spaCy for NLP-based extraction.
- **Trusted Fact Base:** Loads and embeds facts from a trusted source (e.g., PIB RSS feed).
- **Embedding & Retrieval:** Uses Sentence Transformers and a vector store (ChromaDB recommended for cloud).
- **LLM-Powered Comparison:** Uses OpenAI GPT-4 for classification and reasoning.
- **Interactive UI:** Built with Streamlit for easy user interaction.

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Aakanksha-Sai/LLM-Powered-Fact-Checker-with-Custom-Embedding-Based-Retrieval.git
cd llm-powered-fact-checker

text

### 2. Install Dependencies

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

text

**Example `requirements.txt`:**
streamlit
sentence-transformers
pandas
requests
chromadb
openai
spacy
python-dotenv

text

Download the spaCy model:

python -m spacy download en_core_web_sm

text

### 3. Set Up OpenAI API Key

Create a `.env` file in the project root:

OPENAI_API_KEY=your_api_key_here

text

Or set it as an environment variable:

export OPENAI_API_KEY=your_api_key_here

text

### 4. Run the App

streamlit run streamlit_app.py

text

## Usage

1. **Open the app** in your browser (usually at `http://localhost:8501`).
2. **Enter a short news claim or statement** (see sample below).
3. **Click "Check Fact"** to see the verdict, evidence, and reasoning.

## Sample Input/Output

### Sample Input

The Indian government has announced free electricity to all farmers starting July 2025.

text

### Sample Output

{
"verdict": "Unverifiable",
"evidence": [],
"reasoning": "No semantically similar facts were retrieved to verify the claim."
}

text

*(If evidence is found, the output will include the relevant facts and a different verdict.)*

## Sample Files

- **Input:** [`sample_input.txt`](sample_input.txt)
The Indian government has announced free electricity to all farmers starting July 2025.

text
- **Output:** [`fact_check_result.json`](fact_check_result.json)
{
"verdict": "Unverifiable",
"evidence": [],
"reasoning": "No semantically similar facts were retrieved to verify the claim."
}

text

## Troubleshooting

- **If you get a `ModuleNotFoundError` for `faiss`, switch to ChromaDB as described above.**
- **Make sure your OpenAI API key is set correctly.**
- **Ensure the spaCy model is downloaded.**

