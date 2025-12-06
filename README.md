SiratGPT is a Flask-based Retrieval-Augmented Generation (RAG) chatbot that answers questions using Quran content.
This project uses:

Pinecone (vector DB) — stores embeddings and text chunks

Local embedding model (sentence-transformers / MiniLM) for indexing & queries

Gemini Flash (via langchain-google-genai) for final LLM responses

Flask + HTML frontend

Deployed on HuggingFace Spaces (Docker) — free + supports local models

Important: PDFs are used only for indexing locally.
You do NOT upload your PDFs to the deployed app.

Repository structure
siratgpt/
├─ app.py                 # Flask server (production)
├─ build_index.py         # Run locally to create & upload vectors to Pinecone
├─ requirements.txt       # Dependencies for HF Spaces and local dev
├─ Dockerfile             # Required for HuggingFace Docker Space
├─ README.md
├─ templates/
│   └─ index.html         # Your frontend UI
├─ static/
│   └─ ...                # CSS / JS / Images
└─ pdfs/                  # Local PDFs (NOT uploaded to HuggingFace)

Quick Start (Local Development)

Use these steps to index PDFs and test your app locally.

1. Create & activate a Python virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

2. Install required packages
pip install -r requirements.txt


For local indexing (ONLY needed locally):

pip install sentence-transformers torch


NOTE: requirements.txt already contains all packages used on HuggingFace Spaces.
Locally, your system may need PyTorch installed separately.

3. Create a .env file (DO NOT COMMIT IT)

Create a .env file in the project root:

PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=sirat-index
GOOGLE_API_KEY=your_google_api_key
PORT=7860

4. Add your PDF files into the pdfs/ folder

This folder is ONLY used locally for building the index.

5. Build the index (upload vectors to Pinecone)
python build_index.py


This script:

Loads PDFs

Splits into text chunks

Generates embeddings using all-MiniLM-L6-v2

Uploads embeddings to Pinecone

You only need to run this again when PDFs change.

6. Run the app locally
python app.py


Visit the app at:

👉 http://127.0.0.1:7860