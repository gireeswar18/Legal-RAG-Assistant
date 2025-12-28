# Legal RAG Assistant 📚⚖️

A Retrieval-Augmented Generation (RAG) based legal assistant that answers questions **strictly from a provided legal document**.

## Features
- PDF-based document ingestion
- Semantic search using FAISS
- Grounded answers using Mistral (via Ollama)
- Streamlit frontend
- Avoids hallucination by design

## Tech Stack
- Python
- FAISS
- SentenceTransformers
- Mistral (Ollama)
- Streamlit

## How it works
1. Legal PDF → text extraction
2. Text chunking
3. Embedding + FAISS indexing
4. Retrieve relevant chunks
5. Generate grounded answer


## Run Locally

This project uses Mistral via Ollama for local LLM inference.

### 1. Install Ollama

Download and install Ollama from:

https://ollama.com

### 2. Pull the Mistral model

After installing Ollama, run:

```
ollama pull mistral
```

### 3. Start Ollama (if not running)
```
ollama serve
```

### 4. Using Your Own Document

This repository does not include legal documents or generated indexes.

To use the project:

1. Place your Legal data PDF file inside the `data/` folder `data/your_document.pdf` (make sure to update the file name in your code)

2. Extract text from the PDF
```
python rag/extract_pdf.py
```

3. Build the FAISS index

```
python rag/build_index.py
```

### 5. Start the app

```bash
pip install -r requirements.txt
streamlit run app.py
```