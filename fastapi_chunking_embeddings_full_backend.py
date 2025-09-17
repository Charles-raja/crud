# embedding-app/backend
# Full backend implementation (FastAPI) with multiple chunking strategies,
# summarization-based embedding, and multi-modal parent-child linking.
# ---------------------------------------------------------------------
# Folder structure (single-file view below contains all files you need):
#
# embedding-app/
# ├── backend/
# │   ├── main.py
# │   ├── requirements.txt
# │   ├── services/
# │   │   ├── file_loader.py
# │   │   ├── chunking.py
# │   │   └── embedding.py
# │   └── data/   (uploaded files temporary storage)
# └── frontend/   (see notes below for Next.js snippets)
#
# IMPORTANT: This code is ready-to-run but assumes you will install the
# python packages from requirements.txt and set environment variables
# for any API-based options (OPENAI_API_KEY if you use OpenAI for summarization
# and/or embeddings).
# ---------------------------------------------------------------------

# -------------------------------
# backend/requirements.txt
# -------------------------------
#
# fastapi
# uvicorn[standard]
# python-multipart
# chromadb
# langchain
# sentence-transformers
# pandas
# python-magic
# python-docx
# PyMuPDF
# tiktoken
# openai
# aiofiles
# pydantic
#
# Install with:
# pip install -r requirements.txt

# -------------------------------
# backend/services/file_loader.py
# -------------------------------

"""Utilities to read different file types and return plain text or DataFrame."""
import os
import pandas as pd
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import magic
from typing import Tuple, Union


def read_file(path: str) -> Tuple[str, str]:
    """
    Read a file and return (file_type, content).
    file_type: 'csv', 'pdf', 'txt', 'docx', 'unknown'
    content: for csv returns a pandas.DataFrame, otherwise a text string.
    """
    mime = magic.from_file(path, mime=True)
    ext = os.path.splitext(path)[1].lower().lstrip('.')

    if ext == 'csv' or 'csv' in mime:
        df = pd.read_csv(path)
        return 'csv', df

    if ext in ('txt',) or 'text' in mime:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return 'txt', f.read()

    if ext in ('pdf',) or 'pdf' in mime:
        text = []
        with fitz.open(path) as pdf:
            for page in pdf:
                text.append(page.get_text())
        return 'pdf', "\n".join(text)

    if ext in ('docx',) or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in mime:
        doc = DocxDocument(path)
        return 'docx', "\n".join([p.text for p in doc.paragraphs])

    return 'unknown', ''


# -------------------------------
# backend/services/chunking.py
# -------------------------------

"""
Provides several chunking strategies:
- fixed_size_chunking: pure character/token length based
- recursive_chunking: LangChain's RecursiveCharacterTextSplitter
- semantic_chunking: tries to split by semantic boundaries (headings, paragraphs)
- csv_row_based_chunking: treat each CSV row as a document
- csv_column_based_chunking: embed only specified column
- multimodal_table_chunks: create table-specific summaries and link to parent id

All functions return a list of chunks (strings) or list of (chunk, metadata) pairs
for richer metadata use (parent_id, chunk_index, chunk_type).
"""
from typing import List, Tuple, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import re


def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split by fixed character size with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks


def recursive_chunking(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Use LangChain's recursive splitter to preserve semantic boundaries when possible."""
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def semantic_chunking(text: str, max_chunk_words: int = 200) -> List[str]:
    """
    A heuristic semantic splitter:
    1) Try splitting on headings (lines with ALL CAPS or starting with 'Chapter'/'Section' or long title lines)
    2) Then split paragraphs (\n\n)
    3) If still too long, apply recursive splitter
    This is a heuristic approach that works decently for reports and articles without an LLM.
    """
    if not text:
        return []

    # Normalize newlines
    t = re.sub(r"\r\n", "\n", text)

    # Step 1: split on headings-like patterns
    pattern = re.compile(r"(?m)^\s*(?:Chapter|SECTION|Section|CHAPTER|PART|[A-Z\s]{4,}):?\s*$")
    # If headings exist, split there
    if pattern.search(t):
        parts = pattern.split(t)
    else:
        # split by double-newline paragraphs
        parts = t.split("\n\n")

    # Clean parts and merge small paragraphs until they reach max_chunk_words
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        cleaned.append(p)

    # Merge small parts to reach max_chunk_words
    chunks = []
    current = ""
    for p in cleaned:
        cur_words = len(current.split())
        p_words = len(p.split())
        if cur_words + p_words <= max_chunk_words or not current:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)

    # If any chunk is still too large, fallback to recursive chunking for that chunk
    final = []
    for c in chunks:
        if len(c.split()) > max_chunk_words * 3:  # arbitrary threshold
            final.extend(recursive_chunking(c))
        else:
            final.append(c)

    return final


def csv_row_based_chunking(df) -> List[Tuple[str, Dict[str, Any]]]:
    """Treat each CSV row as a document. Returns list of (text, metadata)."""
    docs = []
    for idx, row in df.iterrows():
        # Combine non-null values into a single text
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        metadata = {"row_index": int(idx)}
        # if there's an 'id' column use it as parent id
        if 'id' in df.columns:
            metadata['parent_id'] = str(row['id'])
        docs.append((row_text, metadata))
    return docs


def csv_column_based_chunking(df, column_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Take text from one column and return items with metadata linking to row index."""
    docs = []
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not in CSV")
    for idx, cell in df[column_name].astype(str).items():
        docs.append((cell, {"row_index": int(idx)}))
    return docs


def multimodal_table_chunks(df, text_columns: List[str] = None, summary_fn=None) -> List[Tuple[str, Dict[str, Any]]]:
    """Create table-based text (e.g., for each row produce a short textual summary)
    summary_fn: optional function(row) -> str to produce a compact summary for the row.
    Returns (text, metadata) pairs where metadata contains parent_id (row id) if available.
    """
    import pandas as pd
    docs = []
    text_columns = text_columns or [c for c in df.columns if df[c].dtype == object][:2]  # pick first text cols
    for idx, row in df.iterrows():
        if summary_fn:
            txt = summary_fn(row)
        else:
            # simple summary heuristic: join important text columns
            parts = [f"{col}: {row[col]}" for col in text_columns if pd.notnull(row[col])]
            txt = "; ".join(parts)
        metadata = {"row_index": int(idx)}
        if 'id' in df.columns:
            metadata['parent_id'] = str(row['id'])
        docs.append((txt, metadata))
    return docs


# -------------------------------
# backend/services/embedding.py
# -------------------------------

"""
Embedding and storage layer. Uses ChromaDB for vector storage and supports
both local sentence-transformers embeddings and OpenAI embeddings.
Also implements summarization-based embedding using OpenAI's chat completion
(if OPENAI_API_KEY is set).

Design choices:
- Each stored vector has metadata including: parent_id (optional), chunk_type, chunk_index
- When multimodal/parent-child is used, parent documents are stored as metadata only
  and vectors are child chunks that include parent_id for retrieval.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid

# Optional OpenAI usage
import openai

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Choose storage/persistence location for Chroma
PERSIST_DIR = os.environ.get('CHROMA_PERSIST_DIR', 'chroma_persist')

# Initialize Chroma client with local persistence
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))

# Create or get collection
COLLECTION_NAME = os.environ.get('CHROMA_COLLECTION', 'documents')
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(COLLECTION_NAME)

# Local model fallback
_local_model = None

def _ensure_local_model():
    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _local_model


def _embed_with_local_model(texts: List[str]) -> List[List[float]]:
    model = _ensure_local_model()
    embs = model.encode(texts, show_progress_bar=False)
    return [e.tolist() if isinstance(e, (np.ndarray,)) else list(e) for e in embs]


def _embed_with_openai(texts: List[str], model_name: str = 'text-embedding-3-small') -> List[List[float]]:
    # Uses OpenAI embeddings API (batches texts)
    # Note: requires OPENAI_API_KEY in env
    resp = openai.Embedding.create(input=texts, model=model_name)
    return [r['embedding'] for r in resp['data']]


def summarize_texts_with_openai(texts: List[str], prompt_prefix: str = "Summarize the following text in 2-3 sentences:") -> List[str]:
    """Use OpenAI ChatCompletion to summarize each text chunk. Returns list of summaries."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set for summarization")
    summaries = []
    for t in texts:
        prompt = f"{prompt_prefix}\n\nText:\n{t}\n\nSummary:" 
        # Use the chat completions API (gpt-3.5-turbo or gpt-4 if available)
        resp = openai.ChatCompletion.create(
            model=os.environ.get('OPENAI_SUMMARY_MODEL', 'gpt-3.5-turbo'),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        out = resp['choices'][0]['message']['content'].strip()
        summaries.append(out)
    return summaries


def embed_and_store(
    items: List[Tuple[str, Dict[str, Any]]],
    embed_backend: str = 'local',
    use_summary: bool = False,
    summary_backend: str = 'openai'
) -> Dict[str, Any]:
    """
    items: list of (text, metadata)
    embed_backend: 'local' or 'openai'
    use_summary: if True, first summarize each text then embed summaries
    Returns: details about stored vectors
    """

    texts = [t for t, m in items]
    metadatas = [m for t, m in items]

    # Optionally summarize first
    if use_summary:
        if summary_backend == 'openai':
            summaries = summarize_texts_with_openai(texts)
            texts_to_embed = summaries
            # track that these are summaries for metadata
            for md in metadatas:
                md.setdefault('chunk_type', 'summary')
        else:
            # fallback: use first 300 chars as "summary"
            texts_to_embed = [t[:300] for t in texts]
            for md in metadatas:
                md.setdefault('chunk_type', 'truncated_summary')
    else:
        texts_to_embed = texts
        for md in metadatas:
            md.setdefault('chunk_type', 'text')

    # choose embedding backend
    if embed_backend == 'openai':
        if not OPENAI_API_KEY:
            raise RuntimeError('OPENAI_API_KEY not set for openai embedding')
        embeddings = _embed_with_openai(texts_to_embed)
    else:
        embeddings = _embed_with_local_model(texts_to_embed)

    # Generate IDs and upsert into Chroma
    ids = [str(uuid.uuid4()) for _ in texts_to_embed]
    # enrich metadata with parent/child info if not present
    for i, md in enumerate(metadatas):
        md.setdefault('parent_id', md.get('parent_id', md.get('row_index', None)))
        md.setdefault('chunk_index', i)
        md.setdefault('stored_at', 'chroma')

    # Add to collection (use add or upsert depending on version)
    collection.add(
        ids=ids,
        documents=texts_to_embed,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # Persist the client
    try:
        chroma_client.persist()
    except Exception:
        pass

    return {"stored": len(ids), "sample_id": ids[0] if ids else None}


def similarity_search(query: str, top_k: int = 5, embed_backend: str = 'local'):
    """Perform similarity search over collection and return documents + metadata."""
    if embed_backend == 'openai' and OPENAI_API_KEY:
        q_emb = _embed_with_openai([query])[0]
    else:
        q_emb = _embed_with_local_model([query])[0]

    results = collection.query(embedding=[q_emb], n_results=top_k, include=['documents', 'metadatas', 'distances'])
    return results


# -------------------------------
# backend/main.py
# -------------------------------

"""
FastAPI app that accepts file uploads, chunking selection, embedding options,
and provides a search endpoint.
"""
import os
import shutil
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from services.file_loader import read_file
from services.chunking import (
    fixed_size_chunking, recursive_chunking, semantic_chunking,
    csv_row_based_chunking, csv_column_based_chunking, multimodal_table_chunks
)
from services.embedding import embed_and_store, similarity_search

app = FastAPI(title='Embedding Backend')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', 'data')
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post('/upload/')
async def upload_file(
    file: UploadFile,
    chunking_type: str = Form(...),
    column_name: Optional[str] = Form(None),
    embed_backend: str = Form('local'),  # 'local' or 'openai'
    use_summary: bool = Form(False)
):
    """
    Upload a file and process it based on chunking_type.
    chunking_type options:
      - 'fixed'
      - 'recursive'
      - 'semantic'
      - 'csv_row'
      - 'csv_column'
      - 'csv_multimodal'
    embed_backend: 'local' uses sentence-transformers, 'openai' uses OpenAI embeddings
    use_summary: if true, summarization-based embedding will be used (requires OPENAI_API_KEY)
    """
    # Save upload temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, 'wb') as out:
        content = await file.read()
        out.write(content)

    file_type, content = read_file(file_path)

    items = []  # list of (text, metadata)

    try:
        if file_type == 'csv':
            df = content
            if chunking_type == 'csv_row':
                items = csv_row_based_chunking(df)
            elif chunking_type == 'csv_column':
                if not column_name:
                    raise HTTPException(status_code=400, detail='column_name required for csv_column')
                items = csv_column_based_chunking(df, column_name)
            elif chunking_type == 'csv_multimodal':
                # create multimodal table chunks; you can pass a summary function if you want
                items = multimodal_table_chunks(df)
            else:
                raise HTTPException(status_code=400, detail='invalid chunking_type for csv')
        else:
            # text-based files
            text = content
            if chunking_type == 'fixed':
                chunks = fixed_size_chunking(text)
                items = [(c, {}) for c in chunks]
            elif chunking_type == 'recursive':
                chunks = recursive_chunking(text)
                items = [(c, {}) for c in chunks]
            elif chunking_type == 'semantic':
                chunks = semantic_chunking(text)
                items = [(c, {}) for c in chunks]
            else:
                raise HTTPException(status_code=400, detail='invalid chunking_type for non-csv')

        # embed and store
        result = embed_and_store(items, embed_backend=embed_backend, use_summary=use_summary)

        return JSONResponse({"status": "ok", "details": result})
    finally:
        # remove temp file
        try:
            os.remove(file_path)
        except Exception:
            pass


@app.get('/search/')
async def search(q: str, top_k: int = 5, embed_backend: str = 'local'):
    """Search the vector store."""
    results = similarity_search(q, top_k=int(top_k), embed_backend=embed_backend)
    return JSONResponse(results)


# -------------------------------
# Notes for Frontend (Next.js)
# -------------------------------
#
# - Use a file input and a select/dropdown for chunking_type, a checkbox for use_summary,
#   and a text input for column_name when "csv_column" is selected.
# - POST to http://localhost:8000/upload/ with FormData containing:
#     - file (UploadFile)
#     - chunking_type (string)
#     - column_name (optional)
#     - embed_backend (local|openai)
#     - use_summary (true|false)
# - Example JS fetch code snippet (frontend):
#
# const form = new FormData();
# form.append('file', fileInput.files[0]);
# form.append('chunking_type', 'semantic');
# form.append('embed_backend', 'local');
# form.append('use_summary', 'false');
# fetch('http://localhost:8000/upload/', { method: 'POST', body: form })
#
# - For searching:
#   fetch(`http://localhost:8000/search?q=${encodeURIComponent(query)}&top_k=5`)
#
# -------------------------------
# Run the backend:
# -------------------------------
# 1) Create a Python virtual env and install packages from requirements.txt
# 2) Set optional env vars:
#      export OPENAI_API_KEY="sk-..."
#      export CHROMA_PERSIST_DIR="./chroma_persist"
# 3) Start server:
#      uvicorn main:app --reload --port 8000
#
# -------------------------------
# Security & Production notes:
# -------------------------------
# - In production, persist uploaded files to object storage (S3) not local disk.
# - Limit upload file size and validate file types.
# - Use authentication to protect the upload and search endpoints.
# - Chroma persistence directory should be on a disk volume that is backed up.
# - For heavy load use hosted vector DBs (Pinecone, Milvus, Chroma Cloud) and a
#   production-ready embedding service (OpenAI, Cohere, or self-hosted transformers)
#
# ---------------------------------------------------------------------
# End of code document
# ---------------------------------------------------------------------
