# Vector Database for Documents

This project is a **local semantic search engine** built with FastAPI.
It extracts text from uploaded documents, generates embeddings using [SentenceTransformers](https://www.sbert.net/), and provides a REST API for search and management.



## Features

* Upload documents and generate embeddings automatically
* Semantic search across all stored documents or within a single one
* Manage documents: list, delete individually, or reset the entire database
* Persistent storage (embeddings and metadata saved to disk)


## Installation

Clone the repository and set up your environment:

```bash
git clone git@github.com:Ottitsch/vector.git
cd vector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Make sure you have Python 3.9+ installed.

## Usage

### 1. Start the Server

```bash
uvicorn vector_db_server:app --reload --port 8000
```

The server will load any previously saved embeddings and listen on `http://127.0.0.1:8000`.


### 2. Run the CLI Client

In another terminal:

```bash
python client.py
```


## API Endpoints

* `GET /list_pdfs` – list all stored documents
* `POST /upload_pdf` – upload a new document
* `DELETE /delete_pdf/{name}` – delete a specific document and its embeddings
* `POST /reset` – clear all stored data
* `GET /search?query=...&pdf=...&top_k=3` – semantic search (optionally limit to one document)


## Project Layout

```
.
├── client.py                # CLI for interacting with the server
├── vector_db_server.py      # FastAPI application
├── requirements.txt         # Python dependencies
├── pdf/                     # Directory with example pdfs you can store
│   ├── attention.pdf
│   ├── frogs.pdf
│   ├── windows.pdf
├── pdf_store/               # Stores embeddings and metadata
```


## How It Works

1. **Upload** – text is extracted from the document using PyMuPDF.
2. **Embed** – each page is converted into a 384‑dimensional vector using `all-MiniLM-L6-v2`.
3. **Store** – vectors and metadata are saved as `.npy` and `.pkl` in the `pdf_store` directory.
4. **Search** – queries are embedded, and cosine similarity retrieves the best‑matching pages.


## Example Queries and Results

Below are some real searches run against the database, shown exactly as they appeared in the console:

```
Enter your search query: what is a cool frog?
Search in one PDF (enter name) or press Enter for all:

Search Results:
- PDF: frogs | Page: 9 | Score: 0.5501
  Text: Green Frog (Rana clamitans melanota)
Wood Frog (Rana sylvatica)
Southern Leopard Frog (Rana utricularia)
Pickerel Frog (Rana palustris)...
```
---
```
Enter your search query: what does PATH do?
Search in one PDF (enter name) or press Enter for all:

Search Results:
- PDF: windows | Page: 4 | Score: 0.2425
  Text: o
   OPENFILES Query or display open files
p
   PATH     Display or set a search path for executable
files•
   PATHPING Trace route plus network latency and packet
loss
   PAUSE    Suspend processing of a batch file and display
a message•
   PERMS    Show permissions for a user
   PERFMON ...
```
---
```
Enter your search query: what is a transformer?
Search in one PDF (enter name) or press Enter for all:

Search Results:
- PDF: attention | Page: 3 | Score: 0.3813
  Text: Figure 1: The Transformer - model architecture.
wise fully connected feed-forward network. We employ a residual connection [10] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the functi...

- PDF: attention | Page: 8 | Score: 0.2679
  Text: Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
Model
BLEU
Training Cost (FLOPs)
EN-DE
EN-FR
EN-DE
EN-FR
ByteNet [15]
23.75
Deep-Att + PosUnk [32]
39.2
1.0...

- PDF: attention | Page: 9 | Score: 0.2495
  Text: Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base
model. All metrics are on the English-to-German translation development set, newstest2013. Listed
perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to...
```
