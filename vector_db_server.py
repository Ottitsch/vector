import numpy as np
import os
import pickle
from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
import fitz
import shutil

app = FastAPI()

STORE_DIR = "pdf_store"
os.makedirs(STORE_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# in-memory DB: {pdf_name: {"vectors": np.ndarray, "texts": list}}
pdf_db = {}

def build_embeddings_from_pdf(pdf_path, pdf_name):
    """Extract text from a PDF, generate embeddings, save to disk, and update memory."""
    doc = fitz.open(pdf_path)
    texts = []
    vectors = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            texts.append({"page": i + 1, "text": text})
            vectors.append(model.encode(text))

    vectors_np = np.array(vectors)

    # Save embeddings and metadata to disk
    np.save(os.path.join(STORE_DIR, f"{pdf_name}.npy"), vectors_np)
    with open(os.path.join(STORE_DIR, f"{pdf_name}.pkl"), "wb") as f:
        pickle.dump(texts, f)

    # Update memory
    pdf_db[pdf_name] = {"vectors": vectors_np, "texts": texts}
    print(f"Built and saved {len(texts)} embeddings for {pdf_name}.")

def load_all_pdfs():
    """Load all previously saved PDFs into memory on startup."""
    for file in os.listdir(STORE_DIR):
        if file.endswith(".npy"):
            pdf_name = file.replace(".npy", "")
            vectors_np = np.load(os.path.join(STORE_DIR, file))
            with open(os.path.join(STORE_DIR, f"{pdf_name}.pkl"), "rb") as f:
                texts = pickle.load(f)
            pdf_db[pdf_name] = {"vectors": vectors_np, "texts": texts}
    print(f"Loaded {len(pdf_db)} PDFs into memory.")

load_all_pdfs()

@app.get("/search")
def search(query: str, top_k: int = 3, pdf: str = None):
    """Search across all PDFs or just one if specified."""
    if not pdf_db:
        return {"query": query, "results": [], "message": "No PDFs loaded. Upload some first."}

    query_vector = model.encode(query)
    results = []

    for pdf_name, data in pdf_db.items():
        if pdf and pdf_name != pdf:
            continue

        vectors_np = data["vectors"]
        sims = (vectors_np @ query_vector) / (
            np.linalg.norm(vectors_np, axis=1) * np.linalg.norm(query_vector)
        )

        top_idx = np.argsort(-sims)[:top_k]
        for i in top_idx:
            results.append({
                "pdf": pdf_name,
                "page": data["texts"][i]["page"],
                "score": float(sims[i]),
                "text": data["texts"][i]["text"][:300] + "..."
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return {"query": query, "results": results}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a new PDF and build embeddings."""
    pdf_name = os.path.splitext(file.filename)[0]

    saved_pdf_path = os.path.join(STORE_DIR, file.filename)
    with open(saved_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    build_embeddings_from_pdf(saved_pdf_path, pdf_name)

    return {"status": "success", "message": f"Uploaded and indexed {file.filename}"}

@app.get("/list_pdfs")
def list_pdfs():
    """List all currently indexed PDFs."""
    return {"pdfs": list(pdf_db.keys())}

@app.post("/reset")
async def reset_db():
    """Clear all PDFs and embeddings from memory and disk."""
    global pdf_db
    pdf_db = {}

    for file in os.listdir(STORE_DIR):
        os.remove(os.path.join(STORE_DIR, file))

    return {"status": "success", "message": "All PDFs and embeddings have been cleared."}

@app.delete("/delete_pdf/{pdf_name}")
async def delete_pdf(pdf_name: str):
    """Delete one specific PDF and its embeddings."""
    global pdf_db

    # Remove from memory if present
    if pdf_name in pdf_db:
        del pdf_db[pdf_name]

    # Remove files from disk if they exist
    npy_file = os.path.join(STORE_DIR, f"{pdf_name}.npy")
    pkl_file = os.path.join(STORE_DIR, f"{pdf_name}.pkl")
    pdf_file = os.path.join(STORE_DIR, f"{pdf_name}.pdf")

    for file in [npy_file, pkl_file, pdf_file]:
        if os.path.exists(file):
            os.remove(file)

    return {"status": "success", "message": f"{pdf_name} deleted from memory and disk."}

