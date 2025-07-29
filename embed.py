import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import fitz  # PyMuPDF
import csv
from sentence_transformers import SentenceTransformer

doc = fitz.open("attention.pdf")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = []
for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():  # Skip empty pages
        vector = model.encode(text)
        embeddings.append((i + 1, vector))  # (page number, vector)

print(f"Extracted {len(embeddings)} embeddings (one per page).")

with open("embed.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["page"] + [f"dim_{i}" for i in range(len(embeddings[0][1]))]
    writer.writerow(header)

    for page_num, vector in embeddings:
        writer.writerow([page_num] + vector.tolist())

print("Embeddings saved to embed.csv")

