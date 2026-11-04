"""One-time script: read PDFs in data/, chunk them (across page breaks), embed, upload to Pinecone."""
import bisect
import os
import sys
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from tqdm import tqdm

from rag import DIMENSION, INDEX_NAME, get_embedder

DATA_DIR = Path("data")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH = 100


def load_pdf(path):
    """Join all pages into one text so chunks can span page breaks.
    Also return where each page starts, so every chunk can still be mapped to a page."""
    parts, starts, pos = [], [], 0
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        starts.append((pos, page_no))
        parts.append(text)
        pos += len(text) + 2  # +2 for the "\n\n" separator
    return "\n\n".join(parts), starts


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDFs found. Put at least one .pdf file in the data/ folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks, total_pages = [], 0
    for pdf in pdfs:
        text, starts = load_pdf(pdf)
        total_pages += len(starts)
        offsets = [s[0] for s in starts]
        for i, doc in enumerate(splitter.create_documents([text])):
            start = max(doc.metadata.get("start_index", 0), 0)
            page = starts[max(bisect.bisect_right(offsets, start) - 1, 0)][1]
            chunks.append(
                {
                    "id": f"{pdf.name}-c{i}",
                    "text": doc.page_content,
                    "source": pdf.name,
                    "page": page,
                }
            )
    print(f"{len(pdfs)} PDFs, {total_pages} pages -> {len(chunks)} chunks")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    index = pc.Index(INDEX_NAME)

    embedder = get_embedder()
    for start in tqdm(range(0, len(chunks), BATCH), desc="Uploading"):
        batch = chunks[start : start + BATCH]
        vectors = embedder.encode(
            [c["text"] for c in batch], normalize_embeddings=True
        ).tolist()
        index.upsert(
            vectors=[
                (c["id"], v, {"text": c["text"], "source": c["source"], "page": c["page"]})
                for c, v in zip(batch, vectors)
            ]
        )
    print("Done. Your index is ready.")


if __name__ == "__main__":
    main()