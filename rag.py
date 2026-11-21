"""Core RAG logic: embed the question, search Pinecone, add neighbouring chunks, ask the LLM."""
import logging
import os
import time

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "medical-chatbot"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384  # output size of all-MiniLM-L6-v2

NEIGHBORS = 1     # also read this many chunks before and after every hit
MAX_CHUNKS = 14   # upper limit of chunks sent to the LLM

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer the question using ONLY the "
    "numbered context passages. Do not add facts that are not in them. "
    "Answer in 2-5 short sentences, or a short list if the question asks for a list, "
    "and cite passages like [1]. If the question asks for a list or for several things, "
    "include every matching item that appears in the passages and use the passages' own terms. "
    "If the passages contain the answer, just give it and "
    "never say you couldn't find it. Only if none of the passages contain the answer, "
    "reply with exactly: NOT_FOUND"
)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(INDEX_NAME)


def retrieve(question, k=6):
    """Top-k most similar chunks, best first."""
    vec = get_embedder().encode(question, normalize_embeddings=True).tolist()
    res = get_index().query(vector=vec, top_k=k, include_metadata=True)
    hits = []
    for match in res.matches:
        metadata = match.metadata or {}
        text = metadata.get("text")
        source = metadata.get("source")
        page = metadata.get("page")
        if text is None or source is None or page is None:
            logger.warning("Skipping Pinecone match %s with missing metadata", match.id)
            continue
        hits.append({
            "id": match.id,
            "text": text,
            "source": source,
            "page": int(page),
            "score": match.score,
            "context": False,
        })
    return hits


def _chunk_key(chunk_id):
    """'file.pdf-c17' -> ('file.pdf', 17), so neighbouring chunks can be found."""
    base, _, num = chunk_id.rpartition("-c")
    return (base, int(num)) if num.isdigit() else (chunk_id, 0)


def expand_with_neighbors(hits, neighbors=NEIGHBORS, max_chunks=MAX_CHUNKS):
    """Long lists are often split across chunks. Add the chunks next to each hit, then
    put everything back in document order so the LLM reads it as the document does."""
    have = {h["id"] for h in hits}
    wanted = []  # (neighbour id, hit it belongs to), best hits first
    for h in hits:
        base, n = _chunk_key(h["id"])
        for d in range(1, neighbors + 1):
            for i in (n - d, n + d):
                cid = f"{base}-c{i}"
                if i >= 0 and cid not in have and all(cid != w[0] for w in wanted):
                    wanted.append((cid, h))
    wanted = wanted[: max(0, max_chunks - len(hits))]

    extra = []
    if wanted:
        found = get_index().fetch(ids=[w[0] for w in wanted]).vectors
        for cid, parent in wanted:
            v = found.get(cid)
            if v is not None:
                metadata = v.metadata or {}
                text = metadata.get("text")
                source = metadata.get("source")
                page = metadata.get("page")
                if text is None or source is None or page is None:
                    logger.warning("Skipping Pinecone vector %s with missing metadata", cid)
                    continue
                extra.append({
                    "id": cid,
                    "text": text,
                    "source": source,
                    "page": int(page),
                    "score": parent["score"],  # a neighbour inherits the score of the hit it extends
                    "context": True,
                })
    return sorted(hits + extra, key=lambda h: _chunk_key(h["id"]))


def ask_llm(messages):
    """Works with any OpenAI-compatible provider (OpenRouter, Groq, ...).
    Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env (several models can be
    separated by commas; they are tried in order)."""
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    model_list = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or default_model
    models = [m.strip() for m in model_list.split(",")]
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("LLM_API_KEY") or os.environ["OPENROUTER_API_KEY"],
        max_retries=0,  # we do our own retries below
    )
    for model in models:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.1
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                time.sleep(5 * (attempt + 1))  # wait 5s, 10s, 15s, then give up
            except APIStatusError:
                break  # model unavailable, try the next one
    return None


def answer(question, k=6, min_score=0.30):
    hits = [h for h in retrieve(question, k) if h["score"] >= min_score]
    if not hits:
        return "I couldn't find this in the documents.", []
    hits = expand_with_neighbors(hits)

    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    reply = ask_llm(messages)
    if not reply:
        return "The AI service is busy right now. Please try again in a minute.", hits
    if "NOT_FOUND" in reply:
        return "I couldn't find this in the documents.", []
    return reply.strip(), hits