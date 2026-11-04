"""Core RAG logic: embed the question, search Pinecone, ask the LLM."""
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

SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer the question using ONLY the "
    "numbered context passages. Do not add facts that are not in them. "
    "Answer in 2-5 short sentences, or a short list if the question asks for a list, "
    "and cite passages like [1]. If the passages contain the answer, just give it and "
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


def retrieve(question, k=8):
    vec = get_embedder().encode(question, normalize_embeddings=True).tolist()
    res = get_index().query(vector=vec, top_k=k, include_metadata=True)
    return [
        {
            "text": m.metadata["text"],
            "source": m.metadata["source"],
            "page": int(m.metadata["page"]),
            "score": m.score,
        }
        for m in res.matches
    ]


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


def answer(question, k=8, min_score=0.30):
    hits = [h for h in retrieve(question, k) if h["score"] >= min_score]
    if not hits:
        return "I couldn't find this in the documents.", []

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