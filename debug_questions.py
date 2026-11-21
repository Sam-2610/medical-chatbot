"""Find out WHY a question fails: is the answer missing from the retrieved text (retrieval problem),
or present there but missing from the reply (generation / scoring problem)?

Run from your project folder:  python debug_questions.py
"""
import rag

CASES = [
    ("What three strategies for organizing does the book describe?", ["consensus", "take power", "make power"]),
    ("What things should anyone helping you with health care provide?",
     ["access", "information", "choice", "safety", "respect", "privacy", "comfort", "follow"]),
]


def present(keywords, text):
    text = text.lower()
    return [k for k in keywords if k in text]


for question, keywords in CASES:
    print("=" * 78)
    print("QUESTION:", question)
    hits = [h for h in rag.retrieve(question, 6) if h["score"] >= 0.30]
    print(f"\nTop chunks (score >= 0.30): {len(hits)}")
    for h in hits:
        print(f"  {h['source']} p{h['page']}  score {h['score']:.2f}  | {h['text'][:70]!r}")

    in_hits = present(keywords, " ".join(h["text"] for h in hits))
    print(f"\nKeywords found in the retrieved chunks: {len(in_hits)}/{len(keywords)}  {in_hits}")

    if hasattr(rag, "expand_with_neighbors") and hits:
        wider = rag.expand_with_neighbors(hits)
        in_wider = present(keywords, " ".join(h["text"] for h in wider))
        print(f"After adding neighbouring chunks:        {len(in_wider)}/{len(keywords)}  ({len(wider)} chunks)")

    reply, _ = rag.answer(question)
    in_reply = present(keywords, reply.replace(",", ""))
    print(f"\nKeywords found in the ANSWER: {len(in_reply)}/{len(keywords)}  {in_reply}")
    print("ANSWER:", reply)
    print()