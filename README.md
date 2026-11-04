# Medical Chatbot (RAG)

Answers medical questions from your own PDF documents, with source citations.

**Stack:** Python, sentence-transformers (all-MiniLM-L6-v2), Pinecone, OpenRouter, LangChain text splitters, Streamlit

## How it works
1. `ingest.py` reads PDFs, splits them into chunks, embeds them (384-dim) and uploads to Pinecone.
2. `rag.py` embeds a question, retrieves the top-k similar chunks, and asks an LLM to answer only from them.
3. `app.py` is the Streamlit chat UI, showing sources for every answer.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
# put PDFs in data/
python ingest.py
streamlit run app.py
```

## Results
- Documents indexed: 70
- Chunks: 155
- Average response time: 6.48 Seconds
- Test set accuracy (20 questions): 80%
