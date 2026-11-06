# 🩺 Medical Chatbot (RAG)

A retrieval-augmented chatbot that answers health questions **only from a set of PDF guides**, shows the source passages for every answer, and replies "I couldn't find this in the documents" when the answer isn't there.

**Live demo:** https://medical-chatbot2610.streamlit.app

> The demo runs on free hosting. If nobody has visited for a while the app goes to sleep, so the first load can take about a minute, and the first question also has to load the embedding model.

**Stack:** Python, sentence-transformers (all-MiniLM-L6-v2), Pinecone, LangChain text splitters, any OpenAI-compatible LLM API (OpenRouter, Groq), Streamlit

## How it works
1. `ingest.py` reads the PDFs in `data/`, splits them into overlapping chunks (chunks can span page breaks, and each chunk keeps its page number), embeds them (384-dim) and uploads them to Pinecone.
2. `rag.py` embeds the question, retrieves the top-k most similar chunks (cosine similarity), drops weak matches, and asks an LLM to answer using only those chunks.
3. `app.py` is the Streamlit chat UI. It shows the answer and an expandable list of the source passages with file name, page and similarity score.
4. `eval.py` runs a 20-question test set and reports accuracy and average response time.

## Features
- Answers grounded in your documents, with citations like [1], [2] that match the sources panel.
- Refuses out-of-scope questions instead of guessing (tested with 3 off-topic questions).
- Retry and model fallback when the free LLM API is rate-limited. Several models can be listed in `LLM_MODEL`, separated by commas.
- Works with any OpenAI-compatible provider by changing three environment variables.

## Run locally
Python 3.11 or 3.12 is recommended.

```bash
python -m venv venv
venv\Scripts\activate          # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
copy .env.example .env         # Windows (use: cp .env.example .env on Mac/Linux)
# add your keys to .env, then put your PDFs in data/
python ingest.py
streamlit run app.py
```

`.env` settings:
```
PINECONE_API_KEY=your-pinecone-key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your-llm-key
LLM_MODEL=model-name-from-your-provider
```
Never commit `.env`. It is already listed in `.gitignore`.

## Deploy on Streamlit Community Cloud
1. Push the repo to GitHub (without `.env` and without `data/`).
2. On share.streamlit.io, create a new app from the repo, branch `main`, main file `app.py`.
3. Open **Manage app > Settings > Secrets** and paste your keys in TOML format:
   ```toml
   PINECONE_API_KEY = "your-pinecone-key"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_API_KEY = "your-llm-key"
   LLM_MODEL = "model-name-from-your-provider"
   ```
4. Save and reboot the app. Top-level secrets are available to the app as environment variables, which is what `rag.py` reads.
5. Run `ingest.py` once from your own computer. The Pinecone index lives in the cloud, so the deployed app does not need to ingest anything.

Notes for a public deployment:
- Visitors share your LLM API quota. If you see a "service is busy" reply, the free-tier rate limit was hit. Do not put a paid key with no spending limit in a public app.
- The deploy installs large ML libraries, so the first build takes a few minutes.

## Evaluation
`eval.py` asks 20 questions written from the source PDFs: 17 that the documents can answer and 3 that they cannot (cricket, a drug dose, diabetes treatment). An answer passes if it includes enough expected keywords and doesn't contradict itself. Off-topic questions pass only if the bot refuses. Results are saved to `eval_results.csv`.

| Version | Accuracy | What changed |
|---|---|---|
| First run | 75% (15/20) | Baseline prompt, 800-character chunks |
| Current | 80% (16/20) | Cleaner prompt with a `NOT_FOUND` signal, larger chunks, higher top-k, weak-match filter |

The scoring is keyword-based, so it is approximate. Some correct answers worded differently can fail, and I review failures manually.

## Results
- Documents indexed: 3 PDFs (70 pages)
- Chunks: 155
- Average response time: 6.5 seconds (mostly the free LLM API)
- Test set accuracy (20 questions): 80%

## Known limitations
- Answers come only from the three indexed PDFs, so it does not know anything else.
- The test set is small (20 questions) and written by the author, so treat the accuracy as a rough guide.
- List questions that span several passages are the hardest case for the retriever.
- Free LLM tiers have rate limits, so responses can be slow or occasionally busy.

## Data and disclaimer
The chatbot is built on guides published by [Hesperian Health Guides](https://hesperian.org). The PDFs are not included in this repository; download them from Hesperian and place them in `data/`.

This project is a technical demo. It is **not medical advice** and must not replace a doctor or health worker.
