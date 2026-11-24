# YouTube AI Assistant

A Streamlit app that loads the English transcript of a YouTube video and lets you ask questions about it. It uses retrieval-augmented generation (RAG): the transcript is split into chunks, embedded into a local Chroma vector store, and the chunks most relevant to your question are passed to a local LLM, which answers using only that context.

## Features

- Accepts standard `youtube.com/watch?v=`, `youtu.be/`, `/embed/`, `/shorts/`, and `/live/` URLs
- Fetches English captions with `youtube-transcript-api`
- Splits the transcript into 1000-character chunks with 200-character overlap
- Embeds chunks locally with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and stores them in Chroma
- Retrieves the top 4 most similar chunks per question
- Answers with a local Ollama model (`llama3.2`), instructed to say "I don't know based on this video." when the transcript does not contain the answer

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit web app (main entry point) |
| `transcriptRAG.py` | A standalone, script-style version of the same pipeline for testing in the terminal |
| `.gitignore` | Excludes `.env`, virtual environments, and Python cache files |

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running, with the `llama3.2` model pulled:

  ```bash
  ollama pull llama3.2
  ```

- A YouTube video that has English captions enabled

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ShrishtyJaiswal/YTvideostranscript.git
   cd YTvideostranscript
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install streamlit youtube-transcript-api langchain langchain-community langchain-core langchain-text-splitters langchain-openai langchain-huggingface langchain-ollama chromadb sentence-transformers python-dotenv
   ```

   There is no `requirements.txt` yet. The package list above is taken from the imports in the code.

4. (Optional) Create a `.env` file in the project root. The app calls `load_dotenv()`, so any variables set here are loaded at startup. The current code does not require an OpenAI key, because embeddings run locally.

   ```env
   OPENAI_API_KEY=your-key-here
   LANGCHAIN_TRACING_V2=false
   ```

   Never commit `.env`. It is listed in `.gitignore`.

## Usage

Start the app:

```bash
streamlit run app.py
```

Then in your browser:

1. Paste a YouTube URL and click **Load Video Transcript**. The first run downloads the embedding model, which can take a minute.
2. Once the transcript is loaded, type a question and click **Get Answer**.

To run the single-video pipeline in the terminal instead, use:

```bash
python transcriptRAG.py
```

This script uses a hard-coded video ID and a hard-coded question, and prints the result.

## How it works

```
YouTube URL
   │  extract_video_id()
   ▼
Video ID ──► fetch_transcript()  (English captions)
                 │
                 ▼
        RecursiveCharacterTextSplitter (1000 / 200 overlap)
                 │
                 ▼
   HuggingFace embeddings ──► Chroma vector store
                                      │
Question ──► retriever (top 4) ◄──────┘
                 │
                 ▼
      Prompt (context + question) ──► ChatOllama (llama3.2) ──► Answer
```

## Troubleshooting

- **"Captions are disabled for this video."** The video has no captions, so there is no transcript to load.
- **"No English transcript was found for this video."** Captions exist but not in English. The app only requests `en`.
- **Answers fail or the app cannot connect to a model.** Make sure Ollama is running (`ollama serve`) and `llama3.2` has been pulled.
- **Slow first load.** The sentence-transformers model is downloaded the first time it is used and cached afterward.

## Known limitations

- Only English transcripts are supported.
- Only one video is loaded at a time, and the vector store is rebuilt on every load.
- The LLM and embedding model are fixed in the code (`llama3.2` and the MiniLM model).
