# RAG Coding Agent Demo

A visual demonstration of **Retrieval-Augmented Generation (RAG)** applied to a coding agent.

The agent uses **GPT-4o-mini** (ChatGPT/GPT-5 mini) to read, search, and edit a small bundled Python codebase. Every RAG retrieval step and tool call is visualised in real time in the browser.

---

## What You'll See

| Panel | Description |
|-------|-------------|
| 💬 **Chat** | Converse with the coding agent. Animated RAG cards appear inline showing query → retrieved chunks → injected context. |
| 📄 **File Editor** | Syntax-highlighted view of any file in the codebase. Flashes green when the agent writes changes. |
| 📋 **Request Log** | Collapsible cards for every tool call (`list_files`, `read_file`, `write_file`, `search_code`) and every RAG retrieval. |

---

## Project Structure

```
RAG-demo/
├── backend/
│   ├── main.py               # FastAPI app (WebSocket + REST)
│   ├── agent.py              # GPT-4o-mini function-calling loop
│   ├── rag.py                # ChromaDB indexing & retrieval
│   ├── tools.py              # Agent tools
│   └── example_codebase/     # Bundled Python mini-project
│       ├── calculator.py
│       ├── utils.py
│       └── tests.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt
└── .env                      # You create this — see Setup
```

---

## Setup

### 1. Clone / open the project

```bash
cd RAG-demo
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### 4. Run the server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the UI

Visit **http://localhost:8000** in your browser.

---

## Example Prompts

- *"List the files in the codebase"*
- *"What does the Calculator class do?"*
- *"Add a `square_root` method to the Calculator class"*
- *"Search for how factorial is implemented"*
- *"Add a `median` function to utils.py"*
- *"Fix the tests to cover the new methods you added"*

---

## How RAG Works Here

1. **Index** — At startup, each Python file is chunked by function/class and embedded with `text-embedding-3-small` into a local ChromaDB collection.
2. **Retrieve** — When the agent calls `search_code(query)`, your query is embedded and the most semantically similar chunks are returned with cosine similarity scores.
3. **Augment** — Retrieved chunks are passed back to the LLM as tool results, giving it grounded context before it generates a response or makes edits.
4. **Visualise** — Each retrieval step is rendered as an animated card in the Chat panel showing the query, matched chunks (file, line range, similarity score), and the injected context.
