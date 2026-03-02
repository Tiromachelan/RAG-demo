"""FastAPI application: WebSocket chat endpoint + REST file endpoints."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

import rag as rag_module
from agent import run_agent
from tools import CODEBASE_DIR

# Global ChromaDB collection, initialised at startup
_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _collection
    print("Indexing example codebase into ChromaDB…")
    _collection = rag_module.build_index()
    print(f"Index built: {_collection.count()} chunks")
    yield


app = FastAPI(title="RAG Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------- REST: file browser ----------

@app.get("/files")
async def list_files_endpoint():
    files = sorted(p.name for p in CODEBASE_DIR.glob("*.py"))
    return JSONResponse({"files": files})


@app.get("/files/{filename}")
async def get_file_endpoint(filename: str):
    file_path = CODEBASE_DIR / Path(filename).name
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return JSONResponse({"path": filename, "content": file_path.read_text(encoding="utf-8")})


# ---------- WebSocket: agent chat ----------

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            async def send_event(event: dict):
                await websocket.send_text(json.dumps(event))

            try:
                await run_agent(user_message, _collection, send_event)
            except Exception as exc:
                await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass
