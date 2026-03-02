"""Agent tools: file operations and RAG-powered code search."""
from pathlib import Path
from typing import Any

from rag import VectorStore, retrieve

CODEBASE_DIR = Path(__file__).parent / "example_codebase"

# Tool schemas for OpenAI function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all Python files in the example codebase.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the example codebase.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Filename, e.g. calculator.py"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file in the example codebase with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Filename, e.g. calculator.py"},
                    "content": {"type": "string", "description": "Full new content of the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the codebase using semantic (RAG) retrieval. Returns relevant code chunks.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Natural language or code query"}},
                "required": ["query"],
            },
        },
    },
]


def list_files() -> dict[str, Any]:
    files = sorted(p.name for p in CODEBASE_DIR.glob("*.py"))
    return {"files": files}


def read_file(path: str) -> dict[str, Any]:
    file_path = CODEBASE_DIR / Path(path).name  # restrict to codebase dir
    if not file_path.exists():
        return {"error": f"File not found: {path}"}
    return {"path": path, "content": file_path.read_text(encoding="utf-8")}


def write_file(path: str, content: str) -> dict[str, Any]:
    file_path = CODEBASE_DIR / Path(path).name
    file_path.write_text(content, encoding="utf-8")
    return {"path": path, "written": True, "lines": len(content.splitlines())}


def search_code(store: VectorStore, query: str) -> dict[str, Any]:
    chunks = retrieve(store, query)
    return {"query": query, "chunks": chunks}


def dispatch_tool(
    name: str,
    args: dict[str, Any],
    store: VectorStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute a tool call and return (result, rag_events).

    rag_events is non-empty only for search_code calls.
    """
    rag_events: list[dict[str, Any]] = []

    if name == "list_files":
        result = list_files()
    elif name == "read_file":
        result = read_file(args["path"])
    elif name == "write_file":
        result = write_file(args["path"], args["content"])
    elif name == "search_code":
        query = args["query"]
        rag_events.append({"type": "rag_query", "query": query})
        result = search_code(store, query)
        rag_events.append({"type": "rag_results", "query": query, "chunks": result["chunks"]})
    else:
        result = {"error": f"Unknown tool: {name}"}

    return result, rag_events
