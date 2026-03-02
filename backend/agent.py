"""Coding agent: drives the LLM loop and streams events over a WebSocket.

Uses the OpenAI chat completions API directly via httpx — no openai package needed.
"""
import json
import os
from typing import Any, AsyncIterator

import httpx

from rag import VectorStore
from tools import TOOL_SCHEMAS, dispatch_tool

MODEL = "gpt-5-mini"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You are a helpful coding assistant. You have access to a small Python codebase.
When the user asks you to read, edit, or improve code, use the available tools:
- list_files: see what files exist
- read_file: read a file's current content
- search_code: semantically search the codebase (uses RAG retrieval)
- write_file: overwrite a file with new content

Always search or read relevant files before making edits. Explain your changes clearly."""


# ── OpenAI helper streaming ─────────────────────────────────────

async def _stream_chunks(messages: list, tools: list) -> AsyncIterator[dict]:
    """Yield parsed SSE delta objects from the OpenAI streaming chat API."""
    api_key = os.environ["OPENAI_API_KEY"]
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            OPENAI_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages,
                  "tools": tools, "tool_choice": "auto", "stream": True},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                yield json.loads(payload)


# ── Agent loop ─────────────────────────────────────────────────────────────

async def run_agent(
    user_message: str,
    store: VectorStore,
    send_event,  # async callable: (event_dict) -> None
) -> None:
    """Run the agent loop for a single user message, streaming events via send_event."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}
        finish_reason = "stop"

        async for chunk in _stream_chunks(messages, TOOL_SCHEMAS):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason") or finish_reason

            # Stream text tokens
            token = delta.get("content")
            if token:
                accumulated_content += token
                await send_event({"type": "llm_token", "token": token})

            # Accumulate tool call deltas
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.get("id"):
                    accumulated_tool_calls[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    accumulated_tool_calls[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    accumulated_tool_calls[idx]["arguments"] += fn["arguments"]

        # Build assistant message
        if accumulated_tool_calls:
            tool_calls_list = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in accumulated_tool_calls.values()
            ]
            assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls_list}
            if accumulated_content:
                assistant_msg["content"] = accumulated_content
            messages.append(assistant_msg)
        else:
            messages.append({"role": "assistant", "content": accumulated_content})

        # Done if no tool calls
        if not accumulated_tool_calls:
            await send_event({"type": "llm_done", "content": accumulated_content})
            break

        # Execute tool calls
        for tc in accumulated_tool_calls.values():
            tool_name = tc["name"]
            try:
                tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                tool_args = {}

            await send_event({"type": "tool_call", "tool": tool_name,
                               "args": tool_args, "call_id": tc["id"]})

            result, rag_events = dispatch_tool(tool_name, tool_args, store)

            for ev in rag_events:
                await send_event(ev)

            await send_event({"type": "tool_result", "tool": tool_name,
                               "result": result, "call_id": tc["id"]})

            messages.append({"role": "tool", "tool_call_id": tc["id"],
                              "content": json.dumps(result)})
