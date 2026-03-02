"""Coding agent: drives the LLM loop and streams events over a WebSocket."""
import json
from typing import Any, AsyncGenerator

import chromadb
from openai import AsyncOpenAI

from tools import TOOL_SCHEMAS, dispatch_tool

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful coding assistant. You have access to a small Python codebase.
When the user asks you to read, edit, or improve code, use the available tools:
- list_files: see what files exist
- read_file: read a file's current content
- search_code: semantically search the codebase (uses RAG retrieval)
- write_file: overwrite a file with new content

Always search or read relevant files before making edits. Explain your changes clearly."""


async def run_agent(
    user_message: str,
    collection: chromadb.Collection,
    send_event,  # callable: async (event_dict) -> None
) -> None:
    """Run the agent loop for a single user message, streaming events via send_event."""
    client = AsyncOpenAI()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    while True:
        # Stream the LLM response
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            stream=True,
        )

        # Accumulate the streamed response
        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Stream text tokens
            if delta.content:
                accumulated_content += delta.content
                await send_event({"type": "llm_token", "token": delta.content})

            # Accumulate tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": "",
                        }
                    if tc.id:
                        accumulated_tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated_tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

        finish_reason = chunk.choices[0].finish_reason if chunk.choices else "stop"

        # Build assistant message to append to history
        if accumulated_tool_calls:
            tool_calls_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in accumulated_tool_calls.values()
            ]
            assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls_list}
            if accumulated_content:
                assistant_msg["content"] = accumulated_content
            messages.append(assistant_msg)
        else:
            messages.append({"role": "assistant", "content": accumulated_content})

        # If no tool calls, we're done
        if not accumulated_tool_calls or finish_reason == "stop":
            await send_event({"type": "llm_done", "content": accumulated_content})
            break

        # Execute each tool call
        for tc in accumulated_tool_calls.values():
            tool_name = tc["name"]
            try:
                tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                tool_args = {}

            # Emit tool_call event
            await send_event({"type": "tool_call", "tool": tool_name, "args": tool_args, "call_id": tc["id"]})

            # Dispatch tool (may emit RAG events)
            result, rag_events = dispatch_tool(tool_name, tool_args, collection)

            # Emit RAG events before tool_result
            for ev in rag_events:
                await send_event(ev)

            # Emit tool_result event
            await send_event({"type": "tool_result", "tool": tool_name, "result": result, "call_id": tc["id"]})

            # Append tool result to message history
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                }
            )
