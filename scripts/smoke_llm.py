"""Smoke test for main-model tool use (model configured in .env).

Check API credentials and endpoint access, verify ToolSpec schemas, and print the
complete function-calling round trip as a reference for the ReAct loop.
Run: uv run python scripts/smoke_llm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import MODEL, chat  # noqa: E402
from agent.tools._contract import ToolRegistry, ToolSpec, ok  # noqa: E402


# --- Dummy tool: add two numbers ---


class AddInput(BaseModel):
    a: float = Field(description="First number to add")
    b: float = Field(description="Second number to add")


def add_handler(args: AddInput) -> dict:
    return ok({"sum": args.a + args.b})


ADD_SPEC = ToolSpec(
    name="add_numbers",
    description="Add two numbers and return the sum. Call only for an explicit calculation request.",
    input_model=AddInput,
    handler=add_handler,
)


def run() -> None:
    registry = ToolRegistry([ADD_SPEC])

    messages: list[dict] = [
        {"role": "system", "content": "You are an assistant. Use available tools for calculations instead of mental arithmetic."},
        {"role": "user", "content": "Please use the tool to calculate 13 + 29."},
    ]

    print(f"=== model: {MODEL} ===")
    print(">>> Round 1: send the user message; expect a tool_call")

    resp = chat(messages=messages, tools=registry.openai_tools())
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print("⚠️ Model returned without calling a tool:", msg.content)
        return

    print(f"  · Model tool calls:{[tc.function.name for tc in msg.tool_calls]}")
    for tc in msg.tool_calls:
        print(f"    - {tc.function.name}({tc.function.arguments})")

    # Append the assistant tool-call message as required by the OpenAI protocol.
    messages.append(msg.model_dump(exclude_none=True))

    # Execute each tool call and append its result to messages.
    for tc in msg.tool_calls:
        result = registry.dispatch(tc.function.name, tc.function.arguments)
        print(f"  · Tool result:{result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    print("\n>>> Round 2: send tool_result; expect a natural-language reply")
    resp2 = chat(messages=messages, tools=registry.openai_tools())
    final = resp2.choices[0].message.content
    print(f"  · Final model reply:{final}")

    print("\n✅ smoke test passed.")


if __name__ == "__main__":
    run()
