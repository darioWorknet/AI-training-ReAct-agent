# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the agent
python main.py

# Install dependencies
pip install -e .

# Sync dependencies into the venv
uv sync
```

There is no test suite or linter configured.

## Architecture

This is a minimal ReAct agent for a pizza ordering chatbot. The entry point is `main.py`, which selects a backend client based on the `LLM_PROVIDER` env var (`anthropic` or `ollama`) and hands it to `PizzaAgent`.

**Client layer (`app/llm_client.py`, `app/claude_client.py`, `app/ollama_client.py`)**
- `LLMClient` is an ABC defining `send_messages_with_tools(messages, tools) -> LLMResponse`.
- `LLMResponse` and `ContentBlock` are `Protocol` types (structural subtyping) so that the external `anthropic.types.Message` satisfies the interface without modification.
- `ClaudeClient` wraps the Anthropic SDK directly.
- `OllamaClient` uses the OpenAI-compatible endpoint exposed by Ollama. It converts Anthropic-style tool definitions and message history into OpenAI format before sending, and converts the response back via internal `_Message`/`_ContentBlock` shims.

**Agent layer (`app/pizza_agent.py`)**
- `PizzaAgent.act()` implements the ReAct loop: call the LLM, handle `tool_use` stop reason by executing the tool and appending the result to context, then recurse until `end_turn`.
- Tools are defined as `Tool(BaseModel)` instances inline in `pizza_agent.py` and passed as `List[Dict]` to the client on every call.
- Tool execution is a simple `if/elif` dispatch inside `_execute_tool` — there is no tool registry.

**Context (`app/context_window.py`)**
- `ContextWindow` holds `conversation_history` as a plain list of Pydantic models (`UserMessage`, `AssistantMessage`, `ToolUseMessage`, `ToolResultMessage`).
- All messages are serialised via `.model_dump()` before being sent to the client, so the wire format is always Anthropic's message schema.

## Environment

Copy `.env.example` to `.env`. Required variables:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `LLM_PROVIDER` | `anthropic` | Set to `ollama` to use a local model |
| `OLLAMA_BASE_URL` | `http://192.168.1.12:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Model served by Ollama |
