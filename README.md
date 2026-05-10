# A Super Simple ReAct Agent

This project is to help get you setup with a basic ReAct agent as quickly and as simple as possible.

No fancy frameworks, no complex dependencies, just some simple python code using the LLM API directly.

You'll see that it's actually really simple to do.

Using frameworks (like LangChain) can add a lot of complexity, and hide what is actually going on behind the scenes.

If you are going to be building with Agents, it's important to know how they actually work. And by building your own you can have a lot more control over how they behave as well.

A lot of times is probably best to start with a simple setup like this, with no framework included, and maybe just add things like extra dependencies or frameworks later on, if you really see a need.

For more on building out your AI agents, check out the ["12 Factor Agents"](https://github.com/humanlayer/12-factor-agents) guide as well. That is a good read.

## What does the agent do?

It's a conversational agent, with tool calling abilities, which means you can do some pretty powerful things it.

For this example, I have set it up as a Pizza Delivery agent that gets your name, address and creates an order, using the tools it has.

The point is to demonstrate how to build conversation agent you can interact with in natural language, that has the ability to call tools, and process the results from those tools to see what to do do next.

In this example, the agents goal is to:

- get your name
- check if you exist in their registry via a tool call (this is to show an example of a tool call, and will always
  return `False`)
- if you don't exist, to ask for you address
- to ask what pizza you would like
- create an order with your name, address and pizza via a tool call

## How does the LLM decide which tool to use?

This is the part that feels like magic but is actually straightforward once you see it.

### The system prompt sets the goal

When the agent starts, the LLM receives a **system prompt** that defines its role and the task it needs to accomplish — in this case, collecting a name, address, and pizza order, then placing it. This prompt acts like a job description: the LLM knows what it's trying to achieve before the first user message arrives.

### Tools are described in plain language

Every tool is sent to the LLM on every API call as a JSON schema that includes:

- **`name`** — the identifier the LLM uses to invoke it
- **`description`** — plain English explaining what the tool does and when to use it
- **`input_schema`** — the parameters the LLM must supply

The LLM reads these descriptions and reasons about which tool, if any, is appropriate given the current conversation. There is no routing logic in the Python code — the model makes that decision.

### The ReAct loop: reason → act → observe → repeat

The `PizzaAgent.act()` method implements a loop that mirrors how a person would work through a task:

```
User message arrives
       │
       ▼
 LLM is called with full conversation history + tool definitions
       │
       ├─ stop_reason == "end_turn"  →  return the text reply to the user
       │
       └─ stop_reason == "tool_use"  →  the LLM chose a tool
              │
              ▼
        Execute the tool (Python code runs, e.g. looks up a user)
              │
              ▼
        Append tool result to conversation history
              │
              ▼
        Call act() again  ←──────────────────────────────┐
              │                                           │
              └─ (LLM now sees the result and may call    │
                  another tool or produce a text reply) ──┘
```

The key insight: **the full conversation history — including past tool calls and their results — is sent to the LLM on every request**. This is what lets the model "observe" what happened and decide what to do next. The LLM is not stateful; it's the growing `ContextWindow` that gives it memory.

### Where the decision actually lives

When the LLM decides to use a tool, it returns a content block with `type: "tool_use"` instead of plain text. The agent checks `response.stop_reason`:

- `"tool_use"` → extract the tool name and inputs, execute it via `_execute_tool`, append the result, recurse
- `"end_turn"` → the LLM produced a text response, return it to the user

Python code never decides *which* tool to call or *when* — it only decides *how* to execute a tool once the LLM has already made that choice. The intelligence is entirely in the model.

### Why this works without a framework

Most agent frameworks add routing layers, state machines, or planner agents on top of this pattern. But for a single-goal agent like this one, the LLM's own reasoning is sufficient — the system prompt + tool descriptions give it everything it needs to sequence the steps correctly on its own.

---

## The LangGraph version

The repository also includes a second implementation of the same pizza agent built with [LangGraph](https://github.com/langchain-ai/langgraph). It runs on Ollama only and is kept intentionally parallel to the original so you can compare the two approaches side by side.

**Entry point:** `main_langgraph.py`

```bash
python main_langgraph.py
```

**Relevant files:**

| File | Purpose |
|---|---|
| `app/langgraph_agent.py` | Graph definition — nodes, edges, tools |
| `main_langgraph.py` | REPL loop using the compiled graph |

### How it differs from the hand-rolled agent

The same ReAct loop is expressed differently in each version:

| Concept | Hand-rolled (`PizzaAgent`) | LangGraph |
|---|---|---|
| Loop | `act()` calls itself recursively | Explicit nodes connected by conditional edges |
| State | `ContextWindow` (Pydantic list managed manually) | `AgentState` (TypedDict with `add_messages` reducer) |
| Memory across turns | `context.add(...)` on every step | `MemorySaver` checkpointer — automatic |
| Tools | JSON schema dicts + `if/elif` dispatch | `@tool` decorated functions + `ToolNode` |
| Routing logic | `if stop_reason == "tool_use"` in Python | `_should_call_tools` and `_after_tools` edge functions |
| Model binding | Custom `LLMClient` ABC | `ChatOpenAI.bind_tools()` |

### The graph structure

```
START
  │
  ▼
call_llm ──── no tool calls ────► END
  │
  │ tool calls present
  ▼
execute_tools
  │
  ├─ end_conversation called ────► END
  │
  └─ other tools ─────────────────► call_llm
```

`call_llm` sends the full message history plus the system prompt to Ollama. `execute_tools` runs whichever tools the model requested, prints them in the same Rich-styled panels as the original, and routes back or terminates based on whether `end_conversation` was called.

### Configuration

The LangGraph agent reads the same `.env` variables as the existing Ollama client:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://192.168.1.12:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3:1.7b` | Model served by Ollama |

---

## Why did I build this?

There is something kind of complicated of visualising an Agent that can loop and do things on its own. I think imaging
things with recursion is general kind of hard to do.

I wanted to build my own ReAct agent, and searched for how to do that. There were definitiely some helpful guides on the
web, that allowed me to get started and figure it out.

I did notice though, that even those guides added features and methods which completely necessary for getting just a
basic
ReAct agent working. Once I realised how simple one could be, I thought, why not create one and a write up about to show
how simple it can be. And to help other people getting started as well.

In a lot of ways, this is the startup code I wish I had when building out my fist ReAct agent.

I'm going to be working on a blog post/tutorial to go along with this also, that will add a lot of extra helpful detail
as
well, as well as some links to external resource I found really helpful when trying to figure out ReAct agents.

Anyways, let's get strated!
