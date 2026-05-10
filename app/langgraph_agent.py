import os
from typing import Annotated

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.panel import Panel
from typing_extensions import TypedDict

from app import tools as tool_fns

console = Console()

system_prompt = """You are an AI assistant that takes Pizza Deliveries for the popular restaurant, Mamma's Pizzas.

Once you start chatting with a user, get their name, and check if they already exist in our system by using the get_user_information tool.

If they do not exist, make sure to get their address. Even just a city name is enough.

Ask what sort of pizza they would like to order. Once you have all the information you need, use the create_order tool to place the order.

Once placed, let the user know their order number and that their pizza will be delivered soon.
"""


@tool
def get_user_information(name: str) -> str:
    """Get users information from their name."""
    return tool_fns.get_user_information(name)


@tool
def create_order(pizza_description: str, address: str) -> str:
    """Create a new order for a customer."""
    return tool_fns.create_order(pizza_description, address)


@tool
def end_conversation() -> str:
    """End the conversation when the user says goodbye or indicates they are done."""
    return tool_fns.end_conversation()


_tools = [get_user_information, create_order, end_conversation]
_tool_node = ToolNode(_tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_llm() -> ChatOpenAI:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.12:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
    return ChatOpenAI(
        model=model,
        base_url=f"{base_url}/v1",
        api_key="ollama",
    ).bind_tools(_tools)


def _call_llm(state: AgentState):
    console.print(Panel("Calling LLM...", style="blue"))
    llm = _build_llm()
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def _execute_tools(state: AgentState):
    last_ai_message = state["messages"][-1]
    console.print(
        Panel(
            f"LLM called the following tools: {last_ai_message.tool_calls}",
            style="green",
        )
    )
    for tc in last_ai_message.tool_calls:
        console.print(
            Panel(
                f"Tool: {tc['name']}\nInput: {tc['args']}",
                title="🔧 Tool Use",
                border_style="orange3",
            )
        )
    result = _tool_node.invoke(state)
    for msg in result["messages"]:
        console.print(
            Panel(
                f"Result: {msg.content}",
                title="📊 Tool Result",
                border_style="red3",
            )
        )
    return result


def _should_call_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    # Calls "execute_tools" node if the last message contains tool calls, otherwise calls "END" node which ends the graph execution and returns control to the main loop
    return "execute_tools" if last.tool_calls else END


def _after_tools(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            console.print(Panel(f"Last non-tool message: {msg.content}", style="blue"))
            break
        if msg.name == "end_conversation":
            console.print(
                Panel(
                    "Conversation ended by tool. Ending graph execution.", style="red"
                )
            )
            # Calls "END" node, which ends the graph execution and returns control to the main loop
            return END
    console.print(Panel("Returning to LLM after tool execution.", style="green"))
    # Calls "call_llm" node
    return "call_llm"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", _call_llm)
    graph.add_node("execute_tools", _execute_tools)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", _should_call_tools)
    graph.add_conditional_edges("execute_tools", _after_tools)
    return graph.compile(checkpointer=MemorySaver())
