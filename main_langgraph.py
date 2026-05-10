from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from rich.console import Console
from rich.panel import Panel

from app.langgraph_agent import build_graph

load_dotenv()

console = Console()


def main():
    console.print(
        Panel.fit(
            "🍕 Welcome to Mamma's Pizza Delivery Chat! (LangGraph + Ollama)",
            style="bold green",
        )
    )
    console.print(
        "Type [bold red]'quit'[/bold red] or [bold red]'exit'[/bold red] to end the conversation.",
        style="dim",
    )
    console.print("─" * 60, style="dim")

    graph = build_graph()
    config = {"configurable": {"thread_id": "pizza-session"}}

    try:
        while True:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                console.print(
                    "\n🍕 Thanks for choosing Mamma's Pizza! Goodbye!",
                    style="bold green",
                )
                break

            if not user_input:
                continue

            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )

                last_message = result["messages"][-1]
                console.print(
                    f"\n🤖 [bold yellow]Pizza Bot:[/bold yellow] {last_message.content}"
                )

                ended = any(
                    isinstance(m, ToolMessage) and m.name == "end_conversation"
                    for m in result["messages"]
                )
                if ended:
                    break

            except Exception as e:
                console.print(f"\n❌ [bold red]Error:[/bold red] {str(e)}")
                console.print("[dim]Please try again.[/dim]")

    except KeyboardInterrupt:
        console.print(
            "\n\n🍕 Thanks for choosing Mamma's Pizza! Goodbye!", style="bold green"
        )


if __name__ == "__main__":
    main()
