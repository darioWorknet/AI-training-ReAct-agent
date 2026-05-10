import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from app.context_window import ContextWindow
from app.pizza_agent import PizzaAgent, system_prompt

load_dotenv()

console = Console()


def main():
    console.print(
        Panel.fit("🍕 Welcome to Mamma's Pizza Delivery Chat!", style="bold green")
    )
    console.print(
        "Type [bold red]'quit'[/bold red] or [bold red]'exit'[/bold red] to end the conversation.",
        style="dim",
    )
    console.print("─" * 60, style="dim")

    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    if provider == "ollama":
        from app.ollama_client import OllamaClient

        client = OllamaClient(system_prompt=system_prompt)
    else:
        from app.claude_client import ClaudeClient

        client = ClaudeClient(system_prompt=system_prompt)

    context = ContextWindow(conversation_history=[])
    agent = PizzaAgent(context=context, llm_client=client)

    try:
        while True:
            # Get user input
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                console.print(
                    "\n🍕 Thanks for choosing Mamma's Pizza! Goodbye!",
                    style="bold green",
                )
                break

            # Skip empty inputs
            if not user_input:
                continue

            try:
                # Get agent response
                response = agent.send_message(user_input)
                console.print(f"\n🤖 [bold yellow]Pizza Bot:[/bold yellow] {response}")

            except Exception as e:
                console.print(f"\n❌ [bold red]Error:[/bold red] {str(e)}")
                console.print("[dim]Please try again.[/dim]")

    except KeyboardInterrupt:
        console.print(
            "\n\n🍕 Thanks for choosing Mamma's Pizza! Goodbye!", style="bold green"
        )


if __name__ == "__main__":
    main()
