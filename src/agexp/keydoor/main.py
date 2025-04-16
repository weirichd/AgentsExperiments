import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from agexp.keydoor.env import KeyDoorEnv
from agexp.keydoor.agent import CheatingAgent, RandomAgent, LLMAgent, Agent
from agexp.keydoor.structures import Action

from dotenv import load_dotenv


console = Console()


load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=".env.secret", override=True)


def render_grid(grid) -> Table:
    table = Table.grid(padding=0)
    for row in grid:
        row_str = ""
        for tile in row:
            symbol = tile.value
            style = {
                "@": "bold cyan",
                "K": "bold yellow",
                "D": "bold green",
                "#": "grey39",
            }.get(symbol, "white")
            row_str += f"[{style}]{symbol}[/{style}]"
        table.add_row(row_str)
    return table


def render_game_step(grid, prompt: str, response: str):
    console.clear()

    grid_panel = Panel(render_grid(grid), title="Gridworld", box=box.ROUNDED)
    prompt_panel = Panel(Text(prompt), title="Prompt", style="dim", box=box.ROUNDED)
    response_panel = Panel(
        Text(response, style="bold magenta"), title="LLM Response", box=box.ROUNDED
    )

    console.print(grid_panel)
    console.print(prompt_panel)
    console.print(response_panel)


@click.command()
@click.option(
    "--agent",
    type=click.Choice(["cheating", "random", "llm"]),
    default="cheating",
    help="Which agent to run",
)
@click.option(
    "--llm-backend",
    type=click.Choice(["fake", "openai"]),
    default="fake",
    help="Which LLM backend to use",
)
@click.option(
    "--render-rich/--no-render-rich", default=True, help="Fancy Rich-based render"
)
@click.option("--max-iter", type=int, default=25, help="Maximum attempts")
def main(agent: str, llm_backend: str, render_rich: bool, max_iter):
    env = KeyDoorEnv()

    agent_instance: Agent

    if agent == "cheating":
        agent_instance = CheatingAgent()
    elif agent == "random":
        agent_instance = RandomAgent()
    elif agent == "llm":
        agent_instance = LLMAgent(llm_backend)
    else:
        raise ValueError(f"Unsupported agent type: {agent}")

    obs = env.reset()
    done = False

    i = 0

    while not done and i < max_iter:
        agent_instance.observe(obs)
        action: Action = agent_instance.act()
        prompt = getattr(agent_instance, "_format_prompt", lambda: "N/A")()
        obs, done, info = env.step(action)

        if render_rich:
            render_game_step(env.grid, prompt, action.name)
        else:
            print("Action:", action.name)
            env.render()

        i += 1

    if i < max_iter:
        console.print("[bold green]Episode finished.[/bold green]")
    else:
        console.print("[bold red]Episode failed.[/bold red]")


if __name__ == "__main__":
    main()
