from agexp.keydoor.env import KeyDoorEnv
from agexp.keydoor.agent import CheatingAgent, RandomAgent, LLMAgent, Agent

import click


@click.command()
@click.option(
    "--agent",
    "agent_name",
    type=click.Choice(["cheating", "random", "llm"]),
    default="cheating",
    help="Which agent to use",
)
@click.option("--render/--no-render", default=True, help="Render the grid at each step")
@click.option(
    "--max-iter",
    "max_iter",
    default=10_000,
    type=int,
    help="Max number of actions before we give up",
)
@click.option(
    "--llm-backend",
    type=click.Choice(["fake", "openai"]),
    default="fake",
    help="Which LLM backend to use",
)
def main(agent_name: str, render: bool, max_iter: int, llm_backend: str):
    env = KeyDoorEnv()

    agent: Agent

    if agent_name == "cheating":
        agent = CheatingAgent()
    elif agent_name == "random":
        agent = RandomAgent()
    else:
        agent = LLMAgent(llm_backend)

    obs = env.reset()
    done = False

    i = 0
    while not done and i < max_iter:
        if render:
            env.render()
        agent.observe(obs)
        action = agent.act()
        obs, done, info = env.step(action)

        i += 1

    print("Simulation done!")
    if i < max_iter:
        print(f"Completed in {i} steps")
    else:
        print("Failed to complete.")


if __name__ == "__main__":
    main()
