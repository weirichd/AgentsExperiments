from agexp.keydoor.env import KeyDoorEnv
from agexp.keydoor.agent import CheatingAgent, RandomAgent

import click


AGENTS = {
    "cheating": CheatingAgent,
    "random": RandomAgent,
}


@click.command()
@click.option(
    "--agent",
    "agent_name",
    type=click.Choice(AGENTS.keys()),
    default="cheating",
    help="Which agent to use",
)
@click.option("--render/--no-render", default=True, help="Render the grid at each step")
def main(agent_name: str, render: bool):
    env = KeyDoorEnv()
    agent = AGENTS[agent_name]()
    obs = env.reset()
    done = False

    while not done:
        if render:
            env.render()
        agent.observe(obs)
        action = agent.act()
        obs, done, info = env.step(action)

    print("Simulation done!")
    print(f"Completed in {env.n_steps} steps")


if __name__ == "__main__":
    main()
