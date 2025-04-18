from abc import ABC, abstractmethod

from agexp.keydoor.structures import Observation, Action, Tool
from agexp.keydoor.language_model import LanguageModel, FakeLLM, OpenAIChatLLM

import random

from pathlib import Path

import re

SCRIPT_DIR = Path(__file__).parent


class Agent(ABC):
    @abstractmethod
    def observe(self, obs: Observation) -> None:
        """Update internal state with latest environment observation."""
        ...

    @abstractmethod
    def act(self) -> Action:
        """Return the next action to take based on internal state."""
        ...


class CheatingAgent(Agent):
    """An agent that knows exactly how to solve the default game"""

    _winning_actions = [
        Action.MOVE_DOWN,
        Action.MOVE_DOWN,
        Action.MOVE_RIGHT,
        Action.PICK_UP_KEY,
        Action.MOVE_RIGHT,
        Action.MOVE_DOWN,
        Action.OPEN_DOOR,
    ]

    def __init__(self):
        self.actions = (a for a in self._winning_actions)

    def observe(self, obs: Observation) -> None:
        return

    def act(self) -> Action:
        try:
            return next(self.actions)
        except StopIteration:
            raise RuntimeError("Agent has no more actions planned.")


class RandomAgent(Agent):
    """An agent that acts randomly."""

    def observe(self, obs: Observation) -> None:
        pass

    def act(self) -> Action:
        return random.choice(list(Action))


class LLMAgent(Agent):
    """A LLM agent which interacts with the environment in an entirely static way."""

    def __init__(self, backend: str, instructions_file: str = "llm_prompt.txt"):
        self.prompt = ""

        self.llm: LanguageModel
        with open(SCRIPT_DIR / instructions_file) as f:
            self.instructions = f.read()

        if backend == "fake":
            self.llm = FakeLLM()
        elif backend == "openai":
            self.llm = OpenAIChatLLM()
        else:
            raise ValueError(f"Non-supported LLM backend: {backend}")

    def observe(self, obs: Observation) -> None:
        self.obs = obs

    def act(self) -> Action:
        prompt = self._format_prompt()
        raw_response = self.llm.complete_prompt(self.instructions, prompt)
        return self._parse_response(raw_response)

    def _format_prompt(self) -> str:
        return self.obs.as_string

    def _parse_response(self, text: str) -> Action:
        text = text.lower()
        if "up" in text:
            return Action.MOVE_UP
        elif "down" in text:
            return Action.MOVE_DOWN
        elif "left" in text:
            return Action.MOVE_LEFT
        elif "right" in text:
            return Action.MOVE_RIGHT
        elif "pick" in text:
            return Action.PICK_UP_KEY
        elif "open" in text:
            return Action.OPEN_DOOR
        else:
            raise ValueError(f"Unrecognized response: {text}")


class LLMToolAgent(LLMAgent):

    def __init__(self, backend: str, max_tries: int = 10):
        super().__init__(backend, instructions_file="llm_tool_prompt.txt")
        self.tools: list[Tool] = []
        self.last_tool_result: str | None = None
        self.max_tries = 10

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def act(self) -> Action:
        for _ in range(self.max_tries):
            prompt = self._format_prompt()
            response = self.llm.complete_prompt(self.instructions, prompt)

            if self._is_direct_action(response):
                return self._parse_direct_action(response)

            tool_result = self._parse_and_run_tool(response)
            self.last_tool_result = tool_result

        raise RuntimeError(f"Tool loop did not complete after {self.max_tries} tries.")

    def _format_prompt(self) -> str:
        grid_str = self.obs.as_string.strip()
        tool_descriptions = "\n".join(
            f"- {tool.name}(...): {tool.description}" for tool in self.tools
        )

        tool_feedback = (
            f"\n\nPrevious tool result:\n{self.last_tool_result}"
            if self.last_tool_result
            else ""
        )

        return (
            f"Grid:\n{grid_str}\n\n"
            f"has_key: {self.obs.has_key}\n\n"
            f"Available tools:\n{tool_descriptions}\n"
            f"{tool_feedback}\n\n"
            "What will you do next? Respond with a Thought and Action."
        )

    # Stubs

    def _is_direct_action(self, response: str) -> bool:
        return "move" in response or "pick" in response or "open" in response

    def _parse_direct_action(self, text: str) -> Action:
        return super()._parse_response(text)

    def _parse_and_run_tool(self, text: str) -> str:
        match = re.search(r"Action:\s*(\w+)\((.*?)\)", text)
        if not match:
            raise ValueError(f"Could not parse tool call: {text}")

        name, arg = match.group(1), match.group(2)

        tool_map = {tool.name: tool for tool in self.tools}
        if name not in tool_map:
            raise ValueError(f"Unknown tool: {name}")

        return tool_map[name].func(arg.strip())
