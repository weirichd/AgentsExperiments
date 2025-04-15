from abc import ABC, abstractmethod

from agexp.keydoor.structures import Observation, Action
from agexp.keydoor.language_model import LanguageModel, FakeLLM

import random

from textwrap import dedent


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
    _prompt_header = """
    You are an agent in a grid world. Each character has the following meanings:

    @ - You
    # - Wall
    K - Key
    D - Door

    Observation:
    """

    _prompt_footer = """
    What will you do? Say one of:
    move up, move down, move left, move right, pick up key, open door."""

    def __init__(self, backend: str):
        self.prompt = ""

        self.llm: LanguageModel

        if backend == "fake":
            self.llm = FakeLLM()
        elif backend == "openai":
            raise NotImplementedError()
        else:
            raise ValueError(f"Non-supported LLM backend: {backend}")

    def observe(self, obs: Observation) -> None:
        self.obs = obs

    def act(self) -> Action:
        prompt = self._format_prompt()
        raw_response = self.llm.complete_prompt(prompt)
        return self._parse_response(raw_response)

    def _format_prompt(self) -> str:
        prompt = dedent(self._prompt_header)
        prompt += self.obs.as_string
        prompt += dedent(self._prompt_footer)

        return prompt

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
