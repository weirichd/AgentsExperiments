from abc import ABC, abstractmethod

from agexp.keydoor.structures import Observation, Action

import random


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
