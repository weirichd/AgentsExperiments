from dataclasses import dataclass
from enum import Enum


class Tile(Enum):
    EMPTY = " "
    WALL = "#"
    KEY = "K"
    DOOR = "D"
    AGENT = "@"


Grid = list[list[Tile]]


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_RIGHT = 2
    MOVE_LEFT = 3
    PICK_UP_KEY = 4
    OPEN_DOOR = 5


@dataclass
class Observation:
    grid: list[list[Tile]]
    has_key: bool

    @property
    def as_string(self) -> str:
        result = "\n".join(["".join([t.value for t in row]) for row in self.grid])
        result += "\n"
        if self.has_key:
            result += "You have the key."
        else:
            result += "You do not have the key."

        return result
