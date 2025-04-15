from agexp.keydoor.types import Tile, Action, Observation, Grid


def tile_grid_from_string_list(string_list: list[str]) -> Grid:
    return [[Tile(s) for s in string] for string in string_list]


def _get_tile_position(grid: Grid, tile: Tile) -> tuple[int, int]:
    for row_num, row in enumerate(grid):
        if tile in row:
            y = row_num
            x = row.index(tile)

            return x, y

    raise ValueError(f"{tile} not found in this grid")


def get_agent_position(grid: Grid) -> tuple[int, int]:
    return _get_tile_position(grid, Tile.AGENT)


def get_key_position(grid: Grid) -> tuple[int, int]:
    return _get_tile_position(grid, Tile.KEY)


def get_door_position(grid: Grid) -> tuple[int, int]:
    return _get_tile_position(grid, Tile.DOOR)


class KeyDoorEnv:
    """Key Door Game Environment"""

    _initial_grid: list[str] = [
        "#####",
        "#@  #",
        "# # #",
        "# K #",
        "#  D#",
        "#####",
    ]

    def __init__(self):
        self.reset()

    def reset(self) -> Observation:
        self.grid = tile_grid_from_string_list(self._initial_grid)
        self.has_key = False
        self.done = False
        self.agent_position = get_agent_position(self.grid)
        self.key_position = get_key_position(self.grid)
        self.door_position = get_door_position(self.grid)
        self.n_steps = 0
        self.standing_tile = Tile.EMPTY

        return Observation(self.grid, self.has_key)

    def step(self, action: Action) -> tuple[Observation, bool, dict]:
        x, y = self.agent_position

        match action:
            case Action.OPEN_DOOR:
                if self.has_key and self.agent_position == self.door_position:
                    self.done = True
            case Action.PICK_UP_KEY:
                if not self.has_key and self.agent_position == self.key_position:
                    self.has_key = True
                    self.standing_tile = Tile.EMPTY
            case Action.MOVE_UP:
                dest = (x, y - 1)
                self._update_grid(dest)
            case Action.MOVE_DOWN:
                dest = (x, y + 1)
                self._update_grid(dest)
            case Action.MOVE_RIGHT:
                dest = (x - 1, y)
                self._update_grid(dest)
            case Action.MOVE_LEFT:
                dest = (x + 1, y)
                self._update_grid(dest)

        return Observation(self.grid, self.has_key), self.done, {}

    def render(self) -> None:
        for row in self.grid:
            print("".join([t.value for t in row]))

    def _update_grid(self, dest) -> None:
        dx, dy = dest
        x, y = self.agent_position

        if self.grid[dy][dx] == Tile.WALL:  # Do not move onto walls
            return
        # Replace the location where the agent was with the standing tile
        self.grid[y][x] = self.standing_tile
        # Move the agent
        self.standing_tile = self.grid[dy][dx]
        self.grid[dy][dx] = Tile.AGENT
