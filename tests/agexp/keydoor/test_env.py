from agexp.keydoor.env import tile_grid_from_string_list, get_agent_position, KeyDoorEnv

from agexp.keydoor.types import Tile, Action


def test_tile_grid_from_string_list():
    string_list = ["# #", "@KD"]
    expected = [[Tile.WALL, Tile.EMPTY, Tile.WALL], [Tile.AGENT, Tile.KEY, Tile.DOOR]]

    actual = tile_grid_from_string_list(string_list)

    assert actual == expected


def test_get_agent_position():
    grid = [[Tile.WALL, Tile.EMPTY, Tile.WALL], [Tile.AGENT, Tile.KEY, Tile.DOOR]]
    expected = (0, 1)

    actual = get_agent_position(grid)

    assert actual == expected


def test_move_open_door_success():
    keydoor = KeyDoorEnv()
    keydoor.has_key = True
    keydoor.agent_position = keydoor.door_position

    obs, done, info = keydoor.step(Action.OPEN_DOOR)

    assert done


def test_move_open_door_fail_no_key():
    keydoor = KeyDoorEnv()
    keydoor.agent_position = keydoor.door_position

    obs, done, info = keydoor.step(Action.OPEN_DOOR)

    assert not done


def test_move_open_door_fail_wrong_position():
    keydoor = KeyDoorEnv()
    keydoor.has_key = True

    obs, done, info = keydoor.step(Action.OPEN_DOOR)

    assert not done
