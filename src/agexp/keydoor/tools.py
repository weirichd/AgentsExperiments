from agexp.keydoor.structures import Tool, Action


def make_core_tools(env) -> list[Tool]:
    def move(direction: str) -> str:
        direction = direction.strip().lower()
        match direction:
            case "up":
                env.step(Action.MOVE_UP)
                return "Moved up."
            case "down":
                env.step(Action.MOVE_DOWN)
                return "Moved down."
            case "left":
                env.step(Action.MOVE_LEFT)
                return "Moved left."
            case "right":
                env.step(Action.MOVE_RIGHT)
                return "Moved right."
            case _:
                return f"Unknown direction: {direction}"

    def pick_up_key(_: str = "") -> str:
        obs, done, _ = env.step(Action.PICK_UP_KEY)
        return f"Attempted to pick up key. has_key={obs.has_key}"

    def open_door(_: str = "") -> str:
        obs, done, _ = env.step(Action.OPEN_DOOR)
        return f"Attempted to open door. done={done}"

    return [
        Tool("move", "Move the agent in a direction (up, down, left, right)", move),
        Tool("pick_up_key", "Pick up the key if standing on it", pick_up_key),
        Tool("open_door", "Open the door if standing on it", open_door),
    ]
