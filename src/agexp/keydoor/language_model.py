from abc import ABC, abstractmethod
from openai import OpenAI

import sys


class LanguageModel(ABC):
    @abstractmethod
    def complete_prompt(self, prompt: str) -> str:
        pass


class FakeLLM(LanguageModel):
    """A Fake LLM that is actually contolled by the user."""

    def complete_prompt(self, prompt: str) -> str:
        response = input(prompt)

        return response


class OpenAIChatLLM(LanguageModel):
    """Backend which uses OpenAI's API"""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.client = OpenAI()

    def complete_prompt(self, prompt: str) -> str:
        instructions = """
        You are an agent in a grid world. Each character has the following meanings:

        @ - You
        # - Wall
        K - Key
        D - Door

        You can only pick up the key while standing on the key space.
        You will not see the key while you are standing on it.
        You can only open the door while you are standing on the door space.
        You will not see the door while you are standing on it.

        Your responses should be one of:
        move up, move down, move left, move right, pick up key, open door.
        """

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )

        print(response.output_text, file=sys.stderr)

        return response.output_text
