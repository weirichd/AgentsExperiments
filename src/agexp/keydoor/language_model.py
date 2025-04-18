from abc import ABC, abstractmethod
from openai import OpenAI


class LanguageModel(ABC):
    @abstractmethod
    def complete_prompt(self, instructions: str, prompt: str) -> str:
        pass


class FakeLLM(LanguageModel):
    """A Fake LLM that is actually contolled by the user."""

    def complete_prompt(self, instructions: str, prompt: str) -> str:
        print(instructions)
        response = input(prompt)

        return response


class OpenAIChatLLM(LanguageModel):
    """Backend which uses OpenAI's API"""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.client = OpenAI()

    def complete_prompt(self, instructions: str, prompt: str) -> str:

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )

        return response.output_text
