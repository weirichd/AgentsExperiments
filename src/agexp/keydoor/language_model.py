from abc import ABC, abstractmethod


class LanguageModel(ABC):
    @abstractmethod
    def complete_prompt(self, prompt: str) -> str:
        pass


class FakeLLM(LanguageModel):
    """A Fake LLM that is actually contolled by the user."""

    def complete_prompt(self, prompt) -> str:
        response = input(prompt)

        return response
