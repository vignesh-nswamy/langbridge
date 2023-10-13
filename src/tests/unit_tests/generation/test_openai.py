import pytest

from langbridge.generation import OpenAiGeneration
from langbridge.schema import LlmGenerationPrompt


@pytest.fixture
def fake_generation() -> OpenAiGeneration:
    generation = OpenAiGeneration(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0, "max_tokens": 50},
        prompt=LlmGenerationPrompt(
            messages=[
                {"role": "user", "content": "tell me a joke"}
            ]
        ),
        metadata={"index": 0}
    )

    return generation


def test_prompt_tokens_computation(fake_generation: OpenAiGeneration) -> None:
    assert fake_generation.usage.prompt_tokens == 11


def test_completion_tokens(fake_generation: OpenAiGeneration) -> None:
    assert fake_generation.usage.completion_tokens == 50


def test_cost_computations(fake_generation: OpenAiGeneration) -> None:
    assert fake_generation.usage.prompt_cost > 0
    assert fake_generation.usage.completion_cost > 0


def test_invocation(fake_generation: OpenAiGeneration):
