import asyncio
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from openai.openai_object import OpenAIObject

from langbridge.generation import OpenAiGeneration
from langbridge.schema import GenerationResponse, OpenAiGenerationPrompt
from langbridge.trackers import ProgressTracker
from langbridge.callbacks import BaseCallbackManager


@pytest.fixture
def fake_generation() -> OpenAiGeneration:
    generation = OpenAiGeneration(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0, "max_tokens": 50},
        prompt=OpenAiGenerationPrompt(
            messages=[
                {"role": "user", "content": "tell me a joke"}
            ]
        ),
        metadata={"index": 0},
        callback_manager=BaseCallbackManager(handlers=[], run_id=uuid4())
    )

    return generation


def test_prompt_tokens_computation(fake_generation: OpenAiGeneration) -> None:
    assert fake_generation.usage.prompt_tokens == 11


def test_completion_tokens(fake_generation: OpenAiGeneration) -> None:
    assert fake_generation.usage.completion_tokens == 50


def test_cost_computations(fake_generation: OpenAiGeneration) -> None:
    assert fake_generation.usage.prompt_cost > 0
    assert fake_generation.usage.completion_cost > 0


@pytest.mark.asyncio
async def test_execution(
    fake_generation: OpenAiGeneration,
    monkeypatch
) -> None:
    async def mock_call_api() -> OpenAIObject:
        response = OpenAIObject()
        response.id = "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
        response.object = "text_completion"
        response.created = 1589478378
        response.model = "gpt-3.5-turbo"

        choice = OpenAIObject()
        message = OpenAIObject()
        message.role = "assistant"
        message.content = "\n\nThis is indeed a test"
        choice.message = message
        choice.index = 0
        choice.logprobs = None
        choice.finish_reason = "length"
        response.choices = [choice]

        usage = OpenAIObject()
        usage.prompt_tokens = 5
        usage.completion_tokens = 17
        usage.total_tokens = 12
        response.usage = usage

        return response

    # Patch the execute method of the specific instance
    monkeypatch.setattr("langbridge.generation.OpenAiGeneration._call_api", AsyncMock(side_effect=mock_call_api))

    response: GenerationResponse = await fake_generation.invoke(
        retry_queue=asyncio.Queue(),
        progress_tracker=ProgressTracker()
    )

    assert response.completion is not None
    assert fake_generation.usage.completion_tokens == 17
