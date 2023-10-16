import random
from typing import Literal, List

from openai.error import (
    APIError,
    RateLimitError,
    Timeout,
    ServiceUnavailableError
)
from openai.openai_object import OpenAIObject

import pytest
from unittest.mock import AsyncMock

from pydantic import BaseModel, Field

from langbridge.handlers.generation import OpenAiGenerationHandler
from langbridge.schema import GenerationResponse


@pytest.fixture
def fake_basic_handler() -> OpenAiGenerationHandler:
    handler = OpenAiGenerationHandler(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0, "max_tokens": 50},
        inputs=[
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": i}}
            for i in range(100)
        ],
        max_requests_per_minute=100,
        max_tokens_per_minute=20000,
        max_attempts_per_request=1
    )

    return handler


@pytest.fixture
def fake_handler_with_prompt() -> OpenAiGenerationHandler:
    handler = OpenAiGenerationHandler(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0.8, "max_tokens": 50},
        inputs=[
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": i}}
            for i in range(100)
        ],
        base_prompt="Answer if the statement below is True or False",
        max_requests_per_minute=100,
        max_tokens_per_minute=20000,
        max_attempts_per_request=1
    )

    return handler


@pytest.fixture
def fake_handler_with_response_model() -> OpenAiGenerationHandler:
    class ResponseModel(BaseModel):
        answer: Literal["True", "False"] = Field(description="Whether the statement is True or False")
        reason: str = Field(description="A detailed reason why the statement is True or False")

    handler = OpenAiGenerationHandler(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0.8, "max_tokens": 50},
        inputs=[
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": i}}
            for i in range(100)
        ],
        base_prompt="Answer if the statement below is True or False",
        response_model=ResponseModel,
        max_requests_per_minute=100,
        max_tokens_per_minute=20000,
        max_attempts_per_request=1
    )

    return handler


def test_cost_computations(
    fake_basic_handler: OpenAiGenerationHandler,
    fake_handler_with_prompt: OpenAiGenerationHandler,
    fake_handler_with_response_model: OpenAiGenerationHandler
) -> None:
    assert fake_basic_handler.approximate_cost > 0
    assert fake_handler_with_prompt.approximate_cost > 0
    assert fake_basic_handler.approximate_cost > 0


def test_prompt_tokens_computation(
    fake_basic_handler: OpenAiGenerationHandler,
    fake_handler_with_prompt: OpenAiGenerationHandler,
    fake_handler_with_response_model: OpenAiGenerationHandler
) -> None:
    assert fake_basic_handler.approximate_tokens == 1700
    assert fake_handler_with_prompt.approximate_tokens == 2700
    assert fake_handler_with_response_model.approximate_tokens == 22500


@pytest.mark.asyncio
async def test_execution(
    fake_handler_with_response_model: OpenAiGenerationHandler,
    monkeypatch
) -> None:
    async def mock_call_api() -> OpenAIObject:
        roll = random.random()

        if roll < 0.8:
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
            usage.total_tokens = 22
            response.usage = usage

            return response

        elif roll < 0.85:
            raise APIError("API Error occurred")

        elif roll < 0.9:
            raise RateLimitError("Rate limit reached")

        elif roll < 0.95:
            raise Timeout("Request timed out")

        else:
            raise ServiceUnavailableError("Service is unavailable")

    # Patch the execute method of the specific instance
    monkeypatch.setattr("langbridge.generation.OpenAiGeneration._call_api", AsyncMock(side_effect=mock_call_api))

    responses: List[GenerationResponse] = await fake_handler_with_response_model.execute()

    progress_tracker = fake_handler_with_response_model.progress_tracker

    assert progress_tracker.num_tasks_in_progress == 0
    assert progress_tracker.num_tasks_succeeded > 0
    assert progress_tracker.num_tasks_failed > 0
    assert progress_tracker.num_rate_limit_errors > 0
    assert progress_tracker.num_api_errors > 0
    assert progress_tracker.num_other_errors > 0
