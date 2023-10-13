import random
from typing import Literal, List

import pytest

from pydantic import BaseModel, Field

from langbridge.handlers import OpenAiGenerationHandler
from langbridge.schema import OpenAiChatGenerationResponse


@pytest.fixture
def fake_basic_handler() -> OpenAiGenerationHandler:
    handler = OpenAiGenerationHandler(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0, "max_tokens": 50},
        inputs=[
            {"text": "The speed of light is the same in all media.", "metadata": {"index": 0}},
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": 1}}
        ],
        max_requests_per_minute=100,
        max_tokens_per_minute=20000
    )

    return handler


@pytest.fixture
def fake_handler_with_prompt() -> OpenAiGenerationHandler:
    handler = OpenAiGenerationHandler(
        model="gpt-3.5-turbo",
        model_parameters={"temperature": 0.8, "max_tokens": 50},
        inputs=[
            {"text": "The speed of light is the same in all media.", "metadata": {"index": 0}},
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": 1}}
        ],
        base_prompt="Answer if the statement below is True or False",
        max_requests_per_minute=100,
        max_tokens_per_minute=20000
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
            {"text": "The speed of light is the same in all media.", "metadata": {"index": 0}},
            {"text": "Conduction is the only form of heat transfer.", "metadata": {"index": 1}}
        ],
        base_prompt="Answer if the statement below is True or False",
        response_model=ResponseModel,
        max_requests_per_minute=100,
        max_tokens_per_minute=20000
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
    assert fake_basic_handler.approximate_tokens == 35
    assert fake_handler_with_prompt.approximate_tokens == 55
    assert fake_handler_with_response_model.approximate_tokens == 451


@pytest.mark.asyncio
async def test_execution(
    fake_handler_with_response_model: OpenAiGenerationHandler
) -> None:
    from langbridge.settings import get_openai_settings

    responses: List[OpenAiChatGenerationResponse] = await fake_handler_with_response_model.execute()

    assert len(responses) == 2 and random.sample(responses, 1)[0].choices[0].message.content is not None
