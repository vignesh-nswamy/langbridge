from typing import Literal

import pytest

from pydantic import BaseModel, Field

from langbridge.handlers import OpenAiGenerationHandler


@pytest.fixture
def fake_basic_handler() -> GenerationHandler:
    handler = GenerationHandler(
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
def fake_handler_with_prompt() -> GenerationHandler:
    handler = GenerationHandler(
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
def fake_handler_with_response_model() -> GenerationHandler:
    class ResponseModel(BaseModel):
        answer: Literal["True", "False"] = Field(description="Whether the statement is True or False")
        reason: str = Field(description="A detailed reason why the statement is True or False")

    handler = GenerationHandler(
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
    fake_basic_handler: GenerationHandler,
    fake_handler_with_prompt: GenerationHandler,
    fake_handler_with_response_model: GenerationHandler
) -> None:
    assert fake_basic_handler.approximate_cost > 0
    assert fake_handler_with_prompt.approximate_cost > 0
    assert fake_basic_handler.approximate_cost > 0


def test_prompt_tokens_computation(
    fake_basic_handler: GenerationHandler,
    fake_handler_with_prompt: GenerationHandler,
    fake_handler_with_response_model: GenerationHandler
) -> None:
    assert fake_basic_handler.approximate_tokens == 35
    assert fake_handler_with_prompt.approximate_tokens == 55
    assert fake_handler_with_response_model.approximate_tokens == 451
