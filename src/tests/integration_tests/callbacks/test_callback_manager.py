import random
from pathlib import Path
from typing import Literal, List

import pytest

from pydantic import BaseModel, Field

from langbridge.handlers import OpenAiGenerationHandler
from langbridge.schema import OpenAiChatGenerationResponse
from langbridge.callbacks import FileCallbackHandler
from langbridge.callbacks.analytics import LangfuseCallbackHandler


_tmp_outfile = Path("/tmp/callback-integration-test.txt")


@pytest.fixture
def fake_handler_with_file_callback() -> OpenAiGenerationHandler:
    class ResponseModel(BaseModel):
        answer: Literal["True", "False"] = Field(description="Whether the statement is True or False")
        reason: str = Field(description="A detailed reason why the statement is True or False")

    outfile_callback = FileCallbackHandler(outfile=_tmp_outfile)

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
        max_tokens_per_minute=20000,
        callbacks=[outfile_callback]
    )

    return handler


@pytest.fixture
def fake_handler_with_langfuse_callback() -> OpenAiGenerationHandler:
    class ResponseModel(BaseModel):
        answer: Literal["True", "False"] = Field(description="Whether the statement is True or False")
        reason: str = Field(description="A detailed reason why the statement is True or False")

    langfuse_callback = LangfuseCallbackHandler()

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
        max_tokens_per_minute=20000,
        callbacks=[langfuse_callback]
    )

    return handler


@pytest.mark.asyncio
async def test_outfile_callback(
    fake_handler_with_file_callback: OpenAiGenerationHandler,
) -> None:
    from langbridge.settings import get_openai_settings

    _ = get_openai_settings()

    responses: List[OpenAiChatGenerationResponse] = await fake_handler_with_file_callback.execute()

    assert _tmp_outfile.exists()
    assert _tmp_outfile.stat().st_size > 0

    _tmp_outfile.unlink()


@pytest.mark.asyncio
async def test_langfuse_callback(
    fake_handler_with_langfuse_callback: OpenAiGenerationHandler,
) -> None:
    from langbridge.settings import get_openai_settings

    _ = get_openai_settings()

    responses: List[OpenAiChatGenerationResponse] = await fake_handler_with_langfuse_callback.execute()

    langfuse_callback_handler: LangfuseCallbackHandler = fake_handler_with_langfuse_callback.callback_manager.handlers[0]

    assert len(langfuse_callback_handler.runs) == 2

