from typing import Optional

from pydantic import BaseModel, Field


class OpenAiChatCompletionParameters(BaseModel):
    temperature: float = Field(default=0, le=1, ge=0)
    top_p: float = Field(default=1)
    n: int = Field(default=1)
    # `max_tokens` capped at 1000. Default for the OpenAI API is inf
    max_tokens: int = Field(default=1000)
    presence_penalty: float = Field(default=0, le=2, ge=-2)
    frequency_penalty: float = Field(default=0, le=2, ge=-2)


class AnthropicCompletion(BaseModel):
    temperature: float = Field(default=0, le=1, ge=0)
    top_p: Optional[float] = Field(default=1)
    top_k: Optional[float] = Field(default=1)
    max_tokens_to_sample: int = Field(default=1000)
