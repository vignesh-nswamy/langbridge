from typing import Literal, List, Dict, Optional, Any

from pydantic import BaseModel, Field

from langfuse.model import LlmUsage


class OpenAiMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(default="user")
    content: str


class OpenAiChatGenerationChoice(BaseModel):
    message: OpenAiMessage
    index: int
    logprobs: Optional[float]
    finish_reason: str


class OpenAiChatGenerationResponse(BaseModel):
    id: str
    model: str
    choices: List[OpenAiChatGenerationChoice]
    usage: LlmUsage


class OpenAiGenerationPrompt(BaseModel):
    messages: List[OpenAiMessage]
