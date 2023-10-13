from typing import Literal, List, Dict, Optional, Any

from pydantic import BaseModel, Field


class GenerationHandlerInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]]


class OpenAiMessage(BaseModel):
    role: Literal["system", "user"] = Field(default="user")
    content: str

