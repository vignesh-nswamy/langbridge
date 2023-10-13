import asyncio
from pathlib import Path
from uuid import uuid4, UUID
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, validator

from langbridge.schema import OpenAiGenerationPrompt
from langbridge.trackers import Usage, ProgressTracker
from langbridge.parameters import OpenAiChatCompletionParameters


class BaseGeneration(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model: str
    model_parameters: OpenAiChatCompletionParameters
    prompt: OpenAiGenerationPrompt
    metadata: Optional[Dict[str, Any]]
    max_attempts: Optional[int] = Field(default=3)
    usage: Optional[Usage]

    @validator("usage", pre=True, always=True)
    def resolve_usage(cls, v: Usage, values: Dict[str, Any]) -> Usage:
        raise NotImplemented

    async def _call_api(self) -> Dict[str, Any]:
        raise NotImplemented

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplemented

    async def invoke(
        self,
        retry_queue: asyncio.Queue,
        progress_tracker: ProgressTracker,
        outfile: Optional[Path] = None
    ) -> Dict[str, Any]:
        raise NotImplemented

    # class Config:
    #     arbitrary_types_allowed = True
