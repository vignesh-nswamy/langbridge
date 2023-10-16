import asyncio
from pathlib import Path
from uuid import uuid4, UUID
from typing import Dict, Any, Optional, Union

from pydantic import BaseModel, Field, validator

from langbridge.schema import OpenAiGenerationPrompt
from langbridge.callbacks import BaseCallbackManager
from langbridge.trackers import Usage, ProgressTracker
from langbridge.parameters import OpenAiChatCompletionParameters


class BaseGeneration(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model: str
    model_parameters: Union[OpenAiChatCompletionParameters]
    prompt: Any
    metadata: Optional[Dict[str, Any]]
    max_attempts: Optional[int] = Field(default=3)
    usage: Optional[Usage]
    callback_manager: Optional[BaseCallbackManager]

    class Config:
        arbitrary_types_allowed = True

    @validator("usage", pre=True, always=True)
    def resolve_usage(cls, v: Usage, values: Dict[str, Any]) -> Usage:
        raise NotImplemented

    async def _call_api(self) -> Any:
        raise NotImplemented

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplemented

    async def invoke(
        self,
        retry_queue: asyncio.Queue,
        progress_tracker: ProgressTracker,
    ) -> Dict[str, Any]:
        raise NotImplemented

