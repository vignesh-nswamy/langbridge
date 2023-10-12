from typing import Optional
from functools import lru_cache

from pydantic import BaseSettings, Field


class OpenAiSettings(BaseSettings):
    """
    Settings required to make API calls to OpenAI models
    """
    openai_key: str = Field(env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(env="OPENAI_ORG_ID")


@lru_cache
def get_openai_settings() -> OpenAiSettings:
    return OpenAiSettings()
