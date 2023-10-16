from typing import Optional
from functools import lru_cache

from pydantic import BaseSettings, Field


class AnthropicSettings(BaseSettings):
    """
    Settings required to make API calls to Anthropic models
    """
    anthropic_api_key: str = Field(env="ANTHROPIC_API_KEY")


@lru_cache
def get_anthropic_settings() -> AnthropicSettings:
    return AnthropicSettings()
