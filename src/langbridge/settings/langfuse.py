from typing import Optional
from functools import lru_cache

from pydantic import BaseSettings, Field


class LangfuseSettings(BaseSettings):
    """
    Settings to connect to Langfuse server
    """
    host: Optional[str] = Field(env="LANGFUSE_HOST")
    secret_key: Optional[str] = Field(env="LANGFUSE_SECRET_KEY")
    public_key: Optional[str] = Field(env="LANGFUSE_PUBLIC_KEY")


@lru_cache
def get_langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings()