from typing import Optional

from pydantic import BaseSettings, Field


class OpenAiSettings(BaseSettings):
    openai_key: str = Field(env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(env="OPENAI_ORG_ID")
