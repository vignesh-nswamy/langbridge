from typing import Literal, List, Dict, Optional, Any

from pydantic import BaseModel, Field, root_validator


class GenerationHandlerInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]]

    @root_validator(pre=True)
    def extract_metadata(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        text = values.pop("text")
        return {
            "text": text,
            "metadata": values
        }


class OpenAiMessage(BaseModel):
    role: Literal["system", "user"] = Field(default="user")
    content: str
