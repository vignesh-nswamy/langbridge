from typing import Optional, Dict, Any


import tiktoken
from pydantic import BaseModel, Field, validator


class BaseModelParams(BaseModel):
    """
    OpenAI model parameters that control how the model generates responses
    """
    model: str
    encoding: Optional[tiktoken.Encoding] = Field(exclude=True)

    @validator("encoding", always=True)
    def get_encoding(cls, v: tiktoken.Encoding, values: Dict[str, Any]):
        return v if v else tiktoken.encoding_for_model(
            values["model"]
        )

    class Config:
        arbitrary_types_allowed = True


class ChatModelParams(BaseModelParams):
    """
    Parameters specific to Chat models
    """
    temperature: float = Field(default=0, le=1, ge=0)
    top_p: float = Field(default=1)
    n: int = Field(default=1)
    # `max_tokens` capped at 4000. Default for the OpenAI API is inf
    max_tokens: int = Field(default=1000)
    presence_penalty: float = Field(default=0, le=2, ge=-2)
    frequency_penalty: float = Field(default=0, le=2, ge=-2)


class CompletionModelParams(BaseModelParams):
    """
    Parameters specific to Completion models
    """
    pass


class EmbeddingModelParams(BaseModelParams):
    """
    Parameters specific to Embedding models
    """
    pass
