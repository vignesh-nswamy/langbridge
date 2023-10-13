from typing import Literal, List, Dict, Optional, Any

from pydantic import BaseModel, Field

from langfuse.model import LlmUsage


class GenerationResponse(BaseModel):
    id: str
    completion: str
    metadata: Optional[Dict[str, Any]]
    usage: Optional[LlmUsage]
