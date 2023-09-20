from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, validator


class ApiConsumptionTracker(BaseModel):
    num_input_tokens: int
    num_max_output_tokens: int
    num_total_tokens: Optional[int]
    total_cost: Optional[float]

    @validator("num_total_tokens", always=True)
    def compute_total_tokens(cls, v: int, values: Dict[str, Any]):
        return values["num_input_tokens"] + values["num_max_output_tokens"] if not v \
            else v
