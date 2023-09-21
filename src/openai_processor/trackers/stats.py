from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, validator
from langchain.callbacks.openai_info import get_openai_token_cost_for_model


class ApiConsumptionTracker(BaseModel):
    model_name: str
    num_input_tokens: int
    num_max_output_tokens: int
    num_total_tokens: Optional[int]
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]

    class Config:
        validate_assignment = True

    @validator("num_total_tokens", always=True)
    def compute_total_tokens(cls, v: int, values: Dict[str, Any]):
        return values["num_input_tokens"] + values["num_max_output_tokens"] if not v \
            else v

    @validator("input_cost", always=True)
    def compute_input_cost(cls, _, values: Dict[str, Any]) -> float:
        return get_openai_token_cost_for_model(
            model_name=values["model_name"],
            num_tokens=values["num_input_tokens"],
            is_completion=False
        )

    @validator("output_cost", always=True)
    def compute_output_cost(cls, _, values: Dict[str, Any]) -> float:
        return get_openai_token_cost_for_model(
            model_name=values["model_name"],
            num_tokens=values["num_max_output_tokens"],
            is_completion=True
        )

    @validator("total_cost", always=True)
    def compute_total_cost(cls, _, values: Dict[str, Any]) -> float:
        return values["input_cost"] + values["output_cost"]
