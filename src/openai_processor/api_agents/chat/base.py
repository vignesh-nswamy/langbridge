import json
from typing import Dict, Any

import openai

from pydantic import BaseModel, Field, validator

from openai_processor.utils import get_logger
from openai_processor.trackers import ApiConsumptionTracker
from openai_processor.model_params.params import ChatModelParams
from openai_processor.api_agents import BaseApiAgent


_logger = get_logger()


class ChatCompletionApiAgent(BaseApiAgent):
    model_params: ChatModelParams

    async def _call_api(self):
        return await openai.ChatCompletion.acreate(
            messages=[
                {
                    "role": "user",
                    "content": self.input
                }
            ],
            **self.model_params.dict()
        )

    def _post_process(self, response):
        # Update consumption stats
        self.consumption = self.consumption.parse_obj(
            {
                **self.consumption.dict(),
                "num_max_output_tokens": response["usage"]["completion_tokens"]
            }
        )

        try:
            output = json.loads(response["choices"][0]["message"]["content"])
        except json.JSONDecodeError as jde:
            _logger.error(f"Request f{str(self.uuid)} could not be JSON decoded")
            output = response["choices"][0]["message"]["content"]
        return {
            "metadata": self.metadata,
            "output": output,
            "tokens_consumed": response["usage"],
            "total_cost": self.consumption.total_cost,
            "uuid": str(self.uuid)
        }

    @validator("consumption", pre=True, always=True)
    def compute_consumption(cls, _, values: Dict[str, Any]) -> ApiConsumptionTracker:
        model_params: ChatModelParams = values["model_params"]
        response_tokens = model_params.max_tokens

        request_tokens = (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            + len(model_params.encoding.encode(values["input"]))
            + 2  # every reply is primed with <im_start>assistant
        )

        return ApiConsumptionTracker(
            model_name=model_params.model,
            num_input_tokens=request_tokens,
            num_max_output_tokens=response_tokens
        )