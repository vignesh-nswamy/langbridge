import time
import json
import asyncio
from pathlib import Path
from uuid import uuid4, UUID
from typing import Dict, Any, Optional

import openai

from pydantic.main import ModelMetaclass
from pydantic import BaseModel, Field, validator, root_validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from openai_processor.utils import get_logger
from openai_processor.trackers import ApiStatusTracker, ApiConsumptionTracker
from openai_processor.model_params.params import ChatModelParams, EmbeddingModelParams, _ModelParams


_logger = get_logger()


class _ApiRequest(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    text: str
    response_model: Optional[ModelMetaclass]
    prompt: Optional[str]
    max_attempts: int = Field(default=5)
    metadata: Optional[Dict[str, Any]]
    # Variables that should be overriden
    model_params: _ModelParams
    # TODO: Make the fields below read-only
    input: Optional[str]
    consumption: Optional[ApiConsumptionTracker]

    class Config:
        arbitrary_types_allowed = True

    @validator("prompt")
    def augment_prompt(cls, v: str, values: Dict[str, Any]):
        if not v: return
        if values.get("response_model") is not None:
            return v + "\n{format_instructions}" + "\n{text}"
        else:
            return v + "\n{text}"

    @validator("input", always=True)
    def resolve_input(cls, _, values: Dict[str, Any]):
        if values.get("prompt") and values.get("response_model"):
            parser = PydanticOutputParser(pydantic_object=values["response_model"])
            prompt = PromptTemplate(
                template=values["prompt"],
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
        elif values.get("prompt"):
            prompt = PromptTemplate(
                template=values["prompt"],
                input_variables=["text"]
            )
        else:
            return values["text"]

        return prompt.format_prompt(text=values["text"]).to_string()

    async def _call_api(self):
        raise NotImplemented

    def _post_process(self, response):
        raise NotImplemented

    async def initiate(
        self,
        retry_queue: asyncio.Queue,
        statustracker: ApiStatusTracker,
        outfile: Optional[Path] = None
    ):
        error = False
        try:
            response = await self._call_api()
        except openai.error.APIError as ae:
            error = True
            statustracker.num_api_errors += 1
        except openai.error.RateLimitError as re:
            error = True
            statustracker.time_last_rate_limit_error = time.time()
            statustracker.num_rate_limit_errors += 1
        except openai.error.Timeout as te:
            error = True
            statustracker.num_other_errors += 1
        except openai.error.ServiceUnavailableError as se:
            error = True
            statustracker.num_api_errors += 1
        except (
            openai.error.APIConnectionError,
            openai.error.InvalidRequestError,
            openai.error.AuthenticationError,
            openai.error.PermissionError
        ) as e:
            error = True
            raise e
        except Exception as e:
            error = True
            raise e

        if error:
            if self.max_attempts:
                retry_queue.put_nowait(self)
            else:
                statustracker.num_tasks_in_progress -= 1
                statustracker.num_tasks_failed += 1
        else:
            statustracker.num_tasks_in_progress -= 1
            statustracker.num_tasks_succeeded += 1

            processed_response = self._post_process(response)
            if outfile:
                with open(outfile, "a") as outf:
                    outf.write(json.dumps(processed_response) + "\n")
            else:
                return processed_response


class ChatCompletionApiRequest(_ApiRequest):
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
        try:
            output = json.loads(response["choices"][0]["message"]["content"])
        except json.JSONDecodeError as jde:
            _logger.error(f"Request f{str(self.uuid)} could not be JSON decoded")
            output = response["choices"][0]["message"]["content"]
        return {
            "uuid": str(self.uuid),
            "tokens_consumed": response["usage"],
            # "total_cost": compute_cost(
            #     model_name=response["model"],
            #     num_prompt_tokens=response["usage"]["prompt_tokens"],
            #     num_completion_tokens=response["usage"]["completion_tokens"]
            # ),
            "output": output,
            "metadata": self.metadata
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
            num_input_tokens=request_tokens,
            num_max_output_tokens=response_tokens
        )


class EmbeddingApiRequest(_ApiRequest):
    model_params: EmbeddingModelParams

    @root_validator(pre=True)
    def check_prompt_response_templates_omitted(cls, values: Dict[str, Any]):
        if "prompt" in values:
            raise ValueError("Embedding Model does not allow `prompt`")
        if "response_model" in values:
            raise ValueError("Embedding Model does not allow `response_model`")
