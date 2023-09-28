import uuid
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

import openai

from pydantic.main import ModelMetaclass
from pydantic import BaseModel, Field, validator, root_validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from langfuse.client import StatefulTraceClient
from langfuse.model import CreateGeneration, InitialGeneration

from openai_processor.utils import get_logger
from openai_processor.model_params import BaseModelParams, ChatModelParams
from openai_processor.trackers import Usage, ProgressTracker


_logger = get_logger()


class Generation(InitialGeneration):
    id: str = Field(default_factory=uuid.uuid4)
    model_parameters: BaseModelParams
    response_model: Optional[ModelMetaclass]
    base_prompt: Optional[str]
    prompt: str
    max_attempts: int = Field(default=5, exclude=True)
    status_message: Literal["Pending", "Initiated", "Done", "Error"] = Field(default="Pending")
    usage: Optional[Usage]
    trace: Optional[StatefulTraceClient] = Field(exclude=True)

    class Config:
        frozen = False
        arbitrary_types_allowed = True
        allow_mutation = True

    # TODO: Don't use `root_validator` for resolving `prompt`
    @root_validator(pre=True)
    def root(cls, values):
        base_prompt = values.get("base_prompt")
        response_model = values.get("response_model")
        input = values["input"]

        if base_prompt and response_model:
            parser = PydanticOutputParser(pydantic_object=response_model)
            values["prompt"] = PromptTemplate(
                template=base_prompt + "\n{format_instructions}" + "\n{text}",
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            ).format_prompt(text=input).to_string()
        elif base_prompt:
            values["prompt"] = PromptTemplate(
                template=base_prompt + "\n{text}",
                input_variables=["text"]
            ).format_prompt(text=input).to_string()
        else:
            values["prompt"] = values["input"]

        return values

    @validator("id", pre=True, always=True)
    def to_string(cls, v):
        return str(v)

    @validator("model")
    def resolve_model(cls, v, values: Dict[str, Any]) -> str:
        return v if v else values["model_parameters"].model

    async def _update_trace(self):
        if self.trace is not None:
            self.trace.generation(
                self
            )

    async def _call_api(self):
        raise NotImplemented

    def _post_process(self, response):
        raise NotImplemented

    async def invoke(
        self,
        retry_queue: asyncio.Queue,
        statustracker: ProgressTracker,
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
                self.status_message = "Error"
                await self._update_trace()

                statustracker.num_tasks_in_progress -= 1
                statustracker.num_tasks_failed += 1
        else:
            statustracker.num_tasks_in_progress -= 1
            statustracker.num_tasks_succeeded += 1

            processed_response = await self._post_process(response)
            if outfile:
                with open(outfile, "a") as outf:
                    outf.write(json.dumps(processed_response) + "\n")
            else:
                return processed_response


class ChatGeneration(Generation):
    model_parameters: ChatModelParams

    async def _call_api(self):
        self.status_message = "Initiated"
        await self._update_trace()

        return await openai.ChatCompletion.acreate(
            messages=[
                {
                    "role": "user",
                    "content": self.prompt
                }
            ],
            **self.model_parameters.dict()
        )

    async def _post_process(self, response):
        # Update Usage
        self.usage = Usage(
            model_name=self.usage.model_name,
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"]
        )

        self.completion = response["choices"][0]["message"]["content"]
        self.status_message = "Done"
        await self._update_trace()

        try:
            output = json.loads(self.completion)
        except json.JSONDecodeError as jde:
            _logger.error(f"Request f{self.id} could not be JSON decoded")
            output = self.completion

        return {
            "metadata": self.metadata,
            "output": self.completion,
            "tokens_consumed": response["usage"],
            "total_cost": self.usage.total_cost,
            "uuid": str(self.id)
        }

    @validator("usage", pre=True, always=True)
    def compute_usage(cls, _, values: Dict[str, Any]) -> Usage:
        model_params: ChatModelParams = values["model_parameters"]
        completion_tokens = model_params.max_tokens

        prompt_tokens = (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            + len(model_params.encoding.encode(values["prompt"]))
            + 2  # every reply is primed with <im_start>assistant
        )

        return Usage(
            model_name=model_params.model,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens
        )