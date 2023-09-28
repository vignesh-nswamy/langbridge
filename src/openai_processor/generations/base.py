import uuid
import time
from datetime import datetime as dt
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

import openai

from pydantic.main import ModelMetaclass
from pydantic import Field, validator, root_validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from langfuse.client import StatefulTraceClient
from langfuse.model import InitialGeneration

from openai_processor.utils import get_logger
from openai_processor.model_params import BaseModelParams, ChatModelParams
from openai_processor.trackers import Usage, ProgressTracker


_logger = get_logger()


class Generation(InitialGeneration):
    """
    The `Generation` class is used for handling the entire lifecycle of interacting with OpenAI models. It performs the following core tasks:

    - Pre-process Inputs: It transforms the raw inputs into formatted prompts that the model can understand.
    If response schemas are provided, they will be incorporated into this process.

    - Manage Responses: It handles the responses obtained from OpenAI models.
    This includes error handling for different types of API errors and providing appropriate retry mechanisms.

    - Post-process Outputs: After receiving a response from an OpenAI model, it performs necessary transformations
    which may include parsing the response according to a schema, performing some validations or any other business logic needed.

    - Persist Outputs: It provides mechanisms to save the outputs to a file on disk.
    This is especially useful for large jobs where persisting results immediately is important.

    - Log Outputs: It logs the outputs to the Langfuse trace system.
    This can be used for debugging or tracking the progress.
    """
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

    # TODO: Don't use `root_validator` for resolving `prompt` and `model`
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

        values["model"] = values["model_parameters"].model

        return values

    @validator("id", pre=True, always=True)
    def to_string(cls, v):
        return str(v)

    @validator("model")
    def resolve_model(cls, v, values: Dict[str, Any]) -> str:
        return v if v else values["model_parameters"].model

    def _update_trace(self):
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
        progress_tracker: ProgressTracker,
        outfile: Optional[Path] = None
    ):
        """
        Async method to initiate the calling of the API for a text generation task and
        manage error handling and retries. It performs post-processing on the response,
        updates the task status, and writes the result to an outfile if one is provided.

        Args:
            retry_queue (asyncio.Queue): A queue object to manage tasks retry in case of API errors.
            progress_tracker (ProgressTracker): A tracker object to keep track of the progress of asynchronous tasks.
            outfile (Optional[Path], optional): Outfile path where the result has to be written. Defaults to None.

        Raises:
            openai.error.APIConnectionError
            openai.error.InvalidRequestError
            openai.error.AuthenticationError
            openai.error.PermissionError: Reraised exceptions from calling the OpenAI API.

        Returns:
            processed_response: The processed response of the text generation task if no outfile path is provided.
        """
        error = False
        try:
            response = await self._call_api()
            self.end_time = dt.now()
        except openai.error.APIError as ae:
            error = True
            progress_tracker.num_api_errors += 1
        except openai.error.RateLimitError as re:
            error = True
            progress_tracker.time_last_rate_limit_error = time.time()
            progress_tracker.num_rate_limit_errors += 1
        except openai.error.Timeout as te:
            error = True
            progress_tracker.num_other_errors += 1
        except openai.error.ServiceUnavailableError as se:
            error = True
            progress_tracker.num_api_errors += 1
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
                self._update_trace()

                progress_tracker.num_tasks_in_progress -= 1
                progress_tracker.num_tasks_failed += 1
        else:
            progress_tracker.num_tasks_in_progress -= 1
            progress_tracker.num_tasks_succeeded += 1

            processed_response = await self._post_process(response)
            if outfile:
                with open(outfile, "a") as outf:
                    outf.write(json.dumps(processed_response) + "\n")
            else:
                return processed_response


class ChatGeneration(Generation):
    """
    `Generation` class for interacting specifically with OpenAI Chat models.
    """
    model_parameters: ChatModelParams

    async def _call_api(self):
        """
        Async method to call the OpenAI Chat Completion API

        Raises:
            openai.error.APIError
            openai.error.RateLimitError
            openai.error.Timeout
            openai.error.ServiceUnavailableError
            openai.error.APIConnectionError
            openai.error.InvalidRequestError
            openai.error.AuthenticationError
            openai.error.PermissionError

        Returns:
            A Future object that represents the API response.
            This object will be `await`ed on to extract the response once it is available.
        """
        self.status_message = "Initiated"
        self._update_trace()

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
        self._update_trace()

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
