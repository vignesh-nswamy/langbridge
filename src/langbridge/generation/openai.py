import json
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

import openai
from openai.openai_object import OpenAIObject

import tiktoken
from pydantic import Field, validator

from langchain.callbacks.openai_info import get_openai_token_cost_for_model

from .base import BaseGeneration
from langbridge.trackers import Usage, ProgressTracker
from langbridge.schema import OpenAiChatGenerationResponse, OpenAiGenerationPrompt, GenerationResponse


class OpenAiGeneration(BaseGeneration):
    prompt: OpenAiGenerationPrompt

    async def _call_api(self) -> OpenAIObject:
        if len(self.functions):
            return await openai.ChatCompletion.acreate(
                messages=[
                    message.dict()
                    for message in self.prompt.messages
                ],
                model=self.model,
                **self.model_parameters.dict(),
                functions=self.functions,
                function_call="auto"
            )
        else:
            return await openai.ChatCompletion.acreate(
                messages=[
                    message.dict()
                    for message in self.prompt.messages
                ],
                model=self.model,
                **self.model_parameters.dict()
            )

    @validator("usage", pre=True, always=True)
    def resolve_usage(cls, v: Usage, values: Dict[str, Any]) -> Usage:
        if v: return v

        encoder = tiktoken.encoding_for_model(values["model"])

        prompt_tokens = 0
        prompt: OpenAiGenerationPrompt = values["prompt"]
        for message in prompt.messages:
            prompt_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            prompt_tokens += len(encoder.encode(message.role)) + len(encoder.encode(message.content))
        prompt_tokens += 2  # every reply is primed with <im_start>assistant

        # Compute tokens for function calls
        # TODO: For the love of God, find a better way to estimate the number of function call tokens
        function_call_tokens = 0
        for function in values["functions"]:
            function_tokens = len(encoder.encode(function["name"]))
            function_tokens += len(encoder.encode(function["description"]))

            parameters = function.get("parameters", {})

            if "type" in parameters:
                function_tokens += len(encoder.encode("type"))
                function_tokens += len(encoder.encode(parameters["type"]))

            properties = parameters.get("properties", {})
            for property_key, property_value in properties.items():
                function_tokens += len(encoder.encode(property_key))
                for field, value in property_value.items():
                    if field in {"type", "description"}:
                        function_tokens += len(encoder.encode(value)) + 2
                    elif field == "enum":
                        function_tokens += sum(len(encoder.encode(o)) + 3 for o in value) - 3

            function_tokens += 11  # adjust as necessary based on actual token costs
            function_call_tokens += function_tokens

        # Hacky goddamn crap to make function call tokens estimation match OpenAI estimation
        function_call_tokens = function_call_tokens + 12 - 2 * (len(values["functions"]) - 2) \
            if function_call_tokens else function_call_tokens

        prompt_tokens += function_call_tokens

        completion_tokens = values["model_parameters"].max_tokens

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost=get_openai_token_cost_for_model(
                model_name=values["model"],
                num_tokens=prompt_tokens,
                is_completion=False
            ),
            completion_cost=get_openai_token_cost_for_model(
                model_name=values["model"],
                num_tokens=completion_tokens,
                is_completion=True
            )
        )

    def _update_usage(self, response: GenerationResponse) -> None:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost=get_openai_token_cost_for_model(
                model_name=self.model,
                num_tokens=prompt_tokens,
                is_completion=False
            ),
            completion_cost=get_openai_token_cost_for_model(
                model_name=self.model,
                num_tokens=completion_tokens,
                is_completion=True
            )
        )

    async def invoke(
        self,
        retry_queue: asyncio.Queue,
        progress_tracker: ProgressTracker,
    ) -> GenerationResponse:
        self.callback_manager.on_llm_start(
            self.dict()
        )

        error = False
        try:
            response: OpenAIObject = await self._call_api()
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

            if self.callback_manager:
                self.callback_manager.on_llm_error(
                    error=e,
                    run_id=self.id
                )

            raise e
        except Exception as e:
            error = True

            if self.callback_manager:
                self.callback_manager.on_llm_error(
                    error=e,
                    run_id=self.id
                )

            raise e

        if error:
            if self.max_attempts:
                retry_queue.put_nowait(self)
            else:
                progress_tracker.num_tasks_in_progress -= 1
                progress_tracker.num_tasks_failed += 1
        else:
            completion = json.dumps({
                "name": response.choices[0].message.function_call.name,
                "arguments": response.choices[0].message.function_call.arguments
            }) if "function_call" in response.choices[0].message \
                else response.choices[0].message.content
            response: GenerationResponse = GenerationResponse(
                id=str(self.id),
                completion=completion,
                usage=response.usage.to_dict(),
                metadata=self.metadata
            )
            self._update_usage(response)

            progress_tracker.num_tasks_in_progress -= 1
            progress_tracker.num_tasks_succeeded += 1

            if self.callback_manager:
                self.callback_manager.on_llm_end(
                    response=response.dict(),
                    run_id=self.id
                )

            return response
