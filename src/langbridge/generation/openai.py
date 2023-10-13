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
from langbridge.schema import OpenAiChatGenerationResponse, OpenAiGenerationPrompt


class OpenAiGeneration(BaseGeneration):
    async def _call_api(self) -> OpenAIObject:
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

    def _update_usage(self, response: OpenAiChatGenerationResponse) -> None:
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
        outfile: Optional[Path] = None
    ) -> OpenAiChatGenerationResponse:
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
            raise e
        except Exception as e:
            error = True
            raise e

        if error:
            if self.max_attempts:
                retry_queue.put_nowait(self)
            else:
                self.status_message = "Error"

                progress_tracker.num_tasks_in_progress -= 1
                progress_tracker.num_tasks_failed += 1
        else:
            response: OpenAiChatGenerationResponse = OpenAiChatGenerationResponse(
                **response.to_dict_recursive()
            )
            self._update_usage(response)

            progress_tracker.num_tasks_in_progress -= 1
            progress_tracker.num_tasks_succeeded += 1

            if outfile:
                with open(outfile, "a") as out_f:
                    out_f.write(json.dumps(response.dict()) + "\n")

            return response
