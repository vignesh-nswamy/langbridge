import time
import asyncio
from pathlib import Path
from asyncio import Queue
from typing import List, Optional, Dict, Any, Union

from pydantic.main import ModelMetaclass
from pydantic import BaseModel, Field, validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from langbridge.generation import BaseGeneration, OpenAiGeneration
from langbridge.trackers import ProgressTracker
from langbridge.utils import get_logger
from langbridge.schema import GenerationHandlerInput, OpenAiGenerationPrompt


_logger = get_logger()


class BaseGenerationHandler(BaseModel):
    model: str
    model_parameters: Dict[str, Any]
    inputs: List[GenerationHandlerInput]
    base_prompt: Optional[str]
    response_model: Optional[ModelMetaclass]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_attempts_per_request: int = Field(default=3)
    generations: Optional[List[BaseGeneration]]
    retry_queue: Optional[Queue] = Field(default=Queue())
    progress_tracker: Optional[ProgressTracker] = Field(default=ProgressTracker())
    # Read-only Fields
    approximate_tokens: Optional[int] = Field(description="Approximate number of prompt tokens in all the requests to be made")
    approximate_cost: Optional[float] = Field(description="Approximate cost for all the requests to be made")

    class Config:
        arbitrary_types_allowed = True

    @validator("generations", always=True)
    def resolve_generations(cls, v: List[BaseGeneration], values: Dict[str, Any]) -> List[BaseGeneration]:
        raise NotImplemented

    @validator("approximate_tokens", always=True)
    def compute_approximate_tokens(cls, _, values: Dict[str, Any]) -> int:
        return sum(
            [
                generation.usage.prompt_tokens
                for generation in values["generations"]
            ]
        )

    @validator("approximate_cost", always=True)
    def compute_approximate_cost(cls, _, values: Dict[str, Any]) -> int:
        return sum(
            [
                generation.usage.total_cost
                for generation in values["generations"]
            ]
        )

    async def execute(
        self,
        outfile: Optional[Path] = None
    ):
        generations = iter(self.generations)

        rate_limit_pause = 15
        loop_sleep = 0.001

        next_request: BaseGeneration = None

        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        all_requests_exhausted = False

        tasks = []
        while True:
            # Get the next request from the Retry Queue or the Iterator
            if next_request is None:
                if not self.retry_queue.empty():
                    next_request = self.retry_queue.get_nowait()
                elif not all_requests_exhausted:
                    try:
                        next_request = next(generations)
                        self.progress_tracker.num_tasks_in_progress += 1
                        self.progress_tracker.num_tasks_initiated += 1
                    except StopIteration:
                        _logger.info("All API calls have been initiated. Waiting for responses...")
                        all_requests_exhausted = True

            # Process next request
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
                self.max_requests_per_minute
            )
            available_token_capacity = min(
                available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
                self.max_tokens_per_minute
            )
            last_update_time = current_time

            if next_request:
                next_request_tokens = next_request.usage.total_tokens

                if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.max_attempts -= 1

                    tasks.append(
                        asyncio.create_task(
                            next_request.invoke(
                                retry_queue=self.retry_queue,
                                progress_tracker=self.progress_tracker,
                                outfile=outfile
                            )
                        )
                    )
                    # Reset `next_request` to None
                    next_request = None

            if self.progress_tracker.num_tasks_in_progress == 0:
                break

            # Main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(loop_sleep)

            # If a rate limit error was thrown, pause to cool down
            seconds_since_rate_limit_error = time.time() - self.progress_tracker.time_last_rate_limit_error
            if seconds_since_rate_limit_error < rate_limit_pause:
                await asyncio.sleep(
                    rate_limit_pause - seconds_since_rate_limit_error
                )

        if not outfile:
            results = await asyncio.gather(*tasks)
            return results


class OpenAiGenerationHandler(BaseGenerationHandler):
    @validator("generations", always=True)
    def resolve_generations(cls, v: List[OpenAiGeneration], values: Dict[str, Any]) -> List[OpenAiGeneration]:
        if v: return v

        inputs: List[GenerationHandlerInput] = values["inputs"]
        base_prompt = values.get("base_prompt")
        response_model = values.get("response_model")

        if base_prompt and response_model:
            parser = PydanticOutputParser(pydantic_object=response_model)
            prompt_template = PromptTemplate(
                template=base_prompt + "\n{format_instructions}" + "\n{text}",
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
        elif base_prompt:
            prompt_template = PromptTemplate(
                template=base_prompt + "\n{text}",
                input_variables=["text"]
            )
        else:
            prompt_template = PromptTemplate(
                template="{text}",
                input_variables=["text"]
            )

        return [
            OpenAiGeneration(
                model=values["model"],
                model_parameters=values["model_parameters"],
                prompt=OpenAiGenerationPrompt(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template.format_prompt(
                                text=inp.text
                            ).to_string()
                        }
                    ]
                ),
                metadata=inp.metadata,
                max_attempts=values.get("max_attempts_per_request")
            )
            for inp in inputs
        ]
