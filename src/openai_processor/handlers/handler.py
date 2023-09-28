import asyncio
import time
from asyncio import Queue
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import openai
from pydantic import BaseModel, Field, validator

from openai_processor.trackers import ProgressTracker
from openai_processor.generations import Generation, ChatGeneration
from openai_processor.utils import get_logger
from openai_processor.settings import get_openai_settings


_logger = get_logger()
_settings = get_openai_settings()
openai.api_key = _settings.openai_key


class RequestsHandler(BaseModel):
    generations: List[Generation]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    status_tracker: ProgressTracker = Field(default=ProgressTracker())
    retry_queue: Queue = Field(default=Queue())
    outfile: Optional[Path]
    # TODO: Make the fields below read-only
    total_requests: Optional[int]
    total_tokens: Optional[int]
    total_cost: Optional[float]

    @validator("total_requests", always=True)
    def compute_total_requests(cls, _, values: Dict[str, Any]):
        return len(values["generations"])

    @validator("total_tokens", always=True)
    def compute_total_tokens(cls, _, values: Dict[str, Any]):
        return sum([
            r.usage.total_tokens
            for r in values["generations"]
        ])

    @validator("total_cost", always=True)
    def compute_total_cost(cls, _, values: Dict[str, Any]):
        return sum([
            r.usage.total_cost
            for r in values["generations"]
        ])

    class Config:
        arbitrary_types_allowed = True

    async def execute(
        self
    ):
        requests_iterator = iter(self.generations)

        rate_limit_pause = 15
        loop_sleep = 0.001

        next_request: Generation = None

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
                        next_request = next(requests_iterator)
                        self.status_tracker.num_tasks_in_progress += 1
                        self.status_tracker.num_tasks_initiated += 1
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
                                statustracker=self.status_tracker,
                                outfile=self.outfile
                            )
                        )
                    )
                    # Reset `next_request` to None
                    next_request = None

            if self.status_tracker.num_tasks_in_progress == 0:
                break

            # Main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(loop_sleep)

            # If a rate limit error was thrown, pause to cool down
            seconds_since_rate_limit_error = time.time() - self.status_tracker.time_last_rate_limit_error
            if seconds_since_rate_limit_error < rate_limit_pause:
                await asyncio.sleep(
                    rate_limit_pause - seconds_since_rate_limit_error
                )

        if not self.outfile:
            results = await asyncio.gather(*tasks)
            return results


class ChatRequestHandler(RequestsHandler):
    generations: List[ChatGeneration]
