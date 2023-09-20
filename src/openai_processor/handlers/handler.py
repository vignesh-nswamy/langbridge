import asyncio
import time
from asyncio import Queue
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

from openai_processor.trackers.statustracker import ApiStatusTracker
from openai_processor.api_requests.api_requests import (
    _ApiRequest,
    ChatCompletionApiRequest
)
from openai_processor.utils import get_logger


_logger = get_logger()


class RequestsHandler(BaseModel):
    api_requests: List[_ApiRequest]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    status_tracker: ApiStatusTracker = Field(default=ApiStatusTracker())
    retry_queue: Queue = Field(default=Queue())
    outfile: Optional[Path]
    # TODO: Make the fields below read-only
    total_requests: Optional[int]
    total_tokens: Optional[int]

    @validator("total_requests", always=True)
    def compute_total_requests(cls, _, values: Dict[str, Any]):
        return len(values["api_requests"])

    @validator("total_tokens", always=True)
    def compute_total_tokens(cls, _, values: Dict[str, Any]):
        api_requests = values["api_requests"]

        return sum([
            r.consumption.num_total_tokens
            for r in api_requests
        ])

    class Config:
        arbitrary_types_allowed = True

    async def execute(
        self
    ):
        requests_iterator = iter(self.api_requests)

        rate_limit_pause = 15
        loop_sleep = 0.001

        next_request: _ApiRequest = None

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
                next_request_tokens = next_request.consumption.num_total_tokens

                if available_request_capacity >= 1 and available_token_capacity >= next_request_tokens:
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.max_attempts -= 1

                    tasks.append(
                        asyncio.create_task(
                            next_request.initiate(
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
    api_requests: List[ChatCompletionApiRequest]
