import asyncio
import time
from asyncio import Queue
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import openai
from pydantic import BaseModel, Field, validator

from langbridge.trackers import ProgressTracker
from langbridge.generations import Generation, ChatGeneration
from langbridge.utils import get_logger
from langbridge.settings import get_openai_settings


_logger = get_logger()
_settings = get_openai_settings()
openai.api_key = _settings.openai_key


class RequestsHandler(BaseModel):
    generations: List[Generation]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    progress_tracker: ProgressTracker = Field(default=ProgressTracker())
    retry_queue: Queue = Field(default=Queue())
    outfile: Optional[Path]
    # TODO: Make the fields below read-only
    total_requests: Optional[int]
    total_tokens: Optional[int]
    total_cost: Optional[float]

    """
    This is a RequestsHandler to handle parallel OpenAI generation requests.
    The Responsibilities of RequestsHandler class include:
    - Initiating multiple requests.
    - Keeping a track of the progress of every operation.
    - Keeping a track of every retry operation.
    - Optionally outputting the results of every operation to the specified path.
    - Limiting the frequency of the requests to not violate the max requests per minute threshold.
    - Regulating the consumption of tokens to not exceed the max tokens per minute limit.
    - Considering each 'Generation' as an API call/request made and processing it individually.

    Note:
    - If no outfile path is provided, the results of the API calls will be directly returned.
    - If an outfile path is provided, the results of the API calls are stored at the specified location.
    - The maximum number of requests and tokens allowed per minute are specified during the class initialization.
    - A rate limit error will result in a pause specified by rate_limit_pause and retries will be handled accordingly.
    """

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
        """
        This is an asynchronous method that executes the various requests stored in the 'generations' attribute of
        the RequestsHandler class.

        Process Flow:
        - Checks which request needs to be processed next: a request from the retry queue or a new request.
        - Then, it calculates the available_request_capacity and available_token_capacity based on the time passed since the last update.
        - If there is available capacity, the method proceeds to process the next request and decreases the available capacities accordingly.
        - If a rate limit error is encountered, the method pauses for the time denoted by rate_limit_pause and then re-attempts the failed request.
        - This process continues until all requests have been processed.

        Returns:
            The results of all API calls as a list if no outfile path is provided. Otherwise, it saves the results to the specified outfile path and returns None.
        """
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
                                outfile=self.outfile
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

        if not self.outfile:
            results = await asyncio.gather(*tasks)
            return results


class ChatRequestHandler(RequestsHandler):
    """
    A type of RequestHandler specifically for handling requests to OpenAI Chat models
    """
    generations: List[ChatGeneration]
