import os
import json
from uuid import UUID
from datetime import datetime
from typing import Dict, Any, Optional, Union

from langfuse.api.resources.commons.types.observation_level import ObservationLevel
from langfuse.client import Langfuse, StatefulGenerationClient, StatefulTraceClient
from langfuse.model import CreateGeneration, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan

from .. import BaseCallbackHandler

from langbridge.utils import get_logger


_logger = get_logger()


class LangfuseCallbackHandler(BaseCallbackHandler):
    trace: Optional[StatefulTraceClient]
    langfuse: Optional[Langfuse]

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = None,
        debug: bool = False,
        trace: StatefulTraceClient = None,
        release: Optional[str] = None,
    ) -> None:
        prioritized_public_key = public_key if public_key else os.environ.get("LANGFUSE_PUBLIC_KEY")
        prioritized_secret_key = secret_key if secret_key else os.environ.get("LANGFUSE_SECRET_KEY")
        prioritized_host = host if host else os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if trace:
            self.trace = trace
            self.runs: Dict[UUID, StatefulGenerationClient] = {}

        elif prioritized_public_key and prioritized_secret_key:
            self.trace = None
            self.langfuse = Langfuse(
                public_key=prioritized_public_key,
                secret_key=prioritized_secret_key,
                host=prioritized_host,
                debug=debug,
                release=release,
            )
            self.runs: Dict[UUID, StatefulGenerationClient] = {}
        else:
            _logger.error("Either provide a stateful langfuse object or both public_key and secret_key.")
            raise ValueError("Either provide a stateful langfuse object or both public_key and secret_key.")

    def on_run_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        self.__generate_trace(serialized)

    def on_run_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any
    ):
        raise NotImplemented

    def on_run_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        self.trace.task_manager.flush()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        prompt = {
            "messages": serialized["prompt"],
            "functions": serialized["functions"]
        } if len(serialized["functions"]) else serialized["prompt"]
        start_time = datetime.now()
        self.runs[serialized["id"]] = self.trace.generation(
            CreateGeneration(
                id=str(serialized["id"]),
                model=serialized["model"],
                model_parameters=serialized["model_parameters"],
                start_time=start_time,
                completion_start_time=start_time,
                prompt=prompt,
                usage=serialized["usage"],
                level=ObservationLevel.DEFAULT
            )
        )

    def __generate_trace(
        self,
        serialized: Dict[str, Any],
    ):
        response_model = serialized["response_model"].schema() if serialized.get("response_model") else None
        if self.trace is None:
            self.trace = self.langfuse.trace(
                CreateTrace(
                    id=str(serialized["id"]),
                    metadata={
                        "base_prompt": serialized.get("base_prompt", None),
                        "function_calls": serialized.get("functions", []),
                        "response_model": response_model,
                        "num_llm_calls": len(serialized["generations"])
                    }
                )
            )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any
    ):
        self.runs[run_id] = self.runs[run_id].update(
            UpdateGeneration(
                level=ObservationLevel.ERROR,
                status_message=str(error),
                end_time=datetime.now()
            )
        )

    def on_llm_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        ** kwargs: Any
    ):
        completion = response.get("completion")
        completion = json.dumps(completion) if isinstance(completion, dict) else completion

        self.runs[run_id] = self.runs[run_id].update(
            UpdateGeneration(
                completion=completion,
                status_message="Success",
                end_time=datetime.now()
            )
        )
