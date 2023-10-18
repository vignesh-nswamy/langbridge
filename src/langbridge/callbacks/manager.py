from uuid import UUID
from typing import List, Any, Optional, Literal, Dict, Union

from .base import BaseCallbackHandler, RunManagerMixIn, LlmManagerMixIn
from langbridge.utils import get_logger


_logger = get_logger()


def _handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: Literal[
        "on_run_start",
        "on_run_error",
        "on_run_end",
        "on_llm_start",
        "on_llm_error",
        "on_llm_end"
    ],
    *args: Any,
    **kwargs: Any,
) -> None:
    for handler in handlers:
        try:
            getattr(handler, event_name)(*args, **kwargs)
        except NotImplementedError as e:
            handler_name = handler.__class__.__name__
            _logger.warning(
                f"NotImplementedError in {handler_name}.{event_name}"
                f" callback: {e}"
            )
        except Exception as e:
            _logger.warning(
                f"Error in {handler.__class__.__name__}.{event_name} callback: {e}"
            )
            if handler.raise_error:
                raise e


class BaseCallbackManager(
    RunManagerMixIn,
    LlmManagerMixIn
):
    """Base class for callback manager."""

    def __init__(
        self,
        *,
        run_id: UUID,
        handlers: List[BaseCallbackHandler],
    ) -> None:
        self.run_id = run_id
        self.handlers = handlers

    def on_run_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        _handle_event(
            handlers=self.handlers,
            event_name="on_run_start",
            serialized=serialized,
            **kwargs
        )

    def on_run_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any
    ):
        _handle_event(
            handlers=self.handlers,
            event_name="on_run_error",
            error=error,
            run_id=run_id,
            **kwargs
        )

    def on_run_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        _handle_event(
            handlers=self.handlers,
            event_name="on_run_end",
            response=response,
            run_id=run_id,
            **kwargs
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        _handle_event(
            handlers=self.handlers,
            event_name="on_llm_start",
            serialized=serialized,
            **kwargs
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        _handle_event(
            handlers=self.handlers,
            event_name="on_llm_error",
            error=error,
            run_id=run_id,
            **kwargs
        )

    def on_llm_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        ** kwargs: Any
    ):
        _handle_event(
            handlers=self.handlers,
            event_name="on_llm_end",
            response=response,
            run_id=run_id,
            **kwargs
        )
