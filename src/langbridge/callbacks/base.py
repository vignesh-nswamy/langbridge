from uuid import UUID
from typing import Dict, Any, Union

from langchain.callbacks import StdOutCallbackHandler
from langfuse.callback import CallbackHandler
from langchain.callbacks import FileCallbackHandler


class RunManagerMixIn:
    def on_run_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        """Run when a `Run` starts"""

    def on_run_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any
    ):
        """Run when a `Run` encounters an error"""

    def on_run_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        """Run when a `Run` ends"""


class LlmManagerMixIn:
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        """Run when an API call is invoked"""

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        """Run when API errors are encountered"""

    def on_llm_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        ** kwargs: Any
    ):
        """Run when a Generation request is completed"""


class BaseCallbackHandler(
    RunManagerMixIn,
    LlmManagerMixIn
):
    raise_error: bool = False
