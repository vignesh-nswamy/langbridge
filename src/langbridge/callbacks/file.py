import json
from uuid import UUID
from pathlib import Path
from typing import Dict, Any, Union

from . import BaseCallbackHandler


class FileCallbackHandler(BaseCallbackHandler):
    """Log responses to an output file"""
    def __init__(self, outfile: Path):
        self.outfile = outfile

    def on_run_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        pass

    def on_run_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any
    ):
        pass

    def on_run_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        pass

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        **kwargs: Any
    ):
        pass

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any
    ):
        with open(self.outfile, "a") as out_f:
            out_f.write(json.dumps(
                {
                    "id": run_id,
                    "error": True,
                    "message": str(error)
                }
            ) + "\n")

    def on_llm_end(
        self,
        response: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any
    ):
        with open(self.outfile, "a") as out_f:
            out_f.write(json.dumps(response) + "\n")
