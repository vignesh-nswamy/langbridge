import uuid
from hashlib import sha1

import json
import asyncio
import logging
from pathlib import Path
from typing import *

import rich.progress
import typer
from pydantic import Field, create_model
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from langfuse import Langfuse
from langfuse.model import CreateTrace

from openai_processor.generations import ChatGeneration
from openai_processor.handlers import ChatRequestHandler
from openai_processor.model_params import ChatModelParams
from openai_processor.settings import get_langfuse_settings
from openai_processor.utils import get_logger

console = Console(width=100)

logging.basicConfig(
    level="NOTSET",
    handlers=[
        RichHandler(level="INFO", console=console, rich_tracebacks=True)
    ]
)
_logger = get_logger()

_langfuse_settings = get_langfuse_settings()
_langfuse = Langfuse(
    host=_langfuse_settings.host,
    secret_key=_langfuse_settings.secret_key,
    public_key=_langfuse_settings.public_key
)


def process(
    model: str = typer.Option(help="Name of model to use for API calls"),
    infile: Path = typer.Option(help="Path to a `jsonl` file containing the input texts and optional metadata",
                                exists=True),
    outfile: Path = typer.Option(help="Path to a `jsonl` file to write the outputs to"),
    prompt_file: Path = typer.Option(default=None, help="Path to file containing the prompt", exists=True),
    response_format_file: Path = typer.Option(default=None, help="Path to file containing the response format json",
                                              exists=True),
    max_response_tokens: int = typer.Option(help="Maximum response context length"),
    max_requests_per_minute: int = typer.Option(default=100, help="Maximum number of requests per minute"),
    max_tokens_per_minute: int = typer.Option(default=39500, help="Maximum number of tokens per minute"),
    max_attempts_per_request: int = typer.Option(default=5, help="Maximum number of attempts per request"),
    trace_name: Optional[str] = typer.Option(default=None, help="Langfuse trace name. If not provided, trace will not be used to track generations")
):
    """
    This method is responsible for processing requests using the specified model via API calls. It reads input texts and 
    optional metadata from a provided .jsonl file and builds a list of generations using the provided prompt and response format files.

    Args:
        model: Name of the model to use for API calls (default is None).
        infile: Path to a .jsonl file containing the input texts and optional metadata. The file must exist.
        outfile: Path to a .jsonl file where the outputs will be written.
        prompt_file: Path to a file containing the prompt, if exists.
        response_format_file: Path to a .jsonl file containing the response format json, if exists.
        max_response_tokens: Maximum response context length.
        max_requests_per_minute: Maximum number of requests per minute (default is 100).
        max_tokens_per_minute: Maximum number of tokens per minute (default is 39500).
        max_attempts_per_request: Maximum number of attempts per request (default is 5).
        trace_name: Langfuse trace name. If not provided, trace will not be used to track generations.
    """
    console.print(
        Panel(
            "[bold]Welcome to the OpenAI Processor CLI![/bold]", box=box.DOUBLE,
        )
    )

    model_params = ChatModelParams(
        model=model,
        temperature=0,
        max_tokens=max_response_tokens
    )

    with rich.progress.open(infile, "r", description="Reading input file...", console=console) as f:
        lines: List[dict] = [
            json.loads(line.strip())
            for line in f.readlines()
        ]

    with rich.progress.open(prompt_file, "r", description="Reading prompt file...", console=console) as pf:
        prompt = pf.read()

    response_model = None
    if response_format_file:
        with rich.progress.open(response_format_file, "r", description="Reading response schema...",
                                console=console) as schema_file:
            schema = json.load(schema_file)

        fields = {
            field_name: (
                eval(properties["type"]),
                Field(
                    default=properties.get("default", None),
                    description=properties["description"]
                )
            )
            for field_name, properties in schema.items()
        }

        response_model = create_model(
            "ResponseModel",
            **fields
        )

    trace = _langfuse.trace(
        CreateTrace(
            id=str(
                uuid.UUID(
                    bytes=sha1(
                        str.encode(trace_name)
                    ).digest()[:16],
                    version=4
                )
            ),
            name=trace_name
        )
    ) if trace_name else None

    generations = [
        ChatGeneration(
            input=line.pop("text"),
            metadata={"index": i, **line},
            base_prompt=prompt,
            response_model=response_model,
            model_parameters=model_params,
            max_attempts=max_attempts_per_request,
            trace=trace
        )
        for i, line in enumerate(lines)
    ]

    handler = ChatRequestHandler(
        generations=generations,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        outfile=outfile
    )

    _ = typer.confirm(
        f"{handler.total_requests} requests are scheduled, "
        f"collectively containing {handler.total_tokens} tokens."
        f"Total approximate cost is ${round(handler.total_cost, 2)}."
        f" Proceed?",
        abort=True
    )
    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.TimeElapsedColumn(),
        console=console
    ) as progress:
        progress.add_task(description="Initiating API calls and waiting for responses...")
        asyncio.run(
            handler.execute()
        )

    _logger.info("All responses have been written. Waiting for langfuse to finish logging...")
    _langfuse.flush()
