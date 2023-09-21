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

from openai_processor.api_requests import ChatCompletionApiRequest
from openai_processor.handlers import ChatRequestHandler
from openai_processor.model_params import ChatModelParams

console = Console(width=100)

logging.basicConfig(
    level="NOTSET",
    handlers=[
        RichHandler(level="INFO", console=console, rich_tracebacks=True)
    ]
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
    max_attempts_per_request: int = typer.Option(default=5, help="Maximum number of attempts per request")
):
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

    api_requests = [
        ChatCompletionApiRequest(
            text=line.pop("text"),
            metadata={"index": i, **line},
            prompt=prompt,
            response_model=response_model,
            model_params=model_params,
            max_attempts=max_attempts_per_request
        )
        for i, line in enumerate(lines)
    ]

    handler = ChatRequestHandler(
        api_requests=api_requests,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        outfile=outfile
    )

    _ = typer.confirm(
        f"{handler.total_requests} requests are scheduled, "
        f"collectively containing {handler.total_tokens} tokens."
        f"Total approximate cost is {round(handler.total_cost, 2)}."
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
