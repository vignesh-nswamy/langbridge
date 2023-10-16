import time
import uuid
import asyncio
from pathlib import Path
from asyncio import Queue
from typing import List, Optional, Dict, Any, Union

from anthropic import HUMAN_PROMPT, AI_PROMPT

from pydantic import BaseModel, Field, validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .base import BaseGenerationHandler
from langbridge.generation import AnthropicGeneration
from langbridge.utils import get_logger
from langbridge.schema import GenerationHandlerInput
from langbridge.settings import get_anthropic_settings


_logger = get_logger()


class AnthropicGenerationHandler(BaseGenerationHandler):
    @validator("generations", always=True)
    def resolve_generations(cls, v: List[AnthropicGeneration], values: Dict[str, Any]) -> List[AnthropicGeneration]:
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
            AnthropicGeneration(
                model=values["model"],
                model_parameters=values["model_parameters"],
                prompt=HUMAN_PROMPT + " " + prompt_template.format_prompt(text=inp.text).to_string() + AI_PROMPT,
                metadata=inp.metadata,
                max_attempts=values.get("max_attempts_per_request"),
                callback_manager=values.get("callback_manager")
            )
            for inp in inputs
        ]
