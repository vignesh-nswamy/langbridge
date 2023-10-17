import time
import uuid
import asyncio
from pathlib import Path
from asyncio import Queue
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel, Field, validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from .base import BaseGenerationHandler
from langbridge.generation import OpenAiGeneration
from langbridge.utils import get_logger
from langbridge.schema import GenerationHandlerInput, OpenAiGenerationPrompt
from langbridge.parameters import OpenAiChatCompletionParameters
from langbridge.settings import get_openai_settings


_logger = get_logger()


class OpenAiGenerationHandler(BaseGenerationHandler):
    model_parameters: OpenAiChatCompletionParameters

    @validator("generations", always=True)
    def resolve_generations(cls, v: List[OpenAiGeneration], values: Dict[str, Any]) -> List[OpenAiGeneration]:
        if v: return v

        inputs: List[GenerationHandlerInput] = values["inputs"]
        base_prompt = values.get("base_prompt")

        # TODO: Remove `response_model` support
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
            OpenAiGeneration(
                model=values["model"],
                model_parameters=values["model_parameters"],
                prompt=OpenAiGenerationPrompt(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template.format_prompt(
                                text=inp.text
                            ).to_string()
                        }
                    ]
                ),
                metadata=inp.metadata,
                max_attempts=values.get("max_attempts_per_request"),
                callback_manager=values.get("callback_manager"),
                functions=values["functions"]
            )
            for inp in inputs
        ]
