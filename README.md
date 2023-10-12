# 🤖OpenAI Processor
A package to call OpenAI models without having to worry about rate limits. It also seamlessly integrates with Langfuse, 
providing an interface for analytics and to track / log API calls and their costs.</br>
</br>
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/python-v3.9+-blue.svg) [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
---
## 🚀Getting Started
### 📋Prerequisites
* Python 3.9+ 🐍
* [Poetry](https://python-poetry.org/) <img src="https://python-poetry.org/images/logo-origami.svg" width="10" height="10">
* [Langfuse Server](https://langfuse.com/) [Optional] 🪢

### 💾Installation
Clone the repository
```bash
git clone https://github.com/vignesh-nswamy/openai-processor.git
cd openai-processor
```

Install package and dependencies
```bash
poetry install --without dev
```
---
## 🛠 Usage
The framework can be used both as a CLI and a standalone python package. </br>
If you need analytics and tracking, make sure you have a LangFuse server running.</br>
Refer to [LangFuse Docs](https://langfuse.com/docs/get-started) for more details.

### 💻 As a CLI
```bash
export LANGFUSE_HOST=<langfuse_host_ip>
export LANGFUSE_SECRET_KEY=<langfuse_project_secret_key>
export LANGFUSE_PUBLIC_KEY=<langfuse_project_public_key>
export OPENAI_API_KEY=<openai_api_key>

openai-processor process --model gpt-4 \
  --infile ./examples/input.jsonl \
  --outfile ./examples/output.jsonl \
  --prompt-file ./examples/prompt.txt \
  --response-format-file ./examples/response-format.json \
  --max-response-tokens 75 \
  --max-requests-per-minute 100 \
  --max-tokens-per-minute 39500 \
  --max-attempts-per-request 3 \
  --trace-name openai-processor-test
```

### 📦 As a Python Package
```python
import os
import asyncio
from typing import Literal

from openai_processor.generations import ChatGeneration
from openai_processor.handlers import ChatRequestHandler
from openai_processor.model_params import ChatModelParams

from langfuse import Langfuse
from langfuse.model import CreateTrace

from pydantic import BaseModel, Field

params = ChatModelParams(
    model="gpt-4",
    temperature=0,
    max_tokens=100
)

langfuse = Langfuse(
    host=os.environ["LANGFUSE_HOST"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"]
)
trace = langfuse.trace(
    CreateTrace(
        name="trace-name"
    )
)

class ResponseModel(BaseModel):
    answer: Literal["True", "False"] = Field(description="Whether the statement is True or False")
    reason: str = Field(description="A detailed reason why the statement is True or False")

if __name__ == "__main__":
    generations = [
        ChatGeneration(
            input=line.pop("text"),
            metadata={"index": i, **line},
            base_prompt="""
            Answer if the statement below is True or False
            """,
            response_model=ResponseModel,
            model_parameters=params,
            max_attempts=3,
            trace=trace
        )
        for i, line in enumerate(
            [
                "The speed of light is the same in all media.",
                "Kinetic energy is always a positive quantity."
            ]
        )
    ]

    handler = ChatRequestHandler(
        generations=generations,
        max_requests_per_minute=100,
        max_tokens_per_minute=39500
    )

    results = asyncio.run(
        handler.execute()
    )

    langfuse.flush()
```
---
## 👨‍💻Contributing
Wanna pitch in? Awesome! Here's how:
1. Clone the repo 👾
2. Create a feature branch (`git checkout -b feature/cool-stuff`) 🌿
3. Commit your changes (`git commit -m 'did cool stuff'`) 🛠
4. Push (`git push origin feature/cool-stuff`)
5. Open a PR ✅
---
## 📜License
Distributed under the MIT License. Check out `LICENSE` for more information.

---
## 🐛Reporting Problems
Got issues or feature requests?, [open an issue](https://github.com/vignesh-nswamy/openai-processor/issues) right away!
