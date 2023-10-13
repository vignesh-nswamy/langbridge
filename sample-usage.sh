#!/bin/bash

if [ -f ./.env ]; then
  export $(grep -v '^#' ./.env | xargs)
  echo "Variables exported from .env."
else
  echo "Error: .env file not found."
  exit 1
fi

langbridge generation --service openai \
  --model gpt-3.5-turbo \
  --infile ./examples/input.jsonl \
  --outfile ./examples/output.jsonl \
  --prompt-file ./examples/prompt.txt \
  --response-format-file ./examples/response-format.json \
  --model-parameters '{"max_tokens": 75, "temperature": 0}' \
  --max-requests-per-minute 100 \
  --max-tokens-per-minute 39500 \
  --max-attempts-per-request 3 \
  --analytics-backend langfuse
