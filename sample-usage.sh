#!/bin/bash

if [ -f ./.env ]; then
  export $(grep -v '^#' ./.env | xargs)
  echo "Variables exported from .env."
else
  echo "Error: .env file not found."
  exit 1
fi

openai-processor process --model gpt-4 \
  --infile ./examples/input.jsonl \
  --outfile ./examples/output.jsonl \
  --prompt-file ./examples/prompt.txt \
  --response-format-file ./examples/response-format.json \
  --max-response-tokens 300 \
  --max-requests-per-minute 100 \
  --max-tokens-per-minute 39500 \
  --max-attempts-per-request 3
