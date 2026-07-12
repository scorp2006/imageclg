# image-qa

Multimodal image question-answering API using Gemini.

## Endpoint

`POST /answer-image`

Request: `{"image_base64": "...", "question": "..."}`
Response: `{"answer": "..."}`

## Env

`GEMINI_API_KEY` must be set.
