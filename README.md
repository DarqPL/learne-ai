# LearnE AI Service

Flask service for AI-assisted learning path generation.

## Environment

- `GEMINI_API_KEY`: required for `POST /learning-path/generate`.
- `GEMINI_MODEL`: optional Gemini model name. Defaults to `gemini-1.5-flash`.

## Development

```bash
python -m pip install -r requirements.txt
flask --app app run --host 0.0.0.0 --port 5000
```

Run tests:

```bash
python -m pytest
```

Build and run the development container:

```bash
docker build -f Dockerfile.dev -t learne-ai-dev .
docker run --rm -e GEMINI_API_KEY=your-key learne-ai-dev
```

In the full project, this service is intended to run inside Docker Compose and be called by backend through the internal URL `http://ai-service:5000`. It should not be exposed directly to frontend clients.

## Endpoints

### `GET /health`

Returns service health.

```json
{"status": "ok"}
```

### `POST /learning-path/generate`

Generates a filtered learning path from candidate lessons and constraints.

Request body:

```json
{
  "candidateLessons": [
    {"lessonId": 12, "lessonTitle": "Present Simple - Yes/No Questions"}
  ],
  "constraints": {
    "allowedLessonIds": [12]
  }
}
```

Response body:

```json
{
  "weaknesses": ["arrays"],
  "recommendations": [
    {"lessonId": 12, "score": 0.85, "reason": "Practice question formation"}
  ]
}
```

Errors:

- `400`: invalid JSON or missing/invalid `candidateLessons` or `constraints.allowedLessonIds`.
- `503`: `GEMINI_API_KEY` is not configured.
- `502`: Gemini call fails, returns invalid JSON, or returns an invalid response shape.
