# LearnE AI Service

Flask service for AI-assisted learning path generation.

## Environment

- `LLM_BASE_URL`: 9router OpenAI-compatible API base URL. In Docker Compose this should be `http://9router:20128/v1`. When running AI Service directly on the host and 9router directly on the host, use `http://localhost:20128/v1`.
- `LLM_API_KEY`: API key copied from the 9router dashboard. Required for real generation.
- `LLM_DEFAULT_MODEL`: fallback model when a module-specific model is not configured. The development example is `kr/claude-sonnet-4.5` from the 9router quick-start docs.
- `LLM_GRAMMAR_MODEL`: optional model override for `POST /grammar-checks/check`.
- `LLM_GRAMMAR_FALLBACK_MODELS`: comma-separated fallback models for Grammar Check, tried in order.
- `LLM_AI_TUTOR_MODEL`: optional model override for `POST /ai-tutor/respond`.
- `LLM_AI_TUTOR_FALLBACK_MODELS`: comma-separated fallback models for AI Tutor, tried in order.
- `LLM_LEARNING_PATH_MODEL`: optional model override for `POST /learning-path/generate`.
- `LLM_LEARNING_PATH_FALLBACK_MODELS`: comma-separated fallback models for Learning Path, tried in order.
- `LLM_COURSE_RECOMMENDATION_MODEL`: optional model override for `POST /course-recommendations/generate`.
- `LLM_COURSE_RECOMMENDATION_FALLBACK_MODELS`: comma-separated fallback models for Course Recommendation, tried in order.
- `LLM_TIMEOUT_MS`: outbound timeout for AI Service calls to 9router. Defaults to `8000` in code; Docker examples use `30000` for free models.

Fallback model values are comma-separated. Empty entries are ignored, and duplicate models are skipped while preserving order.

9router itself is configured from `infra/.env`:

- `NINE_ROUTER_INITIAL_PASSWORD`: initial dashboard password for the 9router container. Change the example value before exposing the dashboard beyond localhost.

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
docker run --rm -e LLM_API_KEY=your-key learne-ai-dev
```

## 9router Development

In the full Docker Compose environment, 9router runs as a separate container and publishes its dashboard/API on the host at `http://localhost:20128`. The Compose file binds this port to `127.0.0.1` so it is local-only by default.

Set `NINE_ROUTER_INITIAL_PASSWORD` in `infra/.env` before first startup. The dashboard uses this initial password until you change it from the local dashboard.

AI Service must not call `http://localhost:20128/v1` from inside Docker. Inside Docker, `localhost` points to the AI Service container itself. Use Docker service DNS instead:

```env
LLM_BASE_URL=http://9router:20128/v1
```

Use the 9router dashboard to connect providers and copy the generated API key into `infra/.env` as `LLM_API_KEY`.

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
- `503`: `LLM_API_KEY` or an LLM model is not configured.
- `502`: LLM call fails, returns invalid JSON, returns an invalid response shape, or returns no valid recommendations.

### `POST /course-recommendations/generate`

Generates filtered course recommendations from candidate courses and constraints.

Request body:

```json
{
  "candidateCourses": [
    {"courseId": 10, "title": "Present Simple Grammar"}
  ],
  "constraints": {
    "allowedCourseIds": [10]
  }
}
```

Response body:

```json
{
  "weaknesses": ["present simple"],
  "recommendations": [
    {"courseId": 10, "score": 0.94, "reason": "Grammar fit"}
  ]
}
```

Errors:

- `400`: invalid JSON or missing/invalid `candidateCourses` or `constraints.allowedCourseIds`.
- `503`: `LLM_API_KEY` or an LLM model is not configured.
- `502`: LLM call fails, returns invalid JSON, returns an invalid response shape, or returns no valid recommendations.
