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
- `LLM_COURSE_CONTENT_MODEL`: optional model override for `POST /course-content/enrich-knowledge-points`.
- `LLM_COURSE_CONTENT_FALLBACK_MODELS`: comma-separated fallback models for course content metadata enrichment.
- `LLM_TIMEOUT_MS`: outbound timeout for AI Service calls to 9router. Defaults to `8000` in code; Docker examples use `30000` for free models.
- `LLM_ENRICHMENT_BUDGET_MS`: course enrichment's **total** monotonic generation budget, including all configured fallback models, response parsing, and validation. Defaults to `110000`; must be a positive integer no greater than `110000` (small values are supported for tests). Each attempt is limited to the smaller of `LLM_TIMEOUT_MS` and the remaining total budget.
- `GUNICORN_TIMEOUT`: production worker watchdog in seconds, default `140`. This is not itself a per-request deadline. Development Compose runs Flask and does not use the Gunicorn watchdog.
- `AI_SERVICE_INTERNAL_TOKEN`: shared internal service token. When configured, all generation endpoints and course enrichment capabilities require `X-Internal-Service-Token`. Missing or incorrect tokens return `401` before contract selection or body validation. An unset/blank token preserves legacy development access; production must configure this token externally. `/health` remains public.

Course enrichment uses a dedicated HTTP/HTTPS adapter with verified TLS, no redirects,
debug output disabled, and a 1 MiB response-body limit. Its four daemon I/O workers
have bounded admission: blocked DNS retains its actual worker slot after the caller
times out; saturated admission fails promptly without attempting fallback models.
Cancellation shuts down the owned socket, including stalled HTTP reads or TLS
handshakes. Native DNS itself cannot be interrupted; after cancellation a late DNS
result cannot connect or send. The adapter does not use executor exit-time joins.
Other generation endpoints retain their existing requests transport.

The production timeout layers are **110 seconds** for the total Python enrichment
budget, **120 seconds** for the dedicated backend HTTP/job deadline, **140 seconds**
for the Gunicorn worker watchdog, and a **180-second** AI job lease. Short test budgets
are supported, but production overrides must keep these layers consistent.
`LLM_TIMEOUT_MS` remains a global per-attempt ceiling, clamped to the remaining total
enrichment budget; fallback models do not multiply or reset that budget. Metadata
parsing/validation is pure and bounded, not execution of generated content. Keep raw
provider diagnostics, request/response payloads and internal tokens out of logs.

Course enrichment errors have static bodies and stable codes: local missing/invalid
configuration returns `503 ENRICHMENT_CONFIGURATION_UNAVAILABLE`, total deadline
exhaustion returns `502 ENRICHMENT_DEADLINE_EXCEEDED`, and upstream, response-validation,
or temporary I/O-capacity failures return `502 ENRICHMENT_GENERATION_FAILED`. Successful
responses retain `items` and the executed contract version headers.

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

### `POST /course-content/enrich-knowledge-points`

Internal endpoint called by the backend when a teacher submits a course for review. It receives one course context and
up to 25 questions per request, then returns one concise `knowledgePoint` label per question. Requests must include the
`X-Internal-Service-Token` header when `AI_SERVICE_INTERNAL_TOKEN` is configured.

The JSON body remains strictly `course` and `questions` (at most 25 questions); version fields are not accepted in the body.
Select a contract using **both** `X-Enrichment-Prompt-Version` and `X-Enrichment-Response-Version`.
Both headers absent selects the legacy pair `("1", "1")`. Versions are canonical positive ASCII numeric strings
from `1` to `999999999`, without leading zeroes or whitespace. A partial, blank, or malformed pair returns
`400` with `{"error":"Invalid enrichment contract version headers"}`. A well-formed but unregistered tuple returns
`409` with `{"code":"ENRICHMENT_CONTRACT_UNSUPPORTED","error":"Unsupported enrichment contract"}`.
Selection occurs after token authentication and before body validation, so an unsupported tuple with `{}` still returns `409`.
Neither rejection calls the LLM. Successful responses echo the **executed** pair in those same two headers;
the `items` body is unchanged. Errors do not echo supplied version headers.

The initial registered tuple is exactly `("1", "1")`, using the existing Flask prompt builder, JSON parsing,
response validation and label normalization. Only explicitly registered tuples are supported: registering
`("1", "1")` and `("2", "2")` does not support `("1", "2")`. There is no latest-version fallback.
Registration is application startup code, never an HTTP API; the registry has no dependency on Flask or providers.

Contract versions are immutable across deployments: do not change a version's prompt or response schema/normalization
semantics in place. Register new versions for such changes and retain old pairs while backend jobs using them remain active.
Model routing/fallback order may remain configurable and outputs nondeterministic, but this must not change the prompt
or schema under an existing version. Before deployment, including AI-only updates,
verify the **target AI image and every replica** support all active exact pairs. A probe
of one current load-balanced URL is not proof. Retain required version implementations,
the backend decryption keyring and database queue through rollback. Active submissions
must finish under code compatible with their persisted envelopes and course locks.

The backend C2 capability gate caches compatibility for AI-only admission after application
readiness with workers/AI enabled: five-second GET, healthy refresh after 15 seconds,
failure backoff 5/10/30/60 seconds. Missing support pauses AI, not mail. Required contracts
are V1 union active database contracts supplied by `SubmissionRepository` from
`QUEUED`/`PROCESSING` attempts. Capabilities are not LLM-provider health checks. Runtime
unsupported tuples remain typed failures with no automatic version fallback; backend
finalization owns terminal state and unlock handling. Java and Python wire versions both use
`[1-9][0-9]{0,8}`; the local policy identifier belongs only to the backend.
Course background submission and UI acceptance/status integration are implemented.
The backend returns 202 for durable acceptance; generation is performed by registered
workers using the persisted tuple, not replayed through a synchronous endpoint.

For **every deployment, including AI-only image changes**, follow the
[metadata and cutover runbook](../infra/README.md#r5-queue-cutover-and-rollback).
Inventory nonterminal queue type/payload versions and active policy/prompt/response
tuples using operator-supplied current or isolated metadata. Compare the target backend
registry/local policy/response validators and the **target AI image plus every replica**
against the full required set, including V1. Probe each private instance directly;
a health check or one load-balanced capability response does not prove compatibility.
An incompatible or unverified target blocks cutover. Unknown work stays unclaimed and
observable through unsupported-version metrics; never relabel persisted versions to
force admission. Keep old prompt/response implementations and backend crypto keys.

Rollback first stops claims and drains bounded calls while retaining the queue, locks,
tables and persistent volumes. Use an image that understands active envelopes and locking;
finish or terminalize them through that implementation before considering older code.
Never use a synchronous generation bypass, queue deletion or historical replay to roll back.
The documented Windows/Docker Desktop evidence establishes no native Linux capacity,
1,000-user support, or adequacy of the production AI **384 MiB** container limit.

Response body:

```json
{
  "items": [
    {"questionId": 987, "knowledgePoint": "listening_dates"}
  ]
}
```

Errors:

- `400`: invalid JSON, missing required course/question fields, duplicate IDs, or a batch larger than 25 questions.
- `400`: partial, blank, or malformed version headers.
- `401`: configured internal token is missing or incorrect.
- `409`: well-formed contract tuple is unsupported (`ENRICHMENT_CONTRACT_UNSUPPORTED`).
- `503`: `LLM_API_KEY` or an LLM model is not configured.
- `502`: the LLM call fails or returns malformed, incomplete, duplicate, unknown, blank, or oversized metadata.

### `GET /internal/capabilities/course-enrichment`

Returns stable, numerically sorted **registered tuples**, with no model/provider/key/prompt metadata and no generation:

```json
{"supportedContracts":[{"promptVersion":"1","responseVersion":"1"}]}
```

Uses the same internal-token protection and empty-token development behavior as generation endpoints.
This is an internal service route on the existing service port; it does not add public exposure.
