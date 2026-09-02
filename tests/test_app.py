import json

import pytest

from app import build_learning_path_prompt, create_app, parse_json_response


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_generate_rejects_missing_json(client):
    response = client.post("/learning-path/generate", data="not-json")

    assert response.status_code == 400
    assert "JSON" in response.get_json()["error"]


def test_parse_json_response_strips_fenced_json():
    payload = parse_json_response(
        """```json
        {"weaknesses": ["loops"], "recommendations": []}
        ```"""
    )

    assert payload == {"weaknesses": ["loops"], "recommendations": []}


def test_prompt_requires_numeric_lesson_id():
    prompt = build_learning_path_prompt({"constraints": {"allowedLessonIds": [12]}})

    assert '"lessonId": 123' in prompt
    assert "Do not return lessonId as a string" in prompt


def test_generate_rejects_invalid_candidate_payload(client):
    response = client.post(
        "/learning-path/generate",
        json={"candidateLessons": "bad", "constraints": {"allowedLessonIds": ["l1"]}},
    )

    assert response.status_code == 400
    assert "candidateLessons" in response.get_json()["error"]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"constraints": {"allowedLessonIds": ["l1"]}}, "candidateLessons"),
        ({"candidateLessons": [], "constraints": {"allowedLessonIds": ["l1"]}}, "candidateLessons"),
        ({"candidateLessons": [{"id": "l1"}], "constraints": {}}, "constraints.allowedLessonIds"),
        (
            {"candidateLessons": [{"id": "l1"}], "constraints": {"allowedLessonIds": []}},
            "constraints.allowedLessonIds",
        ),
    ],
)
def test_generate_rejects_missing_or_empty_required_arrays(client, payload, expected_error):
    response = client.post("/learning-path/generate", json=payload)

    assert response.status_code == 400
    assert expected_error in response.get_json()["error"]


def test_generate_filters_invalid_and_duplicate_recommendations(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "weaknesses": ["arrays"],
                "recommendations": [
                    {"lessonId": 12, "score": 0.8, "reason": "Practice arrays"},
                    {"lessonId": "12", "score": 0.75, "reason": "Stringified numeric ID"},
                    {"lessonId": "l2", "score": 0.7, "reason": "Not allowed"},
                    {"lessonId": 12, "score": 0.6, "reason": "Duplicate"},
                    {"lessonId": "l3", "score": 2, "reason": "Bad score"},
                    {"lessonId": "l4", "score": 0.5},
                    {"lessonId": "l5", "score": 0.4, "reason": "   "},
                ],
            }
        )

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/learning-path/generate",
        json={
            "candidateLessons": [
                {"id": 12, "title": "Arrays"},
                {"id": "l2", "title": "Loops"},
                {"id": "l3", "title": "Functions"},
                {"id": "l4", "title": "Objects"},
                {"id": "l5", "title": "Classes"},
            ],
            "constraints": {"allowedLessonIds": [12, "l4", "l5"]},
        },
    )

    assert response.status_code == 200
    response_json = response.get_json()
    assert isinstance(response_json["recommendations"][0]["lessonId"], int)
    assert response_json == {
        "weaknesses": ["arrays"],
        "recommendations": [
            {"lessonId": 12, "score": 0.8, "reason": "Practice arrays"},
            {
                "lessonId": "l4",
                "score": 0.5,
                "reason": "Recommended based on recent learning history.",
            },
            {
                "lessonId": "l5",
                "score": 0.4,
                "reason": "Recommended based on recent learning history.",
            },
        ],
    }


def test_generate_returns_bad_gateway_when_filtering_removes_all_recommendations(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "weaknesses": ["arrays"],
                "recommendations": [
                    {"lessonId": "l2", "score": 0.8, "reason": "Not allowed"},
                    {"lessonId": "l1", "score": 2, "reason": "Bad score"},
                ],
            }
        )

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/learning-path/generate",
        json={
            "candidateLessons": [{"id": "l1", "title": "Arrays"}],
            "constraints": {"allowedLessonIds": ["l1"]},
        },
    )

    assert response.status_code == 502
    assert "valid recommendations" in response.get_json()["error"]


def test_generate_hides_raw_gemini_exception_details(client, monkeypatch):
    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            raise RuntimeError("secret upstream token detail")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/learning-path/generate",
        json={
            "candidateLessons": [{"id": "l1", "title": "Arrays"}],
            "constraints": {"allowedLessonIds": ["l1"]},
        },
    )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Gemini generation failed"}


def test_generate_returns_bad_gateway_when_gemini_json_is_not_object(client, monkeypatch):
    class FakeResponse:
        text = '[{"lessonId": "l1", "score": 0.8}]'

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/learning-path/generate",
        json={
            "candidateLessons": [{"id": "l1", "title": "Arrays"}],
            "constraints": {"allowedLessonIds": ["l1"]},
        },
    )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Gemini returned invalid response shape"}
