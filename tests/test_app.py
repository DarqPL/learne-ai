import json

import pytest

import app as ai_app
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


def test_course_recommendation_prompt_requires_numeric_course_id_and_multiple_courses():
    prompt = ai_app.build_course_recommendation_prompt({"constraints": {"allowedCourseIds": [10, 11]}})

    assert '"courseId": 123' in prompt
    assert "courseId must be a number from constraints.allowedCourseIds" in prompt
    assert "Do not return courseId as a string" in prompt
    assert "Return multiple courses when multiple weaknesses map to different courses" in prompt
    assert "Do not include markdown, commentary, or extra keys" in prompt


def test_generate_rejects_invalid_candidate_payload(client):
    response = client.post(
        "/learning-path/generate",
        json={"candidateLessons": "bad", "constraints": {"allowedLessonIds": ["l1"]}},
    )

    assert response.status_code == 400
    assert "candidateLessons" in response.get_json()["error"]


def test_generate_course_recommendations_rejects_invalid_payload(client):
    response = client.post(
        "/course-recommendations/generate",
        json={"candidateCourses": [], "constraints": {"allowedCourseIds": [10]}},
    )

    assert response.status_code == 400
    assert "candidateCourses" in response.get_json()["error"]


def test_generate_course_recommendations_rejects_non_numeric_allowed_ids(client):
    response = client.post(
        "/course-recommendations/generate",
        json={"candidateCourses": [{"courseId": 10}], "constraints": {"allowedCourseIds": ["10"]}},
    )

    assert response.status_code == 400
    assert "constraints.allowedCourseIds" in response.get_json()["error"]


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


def test_generate_course_recommendations_filters_invalid_and_duplicate_courses(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "weaknesses": ["present simple"],
                "recommendations": [
                    {"courseId": 10, "score": 0.94, "reason": "Grammar fit"},
                    {"courseId": "10", "score": 0.9, "reason": "Wrong type"},
                    {"courseId": 10, "score": 0.8, "reason": "Duplicate"},
                    {"courseId": 11, "score": 2, "reason": "Bad score"},
                    {"courseId": 12, "score": 0.7, "reason": "Not allowed"},
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
        "/course-recommendations/generate",
        json={
            "candidateCourses": [{"courseId": 10, "title": "Grammar"}, {"courseId": 11, "title": "Vocabulary"}],
            "constraints": {"allowedCourseIds": [10, 11]},
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "weaknesses": ["present simple"],
        "recommendations": [{"courseId": 10, "score": 0.94, "reason": "Grammar fit"}],
    }


@pytest.mark.parametrize("recommendations", [None, {"courseId": 10}, "bad"])
def test_generate_course_recommendations_returns_bad_gateway_for_malformed_recommendations(
    client, monkeypatch, recommendations
):
    class FakeResponse:
        text = json.dumps({"weaknesses": ["present simple"], "recommendations": recommendations})

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/course-recommendations/generate",
        json={
            "candidateCourses": [{"courseId": 10, "title": "Grammar"}],
            "constraints": {"allowedCourseIds": [10]},
        },
    )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Gemini returned invalid response shape"}


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


def test_grammar_check_rejects_missing_json(client):
    response = client.post("/grammar-checks/check", data="not-json")

    assert response.status_code == 400
    assert "JSON" in response.get_json()["error"]


def test_grammar_check_rejects_blank_input(client):
    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "   ", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 400
    assert "inputText" in response.get_json()["error"]


def test_grammar_check_requires_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.get_json()["error"]


def test_grammar_check_rejects_invalid_constraints(client):
    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["BAD_TYPE"], "maxErrors": 20}},
    )

    assert response.status_code == 400
    assert "allowedErrorTypes" in response.get_json()["error"]


def test_grammar_check_returns_valid_errors(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {
                        "errorText": "want order",
                        "errorType": "GRAMMAR",
                        "suggestion": "want to order",
                        "explanation": "After want, use to plus verb.",
                        "startPos": 2,
                        "endPos": 12,
                    }
                ]
            }
        )

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            assert "Do not include markdown" in prompt
            assert "correctedText" in prompt
            assert "overallFeedback" in prompt
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/grammar-checks/check",
        json={
            "inputText": "I want order pizza.",
            "constraints": {"allowedErrorTypes": ["GRAMMAR", "SPELLING"], "maxErrors": 20},
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "errors": [
            {
                "errorText": "want order",
                "errorType": "GRAMMAR",
                "suggestion": "want to order",
                "explanation": "After want, use to plus verb.",
                "startPos": 2,
                "endPos": 12,
            }
        ]
    }


def test_grammar_check_accepts_empty_errors(client, monkeypatch):
    class FakeResponse:
        text = json.dumps({"errors": []})

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "This sentence is correct.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {"errors": []}


def test_grammar_check_accepts_fenced_json(client, monkeypatch):
    class FakeResponse:
        text = """```json
        {"errors": []}
        ```"""

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "This sentence is correct.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {"errors": []}


def test_grammar_check_filters_invalid_items_sorts_and_limits(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {
                        "errorText": "pizza yesterday",
                        "errorType": "GRAMMAR",
                        "suggestion": "pizza yesterday?",
                        "explanation": "Question punctuation is missing.",
                        "startPos": 13,
                        "endPos": 28,
                    },
                    {"errorText": "bad", "errorType": "BAD_TYPE", "suggestion": "x", "explanation": "x", "startPos": 0, "endPos": 3},
                    {"errorText": "bad", "errorType": "GRAMMAR", "suggestion": "x", "explanation": "x", "startPos": -1, "endPos": 3},
                    {"errorText": "bad", "errorType": "GRAMMAR", "suggestion": "x", "explanation": "x", "startPos": 0, "endPos": 99},
                    {"errorText": "bad", "errorType": "GRAMMAR", "suggestion": "x", "explanation": "x", "startPos": True, "endPos": 3},
                    {"errorText": "want order", "errorType": "GRAMMAR", "suggestion": "want to order", "explanation": "Use to.", "startPos": 2, "endPos": 12},
                ]
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
        "/grammar-checks/check",
        json={"inputText": "I want order pizza yesterday.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 1}},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "errors": [
            {
                "errorText": "want order",
                "errorType": "GRAMMAR",
                "suggestion": "want to order",
                "explanation": "Use to.",
                "startPos": 2,
                "endPos": 12,
            }
        ]
    }


def test_grammar_check_filters_oversized_explanation(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {
                        "errorText": "want order",
                        "errorType": "GRAMMAR",
                        "suggestion": "want to order",
                        "explanation": "x" * 501,
                        "startPos": 2,
                        "endPos": 12,
                    }
                ]
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
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {"errors": []}


def test_grammar_check_filters_oversized_error_text_and_suggestion(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {
                        "errorText": "x" * 501,
                        "errorType": "GRAMMAR",
                        "suggestion": "y",
                        "explanation": "Too long original span.",
                        "startPos": 0,
                        "endPos": 501,
                    },
                    {
                        "errorText": "bad",
                        "errorType": "GRAMMAR",
                        "suggestion": "x" * 501,
                        "explanation": "Too long suggestion.",
                        "startPos": 502,
                        "endPos": 505,
                    },
                ]
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
        "/grammar-checks/check",
        json={"inputText": f"{'x' * 501} bad", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {"errors": []}


def test_grammar_check_rejects_too_long_input(client):
    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "x" * 2001, "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 400
    assert "inputText" in response.get_json()["error"]


@pytest.mark.parametrize("errors", [None, {"errorText": "bad"}, "bad"])
def test_grammar_check_returns_bad_gateway_for_missing_or_malformed_errors(client, monkeypatch, errors):
    class FakeResponse:
        text = json.dumps({} if errors is None else {"errors": errors})

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Gemini returned invalid response shape"}


def test_grammar_check_returns_bad_gateway_when_gemini_generation_fails(client, monkeypatch):
    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            raise RuntimeError("provider secret detail")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Gemini generation failed"}


def test_grammar_check_rejects_unhashable_allowed_error_type(client):
    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": [{}], "maxErrors": 20}},
    )

    assert response.status_code == 400
    assert "allowedErrorTypes" in response.get_json()["error"]


def test_grammar_check_filters_unhashable_and_non_string_gemini_error_types(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {"errorText": "bad", "errorType": {}, "suggestion": "x", "explanation": "x", "startPos": 0, "endPos": 3},
                    {"errorText": "bad", "errorType": 123, "suggestion": "x", "explanation": "x", "startPos": 0, "endPos": 3},
                    {
                        "errorText": "want order",
                        "errorType": "GRAMMAR",
                        "suggestion": "want to order",
                        "explanation": "Use to.",
                        "startPos": 2,
                        "endPos": 12,
                    },
                ]
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
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "errors": [
            {
                "errorText": "want order",
                "errorType": "GRAMMAR",
                "suggestion": "want to order",
                "explanation": "Use to.",
                "startPos": 2,
                "endPos": 12,
            }
        ]
    }


def test_grammar_check_preserves_leading_space_offsets(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {
                        "errorText": "want order",
                        "errorType": "GRAMMAR",
                        "suggestion": "want to order",
                        "explanation": "Use to.",
                        "startPos": 4,
                        "endPos": 14,
                    }
                ]
            }
        )

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            assert '"inputText": "  I want order pizza."' in prompt
            return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("app.genai.configure", lambda api_key: None)
    monkeypatch.setattr("app.genai.GenerativeModel", FakeModel)

    response = client.post(
        "/grammar-checks/check",
        json={"inputText": "  I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "errors": [
            {
                "errorText": "want order",
                "errorType": "GRAMMAR",
                "suggestion": "want to order",
                "explanation": "Use to.",
                "startPos": 4,
                "endPos": 14,
            }
        ]
    }


def test_grammar_check_filters_mismatched_error_text_and_span(client, monkeypatch):
    class FakeResponse:
        text = json.dumps(
            {
                "errors": [
                    {
                        "errorText": "want order",
                        "errorType": "GRAMMAR",
                        "suggestion": "want to order",
                        "explanation": "Use to.",
                        "startPos": 0,
                        "endPos": 10,
                    },
                    {
                        "errorText": "order",
                        "errorType": "GRAMMAR",
                        "suggestion": "to order",
                        "explanation": "Use to.",
                        "startPos": 7,
                        "endPos": 12,
                    },
                ]
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
        "/grammar-checks/check",
        json={"inputText": "I want order pizza.", "constraints": {"allowedErrorTypes": ["GRAMMAR"], "maxErrors": 20}},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "errors": [
            {
                "errorText": "order",
                "errorType": "GRAMMAR",
                "suggestion": "to order",
                "explanation": "Use to.",
                "startPos": 7,
                "endPos": 12,
            }
        ]
    }
