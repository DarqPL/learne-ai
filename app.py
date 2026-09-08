import json
import os
import re

from flask import Flask, jsonify, request
import google.generativeai as genai


DEFAULT_RECOMMENDATION_REASON = "Recommended based on recent learning history."
GRAMMAR_ERROR_TYPES = {"GRAMMAR", "SPELLING", "PUNCTUATION", "WORD_CHOICE", "STYLE", "OTHER"}
MAX_GRAMMAR_INPUT_LENGTH = 2000
DEFAULT_MAX_GRAMMAR_ERRORS = 20
MAX_GRAMMAR_ERROR_FIELD_LENGTH = 500


def build_learning_path_prompt(payload):
    return f"""You are an AI tutor generating a learning path.

Return only valid JSON with this shape:
{{
  "weaknesses": ["short weakness label"],
  "recommendations": [
    {{"lessonId": 123, "score": 0.0, "reason": "short reason"}}
  ]
}}

Rules:
- lessonId must be a number from constraints.allowedLessonIds. Do not return lessonId as a string.
- score must be a number from 0 to 1.
- Do not include markdown, commentary, or extra keys.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def build_course_recommendation_prompt(payload):
    return f"""You are an AI tutor recommending courses from placement test results.

Return only valid JSON with this shape:
{{
  "weaknesses": ["short weakness label"],
  "recommendations": [
    {{"courseId": 123, "score": 0.0, "reason": "short reason"}}
  ]
}}

Rules:
- courseId must be a number from constraints.allowedCourseIds. Do not return courseId as a string.
- score must be a number from 0 to 1.
- Return multiple courses when multiple weaknesses map to different courses.
- Do not include markdown, commentary, or extra keys.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def build_grammar_check_prompt(payload):
    return f"""You are an English grammar checker.

Return only valid JSON with this shape:
{{
  "errors": [
    {{
      "errorText": "incorrect span from the original input",
      "errorType": "GRAMMAR",
      "suggestion": "correct replacement text",
      "explanation": "short explanation",
      "startPos": 0,
      "endPos": 5
    }}
  ]
}}

Rules:
- errorType must be one of constraints.allowedErrorTypes.
- startPos and endPos use zero-based, end-exclusive offsets for inputText.
- Return an empty errors array when the input is correct.
- Return at most constraints.maxErrors errors.
- Do not include markdown, commentary, correctedText, overallFeedback, or extra keys.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def parse_json_response(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini response was empty")

    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()

    return json.loads(stripped)


def _validate_payload(payload):
    if not isinstance(payload, dict):
        return "Request body must be a JSON object"

    candidate_lessons = payload.get("candidateLessons")
    if not isinstance(candidate_lessons, list):
        return "candidateLessons must be an array"
    if not candidate_lessons:
        return "candidateLessons must be a non-empty array"

    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        return "constraints must be an object"

    allowed_lesson_ids = constraints.get("allowedLessonIds")
    if not isinstance(allowed_lesson_ids, list):
        return "constraints.allowedLessonIds must be an array"
    if not allowed_lesson_ids:
        return "constraints.allowedLessonIds must be a non-empty array"

    return None


def _validate_course_payload(payload):
    if not isinstance(payload, dict):
        return "Request body must be a JSON object"

    candidate_courses = payload.get("candidateCourses")
    if not isinstance(candidate_courses, list):
        return "candidateCourses must be an array"
    if not candidate_courses:
        return "candidateCourses must be a non-empty array"

    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        return "constraints must be an object"

    allowed_course_ids = constraints.get("allowedCourseIds")
    if not isinstance(allowed_course_ids, list):
        return "constraints.allowedCourseIds must be an array"
    if not allowed_course_ids:
        return "constraints.allowedCourseIds must be a non-empty array"
    if any(isinstance(course_id, bool) or not isinstance(course_id, (int, float)) for course_id in allowed_course_ids):
        return "constraints.allowedCourseIds must contain only numbers"

    return None


def _validate_grammar_payload(payload):
    if not isinstance(payload, dict):
        return "Request body must be a JSON object"

    input_text = payload.get("inputText")
    if not isinstance(input_text, str) or not input_text.strip():
        return "inputText must be a non-empty string"
    if len(input_text.strip()) > MAX_GRAMMAR_INPUT_LENGTH:
        return "inputText must be at most 2000 characters"

    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        return "constraints must be an object"

    allowed_error_types = constraints.get("allowedErrorTypes")
    if not isinstance(allowed_error_types, list) or not allowed_error_types:
        return "constraints.allowedErrorTypes must be a non-empty array"
    if any(not isinstance(error_type, str) or error_type not in GRAMMAR_ERROR_TYPES for error_type in allowed_error_types):
        return "constraints.allowedErrorTypes contains an unsupported error type"

    max_errors = constraints.get("maxErrors", DEFAULT_MAX_GRAMMAR_ERRORS)
    if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1 or max_errors > DEFAULT_MAX_GRAMMAR_ERRORS:
        return "constraints.maxErrors must be an integer from 1 to 20"

    return None


def _filter_learning_path(parsed, allowed_lesson_ids):
    allowed = {(type(lesson_id), lesson_id) for lesson_id in allowed_lesson_ids}
    seen = set()
    recommendations = []

    for item in parsed.get("recommendations", []):
        if not isinstance(item, dict):
            continue

        lesson_id = item.get("lessonId")
        lesson_key = (type(lesson_id), lesson_id)
        score = item.get("score")
        if (
            lesson_key not in allowed
            or lesson_key in seen
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or score < 0
            or score > 1
        ):
            continue

        seen.add(lesson_key)
        reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
        recommendations.append(
            {
                "lessonId": lesson_id,
                "score": score,
                "reason": reason.strip() or DEFAULT_RECOMMENDATION_REASON,
            }
        )

    weaknesses = parsed.get("weaknesses", [])
    if not isinstance(weaknesses, list):
        weaknesses = []

    return {
        "weaknesses": [weakness for weakness in weaknesses if isinstance(weakness, str)],
        "recommendations": recommendations,
    }


def _filter_course_recommendations(parsed, allowed_course_ids):
    allowed = set(allowed_course_ids)
    seen = set()
    recommendations = []

    for item in parsed.get("recommendations", []):
        if not isinstance(item, dict):
            continue

        course_id = item.get("courseId")
        score = item.get("score")
        reason = item.get("reason")
        if (
            isinstance(course_id, bool)
            or not isinstance(course_id, (int, float))
            or course_id not in allowed
            or course_id in seen
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or score < 0
            or score > 1
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            continue

        seen.add(course_id)
        recommendations.append(
            {
                "courseId": course_id,
                "score": score,
                "reason": reason.strip(),
            }
        )

    weaknesses = parsed.get("weaknesses", [])
    if not isinstance(weaknesses, list):
        weaknesses = []

    return {
        "weaknesses": [weakness for weakness in weaknesses if isinstance(weakness, str)],
        "recommendations": recommendations,
    }


def _filter_grammar_errors(parsed, input_text, allowed_error_types, max_errors):
    errors = parsed.get("errors")
    if not isinstance(errors, list):
        return None

    filtered = []
    for item in errors:
        if not isinstance(item, dict):
            continue

        error_text = item.get("errorText")
        error_type = item.get("errorType")
        suggestion = item.get("suggestion")
        explanation = item.get("explanation")
        start_pos = item.get("startPos")
        end_pos = item.get("endPos")
        if (
            not isinstance(error_text, str)
            or not error_text.strip()
            or len(error_text.strip()) > MAX_GRAMMAR_ERROR_FIELD_LENGTH
            or not isinstance(error_type, str)
            or error_type not in allowed_error_types
            or not isinstance(suggestion, str)
            or not suggestion.strip()
            or len(suggestion.strip()) > MAX_GRAMMAR_ERROR_FIELD_LENGTH
            or not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation.strip()) > MAX_GRAMMAR_ERROR_FIELD_LENGTH
            or isinstance(start_pos, bool)
            or not isinstance(start_pos, int)
            or isinstance(end_pos, bool)
            or not isinstance(end_pos, int)
            or start_pos < 0
            or end_pos <= start_pos
            or end_pos > len(input_text)
            or input_text[start_pos:end_pos] != error_text
        ):
            continue

        filtered.append(
            {
                "errorText": error_text.strip(),
                "errorType": error_type,
                "suggestion": suggestion.strip(),
                "explanation": explanation.strip(),
                "startPos": start_pos,
                "endPos": end_pos,
            }
        )

    filtered.sort(key=lambda item: (item["startPos"], item["endPos"]))
    return {"errors": filtered[:max_errors]}


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/learning-path/generate")
    def generate_learning_path():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        validation_error = _validate_payload(payload)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY is required for generation"}), 503

        prompt = build_learning_path_prompt(payload)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            parsed = parse_json_response(getattr(response, "text", ""))
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Gemini returned invalid JSON: {exc.msg}"}), 502
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:
            return jsonify({"error": "Gemini generation failed"}), 502

        if not isinstance(parsed, dict):
            return jsonify({"error": "Gemini returned invalid response shape"}), 502

        learning_path = _filter_learning_path(parsed, payload["constraints"]["allowedLessonIds"])
        if not learning_path["recommendations"]:
            return jsonify({"error": "Gemini returned no valid recommendations"}), 502

        return jsonify(learning_path)

    @app.post("/course-recommendations/generate")
    def generate_course_recommendations():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        validation_error = _validate_course_payload(payload)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY is required for generation"}), 503

        prompt = build_course_recommendation_prompt(payload)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            parsed = parse_json_response(getattr(response, "text", ""))
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Gemini returned invalid JSON: {exc.msg}"}), 502
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception as exc:
            return jsonify({"error": "Gemini generation failed"}), 502

        if not isinstance(parsed, dict) or not isinstance(parsed.get("recommendations"), list):
            return jsonify({"error": "Gemini returned invalid response shape"}), 502

        course_recommendations = _filter_course_recommendations(parsed, payload["constraints"]["allowedCourseIds"])
        if not course_recommendations["recommendations"]:
            return jsonify({"error": "Gemini returned no valid recommendations"}), 502

        return jsonify(course_recommendations)

    @app.post("/grammar-checks/check")
    def check_grammar():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        validation_error = _validate_grammar_payload(payload)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY is required for generation"}), 503

        input_text = payload["inputText"]
        prompt = build_grammar_check_prompt(payload)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            parsed = parse_json_response(getattr(response, "text", ""))
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"Gemini returned invalid JSON: {exc.msg}"}), 502
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception:
            return jsonify({"error": "Gemini generation failed"}), 502

        if not isinstance(parsed, dict):
            return jsonify({"error": "Gemini returned invalid response shape"}), 502

        grammar_result = _filter_grammar_errors(
            parsed,
            input_text,
            set(payload["constraints"]["allowedErrorTypes"]),
            payload["constraints"].get("maxErrors", DEFAULT_MAX_GRAMMAR_ERRORS),
        )
        if grammar_result is None:
            return jsonify({"error": "Gemini returned invalid response shape"}), 502

        return jsonify(grammar_result)

    return app


app = create_app()
