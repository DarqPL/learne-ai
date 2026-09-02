import json
import os
import re

from flask import Flask, jsonify, request
import google.generativeai as genai


DEFAULT_RECOMMENDATION_REASON = "Recommended based on recent learning history."


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

    return app


app = create_app()
