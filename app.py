import json
import logging
import os
import re

from flask import Flask, jsonify, request
import requests


DEFAULT_RECOMMENDATION_REASON = "Recommended based on recent learning history."
GRAMMAR_ERROR_TYPES = {"GRAMMAR", "SPELLING", "PUNCTUATION", "WORD_CHOICE", "STYLE", "OTHER"}
MAX_GRAMMAR_INPUT_LENGTH = 2000
DEFAULT_MAX_GRAMMAR_ERRORS = 20
MAX_GRAMMAR_ERROR_FIELD_LENGTH = 500
AI_TUTOR_PRACTICE_TYPES = {"FREE_CHAT", "ROLEPLAY", "WRITING_PRACTICE"}
AI_TUTOR_FEEDBACK_TYPES = {"GRAMMAR", "VOCABULARY", "WORD_CHOICE", "STYLE", "OTHER"}
MAX_AI_TUTOR_TITLE_LENGTH = 120
MAX_AI_TUTOR_MESSAGE_LENGTH = 2000
MAX_AI_TUTOR_RECENT_MESSAGE_LENGTH = 4000
MAX_AI_TUTOR_REPLY_LENGTH = 4000
MAX_AI_TUTOR_FEEDBACK_LENGTH = 1000
MAX_AI_TUTOR_RECENT_MESSAGES = 10
LLM_TASK_MODEL_ENV = {
    "grammar": "LLM_GRAMMAR_MODEL",
    "ai_tutor": "LLM_AI_TUTOR_MODEL",
    "learning_path": "LLM_LEARNING_PATH_MODEL",
    "course_recommendation": "LLM_COURSE_RECOMMENDATION_MODEL",
}
LLM_TASK_FALLBACK_ENV = {
    "grammar": "LLM_GRAMMAR_FALLBACK_MODELS",
    "ai_tutor": "LLM_AI_TUTOR_FALLBACK_MODELS",
    "learning_path": "LLM_LEARNING_PATH_FALLBACK_MODELS",
    "course_recommendation": "LLM_COURSE_RECOMMENDATION_FALLBACK_MODELS",
}
DEFAULT_LLM_BASE_URL = "http://9router:20128/v1"
DEFAULT_LLM_TIMEOUT_MS = 8000
DEFAULT_LLM_TEMPERATURE = 0.2
logger = logging.getLogger(__name__)


class LlmConfigurationError(RuntimeError):
    pass


class LlmGenerationError(RuntimeError):
    pass


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


def build_ai_tutor_prompt(payload):
    return f"""You are a friendly native speaker helping an English learner practice.

Return only valid JSON with this shape:
{{
  "replyText": "natural friendly reply that continues the conversation",
  "feedback": {{
    "errorType": "GRAMMAR",
    "originalText": "student phrase with an issue",
    "correctedText": "natural corrected phrase",
    "explanation": "brief supportive explanation"
  }}
}}

Rules:
- Continue the conversation first; do not sound like a strict examiner.
- feedback may be null when no correction is useful.
- practiceType FREE_CHAT means natural conversation.
- practiceType ROLEPLAY means stay in the scenario from title.
- practiceType WRITING_PRACTICE means respond to the writing helpfully and gently.
- feedback.errorType must be one of GRAMMAR, VOCABULARY, WORD_CHOICE, STYLE, OTHER.
- Do not include markdown, commentary, or extra keys.

Input:
{json.dumps(payload, ensure_ascii=False)}
"""


def resolve_model(task_name):
    task_model_env = LLM_TASK_MODEL_ENV.get(task_name)
    if task_model_env:
        task_model = os.environ.get(task_model_env, "").strip()
        if task_model:
            return task_model

    default_model = os.environ.get("LLM_DEFAULT_MODEL", "").strip()
    if default_model:
        return default_model

    return None


def _append_unique_model(models, model):
    normalized = model.strip()
    if normalized and normalized not in models:
        models.append(normalized)


def resolve_models(task_name):
    models = []
    primary_model = resolve_model(task_name)
    if primary_model:
        _append_unique_model(models, primary_model)

    fallback_env = LLM_TASK_FALLBACK_ENV.get(task_name)
    if fallback_env:
        for model in os.environ.get(fallback_env, "").split(","):
            _append_unique_model(models, model)

    return models


def _llm_timeout_seconds():
    raw_timeout = os.environ.get("LLM_TIMEOUT_MS", str(DEFAULT_LLM_TIMEOUT_MS)).strip()
    try:
        timeout_ms = int(raw_timeout)
    except ValueError:
        timeout_ms = DEFAULT_LLM_TIMEOUT_MS
    if timeout_ms <= 0:
        timeout_ms = DEFAULT_LLM_TIMEOUT_MS
    return timeout_ms / 1000


def _llm_chat_completions_url():
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip().rstrip("/")
    return f"{base_url}/chat/completions"


def generate_llm_text(prompt, task_name, model=None):
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise LlmConfigurationError("LLM_API_KEY is required for generation")

    selected_model = model or resolve_model(task_name)
    if not selected_model:
        raise LlmConfigurationError("LLM model is required for generation")

    try:
        response = requests.post(
            _llm_chat_completions_url(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": selected_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": DEFAULT_LLM_TEMPERATURE,
            },
            timeout=_llm_timeout_seconds(),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise LlmGenerationError("LLM generation failed") from exc
    except ValueError as exc:
        raise LlmGenerationError("LLM generation failed") from exc

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmGenerationError("LLM returned invalid response shape") from exc

    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM response was empty")

    return text


def generate_llm_json(prompt, task_name):
    models = resolve_models(task_name)
    if not models:
        raise LlmConfigurationError("LLM model is required for generation")

    last_error = None
    for model in models:
        try:
            text = generate_llm_text(prompt, task_name, model=model)
            return parse_json_response(text)
        except (json.JSONDecodeError, ValueError, LlmGenerationError) as exc:
            last_error = exc
            logger.warning("LLM model attempt failed task=%s model=%s error=%s", task_name, model, str(exc))

    if isinstance(last_error, json.JSONDecodeError):
        raise last_error
    if isinstance(last_error, ValueError):
        raise last_error
    if isinstance(last_error, LlmGenerationError):
        raise last_error
    raise LlmGenerationError("LLM generation failed")


def generate_valid_llm_result(prompt, task_name, validate_result):
    models = resolve_models(task_name)
    if not models:
        raise LlmConfigurationError("LLM model is required for generation")

    last_error = None
    for model in models:
        try:
            parsed = parse_json_response(generate_llm_text(prompt, task_name, model=model))
            return validate_result(parsed)
        except (json.JSONDecodeError, ValueError, LlmGenerationError) as exc:
            last_error = exc
            logger.warning("LLM model attempt failed task=%s model=%s error=%s", task_name, model, str(exc))

    if isinstance(last_error, json.JSONDecodeError):
        raise last_error
    if isinstance(last_error, ValueError):
        raise last_error
    if isinstance(last_error, LlmGenerationError):
        raise last_error
    raise LlmGenerationError("LLM generation failed")


def parse_json_response(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM response was empty")

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


def _validate_ai_tutor_payload(payload):
    if not isinstance(payload, dict):
        return "Request body must be a JSON object"

    if payload.get("practiceType") not in AI_TUTOR_PRACTICE_TYPES:
        return "practiceType is invalid"

    title = payload.get("title")
    if not isinstance(title, str) or not title.strip() or len(title.strip()) > MAX_AI_TUTOR_TITLE_LENGTH:
        return "title must be a non-empty string up to 120 characters"

    latest_message = payload.get("latestMessage")
    if (
        not isinstance(latest_message, str)
        or not latest_message.strip()
        or len(latest_message.strip()) > MAX_AI_TUTOR_MESSAGE_LENGTH
    ):
        return "latestMessage must be a non-empty string up to 2000 characters"

    recent_messages = payload.get("recentMessages")
    if not isinstance(recent_messages, list) or len(recent_messages) > MAX_AI_TUTOR_RECENT_MESSAGES:
        return "recentMessages must be an array with at most 10 items"

    for item in recent_messages:
        if not isinstance(item, dict):
            return "recentMessages contains an invalid message"
        message = item.get("message")
        if (
            item.get("senderType") not in {"USER", "AI"}
            or not isinstance(message, str)
            or not message.strip()
            or len(message.strip()) > MAX_AI_TUTOR_RECENT_MESSAGE_LENGTH
        ):
            return "recentMessages contains an invalid message"

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


def _filter_ai_tutor_response(parsed):
    if not isinstance(parsed, dict):
        return None

    reply_text = parsed.get("replyText")
    if not isinstance(reply_text, str) or not reply_text.strip() or len(reply_text.strip()) > MAX_AI_TUTOR_REPLY_LENGTH:
        return None

    feedback = parsed.get("feedback")
    normalized_feedback = None
    if isinstance(feedback, dict):
        error_type = feedback.get("errorType")
        original_text = feedback.get("originalText")
        corrected_text = feedback.get("correctedText")
        explanation = feedback.get("explanation")
        if (
            isinstance(error_type, str)
            and error_type in AI_TUTOR_FEEDBACK_TYPES
            and isinstance(original_text, str)
            and original_text.strip()
            and len(original_text.strip()) <= MAX_AI_TUTOR_FEEDBACK_LENGTH
            and isinstance(corrected_text, str)
            and corrected_text.strip()
            and len(corrected_text.strip()) <= MAX_AI_TUTOR_FEEDBACK_LENGTH
            and isinstance(explanation, str)
            and explanation.strip()
            and len(explanation.strip()) <= MAX_AI_TUTOR_FEEDBACK_LENGTH
        ):
            normalized_feedback = {
                "errorType": error_type,
                "originalText": original_text.strip(),
                "correctedText": corrected_text.strip(),
                "explanation": explanation.strip(),
            }

    return {"replyText": reply_text.strip(), "feedback": normalized_feedback}


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

        prompt = build_learning_path_prompt(payload)

        def validate_learning_path(parsed):
            if not isinstance(parsed, dict):
                raise LlmGenerationError("LLM returned invalid response shape")
            learning_path = _filter_learning_path(parsed, payload["constraints"]["allowedLessonIds"])
            if not learning_path["recommendations"]:
                raise LlmGenerationError("LLM returned no valid recommendations")
            return learning_path

        try:
            learning_path = generate_valid_llm_result(prompt, "learning_path", validate_learning_path)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"LLM returned invalid JSON: {exc.msg}"}), 502
        except LlmConfigurationError as exc:
            return jsonify({"error": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except LlmGenerationError as exc:
            return jsonify({"error": str(exc)}), 502

        return jsonify(learning_path)

    @app.post("/course-recommendations/generate")
    def generate_course_recommendations():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        validation_error = _validate_course_payload(payload)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        prompt = build_course_recommendation_prompt(payload)

        def validate_course_recommendations(parsed):
            if not isinstance(parsed, dict) or not isinstance(parsed.get("recommendations"), list):
                raise LlmGenerationError("LLM returned invalid response shape")
            course_recommendations = _filter_course_recommendations(parsed, payload["constraints"]["allowedCourseIds"])
            if not course_recommendations["recommendations"]:
                raise LlmGenerationError("LLM returned no valid recommendations")
            return course_recommendations

        try:
            course_recommendations = generate_valid_llm_result(
                prompt, "course_recommendation", validate_course_recommendations
            )
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"LLM returned invalid JSON: {exc.msg}"}), 502
        except LlmConfigurationError as exc:
            return jsonify({"error": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except LlmGenerationError as exc:
            return jsonify({"error": str(exc)}), 502

        return jsonify(course_recommendations)

    @app.post("/grammar-checks/check")
    def check_grammar():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        validation_error = _validate_grammar_payload(payload)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        input_text = payload["inputText"]
        prompt = build_grammar_check_prompt(payload)

        def validate_grammar(parsed):
            if not isinstance(parsed, dict):
                raise LlmGenerationError("LLM returned invalid response shape")
            grammar_result = _filter_grammar_errors(
                parsed,
                input_text,
                set(payload["constraints"]["allowedErrorTypes"]),
                payload["constraints"].get("maxErrors", DEFAULT_MAX_GRAMMAR_ERRORS),
            )
            if grammar_result is None:
                raise LlmGenerationError("LLM returned invalid response shape")
            return grammar_result

        try:
            grammar_result = generate_valid_llm_result(prompt, "grammar", validate_grammar)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"LLM returned invalid JSON: {exc.msg}"}), 502
        except LlmConfigurationError as exc:
            return jsonify({"error": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except LlmGenerationError as exc:
            return jsonify({"error": str(exc)}), 502

        return jsonify(grammar_result)

    @app.post("/ai-tutor/respond")
    def ai_tutor_respond():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        validation_error = _validate_ai_tutor_payload(payload)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        prompt = build_ai_tutor_prompt(payload)

        def validate_tutor(parsed):
            tutor_result = _filter_ai_tutor_response(parsed)
            if tutor_result is None:
                raise LlmGenerationError("LLM returned invalid response shape")
            return tutor_result

        try:
            tutor_result = generate_valid_llm_result(prompt, "ai_tutor", validate_tutor)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"LLM returned invalid JSON: {exc.msg}"}), 502
        except LlmConfigurationError as exc:
            return jsonify({"error": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        except LlmGenerationError as exc:
            return jsonify({"error": str(exc)}), 502

        return jsonify(tutor_result)

    return app


app = create_app()
