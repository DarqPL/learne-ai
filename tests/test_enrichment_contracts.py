import json

import pytest

import app as ai_app
from test_app import _course_content_payload


ENDPOINT = "/course-content/enrich-knowledge-points"
CAPABILITIES = "/internal/capabilities/course-enrichment"
PROMPT_HEADER = "X-Enrichment-Prompt-Version"
RESPONSE_HEADER = "X-Enrichment-Response-Version"
INVALID_ERROR = {"error": "Invalid enrichment contract version headers"}
UNSUPPORTED_ERROR = {
    "code": "ENRICHMENT_CONTRACT_UNSUPPORTED",
    "error": "Unsupported enrichment contract",
}


def headers(prompt="1", response="1"):
    return {PROMPT_HEADER: prompt, RESPONSE_HEADER: response}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network or generation call")

    monkeypatch.delenv("AI_SERVICE_INTERNAL_TOKEN", raising=False)
    monkeypatch.setattr(ai_app.requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(ai_app, "generate_llm_text", forbidden)
    monkeypatch.setattr(ai_app, "generate_valid_llm_result", forbidden)


@pytest.fixture
def client():
    application = ai_app.create_app()
    application.config.update(TESTING=True)
    return application.test_client()


@pytest.mark.parametrize("selected_headers", [{}, headers()])
def test_default_and_explicit_contract_echo_executed_versions(client, monkeypatch, selected_headers):
    payload = _course_content_payload()
    calls = []

    def generate(prompt, task, validator, deadline=None):
        assert deadline is not None
        calls.append((prompt, task))
        return validator({"items": [{"questionId": 101, "knowledgePoint": "Listening Dates"}]})

    monkeypatch.setattr(ai_app, "generate_valid_llm_result", generate)
    response = client.post(ENDPOINT, json=payload, headers=selected_headers)
    assert response.status_code == 200
    assert response.get_json() == {"items": [{"questionId": 101, "knowledgePoint": "listening_dates"}]}
    assert response.headers[PROMPT_HEADER] == "1"
    assert response.headers[RESPONSE_HEADER] == "1"
    # Flask's test JSON transport sorts object keys before the handler receives them.
    transported_payload = json.loads(client.application.json.dumps(payload))
    assert calls == [(ai_app.build_course_content_prompt(transported_payload), "course_content")]


@pytest.mark.parametrize("selected_headers", [
    {PROMPT_HEADER: "1"}, {RESPONSE_HEADER: "1"}, headers("", "1"),
    headers("1", ""), headers(" ", "1"), headers("01", "1"),
    headers("0", "1"), headers("-1", "1"), headers("1.0", "1"),
    headers("1", "latest"), headers("1, 2", "1"), headers("1", "١"),
    headers("1234567890", "1"), headers(" 1", "1"),
])
def test_invalid_headers_rejected_before_body_and_generation(client, selected_headers):
    response = client.post(ENDPOINT, json={}, headers=selected_headers)
    assert response.status_code == 400
    assert response.get_json() == INVALID_ERROR
    assert PROMPT_HEADER not in response.headers
    assert RESPONSE_HEADER not in response.headers


@pytest.mark.parametrize("payload", [{}, _course_content_payload()])
def test_unsupported_exact_pair_before_body_validation(client, payload):
    response = client.post(ENDPOINT, json=payload, headers=headers("999", "1"))
    assert response.status_code == 409
    assert response.get_json() == UNSUPPORTED_ERROR
    assert PROMPT_HEADER not in response.headers
    assert RESPONSE_HEADER not in response.headers


def test_registry_exact_pairs_and_stable_detached_capabilities():
    from enrichment_contracts import EnrichmentContractRegistry

    registry = EnrichmentContractRegistry()
    build = lambda payload: "prompt"
    validate = lambda parsed, ids: parsed
    registry.register("2", "2", build, validate)
    registry.register("1", "1", build, validate)
    registry.register("10", "10", build, validate)
    contract = registry.resolve("2", "2")
    assert contract.build_prompt is build
    assert contract.validate_result is validate
    assert registry.resolve("1", "2") is None
    assert registry.resolve("999", "1") is None
    expected = [{"promptVersion": v, "responseVersion": v} for v in ("1", "2", "10")]
    assert registry.capabilities() == expected
    registry.capabilities()[0]["promptVersion"] = "mutated"
    assert registry.capabilities() == expected
    with pytest.raises(ValueError, match="already registered"):
        registry.register("1", "1", build, validate)


@pytest.mark.parametrize("version", [None, True, 1, [], "", "0", "01", "-1", " 1", "١", "1234567890"])
def test_registry_rejects_noncanonical_versions(version):
    from enrichment_contracts import EnrichmentContractRegistry

    registry = EnrichmentContractRegistry()
    for pair in ((version, "1"), ("1", version)):
        with pytest.raises(ValueError):
            registry.register(*pair, str, str)
        with pytest.raises(ValueError):
            registry.resolve(*pair)
    assert registry.capabilities() == []


@pytest.mark.parametrize("builder,validator", [(None, str), (str, None), ("prompt", str), (str, {})])
def test_registry_rejects_noncallable_handlers(builder, validator):
    from enrichment_contracts import EnrichmentContractRegistry

    registry = EnrichmentContractRegistry()
    with pytest.raises(TypeError):
        registry.register("1", "1", builder, validator)
    assert registry.capabilities() == []


def test_registered_pairs_select_exact_handlers_and_capabilities(client, monkeypatch):
    registry = client.application.extensions["enrichment_contracts"]
    calls = []

    def validate(parsed, ids):
        calls.append((parsed, ids))
        return {"items": [{"questionId": 101, "knowledgePoint": "second_contract"}]}

    registry.register("2", "2", lambda payload: "second prompt", validate)
    assert client.get(CAPABILITIES).get_json() == {"supportedContracts": [
        {"promptVersion": "1", "responseVersion": "1"},
        {"promptVersion": "2", "responseVersion": "2"},
    ]}
    crosspair = client.post(ENDPOINT, json={}, headers=headers("1", "2"))
    assert crosspair.status_code == 409
    assert crosspair.get_json() == UNSUPPORTED_ERROR

    def generate(prompt, task, validator, deadline=None):
        assert deadline is not None
        assert (prompt, task) == ("second prompt", "course_content")
        return validator({"items": []})

    monkeypatch.setattr(ai_app, "generate_valid_llm_result", generate)
    response = client.post(ENDPOINT, json=_course_content_payload(), headers=headers("2", "2"))
    assert response.status_code == 200
    assert response.get_json()["items"][0]["knowledgePoint"] == "second_contract"
    assert response.headers[PROMPT_HEADER] == response.headers[RESPONSE_HEADER] == "2"
    assert calls == [({"items": []}, [101])]


@pytest.mark.parametrize("token,status", [(None, 401), ("wrong", 401), ("secret", 200)])
def test_capabilities_internal_auth_without_generation(client, monkeypatch, token, status):
    monkeypatch.setenv("AI_SERVICE_INTERNAL_TOKEN", "secret")
    response = client.get(CAPABILITIES, headers={} if token is None else {"X-Internal-Service-Token": token})
    assert response.status_code == status
    assert response.get_json() == ({"supportedContracts": [{"promptVersion": "1", "responseVersion": "1"}]}
                                   if status == 200 else {"error": "Unauthorized"})
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("token", ["", "   "])
def test_capabilities_empty_token_development_behavior(client, monkeypatch, token):
    monkeypatch.setenv("AI_SERVICE_INTERNAL_TOKEN", token)
    assert client.get(CAPABILITIES).status_code == 200


def test_auth_precedes_contract_selection(client, monkeypatch):
    monkeypatch.setenv("AI_SERVICE_INTERNAL_TOKEN", "secret")
    assert client.post(ENDPOINT, json={}, headers=headers("999", "1")).status_code == 401


@pytest.mark.parametrize("mutation", ["body_version", "duplicate", "oversized"])
def test_selected_contract_preserves_strict_body_validation(client, mutation):
    payload = _course_content_payload()
    if mutation == "body_version":
        payload["promptVersion"] = "1"
    elif mutation == "duplicate":
        payload["questions"].append(dict(payload["questions"][0]))
    else:
        payload["questions"] = [dict(payload["questions"][0], questionId=i) for i in range(1, 27)]
    assert client.post(ENDPOINT, json=payload, headers=headers()).status_code == 400


@pytest.mark.parametrize("items,status,label", [
    ([], 502, None),
    ([{"questionId": 101, "knowledgePoint": "x"}] * 2, 502, None),
    ([{"questionId": 101, "knowledgePoint": "x" * 65}], 502, None),
    ([{"questionId": 101, "knowledgePoint": "  " + "X" * 64 + "  "}], 200, "x" * 64),
])
def test_selected_contract_preserves_result_validation(client, monkeypatch, items, status, label):
    monkeypatch.setenv("LLM_COURSE_CONTENT_MODEL", "mock-model")
    monkeypatch.setenv("LLM_COURSE_CONTENT_FALLBACK_MODELS", "")
    monkeypatch.setattr(ai_app, "generate_valid_llm_result", ORIGINAL_GENERATE)
    monkeypatch.setattr(ai_app, "generate_llm_text", lambda *args, **kwargs: json.dumps({"items": items}))
    response = client.post(ENDPOINT, json=_course_content_payload(), headers=headers())
    assert response.status_code == status
    if label:
        assert response.get_json()["items"][0]["knowledgePoint"] == label
    else:
        assert response.get_json() == {"code": "ENRICHMENT_GENERATION_FAILED", "error": "Enrichment generation failed"}


ORIGINAL_GENERATE = ai_app.generate_valid_llm_result
