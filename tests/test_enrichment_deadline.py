import json
import socket
import ssl
import threading
import time
from contextlib import contextmanager

import pytest

import app as ai_app
from test_app import _course_content_payload


@pytest.fixture(autouse=True)
def loopback_network_only(monkeypatch):
    resolve = socket.getaddrinfo
    connect = socket.socket.connect
    def guarded_resolve(host, *args, **kwargs):
        assert host in ("127.0.0.1", "::1"), "Non-loopback DNS forbidden"
        return resolve(host, *args, **kwargs)
    def guarded_connect(sock, address):
        assert address[0] in ("127.0.0.1", "::1"), "Non-loopback connection forbidden"
        return connect(sock, address)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_resolve)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@contextmanager
def peer(handler):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(.1)
    stop = threading.Event()
    sockets = []
    threads = []

    def serve():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            sockets.append(conn)
            conn.settimeout(2)
            thread = threading.Thread(target=handler, args=(conn,), daemon=True)
            threads.append(thread)
            thread.start()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        stop.set()
        listener.close()
        for conn in sockets:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        thread.join(2)
        for worker in threads:
            worker.join(2)


def read_request(conn):
    data = b""
    while b"\r\n\r\n" not in data:
        data += conn.recv(4096)
    head, body = data.split(b"\r\n\r\n", 1)
    size = int(next(line.split(b":", 1)[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")))
    while len(body) < size:
        body += conn.recv(4096)
    return head, json.loads(body)


def transport(**kwargs):
    import enrichment_http
    return enrichment_http.EnrichmentHttpTransport(**kwargs)


@pytest.mark.parametrize("mode", ["headers", "body", "drip", "tls"])
def test_absolute_timeout_closes_actual_socket(mode):
    closed = threading.Event()

    def handle(conn):
        try:
            if mode == "tls":
                conn.recv(4096)
            else:
                read_request(conn)
                if mode != "headers":
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 9999\r\n\r\n")
                if mode == "drip":
                    for _ in range(30):
                        conn.sendall(b"x")
                        time.sleep(.03)
            while conn.recv(4096):
                pass
            closed.set()
        except OSError:
            closed.set()

    import enrichment_http as http
    with peer(handle) as url, transport() as client:
        if mode == "tls":
            url = url.replace("http:", "https:")
        start = time.monotonic()
        with pytest.raises(http.DeadlineExceeded):
            client.post_json(url, {}, {}, deadline=start + .25)
        assert time.monotonic() - start < .8
        assert closed.wait(2), "peer must observe socket shutdown"


def test_dns_timeout_retains_slots_and_never_connects_late():
    import enrichment_http as http
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def resolver(*args, **kwargs):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        return socket.getaddrinfo("127.0.0.1", 9, type=socket.SOCK_STREAM)

    connected = []
    class ForbiddenSocket:
        def __init__(self, *args):
            connected.append(args)
            raise AssertionError("late socket creation")

    client = transport(slots=2, resolver=resolver, socket_factory=ForbiddenSocket)
    try:
        for _ in range(2):
            start = time.monotonic()
            with pytest.raises(http.DeadlineExceeded):
                client.post_json("http://controlled.invalid/v1", {}, {}, deadline=start + .2)
            assert time.monotonic() - start < .7
        assert entered.is_set()
        for _ in range(10):
            with pytest.raises(http.TransportBusy):
                client.post_json("http://controlled.invalid", {}, {}, deadline=time.monotonic() + .2)
        assert len(calls) == 2
        assert len(set(calls)) <= 2
    finally:
        release.set()
        client.close()
    assert not connected


@pytest.mark.parametrize("sse", [False, True])
def test_course_json_and_sse_use_bounded_adapter(monkeypatch, sse):
    content = json.dumps({"items": [{"questionId": 101, "knowledgePoint": "dates"}]})
    envelope = {"choices": [{"message": {"content": content}}]}
    body = ("data: " + json.dumps(envelope) + "\n\ndata: [DONE]\n" if sse else json.dumps(envelope)).encode()
    captured = []
    def handle(conn):
        captured.append(read_request(conn))
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    with peer(handle) as url:
        monkeypatch.setenv("LLM_BASE_URL", url + "/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-secret")
        monkeypatch.setenv("LLM_COURSE_CONTENT_MODEL", "test-model")
        monkeypatch.setenv("LLM_COURSE_CONTENT_FALLBACK_MODELS", "")
        monkeypatch.delenv("AI_SERVICE_INTERNAL_TOKEN", raising=False)
        response = ai_app.create_app().test_client().post("/course-content/enrich-knowledge-points", json=_course_content_payload())
    assert response.status_code == 200
    assert response.headers["X-Enrichment-Prompt-Version"] == "1"
    assert captured[0][1]["model"] == "test-model"
    assert b"Authorization: Bearer test-secret" in captured[0][0]


@pytest.mark.parametrize("value", ["0", "-1", "110001", "secret-invalid", "1.5", ""])
def test_invalid_budget_is_sanitized_configuration_error(monkeypatch, value):
    monkeypatch.setenv("LLM_ENRICHMENT_BUDGET_MS", value)
    monkeypatch.delenv("AI_SERVICE_INTERNAL_TOKEN", raising=False)
    response = ai_app.create_app().test_client().post("/course-content/enrich-knowledge-points", json=_course_content_payload())
    assert response.status_code == 503
    assert response.get_json() == {"code": "ENRICHMENT_CONFIGURATION_UNAVAILABLE", "error": "Enrichment configuration unavailable"}


def test_tls_context_verifies_by_default():
    with transport() as client:
        assert client.tls_context.verify_mode == ssl.CERT_REQUIRED
        assert client.tls_context.check_hostname


@pytest.fixture
def course_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_COURSE_CONTENT_MODEL", "first")
    monkeypatch.setenv("LLM_COURSE_CONTENT_FALLBACK_MODELS", "second,third,fourth,fifth")
    monkeypatch.setenv("LLM_ENRICHMENT_BUDGET_MS", "350")
    monkeypatch.setenv("LLM_TIMEOUT_MS", "130")
    monkeypatch.delenv("AI_SERVICE_INTERNAL_TOKEN", raising=False)


def test_fallback_attempts_share_one_budget_and_close_connections(monkeypatch, course_config):
    calls = []
    closed = []
    def handle(conn):
        _, payload = read_request(conn)
        calls.append(payload["model"])
        try:
            while conn.recv(4096):
                pass
        finally:
            closed.append(payload["model"])
    with peer(handle) as url:
        monkeypatch.setenv("LLM_BASE_URL", url)
        start = time.monotonic()
        response = ai_app.create_app().test_client().post("/course-content/enrich-knowledge-points", json=_course_content_payload())
        elapsed = time.monotonic() - start
    assert response.status_code == 502
    assert response.get_json()["code"] == "ENRICHMENT_DEADLINE_EXCEEDED"
    assert .3 <= elapsed < .8
    assert calls == ["first", "second", "third"]
    assert sorted(closed) == sorted(calls)


@pytest.mark.parametrize("failure", ["json", "validation"])
def test_parse_and_validation_time_reduce_fallback_budget(monkeypatch, course_config, failure):
    calls = []
    original_parse = ai_app.parse_json_response
    def slow_parse(text):
        if failure == "json" and text == "bad":
            time.sleep(.22)
        return original_parse(text)
    monkeypatch.setattr(ai_app, "parse_json_response", slow_parse)
    original_validate = ai_app._filter_course_content_response
    def slow_validate(*args):
        time.sleep(.22)
        return original_validate(*args)
    if failure == "validation":
        monkeypatch.setattr(ai_app, "_filter_course_content_response", slow_validate)
    def handle(conn):
        _, payload = read_request(conn)
        calls.append(payload["model"])
        if len(calls) == 1:
            content = "bad" if failure == "json" else '{"items": []}'
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        else:
            while conn.recv(4096):
                pass
    with peer(handle) as url:
        monkeypatch.setenv("LLM_BASE_URL", url)
        start = time.monotonic()
        response = ai_app.create_app().test_client().post("/course-content/enrich-knowledge-points", json=_course_content_payload())
        elapsed = time.monotonic() - start
    assert response.get_json()["code"] == "ENRICHMENT_DEADLINE_EXCEEDED"
    assert elapsed < .8
    assert calls == ["first", "second"]


@pytest.mark.parametrize("framing", ["length", "chunked", "eof"])
def test_excessive_response_is_bounded_and_socket_closed(framing):
    import enrichment_http as http
    closed = threading.Event()
    def handle(conn):
        read_request(conn)
        body = b"x" * (http.MAX_RESPONSE_BYTES + 1)
        try:
            if framing == "length":
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 9999999\r\n\r\n")
            elif framing == "chunked":
                conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + hex(len(body))[2:].encode() + b"\r\n" + body + b"\r\n0\r\n\r\n")
            else:
                conn.sendall(b"HTTP/1.1 200 OK\r\n\r\n" + body)
            while conn.recv(4096):
                pass
            closed.set()
        except OSError:
            closed.set()
    with peer(handle) as url, transport() as client:
        with pytest.raises(http.TransportError) as error:
            client.post_json(url, {}, {}, deadline=time.monotonic() + .5)
        assert not isinstance(error.value, http.DeadlineExceeded)
        assert closed.wait(2)


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_status_errors_never_follow_redirect_or_retain_provider_data(status):
    import enrichment_http as http
    calls = []
    def handle(conn):
        calls.append(read_request(conn))
        conn.sendall(f"HTTP/1.1 {status} secret-provider-data\r\nLocation: http://external.invalid\r\nContent-Length: 20\r\n\r\nsecret-provider-data".encode())
    with peer(handle) as url, transport() as client:
        with pytest.raises(http.UpstreamStatusError) as caught:
            client.post_json(url, {}, {}, deadline=time.monotonic() + .5)
    assert len(calls) == 1
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_close_interrupts_active_socket_and_caller():
    import enrichment_http as http
    entered = threading.Event()
    closed = threading.Event()
    def handle(conn):
        read_request(conn)
        entered.set()
        while conn.recv(4096):
            pass
        closed.set()
    errors = []
    with peer(handle) as url:
        client = transport()
        def call():
            try:
                client.post_json(url, {}, {}, deadline=time.monotonic() + 2)
            except http.TransportError as error:
                errors.append(error)
        caller = threading.Thread(target=call, daemon=True)
        caller.start()
        try:
            assert entered.wait(1)
            client.close()
            caller.join(1)
            assert not caller.is_alive()
            assert closed.wait(1)
            assert errors
        finally:
            client.close()


def test_flask_dns_timeout_and_busy_are_bounded(monkeypatch, course_config):
    release = threading.Event()
    calls = []
    def resolver(*args, **kwargs):
        calls.append(args)
        assert release.wait(3)
        return socket.getaddrinfo("127.0.0.1", 9, type=socket.SOCK_STREAM)
    with transport(slots=1, resolver=resolver) as client:
        monkeypatch.setattr(ai_app.enrichment_http, "post_json", client.post_json)
        monkeypatch.setenv("LLM_BASE_URL", "http://controlled.invalid")
        monkeypatch.setenv("LLM_TIMEOUT_MS", "500")
        monkeypatch.setenv("LLM_ENRICHMENT_BUDGET_MS", "200")
        flask = ai_app.create_app().test_client()
        try:
            start = time.monotonic()
            response = flask.post("/course-content/enrich-knowledge-points", json=_course_content_payload())
            assert time.monotonic() - start < .7
            assert response.get_json()["code"] == "ENRICHMENT_DEADLINE_EXCEEDED"
            start = time.monotonic()
            for _ in range(5):
                response = flask.post("/course-content/enrich-knowledge-points", json=_course_content_payload())
                assert response.get_json()["code"] == "ENRICHMENT_GENERATION_FAILED"
            assert time.monotonic() - start < .5
            assert len(calls) == 1
        finally:
            release.set()


def test_expired_validation_cannot_start_fallback(monkeypatch, course_config):
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs["model"])
        return "{}"
    def validate(parsed):
        time.sleep(.22)
        raise ValueError("private-provider-data")
    monkeypatch.setattr(ai_app, "generate_llm_text", generate)
    import enrichment_http as http
    with pytest.raises(http.DeadlineExceeded):
        ai_app.generate_valid_llm_result("prompt", "course_content", validate, deadline=time.monotonic() + .2)
    assert calls == ["first"]


@pytest.mark.parametrize("configuration", ["key", "model", "url"])
def test_local_configuration_failure_has_stable_code(monkeypatch, course_config, configuration):
    if configuration == "key":
        monkeypatch.setenv("LLM_API_KEY", "")
    elif configuration == "model":
        monkeypatch.setenv("LLM_COURSE_CONTENT_MODEL", "")
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "")
        monkeypatch.setenv("LLM_COURSE_CONTENT_FALLBACK_MODELS", "")
    else:
        monkeypatch.setenv("LLM_BASE_URL", "ftp://private-provider.invalid")
    response = ai_app.create_app().test_client().post("/course-content/enrich-knowledge-points", json=_course_content_payload())
    assert response.status_code == 503
    assert response.get_json() == {"code": "ENRICHMENT_CONFIGURATION_UNAVAILABLE", "error": "Enrichment configuration unavailable"}


def test_default_total_budget_and_attempt_clamp(monkeypatch, course_config):
    monkeypatch.delenv("LLM_ENRICHMENT_BUDGET_MS")
    start = time.monotonic()
    deadline = ai_app._enrichment_deadline()
    assert 109.9 < deadline - start <= 110.1
    deadlines = []
    def post(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return '{"choices":[{"message":{"content":"{}"}}]}'
    monkeypatch.setattr(ai_app.enrichment_http, "post_json", post)
    ai_app.generate_llm_text("prompt", "course_content", deadline=deadline)
    assert start < deadlines[0] < start + .3
    short = time.monotonic() + .05
    ai_app.generate_llm_text("prompt", "course_content", deadline=short)
    assert deadlines[1] == short


def test_early_rejection_releases_socket_descriptor():
    import enrichment_http as http
    sockets = []
    def factory(*args):
        sock = socket.socket(*args)
        sockets.append(sock)
        return sock
    def handle(conn):
        read_request(conn)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 9999999\r\n\r\n")
        while conn.recv(4096):
            pass
    with peer(handle) as url, transport(socket_factory=factory) as client:
        with pytest.raises(http.TransportError):
            client.post_json(url, {}, {}, deadline=time.monotonic() + .5)
        assert sockets
        assert all(sock.fileno() == -1 for sock in sockets)


def test_prompt_building_does_not_reset_request_budget(monkeypatch, course_config):
    original = ai_app.build_course_content_prompt
    def slow_prompt(payload):
        time.sleep(.22)
        return original(payload)
    monkeypatch.setattr(ai_app, "build_course_content_prompt", slow_prompt)
    monkeypatch.setenv("LLM_ENRICHMENT_BUDGET_MS", "200")
    def forbidden(*args, **kwargs):
        pytest.fail("Expired request must not start model I/O")
    monkeypatch.setattr(ai_app.enrichment_http, "post_json", forbidden)
    response = ai_app.create_app().test_client().post("/course-content/enrich-knowledge-points", json=_course_content_payload())
    assert response.get_json()["code"] == "ENRICHMENT_DEADLINE_EXCEEDED"


@pytest.mark.parametrize("boundary", ["envelope", "content"])
@pytest.mark.parametrize("recover", [True, False], ids=["fallback-recovers", "all-models-invalid"])
def test_deep_json_nesting_is_sanitized_and_uses_fallback(
    monkeypatch, course_config, caplog, boundary, recover
):
    nested = "[" * 20000 + "0" + "]" * 20000
    assert len(nested.encode()) == 40001
    valid_content = json.dumps({"items": [{"questionId": 101, "knowledgePoint": "dates"}]})
    def envelope(content):
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    bad_body = nested.encode() if boundary == "envelope" else envelope(nested)
    assert len(bad_body) < 1024 * 1024
    calls = []
    caught = []
    original_generate = ai_app.generate_valid_llm_result
    def generate(*args, **kwargs):
        try:
            return original_generate(*args, **kwargs)
        except ai_app.LlmGenerationError as error:
            caught.append(error)
            raise
    monkeypatch.setattr(ai_app, "generate_valid_llm_result", generate)
    monkeypatch.setenv("LLM_COURSE_CONTENT_FALLBACK_MODELS", "second")
    monkeypatch.setenv("LLM_ENRICHMENT_BUDGET_MS", "1000")
    monkeypatch.setenv("LLM_TIMEOUT_MS", "500")
    def handle(conn):
        _, payload = read_request(conn)
        calls.append(payload["model"])
        body = envelope(valid_content) if recover and payload["model"] == "second" else bad_body
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    with peer(handle) as url:
        monkeypatch.setenv("LLM_BASE_URL", url)
        application = ai_app.create_app()
        # Exercise Flask's actual 500 handling in RED, rather than propagating exceptions.
        application.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)
        response = application.test_client().post(
            "/course-content/enrich-knowledge-points", json=_course_content_payload()
        )
    assert response.status_code == (200 if recover else 502)
    assert response.is_json
    assert calls == ["first", "second"]
    if recover:
        assert response.get_json() == {"items": [{"questionId": 101, "knowledgePoint": "dates"}]}
        assert response.headers["X-Enrichment-Prompt-Version"] == "1"
        assert response.headers["X-Enrichment-Response-Version"] == "1"
        assert not caught
    else:
        assert response.get_json() == {
            "code": "ENRICHMENT_GENERATION_FAILED", "error": "Enrichment generation failed"
        }
        assert len(caught) == 1
        assert str(caught[0]) == "LLM generation failed"
        assert caught[0].__cause__ is None
        assert caught[0].__suppress_context__
    assert not caplog.records, "Malformed provider data must not generate payload/traceback logs"
