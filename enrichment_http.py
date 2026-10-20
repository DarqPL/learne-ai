"""Bounded course-enrichment I/O. DNS may outlive its caller, never its slot.

Daemon workers deliberately avoid ThreadPoolExecutor's unbounded queue and
interpreter-exit join of potentially uninterruptible native name resolution.
"""
import http.client
import json
import queue
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit


MAX_RESPONSE_BYTES = 1024 * 1024


class TransportError(RuntimeError):
    def __init__(self):
        super().__init__("Enrichment upstream unavailable")


class DeadlineExceeded(TransportError):
    pass


class TransportBusy(TransportError):
    pass


class TransportConfigurationError(TransportError):
    pass


class UpstreamStatusError(TransportError):
    def __init__(self, status):
        super().__init__()
        self.category = "server" if status >= 500 else "client" if status >= 400 else "redirect"


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise DeadlineExceeded()


def _dispose(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()  # SSLSocket.close does not perform TLS unwrap/close_notify.
        except OSError:
            pass


class _Operation:
    def __init__(self, deadline):
        self.deadline = deadline
        self.lock = threading.Lock()
        self.aborted = False
        self.sock = None
        self.done = threading.Event()
        self.result = None
        self.error = None

    def check(self):
        check_deadline(self.deadline)
        if self.aborted:
            raise DeadlineExceeded()

    def own(self, sock):
        with self.lock:
            if self.aborted or time.monotonic() >= self.deadline:
                _dispose(sock)
                raise DeadlineExceeded()
            self.sock = sock

    def abort(self):
        with self.lock:
            self.aborted = True
        self.close_socket()

    def close_socket(self):
        with self.lock:
            sock = self.sock
            self.sock = None
        _dispose(sock)


class EnrichmentHttpTransport:
    def __init__(self, slots=4, resolver=None, socket_factory=None):
        self.resolver = resolver or socket.getaddrinfo
        self.socket_factory = socket_factory or socket.socket
        self.tls_context = ssl.create_default_context()
        self._admission = threading.BoundedSemaphore(slots)
        self._queue = queue.Queue(maxsize=slots)
        self._lock = threading.Lock()
        self._closed = False
        self._active = set()
        self._workers = [threading.Thread(target=self._worker, name=f"enrichment-io-{i}", daemon=True)
                         for i in range(slots)]
        for worker in self._workers:
            worker.start()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            self._closed = True
            active = list(self._active)
        for operation in active:
            operation.abort()
            operation.done.set()
        # Bounded total join; an uncooperative resolver remains a daemon in its slot.
        end = time.monotonic() + .5
        for worker in self._workers:
            worker.join(max(0, end - time.monotonic()))

    def post_json(self, url, headers, payload, *, deadline):
        check_deadline(deadline)
        operation = _Operation(deadline)
        with self._lock:
            if self._closed or not self._admission.acquire(blocking=False):
                raise TransportBusy()
            self._active.add(operation)
            self._queue.put_nowait((operation, url, headers, payload))
        try:
            if not operation.done.wait(max(0, deadline - time.monotonic())):
                raise DeadlineExceeded()
            operation.check()
            if operation.error is not None:
                raise operation.error from None
            return operation.result
        finally:
            operation.abort()

    def _worker(self):
        while True:
            try:
                operation, url, headers, payload = self._queue.get(timeout=.05)
            except queue.Empty:
                with self._lock:
                    if self._closed:
                        return
                continue
            try:
                operation.check()
                operation.result = self._request(operation, url, headers, payload)
            except TransportError as error:
                operation.error = error
            except Exception:
                # Never retain raw provider errors, request data, or exception causes.
                operation.error = TransportError()
            finally:
                operation.close_socket()
                with self._lock:
                    self._active.discard(operation)
                self._admission.release()  # Only the actual worker releases capacity.
                operation.done.set()
                self._queue.task_done()

    def _request(self, operation, url, headers, payload):
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError()
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            host = parsed.hostname
        except ValueError:
            raise TransportConfigurationError() from None
        addresses = self.resolver(host, port, type=socket.SOCK_STREAM)
        operation.check()  # A late DNS result must never connect or send.
        sock = None
        for family, kind, proto, _, address in addresses:
            operation.check()
            sock = self.socket_factory(family, kind, proto)
            operation.own(sock)
            try:
                sock.settimeout(max(.001, operation.deadline - time.monotonic()))
                sock.connect(address)
                break
            except OSError:
                _dispose(sock)
                sock = None
        if sock is None:
            raise TransportError()
        operation.check()
        if parsed.scheme == "https":
            sock = self.tls_context.wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
            operation.own(sock)
            sock.do_handshake()
            operation.check()
        connection = http.client.HTTPConnection(host, port)
        connection.set_debuglevel(0)
        connection.sock = sock
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            operation.check()
            connection.request("POST", path, body=json.dumps(payload).encode("utf-8"), headers=headers)
            response = connection.getresponse()
            operation.check()
            if not 200 <= response.status < 300:
                raise UpstreamStatusError(response.status)
            if response.length is not None and response.length > MAX_RESPONSE_BYTES:
                raise TransportError()
            body = response.read(MAX_RESPONSE_BYTES + 1)
            operation.check()
            if len(body) > MAX_RESPONSE_BYTES:
                raise TransportError()
            return body.decode("utf-8")
        finally:
            operation.close_socket()
            connection.close()


_default_lock = threading.Lock()
_default_transport = None


def post_json(url, headers, payload, *, deadline):
    global _default_transport
    with _default_lock:
        if _default_transport is None:
            _default_transport = EnrichmentHttpTransport()
        transport = _default_transport
    return transport.post_json(url, headers, payload, deadline=deadline)
