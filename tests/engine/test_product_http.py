"""Loopback control-plane tests for the product HTTP transport.

These tests deliberately use a not-ready service.  They verify HTTP policy and
request validation only; they do not fabricate model or GPU inference output.
"""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import subprocess
import threading
import time
from shutil import which

import pytest

from ironmule_product.http_server import create_server


class ControlService:
    def __init__(self) -> None:
        self.stream_calls = 0
        self.cancelled: list[str] = []
        self.closed = False

    def health(self) -> dict[str, object]:
        return {"ready": False, "service": "control-test"}

    def models(self) -> list[dict[str, str]]:
        return [{"id": "gemma-local", "object": "model", "owned_by": "local"}]

    def stream(self, _request):
        self.stream_calls += 1
        raise AssertionError("not-ready control service must not run inference")

    def cancel(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True

    def close(self) -> None:
        self.closed = True


class SaturationService(ControlService):
    """Control-plane service that holds each health handler for saturation."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self._lock = threading.Lock()
        self.all_active = threading.Event()
        self.release = threading.Event()

    def health(self) -> dict[str, object]:
        with self._lock:
            self.active += 1
            if self.active >= 64:
                self.all_active.set()
        self.release.wait(timeout=5)
        with self._lock:
            self.active -= 1
        return super().health()


@pytest.fixture()
def running_server():
    service = ControlService()
    try:
        server = create_server(service, port=0)
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, method: str, path: str, body: object | None = None, headers: dict[str, str] | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    connection.request(method, path, body=encoded, headers=headers or {})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw) if raw else None


def raw_request(server, request_bytes: bytes):
    connection = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
    connection.sendall(request_bytes)
    response = http.client.HTTPResponse(connection)
    response.begin()
    raw = response.read()
    return connection, response, json.loads(raw) if raw else None


def make_tls_material(tmp_path):
    openssl = which("openssl")
    if openssl is None:
        pytest.skip("openssl is unavailable")
    cert = tmp_path / "server-cert.pem"
    key = tmp_path / "server-key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return cert, key


def test_health_is_live_even_when_backend_is_not_ready(running_server):
    server, _service = running_server
    status, payload = request(server, "GET", "/health")
    assert status == 200
    assert payload == {"ready": False, "service": "control-test"}


def test_ready_reports_not_ready_as_503(running_server):
    server, _service = running_server
    status, payload = request(server, "GET", "/ready")
    assert status == 503
    assert payload["ready"] is False


def test_models_is_control_plane_only(running_server):
    server, _service = running_server
    status, payload = request(server, "GET", "/v1/models")
    assert status == 200
    assert payload == {"object": "list", "data": [{"id": "gemma-local", "object": "model", "owned_by": "local"}]}


def test_generation_returns_not_ready_without_calling_service_stream(running_server):
    server, service = running_server
    status, payload = request(
        server,
        "POST",
        "/v1/chat/completions",
        {"model": "gemma-local", "messages": [{"role": "user", "content": "hello"}]},
        {"Content-Type": "application/json"},
    )
    assert status == 503
    assert payload["error"]["code"] == "backend_unavailable"
    assert service.stream_calls == 0


def test_cross_site_origin_is_rejected_before_body_or_model_work(running_server):
    server, service = running_server
    status, payload = request(
        server,
        "POST",
        "/v1/chat/completions",
        {"model": "gemma-local", "messages": [{"role": "user", "content": "hello"}]},
        {"Content-Type": "application/json", "Origin": "https://attacker.example"},
    )
    assert status == 403
    assert payload["error"]["code"] == "origin_forbidden"
    assert service.stream_calls == 0


def test_same_origin_is_allowed_but_still_reports_backend_not_ready(running_server):
    server, service = running_server
    status, payload = request(
        server,
        "POST",
        "/v1/chat/completions",
        {"model": "gemma-local", "messages": [{"role": "user", "content": "hello"}]},
        {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}"},
    )
    assert status == 503
    assert payload["error"]["code"] == "backend_unavailable"
    assert service.stream_calls == 0


def test_tls_loopback_handshake_and_authenticated_control_plane(tmp_path):
    cert, key = make_tls_material(tmp_path)
    service = ControlService()
    try:
        server = create_server(service, port=0, api_key="local-secret", tls_cert=str(cert), tls_key=str(key))
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
        connection = context.wrap_socket(raw, server_hostname="127.0.0.1")
        connection.sendall(
            (
                f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                "Authorization: Bearer local-secret\r\n"
                f"Origin: https://127.0.0.1:{server.server_port}\r\n\r\n"
            ).encode("ascii")
        )
        response = http.client.HTTPResponse(connection)
        response.begin()
        assert response.status == 200
        assert json.loads(response.read()) == {"ready": False, "service": "control-test"}
        assert response.getheader("Access-Control-Allow-Origin") is None
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_incomplete_tls_client_does_not_block_authenticated_health(tmp_path):
    cert, key = make_tls_material(tmp_path)
    service = ControlService()
    try:
        server = create_server(service, port=0, api_key="local-secret", tls_cert=str(cert), tls_key=str(key))
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stalled = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
    try:
        # Let accept() observe the connection while no ClientHello is sent.
        time.sleep(0.1)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
        connection = context.wrap_socket(raw, server_hostname="127.0.0.1")
        connection.sendall(
            (
                f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                "Authorization: Bearer local-secret\r\n\r\n"
            ).encode("ascii")
        )
        response = http.client.HTTPResponse(connection)
        response.begin()
        assert response.status == 200
        assert json.loads(response.read())["ready"] is False
        connection.close()
    finally:
        stalled.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_handler_saturation_returns_429_then_recovers():
    service = SaturationService()
    try:
        server = create_server(service, port=0)
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    held: list[socket.socket] = []
    try:
        request_bytes = f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n\r\n".encode("ascii")
        for _ in range(64):
            connection = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
            connection.sendall(request_bytes)
            held.append(connection)
        assert service.all_active.wait(timeout=5), "64 handler slots did not become active"

        connection, response, payload = raw_request(
            server,
            request_bytes,
        )
        try:
            assert response.status == 429
            assert payload["error"]["code"] == "overloaded"
            assert response.getheader("Connection") == "close"
        finally:
            connection.close()

        service.release.set()
        for connection in held:
            response = http.client.HTTPResponse(connection)
            response.begin()
            assert response.status == 200
            assert json.loads(response.read())["ready"] is False
            connection.close()
        held.clear()

        status, payload = request(server, "GET", "/health")
        assert status == 200
        assert payload["ready"] is False
    finally:
        service.release.set()
        for connection in held:
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_unsupported_parameter_is_rejected_before_readiness_check(running_server):
    server, service = running_server
    status, payload = request(
        server,
        "POST",
        "/v1/chat/completions",
        {"model": "gemma-local", "messages": [{"role": "user", "content": "hello"}], "n": 2},
        {"Content-Type": "application/json"},
    )
    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert service.stream_calls == 0


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"gemma-local","model":"other","messages":[{"role":"user","content":"hello"}]}',
        b'{"model":"gemma-local","messages":[{"role":"user","content":NaN}]}',
        b'{"model":"gemma-local","messages":[{"role":"user","content":"\\ud800"}]}',
    ],
)
def test_json_parser_rejects_duplicate_nonfinite_and_surrogate_values(running_server, body):
    server, service = running_server
    headers = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
    )
    connection, response, payload = raw_request(server, headers)
    try:
        assert response.status == 400
        assert payload["error"]["code"] == "invalid_request"
        assert response.getheader("Connection") == "close"
        assert service.stream_calls == 0
    finally:
        connection.close()


@pytest.mark.parametrize(
    "framing",
    [
        b"",
        b"Content-Length: nope\r\n",
        b"Content-Length: 2\r\nContent-Length: 2\r\n",
        b"Transfer-Encoding: chunked\r\n",
    ],
)
def test_ambiguous_request_framing_is_400_and_closes_connection(running_server, framing):
    server, service = running_server
    body = b"{}"
    request_bytes = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        + framing
        + (b"Content-Length: 2\r\n" if b"Content-Length" not in framing and b"Transfer-Encoding" in framing else b"")
        + b"\r\n"
        + (body if b"Transfer-Encoding" not in framing and b"nope" not in framing else b"")
    )
    connection, response, payload = raw_request(server, request_bytes)
    try:
        assert response.status == 400
        assert payload["error"]["code"] == "invalid_request"
        assert response.getheader("Connection") == "close"
        assert service.stream_calls == 0
    finally:
        connection.close()


def test_post_early_rejection_cannot_be_reused_for_a_pipelined_request(running_server):
    server, service = running_server
    body = b'{"model":"gemma-local","messages":[{"role":"user","content":"hello"}],"n":2}'
    request_bytes = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
        + b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
    )
    connection, response, payload = raw_request(server, request_bytes)
    try:
        assert response.status == 400
        assert payload["error"]["code"] == "invalid_request"
        assert response.getheader("Connection") == "close"
        assert connection.recv(1) == b""
        assert service.stream_calls == 0
    finally:
        connection.close()


def test_duplicate_or_non_loopback_host_is_rejected(running_server):
    server, service = running_server
    body = b"{}"
    request_bytes = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        + f"Host: 127.0.0.1:{server.server_port}\r\nContent-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
    )
    connection, response, payload = raw_request(server, request_bytes)
    try:
        assert response.status == 400
        assert payload["error"]["code"] == "invalid_request"
        assert response.getheader("Connection") == "close"
        assert service.stream_calls == 0
    finally:
        connection.close()


@pytest.mark.parametrize("api_key", ["", "ümlaut", "line\nbreak", "delete\x7f"])
def test_api_key_must_be_nonempty_ascii_without_controls(api_key):
    service = ControlService()
    with pytest.raises(ValueError, match="ASCII"):
        create_server(service, port=0, api_key=api_key)
    assert not service.closed


def test_api_key_is_optional_on_loopback_but_enforced_when_configured():
    service = ControlService()
    try:
        server = create_server(service, port=0, api_key="local-secret")
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = request(server, "GET", "/health")
        assert status == 401
        assert payload["error"]["code"] == "unauthorized"
        status, _ = request(server, "GET", "/health", headers={"Authorization": "Bearer local-secret"})
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert service.closed


def test_duplicate_authorization_headers_are_rejected():
    service = ControlService()
    try:
        server = create_server(service, port=0, api_key="local-secret")
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request_bytes = (
            b"GET /health HTTP/1.1\r\n"
            + f"Host: 127.0.0.1:{server.server_port}\r\nAuthorization: Bearer local-secret\r\nAuthorization: Bearer local-secret\r\n\r\n".encode("ascii")
        )
        connection, response, payload = raw_request(server, request_bytes)
        try:
            assert response.status == 401
            assert payload["error"]["code"] == "unauthorized"
            assert response.getheader("Connection") == "close"
        finally:
            connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_non_ascii_authorization_token_cannot_match_ascii_question_mark_key():
    service = ControlService()
    try:
        server = create_server(service, port=0, api_key="abc?")
    except PermissionError as exc:
        pytest.skip(f"loopback bind denied by test environment: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = request(
            server,
            "GET",
            "/health",
            headers={"Authorization": "Bearer abcé"},
        )
        assert status == 401
        assert payload["error"]["code"] == "unauthorized"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_bind_requires_authentication_and_tls(tmp_path):
    service = ControlService()
    with pytest.raises(ValueError, match="non-loopback"):
        create_server(service, host="0.0.0.0", port=0)
    with pytest.raises(ValueError, match="non-loopback"):
        create_server(service, host="0.0.0.0", port=0, api_key="secret")
    with pytest.raises(FileNotFoundError):
        create_server(
            service,
            host="0.0.0.0",
            port=0,
            api_key="secret",
            tls_cert=str(tmp_path / "missing-cert.pem"),
            tls_key=str(tmp_path / "missing-key.pem"),
        )
    assert not service.closed


def test_literal_ipv6_loopback_uses_native_ipv6_listener():
    service = ControlService()
    try:
        server = create_server(service, host="::1", port=0)
    except (OSError, PermissionError) as exc:
        pytest.skip(f"IPv6 loopback unavailable: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("::1", server.server_port, timeout=3)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"ready": False, "service": "control-test"}
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
