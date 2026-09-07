"""Small, dependency-free HTTP transport for the local product service.

The transport deliberately knows nothing about MLX or model loading.  A service
object is injected by the caller and is only expected to implement the product
control/data contract documented by :func:`create_server`.
"""

from __future__ import annotations

from http import HTTPStatus
import ipaddress
import json
import re
import secrets
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable
from urllib.parse import urlsplit

from .errors import InvalidRequest, ProductError
from .types import GenerationRequest, MAX_REQUEST_BYTES


_SOCKET_TIMEOUT_SECONDS = 15.0
_MAX_ERROR_MESSAGE = 512
_MAX_WORKERS = 64
_DECIMAL = re.compile(r"^[0-9]+$")
_CONTROL = frozenset(chr(value) for value in range(0x20)) | {chr(0x7F)}


def _reject_non_finite(value: str) -> Any:
    raise ValueError("non-finite JSON numbers are not supported")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_surrogates(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise InvalidRequest("request contains invalid Unicode")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_surrogates(key)
            _reject_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item)


_PUBLIC_MESSAGES = {
    "invalid_request": "invalid request",
    "backend_unavailable": "product service is unavailable",
    "overloaded": "request queue is full; retry later",
    "model_not_found": "requested model was not found",
    "request_timeout": "request timed out",
    "request_cancelled": "request cancelled",
    "state_error": "product service failed",
    "product_error": "product request failed",
}
_ERROR_TYPES = {"unauthorized": "authentication_error"}


def _safe_message(error: BaseException) -> str:
    """Return a bounded, user-safe message without exposing a traceback."""

    code = getattr(error, "code", "product_error")
    message = getattr(error, "_public_message", None) or _PUBLIC_MESSAGES.get(code, "product request failed")
    # Error strings can contain malformed Unicode from a backend.  Keep the
    # response serialisable even when the connection is still healthy.
    return str(message).encode("utf-8", "replace").decode("utf-8")[:_MAX_ERROR_MESSAGE]


def _error_payload(error: ProductError) -> dict[str, Any]:
    code = getattr(error, "code", "product_error")
    return {"error": {"message": _safe_message(error), "type": _ERROR_TYPES.get(code, code), "code": code}}


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Hostnames are not reliably resolvable before bind.  A named bind is
        # therefore treated as remote unless it is the explicit localhost name.
        return False


def _parse_authority(value: str, *, default_port: int, allow_dns: bool = False) -> tuple[str, int]:
    """Parse an HTTP Host/origin authority without DNS resolution."""

    if not value or any(character.isspace() or character in _CONTROL for character in value):
        raise ValueError("invalid host authority")
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            raise ValueError("invalid IPv6 host authority")
        host = value[1:closing]
        suffix = value[closing + 1 :]
        if suffix and not suffix.startswith(":"):
            raise ValueError("invalid host authority")
        raw_port = suffix[1:] if suffix else ""
    else:
        if value.count(":") > 1:
            raise ValueError("IPv6 Host values must be bracketed")
        host, separator, raw_port = value.partition(":")
        if not separator:
            raw_port = ""
    if not host:
        raise ValueError("host is empty")
    if raw_port:
        if not _DECIMAL.fullmatch(raw_port):
            raise ValueError("invalid host port")
        port = int(raw_port, 10)
    else:
        port = default_port
    if not 1 <= port <= 65535:
        raise ValueError("invalid host port")
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if (
            len(host) > 253
            or (not allow_dns and host.lower() != "localhost")
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for character in host)
            or any(not label or label[0] == "-" or label[-1] == "-" for label in labels)
        ):
            raise ValueError("invalid host name")
        normalized = host.lower()
    else:
        normalized = parsed.compressed.lower()
    return normalized, port


class _ProductHTTPServer(ThreadingHTTPServer):
    """Threading server carrying the injected service and auth policy."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: Any, api_key: str | None, *, tls_enabled: bool = False, tls_context: ssl.SSLContext | None = None):
        # TCPServer invokes server_close() if bind fails, so initialize state
        # before entering the stdlib constructor.
        self.service = service
        self.api_key = api_key
        self.bind_host = address[0]
        self.tls_enabled = tls_enabled
        self.tls_context = tls_context
        self._service_closed = False
        self._worker_slots = threading.BoundedSemaphore(_MAX_WORKERS)
        super().__init__(address, _ProductRequestHandler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        # ThreadingHTTPServer otherwise creates an unbounded thread per accepted
        # socket.  Reject excess sockets before creating a worker, keeping both
        # active handlers and backend-facing connections bounded at 64.
        if not self._worker_slots.acquire(blocking=False):
            if self.tls_context is not None:
                # The socket has not negotiated TLS because handshakes happen
                # in admitted worker threads. Never send plaintext HTTP bytes
                # to an unnegotiated TLS client from the accept loop.
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                request.close()
                return
            payload = b'{"error":{"message":"server is busy; retry later","type":"overloaded","code":"overloaded"}}'
            response = (
                b"HTTP/1.1 429 Too Many Requests\r\n"
                b"Content-Type: application/json; charset=utf-8\r\n"
                b"Cache-Control: no-store\r\n"
                b"Connection: close\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                + payload
            )
            try:
                request.sendall(response)
            except OSError:
                pass
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            admitted_request: socket.socket = request
            if self.tls_context is not None:
                # Handshake only after a worker slot has been acquired, so an
                # idle/incomplete ClientHello cannot stall accept().
                admitted_request.settimeout(_SOCKET_TIMEOUT_SECONDS)
                admitted_request = self.tls_context.wrap_socket(
                    admitted_request,
                    server_side=True,
                    do_handshake_on_connect=False,
                )
                admitted_request.do_handshake()
            super().process_request_thread(admitted_request, client_address)
        except (ssl.SSLError, socket.timeout, OSError):
            try:
                admitted_request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                admitted_request.close()
            except OSError:
                pass
            if admitted_request is not request:
                request.close()
        finally:
            self._worker_slots.release()

    def server_close(self) -> None:
        if not self._service_closed:
            self._service_closed = True
            try:
                self.service.close()
            except Exception:
                # Shutdown must remain safe even if a backend is already gone.
                pass
        super().server_close()


class _ProductHTTPServerV6(_ProductHTTPServer):
    address_family = socket.AF_INET6


class _ProductRequestHandler(BaseHTTPRequestHandler):
    server: _ProductHTTPServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(_SOCKET_TIMEOUT_SECONDS)

    def log_message(self, _format: str, *_args: Any) -> None:
        # Request paths, query strings and headers can carry secrets.  Product
        # logging is intentionally owned by the caller, not BaseHTTPRequestHandler.
        return

    def _path(self) -> str:
        return urlsplit(self.path).path

    def _validate_boundary(self) -> None:
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            raise _HTTPProductError("request requires exactly one Host header", HTTPStatus.BAD_REQUEST, "invalid_request")
        default_port = 443 if self.server.tls_enabled else 80
        allow_dns = not _is_loopback(self.server.bind_host)
        try:
            host, port = _parse_authority(hosts[0], default_port=default_port, allow_dns=allow_dns)
        except ValueError as exc:
            raise _HTTPProductError("invalid Host header", HTTPStatus.BAD_REQUEST, "invalid_request") from exc
        if _is_loopback(self.server.bind_host) and (not (host == "localhost" or _is_loopback(host)) or port != self.server.server_port):
            raise _HTTPProductError("Host must target this loopback listener", HTTPStatus.BAD_REQUEST, "invalid_request")
        origins = self.headers.get_all("Origin", [])
        if not origins:
            return
        if len(origins) != 1:
            raise _HTTPProductError("origin is ambiguous", HTTPStatus.FORBIDDEN, "origin_forbidden")
        try:
            parsed = urlsplit(origins[0])
            if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
                raise ValueError("invalid origin")
            origin_host, origin_port = _parse_authority(parsed.netloc, default_port=80 if parsed.scheme == "http" else 443, allow_dns=allow_dns)
        except ValueError as exc:
            raise _HTTPProductError("origin is not allowed", HTTPStatus.FORBIDDEN, "origin_forbidden") from exc
        expected_scheme = "https" if self.server.tls_enabled else "http"
        if parsed.scheme != expected_scheme or origin_host != host or origin_port != port:
            raise _HTTPProductError("origin is not allowed", HTTPStatus.FORBIDDEN, "origin_forbidden")

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, error: ProductError, *, close: bool = False) -> None:
        if close:
            self.close_connection = True
        self._send_json(int(getattr(error, "status", 500)), _error_payload(error))

    @staticmethod
    def _finish_reason(done: dict[str, Any]) -> str:
        reason = done.get("finish_reason")
        if reason not in ("stop", "length"):
            raise ProductError("service emitted an invalid finish reason")
        return reason

    def _authenticate(self) -> bool:
        expected = self.server.api_key
        if expected is None:
            return True
        authorization_values = self.headers.get_all("Authorization", [])
        # Multiple credentials are ambiguous; treat them as unauthenticated
        # while still running the same constant-time comparison below.
        value = authorization_values[0] if len(authorization_values) == 1 else ""
        scheme, separator, token = value.partition(" ")
        # Always perform the constant-time comparison for a configured key;
        # scheme/empty-token checks only decide whether the result is accepted.
        token_is_ascii = True
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError:
            # Keep the constant-time comparison on the invalid path too, but
            # never normalize foreign characters into a candidate key.
            token_is_ascii = False
            token_bytes = b""
        digest_matches = secrets.compare_digest(token_bytes, expected.encode("ascii"))
        valid = len(authorization_values) == 1 and token_is_ascii and separator == " " and scheme == "Bearer" and bool(token) and digest_matches
        if not valid:
            self._send_error(
                _HTTPProductError("authentication required", HTTPStatus.UNAUTHORIZED, "unauthorized"),
                close=True,
            )
        return valid

    def _read_body(self) -> bytes:
        transfer_values = self.headers.get_all("Transfer-Encoding", [])
        if transfer_values:
            self.close_connection = True
            raise InvalidRequest("request transfer encoding is not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self.close_connection = True
            raise InvalidRequest("request requires exactly one Content-Length")
        raw_length = lengths[0].strip()
        if not raw_length:
            self.close_connection = True
            raise InvalidRequest("request requires Content-Length")
        if not _DECIMAL.fullmatch(raw_length):
            self.close_connection = True
            raise InvalidRequest("Content-Length must be a valid non-negative integer")
        length = int(raw_length, 10)
        if length > MAX_REQUEST_BYTES:
            self.close_connection = True
            raise _HTTPProductError("request body exceeds 1 MiB", HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large")
        try:
            body = self.rfile.read(length)
        except (OSError, ValueError) as exc:
            self.close_connection = True
            raise InvalidRequest("request body could not be read") from exc
        if len(body) != length:
            self.close_connection = True
            raise _HTTPProductError("request body was truncated", HTTPStatus.BAD_REQUEST, "invalid_request")
        return body

    def _request(self) -> GenerationRequest:
        body = self._read_body()
        try:
            payload = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
            _reject_surrogates(payload)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise InvalidRequest("request body must be valid UTF-8 JSON") from exc
        return GenerationRequest.from_payload(payload, exact=True)

    def _service_ready(self) -> bool:
        result = self.server.service.health()
        return isinstance(result, dict) and result.get("ready") is True

    def do_GET(self) -> None:
        try:
            self._validate_boundary()
        except ProductError as error:
            self._send_error(error, close=True)
            return
        if not self._authenticate():
            return
        path = self._path()
        try:
            if path == "/health":
                # Process liveness is distinct from backend readiness.
                health = self.server.service.health()
                if not isinstance(health, dict):
                    raise ProductError("health response is invalid")
                self._send_json(HTTPStatus.OK, health)
            elif path == "/ready":
                health = self.server.service.health()
                if not isinstance(health, dict):
                    raise ProductError("health response is invalid")
                self._send_json(HTTPStatus.OK if health.get("ready") is True else HTTPStatus.SERVICE_UNAVAILABLE, health)
            elif path == "/v1/models":
                models = self.server.service.models()
                if not isinstance(models, list):
                    raise ProductError("model inventory response is invalid")
                self._send_json(HTTPStatus.OK, {"object": "list", "data": models})
            else:
                self._send_error(_HTTPProductError("unknown endpoint", HTTPStatus.NOT_FOUND, "not_found"))
        except ProductError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return
        except Exception:
            self._send_error(ProductError("product request failed"))

    def do_POST(self) -> None:
        try:
            self._validate_boundary()
        except ProductError as error:
            self._send_error(error, close=True)
            return
        if not self._authenticate():
            return
        if self._path() != "/v1/chat/completions":
            self._send_error(_HTTPProductError("unknown endpoint", HTTPStatus.NOT_FOUND, "not_found"), close=True)
            return
        request: GenerationRequest | None = None
        stream: Iterable[dict[str, Any]] | None = None
        try:
            request = self._request()
            if not self._service_ready():
                raise _HTTPProductError("product service is not ready", HTTPStatus.SERVICE_UNAVAILABLE, "backend_unavailable")
            stream = self.server.service.stream(request)
            if request.stream:
                self._serve_sse(request, stream)
            else:
                self._serve_json_completion(request, stream)
        except ProductError as error:
            # Once SSE headers have gone out, errors must use SSE framing.
            if request is not None and request.stream and self._headers_sent():
                self._finish_sse_error(error, request)
            else:
                self._send_error(error, close=getattr(error, "code", None) in {"invalid_request", "unauthorized"})
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            if request is not None:
                self._cancel(request.request_id)
        except Exception:
            error = ProductError("product request failed")
            if request is not None and request.stream and self._headers_sent():
                self._finish_sse_error(error, request)
            else:
                self._send_error(error)
        finally:
            if stream is not None:
                self._close_stream(stream)

    def _headers_sent(self) -> bool:
        # BaseHTTPRequestHandler has no public flag; this is set by our SSE path.
        return bool(getattr(self, "_sse_started", False))

    def _serve_json_completion(self, request: GenerationRequest, events: Iterable[dict[str, Any]]) -> None:
        text_parts: list[str] = []
        done: dict[str, Any] | None = None
        for event in events:
            kind = event.get("type") if isinstance(event, dict) else None
            if kind == "token":
                text = event.get("text")
                if not isinstance(text, str):
                    raise ProductError("service emitted an invalid token event")
                _reject_surrogates(text)
                text_parts.append(text)
            elif kind == "done":
                done = event
                break
            else:
                raise ProductError("service emitted an invalid completion event")
        if done is None:
            raise ProductError("service ended without a completion event")
        usage = self._usage(done)
        self._send_json(
            HTTPStatus.OK,
            {
                "id": request.request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text_parts)}, "finish_reason": self._finish_reason(done)}],
                "usage": usage,
            },
        )

    def _serve_sse(self, request: GenerationRequest, events: Iterable[dict[str, Any]]) -> None:
        iterator = iter(events)
        try:
            event = next(iterator)
        except StopIteration as exc:
            raise ProductError("service ended without a completion event") from exc
        # Validate the first service event before committing SSE headers.  A
        # synchronous admission/backend failure can therefore remain a normal
        # JSON error response; errors after this point use SSE framing.
        if not isinstance(event, dict) or event.get("type") not in ("token", "done"):
            raise ProductError("service emitted an invalid completion event")
        if event["type"] == "token" and not isinstance(event.get("text"), str):
            raise ProductError("service emitted an invalid token event")
        if event["type"] == "token":
            _reject_surrogates(event["text"])
        if event["type"] == "done":
            self._finish_reason(event)
            self._usage(event)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        # A finite SSE response has no Content-Length or chunked framing.
        # Close it after [DONE] so ordinary HTTP readers see its boundary
        # immediately instead of waiting for the next-request idle timeout.
        self.close_connection = True
        self.send_header("Connection", "close")
        self.end_headers()
        self._sse_started = True
        self._sse_finished = False
        created = int(time.time())
        self._write_sse({"id": request.request_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        while True:
            if not isinstance(event, dict):
                raise ProductError("service emitted an invalid completion event")
            kind = event.get("type")
            if kind == "token":
                text = event.get("text")
                if not isinstance(text, str):
                    raise ProductError("service emitted an invalid token event")
                _reject_surrogates(text)
                self._write_sse({"id": request.request_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
            elif kind == "done":
                chunk = {"id": request.request_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": self._finish_reason(event)}], "usage": self._usage(event)}
                self._write_sse(chunk)
                self._send_sse_done()
                return
            else:
                raise ProductError("service emitted an invalid completion event")
            try:
                event = next(iterator)
            except StopIteration as exc:
                raise ProductError("service ended without a completion event") from exc

    @staticmethod
    def _usage(done: dict[str, Any]) -> dict[str, int]:
        prompt = done.get("prompt_tokens", 0)
        completion = done.get("completion_tokens", 0)
        if type(prompt) is not int or type(completion) is not int or prompt < 0 or completion < 0:
            raise ProductError("service emitted invalid usage")
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    def _write_sse(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
        self.wfile.write(("data: " + encoded + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _send_sse_error(self, error: ProductError) -> None:
        if self._headers_sent() and not getattr(self, "_sse_finished", False):
            self._write_sse(_error_payload(error))

    def _send_sse_done(self) -> None:
        if self._headers_sent() and not getattr(self, "_sse_finished", False):
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self._sse_finished = True

    def _finish_sse_error(self, error: ProductError, request: GenerationRequest) -> None:
        try:
            self._send_sse_error(error)
            self._send_sse_done()
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            self._cancel(request.request_id)

    def _cancel(self, request_id: str) -> None:
        try:
            self.server.service.cancel(request_id)
        except Exception:
            pass

    @staticmethod
    def _close_stream(stream: Iterable[dict[str, Any]]) -> None:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class _HTTPProductError(ProductError):
    def __init__(self, message: str, status: int, code: str):
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self._public_message = message


def create_server(
    service: Any,
    host: str = "127.0.0.1",
    port: int = 8080,
    api_key: str | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> ThreadingHTTPServer:
    """Create a local product HTTP server around an injected service.

    The service must provide ``health()``, ``models()``, ``stream(request)``,
    ``cancel(request_id)`` and ``close()``.  Remote binds require both an API
    key and a TLS certificate/key pair; loopback binds may omit authentication.
    """

    if (tls_cert is None) != (tls_key is None):
        raise ValueError("tls_cert and tls_key must be provided together")
    if api_key is not None and (
        not isinstance(api_key, str)
        or not api_key
        or any(ord(character) > 0x7F or character in _CONTROL for character in api_key)
    ):
        raise ValueError("api_key must be a non-empty ASCII string without control characters")
    if not _is_loopback(host) and (not api_key or not tls_cert or not tls_key):
        raise ValueError("non-loopback server requires api_key and TLS certificate/key")
    context: ssl.SSLContext | None = None
    if tls_cert is not None and tls_key is not None:
        # Validate certificate material before creating/binding a listening
        # socket. A bad remote TLS setup must never briefly expose a port.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(tls_cert, tls_key)
    try:
        literal_host = ipaddress.ip_address(host)
    except ValueError:
        literal_host = None
    server_class = _ProductHTTPServerV6 if literal_host is not None and literal_host.version == 6 else _ProductHTTPServer
    return server_class((host, port), service, api_key, tls_enabled=context is not None, tls_context=context)


__all__ = ["create_server"]
