"""Stable error categories at the local API and worker boundaries."""


class ProductError(Exception):
    code = "product_error"
    status = 500


class InvalidRequest(ProductError):
    code = "invalid_request"
    status = 400


class ModelNotFound(ProductError):
    code = "model_not_found"
    status = 404


class BackendUnavailable(ProductError):
    code = "backend_unavailable"
    status = 503


class Overloaded(ProductError):
    code = "overloaded"
    status = 429


class RequestTimeout(ProductError):
    code = "request_timeout"
    status = 504


class RequestCancelled(ProductError):
    code = "request_cancelled"
    status = 499


class StateError(ProductError):
    code = "state_error"
    status = 500
