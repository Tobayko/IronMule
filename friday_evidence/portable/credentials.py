"""Opt-in reuse of the user's Kaggle MCP credential, only in process memory."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import tomllib


@contextmanager
def configured_kaggle_credentials(config_path: Path | None = None):
    """Never persist or print the token, and never send it to a different host.

    Older user configurations may accidentally place the literal Kaggle token
    in ``bearer_token_env_var``. The explicit CLI opt-in permits recovery of
    that exact Kaggle credential without changing the user's Codex config.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        yield
        return
    source = config_path or Path.home() / ".codex" / "config.toml"
    try:
        settings = tomllib.loads(source.read_text())["mcp_servers"]["kaggle"]
        if settings.get("url") != "https://www.kaggle.com/mcp" or settings.get("enabled", True) is not True:
            raise ValueError
        reference = settings.get("bearer_token_env_var", "")
        token = os.environ.get(reference) if isinstance(reference, str) else None
        if not token and isinstance(reference, str) and re.fullmatch(r"KGAT_[A-Za-z0-9_-]{16,256}", reference):
            token = reference
        if not token:
            header = settings.get("http_headers", {}).get("Authorization", "")
            token = header.removeprefix("Bearer ") if header.startswith("Bearer ") else None
        if not isinstance(token, str) or not re.fullmatch(r"KGAT_[A-Za-z0-9_-]{16,256}", token):
            raise ValueError
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError("kaggle_credential_unavailable") from None
    os.environ["KAGGLE_API_TOKEN"] = token
    try:
        yield
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)

