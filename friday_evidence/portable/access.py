"""Read-only inspection of the configured official Kaggle MCP service."""
from __future__ import annotations

import json
import os

from .credentials import configured_kaggle_credentials


def mcp_inventory(*, read_tool: str | None = None, request: dict | None = None):
    if read_tool is not None and read_tool not in {
        "get_accelerator_quota", "get_user_profile", "get_notebook_session_status",
        "get_notebook_info", "get_dataset_status", "search_notebooks",
    }:
        raise ValueError("read_only_mcp_tool_not_allowed")
    import requests
    with configured_kaggle_credentials():
        with requests.Session() as session:
            session.headers.update({"Authorization": "Bearer " + os.environ["KAGGLE_API_TOKEN"],
                                    "Accept": "application/json, text/event-stream"})
            def rpc(method: str, params: dict, number: int):
                response = session.post("https://www.kaggle.com/mcp", json={
                    "jsonrpc": "2.0", "id": number, "method": method, "params": params},
                    timeout=30, allow_redirects=False)
                if response.status_code != 200:
                    raise RuntimeError(f"kaggle_mcp_http_{response.status_code}")
                if "Mcp-Session-Id" in response.headers:
                    session.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
                if response.headers.get("content-type", "").startswith("text/event-stream"):
                    rows = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                    result = next(row for row in rows if row.get("id") == number)
                else:
                    result = response.json()
                if "error" in result:
                    raise RuntimeError("kaggle_mcp_rpc_error")
                return result["result"]
            rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": {"name": "ironmule-data1-readonly", "version": "1"}}, 1)
            if read_tool is not None:
                return rpc("tools/call", {"name": read_tool, "arguments": {"request": request or {}}}, 2)
            result = rpc("tools/list", {}, 2)
            return [{"name": item["name"], "inputSchema": item.get("inputSchema", {})} for item in result["tools"]]
