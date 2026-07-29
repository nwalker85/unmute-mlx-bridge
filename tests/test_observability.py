from __future__ import annotations

import json

from websockets.datastructures import Headers
from websockets.http11 import Request

from unmute_mlx_bridge.observability import Metrics, ServiceHealth, build_process_request


async def test_build_info_matches_moshi_server_http_contract():
    process_request = build_process_request(
        ServiceHealth(),
        Metrics(),
        "/api/test_streaming",
        frozenset({"public_token"}),
    )

    response = await process_request(None, Request("/api/build_info", Headers()))

    assert response is not None
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    build_info = json.loads(response.body)
    assert set(build_info) == {
        "build_timestamp",
        "build_date",
        "git_branch",
        "git_timestamp",
        "git_date",
        "git_hash",
        "git_describe",
        "rustc_host_triple",
        "rustc_version",
        "cargo_target_triple",
    }
    assert build_info["git_describe"].startswith("unmute-mlx-bridge ")
