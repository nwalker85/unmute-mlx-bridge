from __future__ import annotations

import json

from prometheus_client import generate_latest
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


def test_buffered_tts_metrics_have_stable_names_and_failure_labels():
    metrics = Metrics()
    metrics.buffered_input_characters.observe(11)
    metrics.buffered_audio_seconds.observe(1.5)
    metrics.buffered_synthesis_seconds.observe(2.5)
    metrics.buffered_eos_to_first_emit_seconds.observe(2.6)
    metrics.buffered_turn_failures.labels(reason="generation").inc()

    payload = generate_latest(metrics.registry).decode()

    assert "bridge_tts_buffered_input_characters_count 1.0" in payload
    assert "bridge_tts_buffered_audio_seconds_count 1.0" in payload
    assert "bridge_tts_buffered_synthesis_seconds_count 1.0" in payload
    assert "bridge_tts_buffered_eos_to_first_emit_seconds_count 1.0" in payload
    assert (
        'bridge_tts_buffered_turn_failures_total{reason="generation"} 1.0'
        in payload
    )
