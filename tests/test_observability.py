from __future__ import annotations

import json

from prometheus_client import generate_latest
from websockets.datastructures import Headers
from websockets.http11 import Request

from unmute_mlx_bridge.observability import (
    Metrics,
    ServiceHealth,
    build_process_request,
    check_auth,
)


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


def test_check_auth_accepts_single_valid_auth_id_query_param():
    request = Request("/api/test_streaming?auth_id=public_token", Headers())
    assert check_auth(request, frozenset({"public_token"})) is True


def test_check_auth_rejects_repeated_auth_id_query_param():
    """RAV-1552: `check_auth` used to take `query.get("auth_id", [None])[0]`
    unconditionally -- a repeated `?auth_id=public_token&auth_id=other`
    silently authorized (or rejected) based on whichever value happened to
    come first, with no signal that the request was malformed. This gate
    must reject fail-closed instead, consistent with
    `tts/server.py::_parse_query`'s own repeated-`auth_id=` guard (which runs
    on the same field, after the upgrade this gate controls) -- the two must
    never be able to disagree about whether a given `auth_id=` query string
    is acceptable."""
    request = Request(
        "/api/test_streaming?auth_id=public_token&auth_id=other", Headers()
    )
    assert check_auth(request, frozenset({"public_token"})) is False

    # Order must not matter: putting the valid id first must not sneak past.
    request = Request(
        "/api/test_streaming?auth_id=other&auth_id=public_token", Headers()
    )
    assert check_auth(request, frozenset({"public_token"})) is False


def test_check_auth_rejects_missing_auth_id_query_param():
    request = Request("/api/test_streaming", Headers())
    assert check_auth(request, frozenset({"public_token"})) is False


def test_check_auth_disabled_when_authorized_ids_is_empty():
    request = Request("/api/test_streaming", Headers())
    assert check_auth(request, frozenset()) is True
