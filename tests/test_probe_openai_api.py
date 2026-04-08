"""Focused tests for the OpenAI-compatible probe utility."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import engllm_chat.probe_openai_api as probe


class _IterablePage:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)


class _FakeModelsAPI:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def list(self):
        return SimpleNamespace(data=self._items)


class _FakeClient:
    last_init: dict[str, object] | None = None

    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        type(self).last_init = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
        }
        self.models = _FakeModelsAPI(
            [
                SimpleNamespace(id="gpt-5-mini"),
                SimpleNamespace(id="text-embedding-3-small"),
                SimpleNamespace(id="gpt-image-1"),
                SimpleNamespace(id="gpt-4o-mini-tts"),
            ]
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda: None))


def _make_context(**overrides: object) -> probe.ProbeContext:
    raw: dict[str, object] = {
        "base_url": "https://example.test/v1",
        "text_model": "gpt-5-mini",
        "embedding_model": "text-embedding-3-small",
        "image_model": "gpt-image-1",
        "tts_model": "gpt-4o-mini-tts",
        "include_images": False,
        "include_audio": False,
    }
    raw.update(overrides)
    return probe.ProbeContext(**raw)


def test_probe_helper_selection_and_extraction_functions() -> None:
    root = SimpleNamespace(child=SimpleNamespace(value=3))
    assert probe._resolve_sdk_target(root, ("child", "value")) == 3
    assert probe._resolve_sdk_target(root, ("child", "missing")) is None

    page = SimpleNamespace(data=[SimpleNamespace(id="gpt-5-mini"), object()])
    assert probe._page_items(page) == page.data
    assert probe._page_items(_IterablePage([SimpleNamespace(id="model-a")])) == [
        SimpleNamespace(id="model-a")
    ]
    assert probe._extract_model_ids(page) == ["gpt-5-mini"]

    model_ids = [
        "text-embedding-3-small",
        "gpt-5-mini",
        "gpt-image-1",
        "gpt-4o-mini-tts",
    ]
    assert probe._pick_text_model(model_ids) == "gpt-5-mini"
    assert probe._pick_embedding_model(model_ids) == "text-embedding-3-small"
    assert probe._pick_image_model(model_ids) == "gpt-image-1"
    assert probe._pick_tts_model(model_ids) == "gpt-4o-mini-tts"


def test_probe_error_classification_helpers_cover_major_paths() -> None:
    response_exc = Exception("boom")
    response_exc.status_code = 404  # type: ignore[attr-defined]
    assert probe._extract_status_code(response_exc) == 404

    body_exc = Exception("boom")
    body_exc.body = {"error": "nope"}  # type: ignore[attr-defined]
    assert '"error": "nope"' in probe._extract_error_text(body_exc)

    response_text_exc = Exception("boom")
    response_text_exc.response = SimpleNamespace(status_code=403, text="forbidden")  # type: ignore[attr-defined]
    status, http_status, detail = probe._classify_exception(response_text_exc)
    assert (status, http_status, detail) == ("restricted", 403, "forbidden")

    unavailable_exc = Exception("unknown url")
    unavailable_exc.response = SimpleNamespace(status_code=None, text="unknown url")  # type: ignore[attr-defined]
    assert probe._classify_exception(unavailable_exc)[0] == "unavailable"

    api_timeout = type("APITimeoutError", (Exception,), {})
    timeout_exc = api_timeout("timed out")
    assert probe._classify_exception(timeout_exc)[0] == "indeterminate"

    available_exc = Exception("bad request")
    available_exc.status_code = 422  # type: ignore[attr-defined]
    assert probe._classify_exception(available_exc)[0] == "available"


def test_probe_response_parsing_helpers_validate_expected_payloads() -> None:
    assert probe._build_json_schema_response_format("probe")["type"] == "json_schema"
    assert (
        probe._build_responses_text_format("probe")["format"]["type"] == "json_schema"
    )

    direct = SimpleNamespace(output_text='{"status":"ok","value":1}')
    assert probe._extract_response_output_text(direct) == '{"status":"ok","value":1}'

    nested = SimpleNamespace(
        output=[
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"status":"ok",'},  # noqa: E501
                    {"type": "text", "text": '"value":1}'},
                ],
            }
        ]
    )
    assert probe._extract_response_output_text(nested) == '{"status":"ok","value":1}'

    payload = probe._load_json_object('{"status":"ok","value":1}', source="probe")
    probe._validate_probe_payload(payload, source="probe")

    with pytest.raises(probe.ProbeFailure, match="did not return valid JSON"):
        probe._load_json_object("not-json", source="probe")

    with pytest.raises(probe.ProbeFailure, match="not an object"):
        probe._load_json_object('["bad"]', source="probe")

    with pytest.raises(probe.ProbeFailure, match="unexpected status field"):
        probe._validate_probe_payload({"status": "bad", "value": 1}, source="probe")

    with pytest.raises(probe.ProbeFailure, match="unexpected value field"):
        probe._validate_probe_payload({"status": "ok", "value": 2}, source="probe")


def test_probe_tool_call_extractors_handle_success_and_missing_calls() -> None:
    chat_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="report_probe",
                                arguments='{"status":"ok"}',
                            )
                        )
                    ]
                )
            )
        ]
    )
    assert probe._extract_chat_tool_call(chat_response) == (
        "report_probe",
        {"status": "ok"},
    )

    responses_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="report_probe",
                arguments='{"status":"ok"}',
            )
        ]
    )
    assert probe._extract_responses_tool_call(responses_response) == (
        "report_probe",
        {"status": "ok"},
    )

    with pytest.raises(probe.ProbeFailure, match="returned no tool_calls"):
        probe._extract_chat_tool_call(
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))]
            )
        )

    with pytest.raises(probe.ProbeFailure, match="returned no function_call items"):
        probe._extract_responses_tool_call(SimpleNamespace(output=[]))


def test_probe_runtime_tiers_and_summary_reflect_engllm_chat_requirements() -> None:
    assert probe._is_runtime_required("required_for_engllm_chat") is True
    assert probe._is_runtime_required("optional_for_engllm_chat") is False

    ready_results = [
        probe.ProbeResult(
            name="chat.completions.create",
            description="Chat",
            runtime_tier="required_for_engllm_chat",
            runtime_required=True,
            sdk_path="chat.completions.create",
            cost="low-cost inference",
            status="available",
            detail="ok",
            http_status=200,
            elapsed_ms=1,
        ),
        probe.ProbeResult(
            name="chat.completions.create.structured_output",
            description="Structured",
            runtime_tier="required_for_engllm_chat",
            runtime_required=True,
            sdk_path="chat.completions.create",
            cost="low-cost inference",
            status="available",
            detail="ok",
            http_status=200,
            elapsed_ms=1,
        ),
        probe.ProbeResult(
            name="models.list",
            description="Models",
            runtime_tier="optional_for_engllm_chat",
            runtime_required=False,
            sdk_path="models.list",
            cost="read-only",
            status="unavailable",
            detail="not needed",
            http_status=404,
            elapsed_ms=1,
        ),
    ]
    summary = probe._build_runtime_summary(ready_results, text_model="gpt-5-mini")
    assert summary["status"] == "ready"
    assert summary["blocking_operations"] == []

    not_ready = probe._build_runtime_summary(
        [
            *ready_results[:1],
            ready_results[1].__class__(
                **{**ready_results[1].__dict__, "status": "restricted"}
            ),
        ],
        text_model="gpt-5-mini",
    )
    assert not_ready["status"] == "not_ready"

    indeterminate = probe._build_runtime_summary(
        [
            *ready_results[:1],
            ready_results[1].__class__(
                **{**ready_results[1].__dict__, "status": "indeterminate"}
            ),
        ],
        text_model="gpt-5-mini",
    )
    assert indeterminate["status"] == "indeterminate"

    no_model = probe._build_runtime_summary(ready_results, text_model=None)
    assert no_model["status"] == "not_ready"


def test_probe_individual_api_checks_cover_success_and_precondition_paths() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _record(name: str):
        def inner(**kwargs: object):
            calls.append((name, kwargs))
            return SimpleNamespace(
                output_text='{"status":"ok","value":1}',
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="report_probe",
                        arguments='{"status":"ok"}',
                    )
                ],
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"status":"ok","value":1}',
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="report_probe",
                                        arguments='{"status":"ok"}',
                                    )
                                )
                            ],
                        )
                    )
                ],
            )

        return inner

    client = SimpleNamespace(
        responses=SimpleNamespace(create=_record("responses.create")),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_record("chat.completions.create"))
        ),
        beta=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    parse=_record("beta.chat.completions.parse")
                )
            )
        ),
        embeddings=SimpleNamespace(create=_record("embeddings.create")),
        moderations=SimpleNamespace(create=_record("moderations.create")),
        files=SimpleNamespace(list=lambda: _IterablePage([1, 2])),
        fine_tuning=SimpleNamespace(
            jobs=SimpleNamespace(list=lambda: _IterablePage([1, 2, 3]))
        ),
        batches=SimpleNamespace(list=lambda: _IterablePage([1])),
        vector_stores=SimpleNamespace(list=lambda: _IterablePage([1, 2, 3, 4])),
        images=SimpleNamespace(generate=_record("images.generate")),
        audio=SimpleNamespace(
            speech=SimpleNamespace(create=_record("audio.speech.create"))
        ),
    )

    context = _make_context(include_images=True, include_audio=True)

    assert probe._probe_responses_create(client, context)[0] == 200
    assert probe._probe_responses_structured_output(client, context)[1].startswith(
        "returned schema-valid JSON"
    )
    assert probe._probe_responses_tool_calls(client, context)[1].startswith(
        "returned a valid function_call"
    )
    assert probe._probe_chat_completions_create(client, context)[0] == 200
    assert probe._probe_chat_completions_structured_output(client, context)[0] == 200
    assert probe._probe_beta_chat_completions_parse(client, context)[1].startswith(
        "returned schema-valid JSON payload via beta.parse"
    )
    assert probe._probe_chat_completions_tool_calls(client, context)[0] == 200
    assert probe._probe_embeddings_create(client, context)[0] == 200
    assert probe._probe_moderations_create(client, context)[0] == 200
    assert probe._probe_files_list(client, context)[1] == "listed 2 files on first page"
    assert probe._probe_fine_tuning_jobs_list(client, context)[1] == (
        "listed 3 fine-tuning jobs on first page"
    )
    assert (
        probe._probe_batches_list(client, context)[1]
        == "listed 1 batches on first page"
    )
    assert probe._probe_vector_stores_list(client, context)[1] == (
        "listed 4 vector stores on first page"
    )
    assert probe._probe_images_generate(client, context)[0] == 200
    assert probe._probe_audio_speech_create(client, context)[0] == 200

    with pytest.raises(ValueError, match="no text model configured"):
        probe._probe_responses_create(client, _make_context(text_model=None))
    with pytest.raises(ValueError, match="image probes disabled"):
        probe._probe_images_generate(client, _make_context(include_images=False))
    with pytest.raises(ValueError, match="audio probes disabled"):
        probe._probe_audio_speech_create(client, _make_context(include_audio=False))

    assert any(name == "responses.create" for name, _kwargs in calls)
    assert any(name == "chat.completions.create" for name, _kwargs in calls)
    assert any(name == "beta.chat.completions.parse" for name, _kwargs in calls)


def test_probe_individual_api_checks_reject_unexpected_probe_payloads() -> None:
    bad_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(output_text="", output=[])
        ),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(choices=[])
            )
        ),
    )
    context = _make_context()

    with pytest.raises(probe.ProbeFailure, match="returned no text"):
        probe._probe_responses_structured_output(bad_client, context)

    with pytest.raises(probe.ProbeFailure, match="returned no function_call items"):
        probe._probe_responses_tool_calls(bad_client, context)

    with pytest.raises(probe.ProbeFailure, match="returned no choices"):
        probe._probe_chat_completions_structured_output(bad_client, context)

    with pytest.raises(probe.ProbeFailure, match="returned no tool_calls"):
        probe._probe_chat_completions_tool_calls(
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: SimpleNamespace(
                            choices=[
                                SimpleNamespace(message=SimpleNamespace(tool_calls=[]))
                            ]
                        )
                    )
                )
            ),
            context,
        )

    # beta.chat.completions.parse — empty choices
    with pytest.raises(probe.ProbeFailure, match="returned no choices"):
        probe._probe_beta_chat_completions_parse(
            SimpleNamespace(
                beta=SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(
                            parse=lambda **_kwargs: SimpleNamespace(choices=[])
                        )
                    )
                )
            ),
            context,
        )

    # beta.chat.completions.parse — SDK surface missing
    with pytest.raises(probe.ProbeFailure, match="not found in SDK"):
        probe._probe_beta_chat_completions_parse(
            SimpleNamespace(),
            context,
        )


def test_probe_operation_handles_all_major_result_states() -> None:
    context = _make_context()

    missing = probe._probe_operation(
        SimpleNamespace(),
        probe.OperationSpec(
            name="missing",
            description="Missing target",
            runtime_tier="extra_compatibility_surface",
            sdk_path=("foo", "bar"),
            probe=lambda _client, _ctx: (200, "never"),
            cost="read-only",
        ),
        context,
    )
    assert missing.status == "unavailable"

    skipped = probe._probe_operation(
        SimpleNamespace(files=SimpleNamespace(list=lambda: None)),
        probe.OperationSpec(
            name="files.list",
            description="Manual",
            runtime_tier="extra_compatibility_surface",
            sdk_path=("files", "list"),
            probe=None,
            cost="manual",
        ),
        context,
    )
    assert skipped.status == "skipped"

    value_error = probe._probe_operation(
        SimpleNamespace(models=SimpleNamespace(list=lambda: None)),
        probe.OperationSpec(
            name="models.list",
            description="Needs setup",
            runtime_tier="optional_for_engllm_chat",
            sdk_path=("models", "list"),
            probe=lambda _client, _ctx: (_ for _ in ()).throw(ValueError("not ready")),
            cost="read-only",
        ),
        context,
    )
    assert value_error.status == "skipped"
    assert value_error.detail == "not ready"

    probe_failure = probe._probe_operation(
        SimpleNamespace(models=SimpleNamespace(list=lambda: None)),
        probe.OperationSpec(
            name="models.list",
            description="Probe failure",
            runtime_tier="optional_for_engllm_chat",
            sdk_path=("models", "list"),
            probe=lambda _client, _ctx: (_ for _ in ()).throw(
                probe.ProbeFailure(
                    status="restricted", detail="denied", http_status=403
                )
            ),
            cost="read-only",
        ),
        context,
    )
    assert probe_failure.status == "restricted"
    assert probe_failure.http_status == 403

    class _HTTP404Error(Exception):
        def __init__(self) -> None:
            super().__init__("missing")
            self.status_code = 404

    classified = probe._probe_operation(
        SimpleNamespace(models=SimpleNamespace(list=lambda: None)),
        probe.OperationSpec(
            name="models.list",
            description="Classified exception",
            runtime_tier="optional_for_engllm_chat",
            sdk_path=("models", "list"),
            probe=lambda _client, _ctx: (_ for _ in ()).throw(_HTTP404Error()),
            cost="read-only",
        ),
        context,
    )
    assert classified.status == "unavailable"


def test_probe_operations_with_progress_and_format_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _make_context()
    operations = [
        probe.OperationSpec(
            name="models.list",
            description="List models",
            runtime_tier="optional_for_engllm_chat",
            sdk_path=("models", "list"),
            probe=lambda _client, _ctx: (200, "listed"),
            cost="read-only",
        )
    ]
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: None))

    results = probe._probe_operations_with_progress(
        client,
        operations,
        context,
        progress_enabled=True,
        total_operations=1,
    )

    captured = capsys.readouterr()
    assert "[1/1] probing models.list" in captured.err
    assert "[1/1] available models.list" in captured.err

    table = probe._format_table(results)
    assert "status" in table
    assert "models.list" in table
    assert "listed" in table


def test_probe_build_parser_uses_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    parser = probe._build_parser()
    args = parser.parse_args([])

    assert args.base_url == "https://env.example/v1"
    assert args.api_key == "env-key"


def test_probe_main_handles_missing_sdk(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        probe,
        "_load_openai_sdk",
        lambda: (_ for _ in ()).throw(RuntimeError("sdk missing")),
    )

    rc = probe.main(["--base-url", "https://example.test/v1", "--api-key", "token"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "sdk missing" in captured.err


def test_probe_main_emits_json_report_with_discovered_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(probe, "_load_openai_sdk", lambda: (_FakeClient, "1.2.3"))
    monkeypatch.setattr(
        probe,
        "OPERATIONS",
        (
            probe.OperationSpec(
                name="models.list",
                description="List registered models",
                runtime_tier="optional_for_engllm_chat",
                sdk_path=("models", "list"),
                probe=probe._probe_models_list,
                cost="read-only",
            ),
            probe.OperationSpec(
                name="chat.completions.create",
                description="Chat",
                runtime_tier="required_for_engllm_chat",
                sdk_path=("chat", "completions", "create"),
                probe=lambda _client, context: (
                    200,
                    f"request succeeded with model {context.text_model!r}",
                ),
                cost="low-cost inference",
            ),
        ),
    )

    rc = probe.main(
        [
            "--base-url",
            "https://example.test/v1",
            "--api-key",
            "token",
            "--json",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    report = json.loads(captured.out)
    assert report["base_url"] == "https://example.test/v1"
    assert report["sdk_version"] == "1.2.3"
    assert report["selected_models"]["text_model"] == "gpt-5-mini"
    assert report["results"][0]["name"] == "models.list"
    assert report["results"][0]["runtime_tier"] == "optional_for_engllm_chat"
    assert report["engllm_chat_runtime"]["status"] == "ready"
    assert "summary_by_tier" in report
    assert _FakeClient.last_init == {
        "api_key": "token",
        "base_url": "https://example.test/v1",
        "timeout": 30.0,
    }


def test_probe_main_emits_text_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(probe, "_load_openai_sdk", lambda: (_FakeClient, "1.2.3"))
    monkeypatch.setattr(
        probe,
        "OPERATIONS",
        (
            probe.OperationSpec(
                name="models.list",
                description="List registered models",
                runtime_tier="optional_for_engllm_chat",
                sdk_path=("models", "list"),
                probe=probe._probe_models_list,
                cost="read-only",
            ),
            probe.OperationSpec(
                name="chat.completions.create",
                description="Chat Completions API",
                runtime_tier="required_for_engllm_chat",
                sdk_path=("chat", "completions", "create"),
                probe=lambda _client, _ctx: (200, "chat ok"),
                cost="low-cost inference",
            ),
            probe.OperationSpec(
                name="chat.completions.create.structured_output",
                description="Structured output",
                runtime_tier="required_for_engllm_chat",
                sdk_path=("chat", "completions", "create"),
                probe=lambda _client, _ctx: (200, "structured ok"),
                cost="low-cost inference",
            ),
        ),
    )

    rc = probe.main(
        [
            "--base-url",
            "https://example.test/v1",
            "--api-key",
            "token",
            "--no-progress",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "OpenAI-Compatible API Probe" in captured.out
    assert "Base URL: https://example.test/v1" in captured.out
    assert "engllm-chat runtime: READY" in captured.out
    assert "Required for engllm-chat:" in captured.out
    assert "tier" in captured.out
    assert "models.list" in captured.out
