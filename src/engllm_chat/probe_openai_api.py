#!/usr/bin/env python3
"""Probe an OpenAI-compatible endpoint and report available API surfaces.

This script uses the official OpenAI Python SDK against a caller-supplied
`base_url` and API token. It performs lightweight live probes where possible
and classifies each operation as one of:

- `available`: the endpoint succeeded or clearly exists
- `unavailable`: the endpoint appears missing or unimplemented
- `restricted`: the endpoint exists but this token cannot use it
- `skipped`: the probe was not attempted because a prerequisite was missing
- `indeterminate`: connectivity or server behavior prevented a conclusion

The goal is to help compare OpenAI-compatible endpoints that implement only a
subset of the SDK surface, such as chat completions without the Responses API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

ProbeCallable = Callable[[Any, "ProbeContext"], tuple[int | None, str]]
RuntimeTier = Literal[
    "required_for_engllm_chat",
    "optional_for_engllm_chat",
    "extra_compatibility_surface",
]

_PROBE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "value": {"type": "integer"},
    },
    "required": ["status", "value"],
    "additionalProperties": False,
}

_PROBE_TOOL_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
    },
    "required": ["status"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProbeContext:
    """Resolved runtime inputs for live endpoint probing."""

    base_url: str
    text_model: str | None
    embedding_model: str | None
    image_model: str | None
    tts_model: str | None
    include_images: bool
    include_audio: bool


@dataclass(frozen=True)
class OperationSpec:
    """One SDK operation plus its live-probe implementation."""

    name: str
    description: str
    runtime_tier: RuntimeTier
    sdk_path: tuple[str, ...]
    probe: ProbeCallable | None
    cost: str


@dataclass(frozen=True)
class ProbeResult:
    """Outcome for one probed SDK operation."""

    name: str
    description: str
    runtime_tier: RuntimeTier
    runtime_required: bool
    sdk_path: str
    cost: str
    status: str
    detail: str
    http_status: int | None
    elapsed_ms: int


class ProbeFailure(Exception):
    """A probe reached the endpoint but the feature check failed."""

    def __init__(
        self,
        *,
        status: str,
        detail: str,
        http_status: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.http_status = http_status


def _resolve_sdk_target(root: Any, path: Sequence[str]) -> Any | None:
    current = root
    for segment in path:
        current = getattr(current, segment, None)
        if current is None:
            return None
    return current


def _load_openai_sdk() -> tuple[type[Any], str]:
    try:
        import openai
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The OpenAI SDK is not installed. Install project dependencies or "
            "pip install openai before running this script."
        ) from exc
    return OpenAI, getattr(openai, "__version__", "unknown")


def _extract_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _extract_error_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        try:
            return json.dumps(body, ensure_ascii=True, sort_keys=True)
        except Exception:
            return str(body)
    if isinstance(body, str) and body.strip():
        return body.strip()

    response = getattr(exc, "response", None)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    message = str(exc).strip()
    return message or exc.__class__.__name__


def _object_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _classify_exception(exc: Exception) -> tuple[str, int | None, str]:
    status_code = _extract_status_code(exc)
    detail = _extract_error_text(exc)
    lowered = detail.lower()
    error_type = exc.__class__.__name__

    if error_type in {"APIConnectionError", "APITimeoutError"}:
        return "indeterminate", status_code, detail

    if status_code in {401, 403}:
        return "restricted", status_code, detail

    if status_code in {404, 405, 501}:
        return "unavailable", status_code, detail

    if status_code is not None and status_code >= 500:
        return "indeterminate", status_code, detail

    unavailable_signals = (
        "not found",
        "not implemented",
        "unsupported",
        "unknown url",
        "no route",
        "does not exist",
        "method not allowed",
    )
    if any(signal in lowered for signal in unavailable_signals):
        return "unavailable", status_code, detail

    if status_code in {400, 409, 422, 429}:
        return "available", status_code, detail

    return "indeterminate", status_code, f"{error_type}: {detail}"


def _page_items(page: Any) -> list[Any]:
    data = getattr(page, "data", None)
    if isinstance(data, list):
        return data
    try:
        return list(page)
    except Exception:
        return []


def _extract_model_ids(page: Any) -> list[str]:
    model_ids: list[str] = []
    for item in _page_items(page):
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return model_ids


def _pick_text_model(model_ids: Sequence[str]) -> str | None:
    if not model_ids:
        return None

    preferred_terms = ("gpt", "chat", "instruct", "llama", "qwen", "mistral")
    excluded_terms = (
        "embed",
        "embedding",
        "moderation",
        "tts",
        "whisper",
        "transcribe",
        "image",
        "dall",
    )
    for model_id in model_ids:
        lowered = model_id.lower()
        if any(term in lowered for term in preferred_terms) and not any(
            term in lowered for term in excluded_terms
        ):
            return model_id
    for model_id in model_ids:
        lowered = model_id.lower()
        if not any(term in lowered for term in excluded_terms):
            return model_id
    return None


def _pick_embedding_model(model_ids: Sequence[str]) -> str | None:
    for model_id in model_ids:
        lowered = model_id.lower()
        if "embed" in lowered:
            return model_id
    return None


def _pick_image_model(model_ids: Sequence[str]) -> str | None:
    for model_id in model_ids:
        lowered = model_id.lower()
        if "image" in lowered or "dall" in lowered:
            return model_id
    return None


def _pick_tts_model(model_ids: Sequence[str]) -> str | None:
    for model_id in model_ids:
        lowered = model_id.lower()
        if "tts" in lowered:
            return model_id
    return None


def _build_json_schema_response_format(schema_name: str) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": _PROBE_JSON_SCHEMA,
        },
    }


def _build_responses_text_format(schema_name: str) -> dict[str, object]:
    return {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "schema": _PROBE_JSON_SCHEMA,
        }
    }


def _extract_response_output_text(response: Any) -> str:
    direct_text = _object_get(response, "output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    output = _object_get(response, "output")
    if not isinstance(output, list):
        return ""

    text_parts: list[str] = []
    for item in output:
        if _object_get(item, "type") != "message":
            continue
        content = _object_get(item, "content")
        if not isinstance(content, list):
            continue
        for entry in content:
            if _object_get(entry, "type") in {"output_text", "text"}:
                text = _object_get(entry, "text")
                if isinstance(text, str):
                    text_parts.append(text)
    return "".join(text_parts).strip()


def _load_json_object(raw_text: str, *, source: str) -> dict[str, object]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProbeFailure(
            status="unavailable",
            detail=f"{source} did not return valid JSON: {exc}",
        ) from exc

    if not isinstance(payload, dict):
        raise ProbeFailure(
            status="unavailable",
            detail=f"{source} returned JSON but not an object",
        )
    return payload


def _validate_probe_payload(payload: dict[str, object], *, source: str) -> None:
    if payload.get("status") != "ok":
        raise ProbeFailure(
            status="unavailable",
            detail=f"{source} returned unexpected status field: {payload!r}",
        )
    if payload.get("value") != 1:
        raise ProbeFailure(
            status="unavailable",
            detail=f"{source} returned unexpected value field: {payload!r}",
        )


def _extract_chat_tool_call(
    response: Any,
) -> tuple[str, dict[str, object]]:
    choices = _object_get(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ProbeFailure(
            status="unavailable",
            detail="chat tool-call probe returned no choices",
        )

    message = _object_get(choices[0], "message")
    if message is None:
        raise ProbeFailure(
            status="unavailable",
            detail="chat tool-call probe returned no message",
        )

    tool_calls = _object_get(message, "tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ProbeFailure(
            status="indeterminate",
            detail=(
                "chat tool-call probe returned no tool_calls; "
                "the endpoint accepted the request but the selected model did not "
                "emit a tool call"
            ),
        )

    function = _object_get(tool_calls[0], "function")
    name = _object_get(function, "name")
    arguments = _object_get(function, "arguments")
    if not isinstance(name, str) or not name:
        raise ProbeFailure(
            status="unavailable",
            detail="chat tool-call probe returned no function name",
        )
    if not isinstance(arguments, str) or not arguments.strip():
        raise ProbeFailure(
            status="unavailable",
            detail="chat tool-call probe returned no function arguments",
        )
    return name, _load_json_object(arguments, source="chat tool-call probe")


def _extract_responses_tool_call(
    response: Any,
) -> tuple[str, dict[str, object]]:
    output = _object_get(response, "output")
    if not isinstance(output, list):
        raise ProbeFailure(
            status="unavailable",
            detail="responses tool-call probe returned no output list",
        )

    for item in output:
        item_type = _object_get(item, "type")
        if item_type != "function_call":
            continue
        name = _object_get(item, "name")
        arguments = _object_get(item, "arguments")
        if not isinstance(name, str) or not name:
            raise ProbeFailure(
                status="unavailable",
                detail="responses tool-call probe returned no function name",
            )
        if not isinstance(arguments, str) or not arguments.strip():
            raise ProbeFailure(
                status="unavailable",
                detail="responses tool-call probe returned no function arguments",
            )
        return name, _load_json_object(arguments, source="responses tool-call probe")

    raise ProbeFailure(
        status="indeterminate",
        detail=(
            "responses tool-call probe returned no function_call items; "
            "the endpoint accepted the request but the selected model did not "
            "emit a function call"
        ),
    )


def _probe_models_list(client: Any, _context: ProbeContext) -> tuple[int | None, str]:
    page = client.models.list()
    model_ids = _extract_model_ids(page)
    return 200, f"listed {len(model_ids)} models"


def _probe_responses_create(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.text_model:
        raise ValueError("no text model configured or discovered")
    client.responses.create(
        model=context.text_model,
        input="Reply with OK.",
    )
    return 200, f"request succeeded with model {context.text_model!r}"


def _probe_responses_structured_output(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.text_model:
        raise ValueError("no text model configured or discovered")
    response = client.responses.create(
        model=context.text_model,
        input=("Return only JSON with status set to 'ok' and value set to 1."),
        max_output_tokens=64,
        temperature=0,
        text=_build_responses_text_format("probe_payload"),
    )
    raw_text = _extract_response_output_text(response)
    if not raw_text:
        raise ProbeFailure(
            status="unavailable",
            detail="responses structured-output probe returned no text",
        )
    payload = _load_json_object(raw_text, source="responses structured-output probe")
    _validate_probe_payload(payload, source="responses structured-output probe")
    return 200, "returned schema-valid JSON payload"


def _probe_responses_tool_calls(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.text_model:
        raise ValueError("no text model configured or discovered")
    response = client.responses.create(
        model=context.text_model,
        input="Call the report_probe tool with status set to 'ok'.",
        max_output_tokens=64,
        temperature=0,
        tools=[
            {
                "type": "function",
                "name": "report_probe",
                "description": "Report a successful probe result.",
                "parameters": _PROBE_TOOL_PARAMETERS,
            }
        ],
        tool_choice={"type": "function", "name": "report_probe"},
    )
    tool_name, payload = _extract_responses_tool_call(response)
    if tool_name != "report_probe":
        raise ProbeFailure(
            status="unavailable",
            detail=f"responses tool-call probe called unexpected tool {tool_name!r}",
        )
    if payload.get("status") != "ok":
        raise ProbeFailure(
            status="unavailable",
            detail=f"responses tool-call probe returned unexpected payload {payload!r}",
        )
    return 200, "returned a valid function_call payload"


def _probe_chat_completions_create(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.text_model:
        raise ValueError("no text model configured or discovered")
    client.chat.completions.create(
        model=context.text_model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=8,
        temperature=0,
    )
    return 200, f"request succeeded with model {context.text_model!r}"


def _probe_chat_completions_structured_output(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.text_model:
        raise ValueError("no text model configured or discovered")
    response = client.chat.completions.create(
        model=context.text_model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return only JSON with status set to 'ok' and value set to 1."
                ),
            }
        ],
        max_tokens=64,
        temperature=0,
        response_format=_build_json_schema_response_format("probe_payload"),
    )
    choices = _object_get(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ProbeFailure(
            status="unavailable",
            detail="chat structured-output probe returned no choices",
        )
    message = _object_get(choices[0], "message")
    raw_text = _object_get(message, "content")
    if isinstance(raw_text, list):
        raw_text = "".join(
            str(_object_get(part, "text") or "")
            for part in raw_text
            if _object_get(part, "type") in {"text", "output_text"}
        )
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ProbeFailure(
            status="unavailable",
            detail="chat structured-output probe returned no message content",
        )
    payload = _load_json_object(
        raw_text.strip(),
        source="chat structured-output probe",
    )
    _validate_probe_payload(payload, source="chat structured-output probe")
    return 200, "returned schema-valid JSON payload"


def _probe_chat_completions_tool_calls(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.text_model:
        raise ValueError("no text model configured or discovered")
    response = client.chat.completions.create(
        model=context.text_model,
        messages=[
            {
                "role": "user",
                "content": "Call the report_probe tool with status set to 'ok'.",
            }
        ],
        max_tokens=64,
        temperature=0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "report_probe",
                    "description": "Report a successful probe result.",
                    "parameters": _PROBE_TOOL_PARAMETERS,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "report_probe"}},
    )
    tool_name, payload = _extract_chat_tool_call(response)
    if tool_name != "report_probe":
        raise ProbeFailure(
            status="unavailable",
            detail=f"chat tool-call probe called unexpected tool {tool_name!r}",
        )
    if payload.get("status") != "ok":
        raise ProbeFailure(
            status="unavailable",
            detail=f"chat tool-call probe returned unexpected payload {payload!r}",
        )
    return 200, "returned a valid tool_call payload"


def _probe_embeddings_create(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.embedding_model:
        raise ValueError("no embedding model configured or discovered")
    client.embeddings.create(
        model=context.embedding_model,
        input="ping",
    )
    return 200, f"request succeeded with model {context.embedding_model!r}"


def _probe_moderations_create(
    client: Any, _context: ProbeContext
) -> tuple[int | None, str]:
    client.moderations.create(input="ping")
    return 200, "request succeeded"


def _probe_files_list(client: Any, _context: ProbeContext) -> tuple[int | None, str]:
    page = client.files.list()
    return 200, f"listed {len(_page_items(page))} files on first page"


def _probe_fine_tuning_jobs_list(
    client: Any, _context: ProbeContext
) -> tuple[int | None, str]:
    page = client.fine_tuning.jobs.list()
    return 200, f"listed {len(_page_items(page))} fine-tuning jobs on first page"


def _probe_batches_list(client: Any, _context: ProbeContext) -> tuple[int | None, str]:
    page = client.batches.list()
    return 200, f"listed {len(_page_items(page))} batches on first page"


def _probe_vector_stores_list(
    client: Any, _context: ProbeContext
) -> tuple[int | None, str]:
    page = client.vector_stores.list()
    return 200, f"listed {len(_page_items(page))} vector stores on first page"


def _probe_images_generate(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.include_images:
        raise ValueError("image probes disabled; pass --include-images to enable")
    if not context.image_model:
        raise ValueError("no image model configured or discovered")
    client.images.generate(
        model=context.image_model,
        prompt="A tiny red square on a white background.",
        size="1024x1024",
    )
    return 200, f"request succeeded with model {context.image_model!r}"


def _probe_audio_speech_create(
    client: Any, context: ProbeContext
) -> tuple[int | None, str]:
    if not context.include_audio:
        raise ValueError("audio probes disabled; pass --include-audio to enable")
    if not context.tts_model:
        raise ValueError("no text-to-speech model configured or discovered")
    client.audio.speech.create(
        model=context.tts_model,
        voice="alloy",
        input="OK",
    )
    return 200, f"request succeeded with model {context.tts_model!r}"


# OpenAI docs show that Chat Completions and Responses can both support
# structured outputs and function calling. `engllm-chat` currently uses Chat
# Completions plus structured outputs only, so native tool calling and the
# Responses API are reported as extra compatibility surface rather than runtime
# prerequisites.
OPERATIONS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="models.list",
        description="List registered models",
        runtime_tier="optional_for_engllm_chat",
        sdk_path=("models", "list"),
        probe=_probe_models_list,
        cost="read-only",
    ),
    OperationSpec(
        name="responses.create",
        description="Responses API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("responses", "create"),
        probe=_probe_responses_create,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="responses.create.structured_output",
        description="Responses structured outputs via text.format",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("responses", "create"),
        probe=_probe_responses_structured_output,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="responses.create.tool_calls",
        description="Responses function calling",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("responses", "create"),
        probe=_probe_responses_tool_calls,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="chat.completions.create",
        description="Chat Completions API",
        runtime_tier="required_for_engllm_chat",
        sdk_path=("chat", "completions", "create"),
        probe=_probe_chat_completions_create,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="chat.completions.create.structured_output",
        description="Chat structured outputs via response_format=json_schema",
        runtime_tier="required_for_engllm_chat",
        sdk_path=("chat", "completions", "create"),
        probe=_probe_chat_completions_structured_output,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="chat.completions.create.tool_calls",
        description="Chat function calling",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("chat", "completions", "create"),
        probe=_probe_chat_completions_tool_calls,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="embeddings.create",
        description="Embeddings API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("embeddings", "create"),
        probe=_probe_embeddings_create,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="moderations.create",
        description="Moderations API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("moderations", "create"),
        probe=_probe_moderations_create,
        cost="low-cost inference",
    ),
    OperationSpec(
        name="files.list",
        description="List uploaded files",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("files", "list"),
        probe=_probe_files_list,
        cost="read-only",
    ),
    OperationSpec(
        name="fine_tuning.jobs.list",
        description="List fine-tuning jobs",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("fine_tuning", "jobs", "list"),
        probe=_probe_fine_tuning_jobs_list,
        cost="read-only",
    ),
    OperationSpec(
        name="batches.list",
        description="List batch jobs",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("batches", "list"),
        probe=_probe_batches_list,
        cost="read-only",
    ),
    OperationSpec(
        name="vector_stores.list",
        description="List vector stores",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("vector_stores", "list"),
        probe=_probe_vector_stores_list,
        cost="read-only",
    ),
    OperationSpec(
        name="images.generate",
        description="Image generation API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("images", "generate"),
        probe=_probe_images_generate,
        cost="optional image generation",
    ),
    OperationSpec(
        name="audio.speech.create",
        description="Text-to-speech API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("audio", "speech", "create"),
        probe=_probe_audio_speech_create,
        cost="optional audio generation",
    ),
    OperationSpec(
        name="audio.transcriptions.create",
        description="Audio transcription API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("audio", "transcriptions", "create"),
        probe=None,
        cost="manual probe required",
    ),
    OperationSpec(
        name="audio.translations.create",
        description="Audio translation API",
        runtime_tier="extra_compatibility_surface",
        sdk_path=("audio", "translations", "create"),
        probe=None,
        cost="manual probe required",
    ),
)


def _is_runtime_required(tier: RuntimeTier) -> bool:
    return tier == "required_for_engllm_chat"


def _probe_operation(
    client: Any,
    spec: OperationSpec,
    context: ProbeContext,
) -> ProbeResult:
    target = _resolve_sdk_target(client, spec.sdk_path)
    if target is None:
        return ProbeResult(
            name=spec.name,
            description=spec.description,
            runtime_tier=spec.runtime_tier,
            runtime_required=_is_runtime_required(spec.runtime_tier),
            sdk_path=".".join(spec.sdk_path),
            cost=spec.cost,
            status="unavailable",
            detail="SDK surface missing",
            http_status=None,
            elapsed_ms=0,
        )

    if spec.probe is None:
        return ProbeResult(
            name=spec.name,
            description=spec.description,
            runtime_tier=spec.runtime_tier,
            runtime_required=_is_runtime_required(spec.runtime_tier),
            sdk_path=".".join(spec.sdk_path),
            cost=spec.cost,
            status="skipped",
            detail="probe not implemented because this operation needs file input",
            http_status=None,
            elapsed_ms=0,
        )

    started = time.perf_counter()
    try:
        http_status, detail = spec.probe(client, context)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ProbeResult(
            name=spec.name,
            description=spec.description,
            runtime_tier=spec.runtime_tier,
            runtime_required=_is_runtime_required(spec.runtime_tier),
            sdk_path=".".join(spec.sdk_path),
            cost=spec.cost,
            status="available",
            detail=detail,
            http_status=http_status,
            elapsed_ms=elapsed_ms,
        )
    except ValueError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ProbeResult(
            name=spec.name,
            description=spec.description,
            runtime_tier=spec.runtime_tier,
            runtime_required=_is_runtime_required(spec.runtime_tier),
            sdk_path=".".join(spec.sdk_path),
            cost=spec.cost,
            status="skipped",
            detail=str(exc),
            http_status=None,
            elapsed_ms=elapsed_ms,
        )
    except ProbeFailure as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ProbeResult(
            name=spec.name,
            description=spec.description,
            runtime_tier=spec.runtime_tier,
            runtime_required=_is_runtime_required(spec.runtime_tier),
            sdk_path=".".join(spec.sdk_path),
            cost=spec.cost,
            status=exc.status,
            detail=exc.detail,
            http_status=exc.http_status,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:  # pragma: no cover - network/provider behavior
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status, http_status, detail = _classify_exception(exc)
        return ProbeResult(
            name=spec.name,
            description=spec.description,
            runtime_tier=spec.runtime_tier,
            runtime_required=_is_runtime_required(spec.runtime_tier),
            sdk_path=".".join(spec.sdk_path),
            cost=spec.cost,
            status=status,
            detail=detail,
            http_status=http_status,
            elapsed_ms=elapsed_ms,
        )


def _format_table(results: Sequence[ProbeResult]) -> str:
    headers = ("tier", "status", "operation", "http", "elapsed", "detail")
    rows = [
        (
            result.runtime_tier,
            result.status,
            result.name,
            str(result.http_status) if result.http_status is not None else "-",
            f"{result.elapsed_ms}ms",
            result.detail,
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    lines = [render_row(headers), render_row(tuple("-" * width for width in widths))]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _build_runtime_summary(
    results: Sequence[ProbeResult], *, text_model: str | None
) -> dict[str, object]:
    required_results = [
        result
        for result in results
        if result.runtime_tier == "required_for_engllm_chat"
    ]
    blocking_results = [
        result
        for result in required_results
        if result.status in {"unavailable", "restricted", "skipped"}
    ]
    indeterminate_results = [
        result for result in required_results if result.status == "indeterminate"
    ]

    if blocking_results:
        status = "not_ready"
    elif indeterminate_results:
        status = "indeterminate"
    else:
        status = "ready"

    if not text_model:
        status = "not_ready"

    if status == "ready":
        note = (
            "engllm-chat can run against this endpoint because Chat Completions "
            "and Structured Outputs succeeded for the selected text model."
        )
    elif status == "indeterminate":
        note = (
            "engllm-chat readiness could not be confirmed because a required "
            "chat capability was inconclusive for the selected text model."
        )
    else:
        note = (
            "engllm-chat currently requires Chat Completions and Structured "
            "Outputs on the selected text model. Responses API support and "
            "native function calling are informative extras, not runtime requirements."
        )

    return {
        "status": status,
        "selected_text_model": text_model,
        "required_operations": [
            {
                "name": result.name,
                "status": result.status,
                "detail": result.detail,
            }
            for result in required_results
        ],
        "blocking_operations": [
            {
                "name": result.name,
                "status": result.status,
                "detail": result.detail,
            }
            for result in (*blocking_results, *indeterminate_results)
        ],
        "note": note,
    }


def _build_tier_summary(results: Sequence[ProbeResult]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for tier in (
        "required_for_engllm_chat",
        "optional_for_engllm_chat",
        "extra_compatibility_surface",
    ):
        tier_results = [result for result in results if result.runtime_tier == tier]
        counts: dict[str, int] = {"total": len(tier_results)}
        for status in (
            "available",
            "unavailable",
            "restricted",
            "skipped",
            "indeterminate",
        ):
            counts[status] = sum(
                1 for result in tier_results if result.status == status
            )
        summary[tier] = counts
    return summary


def _format_runtime_status(status: str) -> str:
    return {
        "ready": "READY",
        "not_ready": "NOT READY",
        "indeterminate": "INDETERMINATE",
    }[status]


def _emit_progress(message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    print(message, file=sys.stderr, flush=True)


def _probe_operations_with_progress(
    client: Any,
    operations: Sequence[OperationSpec],
    context: ProbeContext,
    *,
    progress_enabled: bool,
    start_index: int = 1,
    total_operations: int | None = None,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    total = total_operations or len(operations)

    for index, spec in enumerate(operations, start=start_index):
        _emit_progress(
            f"[{index}/{total}] probing {spec.name} ({spec.description})...",
            enabled=progress_enabled,
        )
        result = _probe_operation(client, spec, context)
        results.append(result)
        http_text = str(result.http_status) if result.http_status is not None else "-"
        _emit_progress(
            (
                f"[{index}/{total}] {result.status} {spec.name} "
                f"(http={http_text}, {result.elapsed_ms}ms)"
            ),
            enabled=progress_enabled,
        )

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help=(
            "OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL "
            "or OpenAI's v1 URL."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API token. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--text-model",
        default=None,
        help="Optional override for chat/responses probes.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional override for embedding probes.",
    )
    parser.add_argument(
        "--image-model",
        default=None,
        help="Optional override for image probes.",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="Optional override for text-to-speech probes.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Client timeout in seconds.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Enable image-generation probes. This may incur extra cost.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Enable text-to-speech probes. This may incur extra cost.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress updates on stderr.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("API key is required via --api-key or OPENAI_API_KEY")

    try:
        openai_client_type, sdk_version = _load_openai_sdk()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    client = openai_client_type(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout_seconds,
    )

    initial_context = ProbeContext(
        base_url=args.base_url,
        text_model=args.text_model,
        embedding_model=args.embedding_model,
        image_model=args.image_model,
        tts_model=args.tts_model,
        include_images=args.include_images,
        include_audio=args.include_audio,
    )

    progress_enabled = not args.no_progress
    total_operations = len(OPERATIONS)
    results = _probe_operations_with_progress(
        client,
        (OPERATIONS[0],),
        initial_context,
        progress_enabled=progress_enabled,
        start_index=1,
        total_operations=total_operations,
    )
    models_result = results[0]

    discovered_model_ids: list[str] = []
    if models_result.status == "available":
        _emit_progress(
            "Discovering likely model ids from models.list() output...",
            enabled=progress_enabled,
        )
        try:
            discovered_model_ids = _extract_model_ids(client.models.list())
        except Exception:
            discovered_model_ids = []
        selected_text_model = args.text_model or _pick_text_model(discovered_model_ids)
        selected_embedding_model = args.embedding_model or _pick_embedding_model(
            discovered_model_ids
        )
        selected_image_model = args.image_model or _pick_image_model(
            discovered_model_ids
        )
        selected_tts_model = args.tts_model or _pick_tts_model(discovered_model_ids)
        _emit_progress(
            (
                "Selected models: "
                "text="
                f"{selected_text_model or 'none'}"
                ", "
                "embedding="
                f"{selected_embedding_model or 'none'}"
                ", "
                "image="
                f"{selected_image_model or 'none'}"
                ", "
                "tts="
                f"{selected_tts_model or 'none'}"
            ),
            enabled=progress_enabled,
        )

    context = ProbeContext(
        base_url=args.base_url,
        text_model=args.text_model or _pick_text_model(discovered_model_ids),
        embedding_model=(
            args.embedding_model or _pick_embedding_model(discovered_model_ids)
        ),
        image_model=args.image_model or _pick_image_model(discovered_model_ids),
        tts_model=args.tts_model or _pick_tts_model(discovered_model_ids),
        include_images=args.include_images,
        include_audio=args.include_audio,
    )

    results.extend(
        _probe_operations_with_progress(
            client,
            OPERATIONS[1:],
            context,
            progress_enabled=progress_enabled,
            start_index=2,
            total_operations=total_operations,
        )
    )

    report = {
        "base_url": args.base_url,
        "sdk_version": sdk_version,
        "selected_models": {
            "text_model": context.text_model,
            "embedding_model": context.embedding_model,
            "image_model": context.image_model,
            "tts_model": context.tts_model,
        },
        "engllm_chat_runtime": _build_runtime_summary(
            results,
            text_model=context.text_model,
        ),
        "summary_by_tier": _build_tier_summary(results),
        "results": [asdict(result) for result in results],
        "legend": {
            "available": "operation succeeded or clearly exists",
            "unavailable": "endpoint appears missing or unimplemented",
            "restricted": "endpoint exists but this token cannot use it",
            "skipped": "probe not attempted because a prerequisite was missing",
            "indeterminate": "connectivity or server behavior prevented a conclusion",
        },
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("OpenAI-Compatible API Probe")
    print(f"Base URL: {args.base_url}")
    print(f"OpenAI SDK version: {sdk_version}")
    print("Selected models:")
    print(f"  text_model: {context.text_model or 'none'}")
    print(f"  embedding_model: {context.embedding_model or 'none'}")
    print(f"  image_model: {context.image_model or 'none'}")
    print(f"  tts_model: {context.tts_model or 'none'}")
    print()
    runtime_summary = _build_runtime_summary(results, text_model=context.text_model)
    required_operations = runtime_summary["required_operations"]
    print(
        f"engllm-chat runtime: {_format_runtime_status(str(runtime_summary['status']))}"
    )
    print(f"  {runtime_summary['note']}")
    print("Required for engllm-chat:")
    if isinstance(required_operations, list):
        for item in required_operations:
            if not isinstance(item, dict):
                continue
            print(f"  - {item['name']}: {item['status']} ({item['detail']})")
    print()
    print(_format_table(results))
    print()
    print("Legend:")
    print("  available     operation succeeded or clearly exists")
    print("  unavailable   endpoint appears missing or unimplemented")
    print("  restricted    endpoint exists but this token cannot use it")
    print("  skipped       probe not attempted because a prerequisite was missing")
    print("  indeterminate connectivity or server behavior prevented a conclusion")
    print("Tier meanings:")
    print("  required_for_engllm_chat      needed by the current chat runtime")
    print("  optional_for_engllm_chat      helpful, but not required")
    print(
        "  extra_compatibility_surface   broader API surface not used by chat runtime"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
