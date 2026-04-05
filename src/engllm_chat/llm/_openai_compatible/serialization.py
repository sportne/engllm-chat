"""Message and schema serialization helpers for OpenAI-compatible chat."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, create_model

from engllm_chat.domain.errors import LLMError
from engllm_chat.domain.models import ChatMessage
from engllm_chat.llm.base import ChatToolDefinition


def _serialize_chat_message(message: ChatMessage) -> dict[str, object]:
    if message.role == "tool":
        if message.tool_result is None:
            raise LLMError("Tool chat message missing tool_result")
        return {
            "role": "user",
            "content": (
                "Tool result:\n"
                f"{json.dumps(message.tool_result.model_dump(mode='json'), sort_keys=True)}"
            ),
        }

    payload: dict[str, object] = {"role": message.role}
    if message.role == "assistant" and message.tool_calls:
        request_text = json.dumps(
            [tool_call.model_dump(mode="json") for tool_call in message.tool_calls],
            sort_keys=True,
        )
        content_parts = []
        if message.content is not None:
            content_parts.append(message.content)
        content_parts.append(f"Tool request:\n{request_text}")
        payload["content"] = "\n\n".join(part for part in content_parts if part.strip())
        return payload

    if message.content is not None:
        payload["content"] = message.content
    elif message.role in {"system", "user", "assistant"}:
        payload["content"] = ""
    return payload


def _build_chat_turn_action_model(
    response_model: type[BaseModel],
    tools: list[ChatToolDefinition],
) -> type[BaseModel]:
    final_action_model = create_model(
        f"{response_model.__name__}FinalAction",
        kind=(Literal["final_response"], "final_response"),
        response=(response_model, ...),
    )

    action_field_type: Any = final_action_model
    if tools:
        tool_name_pattern = (
            "^(" + "|".join(re.escape(tool.name) for tool in tools) + ")$"
        )

        tool_action_model = create_model(
            f"{response_model.__name__}ToolAction",
            kind=(Literal["tool_request"], "tool_request"),
            tool_name=(str, Field(pattern=tool_name_pattern)),
            arguments=(dict[str, object], Field(default_factory=dict)),
        )
        action_field_type = Annotated[
            tool_action_model | final_action_model,
            Field(discriminator="kind"),
        ]

    return create_model(
        f"{response_model.__name__}ChatTurnActionEnvelope",
        action=(action_field_type, ...),
    )
