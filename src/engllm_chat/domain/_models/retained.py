"""Retained simple structured models used by mock-provider tests."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import DomainModel
from .responses import ChatCitation


class SectionDraft(DomainModel):
    """Simple structured section draft retained for mock-provider tests."""

    section_id: str
    title: str
    content: str
    evidence_refs: list[dict[str, object]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SectionUpdateProposal(DomainModel):
    """Simple structured update proposal retained for mock-provider tests."""

    section_id: str
    title: str
    existing_text: str
    proposed_text: str
    rationale: str
    uncertainty_list: list[str] = Field(default_factory=list)
    review_priority: Literal["low", "medium", "high"] = "medium"
    evidence_refs: list[dict[str, object]] = Field(default_factory=list)


class QueryAnswer(DomainModel):
    """Simple structured answer retained for mock-provider tests."""

    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
