"""The intermediate ("working") record schema used from 02_build_qa.py through
05b_import_validation.py, before 06_finalise_dataset.py assembles the final
Table 3.5 schema (src/data/schema.py).

Phase 1 is a pipeline of English -> MT Yoruba -> post-edited Yoruba -> validated
Yoruba. Each stage only ever *adds* fields; earlier fields are never
overwritten silently, so the provenance of every string in the final dataset
stays traceable to a specific stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

WORKING_RECORD_FIELDS = [
    "id", "claim_id", "topic", "question_type",
    "question_en", "answer_en",
    "source_ref", "source_name", "source_url",
    "question_yo_mt", "answer_yo_mt", "post_edited",
    "question_yo_final", "answer_yo_final", "editor_id", "notes",
    "validated_ling", "validated_clin", "validation_notes",
]


@dataclass
class WorkingRecord:
    id: str
    claim_id: str
    topic: str
    question_type: str  # "derived" | "elicited" | "out_of_scope"
    question_en: str
    answer_en: str
    source_ref: str
    source_name: str
    source_url: str | None = None
    question_yo_mt: str | None = None
    answer_yo_mt: str | None = None
    post_edited: bool = False
    question_yo_final: str | None = None
    answer_yo_final: str | None = None
    editor_id: str | None = None
    notes: str = ""
    validated_ling: bool = False
    validated_clin: bool = False
    validation_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkingRecord":
        known = {k: d.get(k) for k in WORKING_RECORD_FIELDS if k in d}
        return cls(**known)


def make_id(sequence_number: int) -> str:
    return f"yhqa-{sequence_number:06d}"


def make_claim_id(source_name: str, slug: str, claim_index: int) -> str:
    return f"{source_name}-{slug}-{claim_index:03d}"
