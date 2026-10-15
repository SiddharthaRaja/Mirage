"""The one prompt.

Baseline, SFT, both GRPO arms, evaluation, and the Vantage serving workload
all build their messages through `build_messages`. There is no second copy of
the prompt anywhere in the repo, on purpose: a model trained on one wording and
evaluated or served on another silently loses much of the gain, and that bug is
invisible in the training curves.

The prompt version is stamped into every run manifest. Changing the wording is
a new experiment, not a tweak.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are an expert SQLite developer. Given a database schema and a question, "
    "write one SQLite query that answers the question. Use only tables and columns "
    "that exist in the schema. Return only the SQL, inside a ```sql code block."
)

USER_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}"


@dataclass(frozen=True)
class PromptSpec:
    sample_values: int = 0  # example values per column appended to the schema
    system: str = SYSTEM
    version: str = PROMPT_VERSION


def build_messages(schema_text: str, question: str, spec: PromptSpec | None = None) -> list[dict]:
    spec = spec or PromptSpec()
    return [
        {"role": "system", "content": spec.system},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema_text, question=question)},
    ]


def format_answer(sql: str) -> str:
    """The SFT target. Matches what the extractor expects to find."""
    return f"```sql\n{sql.strip()}\n```"
