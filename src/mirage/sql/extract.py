"""Getting one SQL statement out of whatever the model wrote.

The model is asked for a single query in a ```sql block. It does not always
comply: it adds prose, forgets the fence, writes two queries, or rambles after
the answer. How generously this is parsed is a hidden experimental variable —
a lenient extractor credits answers a strict one rejects — so the rules are
fixed here, documented, and identical for every arm and every model.

Rules, in order:
1. If there is a ```sql (or bare ```) fenced block, take the first one.
2. Otherwise, if a line starts with SELECT/WITH, take from there.
3. Keep only the first statement (up to the first top-level semicolon).
4. Strip whitespace and trailing semicolons.

The result also records *how* it was found, because "the model stopped using
the fence" is itself something training can do and the report should show.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE = re.compile(r"```(?:sql|sqlite)?[ \t]*\n?(.*?)(?:```|$)", re.IGNORECASE | re.DOTALL)
_START = re.compile(r"(?im)^\s*(select|with)\b")


@dataclass(frozen=True)
class Extracted:
    sql: str | None
    how: str  # "fence" | "bare" | "none"
    n_statements: int = 0


def _first_statement(text: str) -> tuple[str, int]:
    """Split on semicolons that are not inside string literals."""
    stmts, buf, quote = [], [], None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == ";":
            s = "".join(buf).strip()
            if s:
                stmts.append(s)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return (stmts[0] if stmts else ""), len(stmts)


def extract_sql(completion: str) -> Extracted:
    if not completion or not completion.strip():
        return Extracted(None, "none")
    m = _FENCE.search(completion)
    if m and m.group(1).strip():
        body, how = m.group(1), "fence"
    else:
        s = _START.search(completion)
        if not s:
            return Extracted(None, "none")
        body, how = completion[s.start():], "bare"
        # bare answers often trail into prose: stop at the first blank line
        body = re.split(r"\n\s*\n", body, maxsplit=1)[0]
    stmt, n = _first_statement(body)
    stmt = stmt.strip().rstrip(";").strip()
    if not stmt:
        return Extracted(None, "none")
    return Extracted(stmt, how, n)


def completion_text(completion) -> str:
    """TRL hands reward functions either plain strings or chat-format lists
    ([{"role": "assistant", "content": ...}]). Accept both."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, list):  # content parts
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)
