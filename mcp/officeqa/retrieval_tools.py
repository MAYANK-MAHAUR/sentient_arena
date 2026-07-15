from __future__ import annotations

import json
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO

from corpus_tools import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    _compact_context_lines,
    _dump_limited_json,
    _row_to_compact,
    _table_to_tsv,
)
from officeqa_cli import auction_windows, cmd_candidates, resolve_root


def quick_retrieve_candidates(
    question: str,
    root: str | None = None,
    max_rows: int = 5,
    max_tables: int = 5,
    max_text: int = 5,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKEN_LIMIT,
    year_start: int | None = None,
    year_end: int | None = None,
) -> str:
    max_rows = min(max(1, max_rows), 6)
    max_tables = min(max(1, max_tables), 6)
    max_text = min(max(0, max_text), 4)
    args = Namespace(
        root=root,
        question=question,
        terms=[],
        phrases=[],
        year_start=year_start,
        year_end=year_end,
        max_rows=max_rows,
        max_tables=max_tables,
        max_text=max_text,
        max_headers=24,
        sample_rows=10,
        context_lines=3,
    )
    buffer = StringIO()
    with redirect_stdout(buffer):
        cmd_candidates(args)
    payload = json.loads(buffer.getvalue())

    for item in payload.get("top_text_hits", []):
        if "context" in item:
            item["context"] = _compact_context_lines(item["context"], max_line_chars=220)
        item.pop("read_command", None)

    for item in payload.get("top_tables", []):
        item["headers"] = [str(header) for header in item.get("headers", [])[:18]]
        item["sample_row_labels"] = [str(label) for label in item.get("sample_row_labels", [])[:8]]
        item.pop("read_command", None)

    for item in payload.get("top_rows", []):
        row = item.pop("row", None)
        headers = item.get("headers", [])
        if isinstance(row, dict):
            item["row"] = _row_to_compact(row, max_cells=18)
            item["row_tsv"] = _table_to_tsv(headers, [row], max_rows=1, max_cells=18)
        item["headers"] = [str(header) for header in headers[:18]]
        item.pop("read_command", None)

    return _dump_limited_json(
        payload,
        list_keys=("top_rows", "top_tables", "top_text_hits"),
        max_context_tokens=max_context_tokens,
    )


def financing_auction_candidates(
    question: str,
    root: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    max_results: int = 8,
) -> str:
    corpus = resolve_root(root)
    candidates = auction_windows(
        root=corpus,
        question=question,
        year_start=year_start,
        year_end=year_end,
    )
    return json.dumps(candidates[:max_results], indent=2)
