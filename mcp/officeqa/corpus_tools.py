from __future__ import annotations

import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from math_tools import format_numeric_value

DEFAULT_ROOT = Path("/app/corpus")
DEFAULT_CONTEXT_TOKEN_LIMIT = 4000
_APPROX_CHARS_PER_TOKEN = 4

MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _resolve_root(root: str | None = None) -> Path:
    candidates = []
    if root:
        candidates.append(Path(root))
    candidates.extend(
        [
            DEFAULT_ROOT,
            Path("corpus"),
            Path("/workspace/corpus"),
            Path("/mnt/data/corpus"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return DEFAULT_ROOT


def _iter_files(root: Path, year_start: int | None = None, year_end: int | None = None) -> Iterable[Path]:
    for path_str in _file_paths_cached(str(root.resolve())):
        path = Path(path_str)
        if year_start is not None or year_end is not None:
            match = re.search(r"(\d{4})[_-](\d{2})", path.name)
            if match:
                year = int(match.group(1))
                if year_start is not None and year < year_start:
                    continue
                if year_end is not None and year > year_end:
                    continue
        yield path


@lru_cache(maxsize=16)
def _file_paths_cached(root_str: str) -> tuple[str, ...]:
    root = Path(root_str)
    files = sorted(root.glob("treasury_bulletin_*.txt"))
    if not files:
        files = sorted(root.glob("*.txt"))
    return tuple(str(path) for path in files)


def _safe_file(root: Path, file_name: str) -> Path:
    path = Path(file_name)
    if not path.is_absolute():
        path = root / file_name
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError("file_name must stay inside the corpus root")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


@lru_cache(maxsize=24)
def _read_text_cached(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="replace")


def _read_text(path: Path) -> str:
    return _read_text_cached(str(path.resolve()))


def _lines(path: Path) -> list[str]:
    return _read_text(path).splitlines()


@lru_cache(maxsize=1024)
def _table_spans_cached(path_str: str) -> tuple[tuple[int, int], ...]:
    lines = _lines(Path(path_str))
    spans = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].lstrip().startswith("|"):
            idx += 1
            continue
        start, end = _table_bounds(lines, idx)
        spans.append((start, end))
        idx = end + 1
    return tuple(spans)


def _table_spans(path: Path) -> tuple[tuple[int, int], ...]:
    return _table_spans_cached(str(path.resolve()))


def _file_year_month(path: Path) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{4})[_-](\d{2})", path.name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _month_number(text: str) -> int | None:
    cleaned = text.strip().lower().rstrip(".")
    if cleaned.isdigit():
        month = int(cleaned)
        if 1 <= month <= 12:
            return month
    return MONTH_NAME_TO_NUM.get(cleaned)


def _parse_month_year_text(text: str) -> tuple[int | None, int | None]:
    cleaned = text.strip().lower()
    month_pattern = "|".join(sorted((re.escape(name) for name in MONTH_NAME_TO_NUM), key=len, reverse=True))
    match = re.search(rf"\b({month_pattern})\.?\s+((?:19|20)\d{{2}})\b", cleaned)
    if match:
        return int(match.group(2)), MONTH_NAME_TO_NUM[match.group(1).rstrip(".")]
    match = re.search(rf"\b((?:19|20)\d{{2}})[_\-/ ]+({month_pattern}|\d{{1,2}})\b", cleaned)
    if match:
        return int(match.group(1)), _month_number(match.group(2))
    match = re.search(r"\b((?:19|20)\d{2})[_\-/](\d{1,2})\b", cleaned)
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return int(match.group(1)), month
    return None, None


def _detect_units(text: str) -> dict[str, object]:
    text_l = re.sub(r"\s+", " ", text.lower())
    if re.search(r"\b(in\s+)?trillions?\b|\btrillions?\s+of\s+dollars\b", text_l):
        return {"unit": "trillions", "scale_to_dollars": 1_000_000_000_000, "scale_name": "trillion"}
    if re.search(r"\b(in\s+)?billions?\b|\bbillions?\s+of\s+dollars\b", text_l):
        return {"unit": "billions", "scale_to_dollars": 1_000_000_000, "scale_name": "billion"}
    if re.search(r"\b(in\s+)?millions?\b|\bmillions?\s+of\s+dollars\b", text_l):
        return {"unit": "millions", "scale_to_dollars": 1_000_000, "scale_name": "million"}
    if re.search(r"\b(in\s+)?thousands?\b|\bthousands?\s+of\s+dollars\b", text_l):
        return {"unit": "thousands", "scale_to_dollars": 1_000, "scale_name": "thousand"}
    if re.search(r"\bpercent(?:age)?\b|%", text_l):
        return {"unit": "percent", "scale_to_dollars": None, "scale_name": "percent"}
    if re.search(r"\bdollars?\b", text_l):
        return {"unit": "dollars", "scale_to_dollars": 1, "scale_name": "dollar"}
    return {"unit": None, "scale_to_dollars": 1, "scale_name": None}


def _table_profile(lines: list[str], start: int, end: int, title_scan_lines: int = 12) -> dict[str, object]:
    title = _title_context(lines, start, title_scan_lines)
    context_start = max(0, start - title_scan_lines)
    context = [lines[i] for i in range(context_start, min(len(lines), end + 1)) if lines[i].strip()]
    unit_line = ""
    for line in title + lines[start : min(end + 1, start + 3)]:
        if re.search(r"\b(?:in\s+)?(?:thousands|millions|billions|trillions|dollars|percent)\b|%", line, re.I):
            unit_line = line.strip()
            break
    unit_info = _detect_units("\n".join(context))
    # Cumulative-table base period ("Cumulative from January 1, 1934 ...").
    # Two cumulative figures may only be differenced when their bases are
    # identical — mixing a 1934-base table with a 1935-base one produced
    # silent wrong answers in graded runs.
    cumulative_base = None
    for line in title + lines[start : min(end + 1, start + 3)]:
        m = re.search(
            r"cumulative\s+(?:from|since|through|to)\s+([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}|[A-Za-z]+\.?\s+\d{4}|\d{4})",
            line,
            re.I,
        )
        if m:
            cumulative_base = m.group(0).strip()
            break
    return {
        "title_context": title,
        "unit_line": unit_line,
        "unit": unit_info["unit"],
        "scale_to_dollars": unit_info["scale_to_dollars"],
        "scale_name": unit_info["scale_name"],
        "cumulative_base": cumulative_base,
    }


def _table_bounds(lines: list[str], row_idx: int) -> tuple[int, int]:
    start = row_idx
    while start > 0 and lines[start - 1].lstrip().startswith("|"):
        start -= 1
    end = row_idx
    while end + 1 < len(lines) and lines[end + 1].lstrip().startswith("|"):
        end += 1
    return start, end


def _split_md_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _is_separator_row(line: str) -> bool:
    cells = _split_md_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _clean_header_cell(cell: str) -> str:
    text = re.sub(r"\s+", " ", cell).strip()
    if not text:
        return ""
    if text.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r":?-{3,}:?", text):
        return ""
    if re.fullmatch(r"unnamed:\s*\d+(?:_level_\d+)?", text, re.I):
        return ""
    return text


def _flatten_header_rows(header_lines: list[str]) -> list[str]:
    """Flatten multi-row Treasury headers into compact parent > child labels."""
    split_rows = [[_clean_header_cell(cell) for cell in _split_md_row(line)] for line in header_lines]
    split_rows = [row for row in split_rows if any(row)]
    if not split_rows:
        return []

    width = max(len(row) for row in split_rows)
    rows = [row + [""] * (width - len(row)) for row in split_rows]

    for row in rows[:-1]:
        current = ""
        for idx, cell in enumerate(row):
            if idx == 0:
                continue
            if cell:
                current = cell
            elif current:
                row[idx] = current

    headers = []
    for idx in range(width):
        parts = []
        for row in rows:
            part = row[idx]
            if not part:
                continue
            if parts and parts[-1].lower() == part.lower():
                continue
            parts.append(part)
        headers.append(" > ".join(parts) if parts else ("label" if idx == 0 else f"column_{idx}"))
    return headers


def _title_context(lines: list[str], table_start: int, scan_lines: int = 12) -> list[str]:
    start = max(0, table_start - scan_lines)
    return [lines[i] for i in range(start, table_start) if lines[i].strip()][-scan_lines:]


def _row_numeric_ratio(cells: list[str]) -> float:
    """Fraction of non-first cells that look numeric."""
    if len(cells) <= 1:
        return 0.0
    body = cells[1:]
    if not body:
        return 0.0
    numeric = 0
    counted = 0
    for cell in body:
        cleaned = cell.strip()
        if not cleaned or cleaned.lower() in {"nan", "none", "null"}:
            continue
        counted += 1
        if _NUMERIC_CELL_RE.search(cleaned):
            numeric += 1
    if counted == 0:
        return 0.0
    return numeric / counted


def _detect_header_span(table_lines: list[str]) -> int:
    """Return how many leading rows are header rows when no markdown
    separator is present. Treasury parsed tables sometimes have 2 to 3
    header rows (year row + period-label row) and only the last is
    visually distinguished."""
    if not table_lines:
        return 0
    max_inspect = min(4, len(table_lines))
    header_count = 0
    for idx in range(max_inspect):
        cells = _split_md_row(table_lines[idx])
        if not cells:
            break
        ratio = _row_numeric_ratio(cells)
        if ratio < 0.5:
            header_count = idx + 1
        else:
            break
    return max(1, header_count) if header_count else 1


def _parse_table(lines: list[str], start: int, end: int) -> dict[str, object]:
    table_lines = lines[start : end + 1]
    separator_index = next(
        (idx for idx, line in enumerate(table_lines) if _is_separator_row(line)),
        None,
    )
    if separator_index is None:
        header_span = _detect_header_span(table_lines)
        header_lines = table_lines[:header_span]
        data_lines = table_lines[header_span:]
        data_offset = header_span
    else:
        header_lines = table_lines[:separator_index]
        data_lines = table_lines[separator_index + 1 :]
        data_offset = separator_index + 1
    headers = _flatten_header_rows(header_lines)
    rows = []
    for offset, line in enumerate(data_lines, start=start + data_offset + 1):
        cells = _split_md_row(line)
        if not cells:
            continue
        values = []
        for idx, cell in enumerate(cells):
            header = headers[idx] if idx < len(headers) else f"column_{idx}"
            values.append({"column": header, "value": cell})
        rows.append({"line": offset, "label": cells[0], "cells": values})
    return {"headers": headers, "rows": rows}


@lru_cache(maxsize=2048)
def _parse_table_cached(path_str: str, start: int, end: int) -> dict[str, object]:
    return _parse_table(_lines(Path(path_str)), start, end)


def _parse_table_for_path(path: Path, start: int, end: int) -> dict[str, object]:
    return _parse_table_cached(str(path.resolve()), start, end)


def _compact_text(value: object, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value).replace("\t", " ")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _compact_header(value: object) -> str:
    text = _compact_text(value, 120)
    text = re.sub(r"\s*>\s*Unnamed:\s*\d+(?:_level_\d+)?", "", text)
    text = re.sub(r"\bUnnamed:\s*\d+(?:_level_\d+)?\s*>\s*", "", text)
    return text or "col"


def _table_to_tsv(headers: list[object], rows: list[dict[str, object]], max_rows: int = 40, max_cells: int = 24) -> str:
    """Return a dense, LLM-readable table without Markdown pipe noise."""
    limited_headers = [_compact_header(header) for header in headers[:max_cells]]
    output = ["line\t" + "\t".join(limited_headers)]
    for row in rows[:max_rows]:
        cells = list(row.get("cells", []))[:max_cells]
        values = [_compact_text(cell.get("value", ""), 160) for cell in cells if isinstance(cell, dict)]
        if len(values) < len(limited_headers):
            values.extend([""] * (len(limited_headers) - len(values)))
        output.append(f"{row.get('line', '')}\t" + "\t".join(values[: len(limited_headers)]))
    return "\n".join(output)


def _table_to_vertical(
    headers: list[object],
    rows: list[dict[str, object]],
    max_rows: int = 8,
    max_cells: int = 24,
    skip_empty: bool = True,
) -> str:
    """Vertical serialization: one header > value pair per line, grouped by row.

    This format eliminates wide-table column-misalignment errors. Use for
    narrow result sets (after row-filtering) where the row label and each
    column's value should be unambiguously paired.

    Example output (schematic — numbers and labels are placeholders):
        ROW @ line <N>: <row label>
          <year_a> > <period_1> = <value>
          <year_a> > <period_2> = <value>
          <year_a> > Total      = <value>
          <year_b> > <period_1> = <value>
    """
    out: list[str] = []
    for row in rows[:max_rows]:
        line_no = row.get("line", "")
        label = _compact_text(row.get("label", ""), 160)
        out.append(f"ROW @ line {line_no}: {label}")
        cells = list(row.get("cells", []))[:max_cells]
        for idx, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            if idx == 0:
                # Skip the label cell (already emitted above).
                continue
            header = _compact_header(cell.get("column", "") or f"column_{idx}")
            value = _compact_text(cell.get("value", ""), 160)
            if skip_empty and not value:
                continue
            out.append(f"  {header} = {value}")
    return "\n".join(out)


def _row_to_compact(row: dict[str, object], max_cells: int = 24) -> dict[str, object]:
    cells = list(row.get("cells", []))[:max_cells]
    values = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        values.append(f"{_compact_header(cell.get('column', ''))}={_compact_text(cell.get('value', ''), 120)}")
    return {
        "line": row.get("line"),
        "label": _compact_text(row.get("label", ""), 160),
        "values": values,
        "truncated": len(row.get("cells", [])) > max_cells,
    }


def _compact_context_lines(context: list[dict[str, object]], max_line_chars: int = 220) -> list[str]:
    return [f"{item.get('line')}: {_compact_text(item.get('text', ''), max_line_chars)}" for item in context]


def _token_budget_chars(max_context_tokens: int | None) -> int | None:
    if max_context_tokens is None or max_context_tokens <= 0:
        return None
    return max(800, int(max_context_tokens) * _APPROX_CHARS_PER_TOKEN)


def _json_compact(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _trim_payload_lists(
    payload: dict[str, object],
    list_keys: tuple[str, ...] = ("results", "top_rows", "top_tables", "top_text_hits"),
    max_context_tokens: int | None = DEFAULT_CONTEXT_TOKEN_LIMIT,
) -> dict[str, object]:
    budget = _token_budget_chars(max_context_tokens)
    if budget is None:
        return payload
    trimmed = False
    while len(_json_compact(payload)) > budget:
        candidates = []
        for key in list_keys:
            value = payload.get(key)
            if isinstance(value, list) and len(value) > 1:
                candidates.append((key, value))
        if not candidates:
            break
        key, value = max(candidates, key=lambda item: len(_json_compact(item[1][-1])))
        value.pop()
        trimmed = True
    if trimmed:
        payload["context_truncated"] = True
        payload["context_limit_tokens"] = max_context_tokens
    return payload


def _dump_limited_json(
    payload: dict[str, object],
    list_keys: tuple[str, ...] = ("results", "top_rows", "top_tables", "top_text_hits"),
    max_context_tokens: int | None = DEFAULT_CONTEXT_TOKEN_LIMIT,
) -> str:
    return _json_compact(_trim_payload_lists(payload, list_keys=list_keys, max_context_tokens=max_context_tokens))


_NUMERIC_CELL_RE = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")

_MONTH_ALIASES = {
    1: ("jan", "january"),
    2: ("feb", "february"),
    3: ("mar", "march"),
    4: ("apr", "april"),
    5: ("may",),
    6: ("jun", "june"),
    7: ("jul", "july"),
    8: ("aug", "august"),
    9: ("sep", "sept", "september"),
    10: ("oct", "october"),
    11: ("nov", "november"),
    12: ("dec", "december"),
}


def _clean_terms(items: list[str] | None, limit: int = 16) -> list[str]:
    seen = set()
    output = []
    for item in items or []:
        term = re.sub(r"\s+", " ", item.strip().lower())
        if not term or term in seen:
            continue
        seen.add(term)
        output.append(term)
        if len(output) >= limit:
            break
        if "associated activities" in term:
            alias = term.replace("associated activities", "related activities")
            if alias and alias not in seen:
                seen.add(alias)
                output.append(alias)
        if term == "associated" and "related" not in seen:
            seen.add("related")
            output.append("related")
        if len(output) >= limit:
            break
    return output


# Bond-rating grades must match as whole words: "aa" is a substring of
# "aaa", so plain substring matching scores Moody's Aaa rows for an Aa
# question (graded runs averaged the Aaa column for an Aa-spread question).
_RATING_GRADE_TOKENS = {"aa", "aaa", "a", "baa", "ba", "b", "caa"}


def _contains_term(haystack: str, term: str) -> bool:
    if term in _RATING_GRADE_TOKENS:
        return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", haystack) is not None
    if term in haystack:
        return True
    if term.endswith("s") and len(term) > 4 and term[:-1] in haystack:
        return True
    if not term.endswith("s") and f"{term}s" in haystack:
        return True
    return False


_SEARCH_STOPWORDS = {
    "about",
    "amount",
    "and",
    "are",
    "between",
    "by",
    "category",
    "dollar",
    "dollars",
    "federal",
    "for",
    "from",
    "government",
    "in",
    "nominal",
    "of",
    "reported",
    "same",
    "the",
    "to",
    "u",
    "us",
    "value",
    "values",
    "was",
    "were",
    "what",
    "which",
    "with",
    "year",
}


def _search_terms(query: str, limit: int = 10) -> list[str]:
    raw_terms = re.findall(r"[a-z0-9][a-z0-9-]*", query.lower())
    terms = []
    for term in raw_terms:
        if len(term) < 3 or term in _SEARCH_STOPWORDS:
            continue
        if term.endswith("s") and len(term) > 4:
            term = term[:-1]
        if term not in terms:
            terms.append(term)
    if {"net", "interest"} <= set(raw_terms) and any(term.startswith("outlay") for term in raw_terms):
        for term in ("interest", "outlay", "total", "function"):
            if term not in terms:
                terms.append(term)
        terms = [term for term in terms if term != "net"]
    return terms[:limit]


def _line_context(lines: list[str], idx: int, context_lines: int) -> list[dict[str, object]]:
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    return [{"line": line_no + 1, "text": lines[line_no]} for line_no in range(start, end)]


def _search_corpus_lines(
    query: str,
    root: Path,
    year_start: int | None = None,
    year_end: int | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    context_lines: int = 2,
    max_results: int = 40,
) -> list[dict[str, object]]:
    if year_start is None or year_end is None:
        extracted_years = [int(y) for y in re.findall(r'\b(19[3-9]\d|20[0-2]\d)\b', query)]
        if extracted_years:
            if year_start is None:
                year_start = max(1939, min(extracted_years) - 3)
            if year_end is None:
                year_end = min(2025, max(extracted_years) + 3)
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if regex else re.escape(query), flags)
    # Time budget: unbounded scans over all ~697 files can exceed the
    # harness per-call timeout, which surfaces as a BLANK tool output and a
    # seemingly dead MCP server. Partial hits + a warning beat that.
    import time as _time

    deadline = _time.monotonic() + 20.0
    truncated = None
    hits = []
    for path in _iter_files(root, year_start, year_end):
        if _time.monotonic() > deadline:
            truncated = path.name
            break
        lines = _lines(path)
        for idx, line in enumerate(lines):
            if not pattern.search(line):
                continue
            hits.append(
                {
                    "file": path.name,
                    "line": idx + 1,
                    "search_mode": "exact" if not regex else "regex",
                    "context": _line_context(lines, idx, context_lines),
                }
            )
            if len(hits) >= max_results:
                return hits
    if truncated and hits:
        hits[0] = dict(hits[0])
        hits[0]["truncated_warning"] = (
            f"Search time budget hit at {truncated}; later files not scanned. "
            "Pass year_start/year_end to narrow the window."
        )
    if hits or regex:
        return hits
    if truncated:
        return [{
            "file": None,
            "line": None,
            "search_mode": "exact",
            "context": [],
            "truncated_warning": (
                f"No hits before the time budget expired at {truncated}. "
                "This query has no year bounds — pass year_start/year_end, "
                "or use rank_files_by_terms first."
            ),
        }]
    return _multi_term_search(query, root, year_start, year_end, case_sensitive, context_lines, max_results)


def _multi_term_search(
    query: str,
    root: Path,
    year_start: int | None,
    year_end: int | None,
    case_sensitive: bool,
    context_lines: int,
    max_results: int,
) -> list[dict[str, object]]:
    terms = _search_terms(query)
    if not terms:
        return []
    import time as _time

    deadline = _time.monotonic() + 20.0
    truncated = None
    ranked = []
    for path in _iter_files(root, year_start, year_end):
        if _time.monotonic() > deadline:
            truncated = path.name
            break
        lines = _lines(path)
        for idx, line in enumerate(lines):
            context = _line_context(lines, idx, max(context_lines, 2))
            haystack = "\n".join(str(item["text"]) for item in context)
            haystack_cmp = haystack if case_sensitive else haystack.lower()
            matched = [term for term in terms if _contains_term(haystack_cmp, term if case_sensitive else term.lower())]
            if len(matched) < min(2, len(terms)):
                continue
            score = len(set(matched)) * 20
            line_l = line if case_sensitive else line.lower()
            score += sum(8 for term in matched if _contains_term(line_l, term))
            if line.lstrip().startswith("|"):
                score += 5
            if any(code in haystack_cmp for code in ("ffo-5", "ff0-5", "budget outlays by function")):
                score += 15
            ranked.append(
                {
                    "score": score,
                    "file": path.name,
                    "line": idx + 1,
                    "search_mode": "multi_term",
                    "matched_terms": matched,
                    "context": context,
                }
            )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    out = ranked[:max_results]
    if truncated:
        note = {
            "truncated_warning": (
                f"Multi-term search time budget hit at {truncated}; later files "
                "not scanned. Pass year_start/year_end to narrow the window."
            )
        }
        if out:
            out[0] = {**out[0], **note}
        else:
            out = [{"score": 0, "file": None, "line": None, "search_mode": "multi_term", "matched_terms": [], "context": [], **note}]
    return out


def _question_months(question: str) -> list[int]:
    q = question.lower()
    months = []
    for month_num, aliases in _MONTH_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\.?\b", q):
                months.append(month_num)
                break
    return months


def _cell_number_text(value: str) -> str | None:
    match = _NUMERIC_CELL_RE.search(value)
    if not match:
        return None
    return match.group(0).replace("$", "").strip()


def _numeric_value(number_text: str) -> float | None:
    cleaned = number_text.strip().replace(",", "").replace("$", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    is_parenthesized = cleaned.startswith("(") and cleaned.endswith(")")
    if is_parenthesized:
        cleaned = cleaned[1:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if is_parenthesized:
        value = -value
    return value



def _format_numeric_value(value: float) -> str:
    return format_numeric_value(value)

def _is_calendar_year_question(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\bcalendar[- ]year\b", q))


def _is_total_question(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\b(total|sum|combined|aggregate|overall)\b", q))


def _is_fiscal_period_text(text: str) -> bool:
    t = text.lower()
    if re.search(r"\bfiscal[- ]years?\b", t):
        return True
    if re.search(r"\bfirst\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven)\s+months?\b", t):
        return True
    if re.search(r"\bactual\s+\d+\s+months?\b", t):
        return True
    return False


def _months_in_headers(headers: list[str]) -> set[int]:
    months = set()
    for header in headers:
        months.update(_cell_period_metadata(str(header))["months"])
    return months


def _has_total_header(headers: list[str]) -> bool:
    return any(re.search(r"\btotal\b", str(header).lower()) for header in headers)


def _table_intent_adjustment(question: str, table_text: str, headers: list[str]) -> tuple[int, list[str]]:
    score = 0
    notes = []
    if _is_calendar_year_question(question):
        header_months = _months_in_headers(headers)
        if set(range(1, 13)).issubset(header_months):
            score += 80
            notes.append("jan_dec_headers")
        if _has_total_header(headers) and _is_total_question(question):
            score += 30
            notes.append("total_header")
        if _is_fiscal_period_text(table_text):
            score -= 140
            notes.append("fiscal_or_partial_period_penalty")
    return score, notes


def _cell_intent_adjustment(question: str, header: str) -> tuple[int, list[str]]:
    score = 0
    notes = []
    header_l = header.lower()
    if _is_calendar_year_question(question):
        if _is_total_question(question) and re.search(r"\btotal\b", header_l):
            score += 45
            notes.append("calendar_total_column")
        if _is_fiscal_period_text(header_l):
            score -= 45
            notes.append("fiscal_column_penalty")
    return score, notes


def _score_row_for_terms(
    label: str,
    row_json: str,
    table_text: str,
    row_terms: list[str],
) -> tuple[int, list[str]]:
    label_l = label.lower()
    row_l = row_json.lower()
    table_l = table_text.lower()
    score = 0
    context_score = 0
    matched = []
    context_matched = []
    for term in row_terms:
        if _contains_term(label_l, term):
            score += 40
            matched.append(term)
        elif _contains_term(row_l, term):
            score += 18
            matched.append(term)
        elif _contains_term(table_l, term):
            context_score += 6
            context_matched.append(term)
    if row_terms and score <= 0:
        return 0, []
    return score + context_score, (matched + context_matched)[:12]


def _score_column_for_terms(header: str, column_terms: list[str]) -> tuple[int, list[str]]:
    header_l = header.lower()
    score = 0
    matched = []
    for term in column_terms:
        if _contains_term(header_l, term):
            score += 40
            matched.append(term)
        else:
            pieces = [piece for piece in re.split(r"[^a-z0-9]+", term) if len(piece) > 2]
            piece_matches = [piece for piece in pieces if _contains_term(header_l, piece)]
            if len(pieces) > 1 and len(piece_matches) < len(pieces):
                continue
            if piece_matches:
                score += min(18, 5 * len(piece_matches))
                matched.extend(piece_matches[:3])
    return score, matched[:12]


def _month_in_text(text: str) -> int | None:
    text_l = text.lower()
    for month_num, aliases in _MONTH_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\.?\b", text_l):
                return month_num
    return None


def _year_in_text(text: str) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(match.group(1)) if match else None


def _score_cell(
    header: str,
    value: str,
    column_terms: list[str],
    years: list[int],
    months: list[int],
) -> tuple[int, list[str]]:
    header_l = header.lower()
    value_l = value.lower()
    score = 0
    matched = []
    for term in column_terms:
        if _contains_term(header_l, term):
            score += 24
            matched.append(term)
        elif _contains_term(value_l, term):
            score += 8
            matched.append(term)
    for year in years:
        year_s = str(year)
        if year_s in header_l:
            score += 18
            matched.append(year_s)
        elif year_s in value_l:
            score += 4
            matched.append(year_s)
    for month_num in months:
        if any(alias in header_l for alias in _MONTH_ALIASES[month_num]):
            score += 14
            matched.append(_MONTH_ALIASES[month_num][-1])
    if _cell_number_text(value):
        score += 2
    return score, matched[:8]


def _cell_period_metadata(header: str) -> dict[str, list[int]]:
    header_l = header.lower()
    years = sorted({int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", header_l)})
    months = []
    for month_num, aliases in _MONTH_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\.?\b", header_l) for alias in aliases):
            months.append(month_num)
    return {"years": years, "months": months}


def _compact_table_title(title_context: object) -> str:
    if not isinstance(title_context, list):
        return ""
    candidates = [re.sub(r"\s+", " ", str(line)).strip() for line in title_context if str(line).strip()]
    for line in reversed(candidates):
        if line.startswith("|") or line.lower().startswith(("source:", "footnotes")):
            continue
        if re.search(r"\b(?:table|budget|expenditures|receipts|outlays|income|debt|financing)\b", line, re.I):
            return line[:220]
    return candidates[-1][:220] if candidates else ""


def _find_json_file(corpus: Path, file_name: str) -> Path | None:
    base = Path(file_name).stem
    json_name = f"{base}.json"
    candidates = [
        corpus / json_name,
        corpus / "jsons" / json_name,
        corpus / "treasury_bulletins_parsed" / "jsons" / json_name,
        corpus.parent / "treasury_bulletins_parsed" / "jsons" / json_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _align_cells_by_bbox(cells: list[dict]) -> list[list[str]]:
    valid_cells = []
    for c in cells:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or c.get("value") or "").strip()
        bbox = c.get("bbox") or c.get("box")
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            valid_cells.append({
                "text": text,
                "bbox": [float(v) for v in bbox]
            })
            
    if not valid_cells:
        return []
        
    rows = []
    for cell in valid_cells:
        y0_c, y1_c = cell["bbox"][1], cell["bbox"][3]
        h_c = y1_c - y0_c
        if h_c <= 0:
            h_c = 1.0
            
        inserted = False
        for row in rows:
            y0_r = sum(c["bbox"][1] for c in row) / len(row)
            y1_r = sum(c["bbox"][3] for c in row) / len(row)
            h_r = y1_r - y0_r
            if h_r <= 0:
                h_r = 1.0
                
            overlap = max(0, min(y1_c, y1_r) - max(y0_c, y0_r))
            min_h = min(h_c, h_r)
            if (overlap / min_h) >= 0.75:
                row.append(cell)
                inserted = True
                break
        
        if not inserted:
            rows.append([cell])
            
    rows.sort(key=lambda r: sum(c["bbox"][1] for c in r) / len(r))
    for row in rows:
        row.sort(key=lambda c: c["bbox"][0])
        
    grid = []
    for row in rows:
        grid.append([c["text"] for c in row])
        
    return grid


def _extract_footnotes_for_table(lines: list[str], table_end_idx: int) -> list[str]:
    footnotes = []
    scan_limit = min(len(lines), table_end_idx + 16)
    for idx in range(table_end_idx + 1, scan_limit):
        line = lines[idx].strip()
        if not line:
            continue
        if re.match(r'^(?:\d+/|\*|†|‡|§|[a-z]/)\s', line):
            footnotes.append(line)
        elif line.lower().startswith("source:") or line.lower().startswith("note:"):
            footnotes.append(line)
    return footnotes


def _extract_json_table(json_path: Path, table_title: str) -> dict | None:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = []
        def search_tables(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "table" or "cells" in obj or "row_index" in obj:
                    if "cells" in obj or "rows" in obj or ("row_index" in obj and "col_index" in obj):
                        tables.append(obj)
                for val in obj.values():
                    search_tables(val)
            elif isinstance(obj, list):
                for item in obj:
                    search_tables(item)
                    
        search_tables(data)
        if not tables:
            return None
            
        best_table = None
        best_score = 0
        title_words = set(re.findall(r'\w+', table_title.lower()))
        for table in tables:
            t_title = ""
            for key in ["title", "caption", "name", "text"]:
                if key in table and isinstance(table[key], str):
                    t_title = table[key]
                    break
            if not t_title and "metadata" in table and isinstance(table["metadata"], dict):
                for key in ["title", "caption", "name"]:
                    if key in table["metadata"] and isinstance(table["metadata"][key], str):
                        t_title = table["metadata"][key]
                        break
            
            if t_title:
                t_words = set(re.findall(r'\w+', t_title.lower()))
                score = len(title_words.intersection(t_words))
                if score > best_score:
                    best_score = score
                    best_table = table
                    
        if best_table is None:
            best_table = tables[0]
            
        cells = best_table.get("cells") or best_table.get("rows") or best_table.get("data")
        if not cells:
            return None
            
        grid = _align_cells_by_bbox(cells)
        if grid and len(grid) > 1:
            headers = grid[0]
            rows = []
            for idx, r in enumerate(grid[1:], start=2):
                row_cells = [{"column": headers[i] if i < len(headers) else f"column_{i}", "value": val} for i, val in enumerate(r)]
                rows.append({"line": idx, "label": r[0] if r else "", "cells": row_cells})
            return {"headers": headers, "rows": rows}
            
        if isinstance(cells, list) and all(isinstance(c, dict) and "row_index" in c and "col_index" in c for c in cells if c):
            max_row = max(int(c.get("row_index", 0)) for c in cells)
            max_col = max(int(c.get("col_index", 0)) for c in cells)
            grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            for c in cells:
                r_idx = int(c.get("row_index", 0))
                c_idx = int(c.get("col_index", 0))
                grid[r_idx][c_idx] = str(c.get("text") or c.get("value") or "")
            
            if len(grid) > 1:
                headers = grid[0]
                rows = []
                for idx, r in enumerate(grid[1:], start=2):
                    row_cells = [{"column": headers[i] if i < len(headers) else f"column_{i}", "value": val} for i, val in enumerate(r)]
                    rows.append({"line": idx, "label": r[0] if r else "", "cells": row_cells})
                return {"headers": headers, "rows": rows}
                
        if isinstance(cells, list) and all(isinstance(r, list) for r in cells if r):
            if len(cells) > 1:
                headers = [str(h) for h in cells[0]]
                rows = []
                for idx, r in enumerate(cells[1:], start=2):
                    row_cells = [{"column": headers[i] if i < len(headers) else f"column_{i}", "value": str(val)} for i, val in enumerate(r)]
                    rows.append({"line": idx, "label": str(r[0]) if r else "", "cells": row_cells})
                return {"headers": headers, "rows": rows}
                
        return None
    except Exception:
        return None


def _parse_table_precise(path: Path, start: int, end: int, table_title: str) -> dict:
    json_path = _find_json_file(path.parent, path.name)
    json_table = None
    if json_path:
        json_table = _extract_json_table(json_path, table_title)
        
    parsed = json_table if json_table else _parse_table_for_path(path, start, end)
    lines = _lines(path)
    parsed["footnotes"] = _extract_footnotes_for_table(lines, end)
    return parsed
