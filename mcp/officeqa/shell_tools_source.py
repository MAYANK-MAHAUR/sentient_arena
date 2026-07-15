#!/usr/bin/env python3
"""OfficeQA shell-callable helper.

Some models prefer ``shell`` over MCP tools. This CLI gives those models a
high-quality route through the same parsing logic as the MCP server — without
embedding any OfficeQA answers, CPI tables, or lookup data.

Usage::

    python3 /tmp/officeqa/tools.py search "<query>" [--year-start Y] [--year-end Y] [--max N]
    python3 /tmp/officeqa/tools.py read   <file> [--row "label"] [--col "header"] [--around L]
    python3 /tmp/officeqa/tools.py table  <file> <line>
    python3 /tmp/officeqa/tools.py calc   "<expression>" [var=value ...]
    python3 /tmp/officeqa/tools.py finalize "<value>"
    python3 /tmp/officeqa/tools.py help

Stdlib-only. The script is installed by the MCP server at import time.
"""
from __future__ import annotations

import ast
import math
import os
import re
import statistics
import sys
from pathlib import Path

CORPUS_ROOT = Path(os.environ.get("OFFICEQA_CORPUS", "/app/corpus"))
ANSWER_PATH = Path(os.environ.get("OFFICEQA_ANSWER_PATH", "/app/answer.txt"))
_CALL_LOG = Path(os.environ.get("OFFICEQA_CALL_COUNTER", "/tmp/officeqa_tool_calls.count"))
_NAG_AT = 5
_STOP_AT = 10


def _bump() -> int:
    """Increment the shared call-count file. Returns the new count."""
    try:
        current = 0
        if _CALL_LOG.exists():
            raw = _CALL_LOG.read_text(encoding="utf-8").strip()
            if raw:
                current = int(raw)
        new = current + 1
        _CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        _CALL_LOG.write_text(str(new), encoding="utf-8")
        return new
    except (OSError, ValueError):
        return 1


def _nag(count: int) -> None:
    """Write a budget warning to stderr when the combined call count
    exceeds the thresholds shared with the MCP server."""
    if count >= _STOP_AT:
        print(
            f"\n*** STOP: {count} combined tool calls (shell+MCP). "
            f"Finalize your answer NOW: "
            f"python3 /tmp/officeqa/tools.py finalize \"<value>\" ***\n",
            file=sys.stderr,
        )
    elif count >= _NAG_AT:
        print(
            f"\n--- BUDGET: {count} combined tool calls (shell+MCP). "
            f"Limit is {_STOP_AT}. Finalize within the next 2 calls. ---\n",
            file=sys.stderr,
        )


def _iter_corpus_files(year_start: int | None = None, year_end: int | None = None):
    if not CORPUS_ROOT.is_dir():
        return
    files = sorted(CORPUS_ROOT.glob("treasury_bulletin_*.txt"))
    if not files:
        files = sorted(CORPUS_ROOT.glob("*.txt"))
    for path in files:
        match = re.search(r"(\d{4})[_-](\d{2})", path.name)
        if match:
            year = int(match.group(1))
            if year_start is not None and year < year_start:
                continue
            if year_end is not None and year > year_end:
                continue
        yield path


def _resolve_file(name: str) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = CORPUS_ROOT / name
    if not p.exists():
        candidate = CORPUS_ROOT / Path(name).name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"file not found: {name}")
    return p


_NUM_RE = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", stripped)]


def _is_sep(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def _table_bounds(lines: list[str], idx: int) -> tuple[int, int]:
    start = idx
    while start > 0 and lines[start - 1].lstrip().startswith("|"):
        start -= 1
    end = idx
    while end + 1 < len(lines) and lines[end + 1].lstrip().startswith("|"):
        end += 1
    return start, end


def _numeric_ratio(cells: list[str]) -> float:
    if len(cells) <= 1:
        return 0.0
    body = cells[1:]
    counted = numeric = 0
    for c in body:
        c = c.strip()
        if not c or c.lower() in {"nan", "none", "null"}:
            continue
        counted += 1
        if _NUM_RE.search(c):
            numeric += 1
    return numeric / counted if counted else 0.0


def _detect_header_span(table_lines: list[str]) -> int:
    span = 0
    for i in range(min(4, len(table_lines))):
        cells = _split_row(table_lines[i])
        if not cells:
            break
        if _numeric_ratio(cells) < 0.5:
            span = i + 1
        else:
            break
    return max(1, span) if span else 1


def _flatten_headers(header_lines: list[str]) -> list[str]:
    rows = [[c.strip() for c in _split_row(ln)] for ln in header_lines]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # Fill parent-header blanks horizontally.
    for r in rows[:-1]:
        current = ""
        for i, c in enumerate(r):
            if i == 0:
                continue
            if c:
                current = c
            elif current:
                r[i] = current
    headers = []
    for j in range(width):
        parts = []
        for r in rows:
            part = r[j]
            if part and (not parts or parts[-1].lower() != part.lower()):
                parts.append(part)
        headers.append(" > ".join(parts) if parts else (f"column_{j}" if j else "label"))
    return headers


def _parse_table(lines: list[str], start: int, end: int) -> tuple[list[str], list[dict]]:
    block = lines[start : end + 1]
    sep = next((i for i, ln in enumerate(block) if _is_sep(ln)), None)
    if sep is None:
        span = _detect_header_span(block)
        head_lines = block[:span]
        data_lines = block[span:]
        data_off = span
    else:
        head_lines = block[:sep]
        data_lines = block[sep + 1 :]
        data_off = sep + 1
    headers = _flatten_headers(head_lines)
    rows = []
    for off, ln in enumerate(data_lines, start=start + data_off + 1):
        cells = _split_row(ln)
        if not cells:
            continue
        rows.append({
            "line": off,
            "label": cells[0],
            "cells": cells,
        })
    return headers, rows


_STOPWORDS = {
    "the", "and", "for", "of", "in", "to", "a", "an", "is", "are", "was",
    "were", "what", "which", "with", "by", "from", "be", "on", "as", "or",
}


def _terms(query: str, limit: int = 8) -> list[str]:
    raw = re.findall(r"[a-z0-9][a-z0-9-]*", query.lower())
    out = []
    for t in raw:
        if len(t) < 3 or t in _STOPWORDS:
            continue
        if t.endswith("s") and len(t) > 4:
            t = t[:-1]
        if t not in out:
            out.append(t)
    return out[:limit]


def cmd_search(argv: list[str]) -> int:
    if not argv:
        print("usage: tools.py search \"<query>\" [--year-start Y] [--year-end Y] [--max N]", file=sys.stderr)
        return 1
    query = argv[0]
    ys = ye = None
    cap = 20
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--year-start" and i + 1 < len(argv):
            ys = int(argv[i + 1]); i += 2
        elif a == "--year-end" and i + 1 < len(argv):
            ye = int(argv[i + 1]); i += 2
        elif a == "--max" and i + 1 < len(argv):
            cap = max(1, int(argv[i + 1])); i += 2
        else:
            i += 1
    terms = _terms(query)
    if not terms:
        # Fallback: literal substring search.
        terms = [query.lower()]
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    hits = []
    for path in _iter_corpus_files(ys, ye):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            line_l = line.lower()
            matched = [t for t in terms if t in line_l]
            if len(matched) < min(2, len(terms)):
                continue
            score = len(set(matched)) * 20
            if line.lstrip().startswith("|"):
                score += 5
            hits.append((score, path.name, idx + 1, line.strip()[:200], matched))
            if len(hits) > cap * 4:
                break
    hits.sort(key=lambda x: x[0], reverse=True)
    if not hits:
        print("(no matches)")
        return 1
    for score, fn, ln, text, matched in hits[:cap]:
        print(f"{fn}:{ln}\t({score}, matched={','.join(matched)})\t{text}")
    return 0


def cmd_table(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: tools.py table <file> <line>", file=sys.stderr)
        return 1
    path = _resolve_file(argv[0])
    target = int(argv[1])
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    idx = min(max(target - 1, 0), len(lines) - 1)
    while idx > 0 and not lines[idx].lstrip().startswith("|"):
        idx -= 1
    if not lines[idx].lstrip().startswith("|"):
        print("(no table at or before line)", file=sys.stderr)
        return 1
    start, end = _table_bounds(lines, idx)
    title_start = max(0, start - 8)
    print("[title context]")
    for ln in lines[title_start:start]:
        if ln.strip():
            print("  " + ln.strip())
    headers, rows = _parse_table(lines, start, end)
    print(f"[headers] {' | '.join(headers)}")
    for r in rows[:60]:
        print(f"L{r['line']}\t" + "\t".join(r["cells"]))
    if len(rows) > 60:
        print(f"(... {len(rows) - 60} more rows ...)")
    return 0


def cmd_read(argv: list[str]) -> int:
    if not argv:
        print("usage: tools.py read <file> [--row \"label\"] [--col \"header\"] [--around L]", file=sys.stderr)
        return 1
    path = _resolve_file(argv[0])
    row_q = col_q = None
    around = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--row" and i + 1 < len(argv):
            row_q = argv[i + 1].lower(); i += 2
        elif a == "--col" and i + 1 < len(argv):
            col_q = argv[i + 1].lower(); i += 2
        elif a == "--around" and i + 1 < len(argv):
            around = int(argv[i + 1]); i += 2
        else:
            i += 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    table_indices = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")]
    if around is not None:
        idx = min(max(around - 1, 0), len(lines) - 1)
        candidates = [t for t in table_indices if abs(t - idx) <= 200]
        if not candidates:
            print("(no table near line)", file=sys.stderr)
            return 1
        starts_seen = set()
        ordered_tables = []
        for t in candidates:
            s, e = _table_bounds(lines, t)
            if s not in starts_seen:
                starts_seen.add(s)
                ordered_tables.append((s, e))
    else:
        starts_seen = set()
        ordered_tables = []
        for t in table_indices:
            s, e = _table_bounds(lines, t)
            if s not in starts_seen:
                starts_seen.add(s)
                ordered_tables.append((s, e))
    printed = 0
    for s, e in ordered_tables:
        headers, rows = _parse_table(lines, s, e)
        rows_match = rows if row_q is None else [r for r in rows if row_q in r["label"].lower()]
        if not rows_match:
            continue
        print(f"[table @ line {s + 1}–{e + 1}]")
        title_start = max(0, s - 6)
        for ln in lines[title_start:s]:
            if ln.strip():
                print("  " + ln.strip())
        for r in rows_match[:6]:
            label = r["label"]
            cells = r["cells"]
            print(f"ROW @ L{r['line']}: {label}")
            for j, c in enumerate(cells):
                if j == 0:
                    continue
                head = headers[j] if j < len(headers) else f"column_{j}"
                if col_q is not None and col_q not in head.lower():
                    continue
                if c.strip():
                    print(f"  {head} = {c}")
            printed += 1
            if printed >= 6:
                return 0
        print()
    if not printed:
        print("(no matching rows)", file=sys.stderr)
        return 1
    return 0


def _safe_eval(expr: str, variables: dict[str, float]) -> float | list[float]:
    def _ev(n):
        if isinstance(n, ast.Expression):
            return _ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in variables:
                return variables[n.id]
            raise ValueError(f"unknown variable: {n.id}")
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                return -_ev(n.operand)
            if isinstance(n.op, ast.UAdd):
                return _ev(n.operand)
        if isinstance(n, ast.BinOp):
            L, R = _ev(n.left), _ev(n.right)
            op = n.op
            if isinstance(op, ast.Add): return L + R
            if isinstance(op, ast.Sub): return L - R
            if isinstance(op, ast.Mult): return L * R
            if isinstance(op, ast.Div):
                if R == 0: raise ValueError("division by zero")
                return L / R
            if isinstance(op, ast.Pow): return L ** R
            if isinstance(op, ast.Mod): return L % R
        if isinstance(n, ast.List):
            return [_ev(e) for e in n.elts]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = n.func.id
            args = [_ev(a) for a in n.args]
            def flat(a):
                out = []
                for x in a:
                    if isinstance(x, list): out.extend(x)
                    else: out.append(x)
                return out
            if fn == "abs": return abs(args[0])
            if fn == "round": return round(args[0], int(args[1])) if len(args) > 1 else round(args[0])
            if fn == "min": return min(flat(args))
            if fn == "max": return max(flat(args))
            if fn == "sum": return sum(flat(args))
            if fn == "mean": v = flat(args); return sum(v) / len(v)
            if fn == "median": return statistics.median(flat(args))
            if fn == "stdev": return statistics.stdev(flat(args))
            if fn == "pstdev": return statistics.pstdev(flat(args))
            if fn == "variance": return statistics.variance(flat(args))
            if fn == "pvariance": return statistics.pvariance(flat(args))
            if fn == "sqrt": return math.sqrt(args[0])
            if fn == "log": return math.log(*args)
            if fn == "exp": return math.exp(args[0])
            if fn == "pct_change":
                if args[0] == 0: raise ValueError("division by zero")
                return (args[1] - args[0]) / args[0] * 100
            if fn == "abs_pct_change":
                if args[0] == 0: raise ValueError("division by zero")
                return abs((args[1] - args[0]) / args[0]) * 100
            if fn == "cagr":
                return ((args[1] / args[0]) ** (1.0 / args[2]) - 1) * 100
        raise ValueError(f"unsupported expression element: {ast.dump(n)}")
    return _ev(ast.parse(expr.replace("^", "**"), mode="eval"))


def cmd_calc(argv: list[str]) -> int:
    if not argv:
        print("usage: tools.py calc \"<expression>\" [var=value ...]", file=sys.stderr)
        return 1
    expr = argv[0]
    variables: dict[str, float] = {}
    for a in argv[1:]:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        try:
            variables[k.strip()] = float(v.strip())
        except ValueError:
            pass
    try:
        result = _safe_eval(expr, variables)
    except Exception as exc:
        print(f"calc error: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


_PROSE_TOKEN_RE = re.compile(
    r"\b(?:because|since|therefore|hence|approximately|approx|roughly|"
    r"the\s+answer|answer\s*[:\-]|based\s+on|according\s+to|"
    r"computed|calculated|reasoning|note|"
    r"as\s+follows|i\.e\.|e\.g\.|"
    r"please|note\s+that|to\s+find|to\s+compute|"
    r"step\s*\d|here\s+is|here's)\b",
    re.IGNORECASE,
)
_DATE_LIKE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.IGNORECASE,
)


def _validate(cleaned: str) -> None:
    if not cleaned:
        raise ValueError("answer must not be empty")
    if len(cleaned) > 250:
        raise ValueError("answer is too long")
    if "\n" in cleaned or "\r" in cleaned:
        raise ValueError("answer must be a single line")
    if _PROSE_TOKEN_RE.search(cleaned):
        raise ValueError("answer looks like prose")
    no_sym = re.sub(r"[0-9+\-.,\[\]\s()%$:/]", "", cleaned)
    if not no_sym:
        return
    unit_re = re.compile(
        r"\b(?:trillions?|billions?|millions?|thousands?|hundreds?|"
        r"percent(?:age)?|dollars?)\b",
        re.IGNORECASE,
    )
    after_units = unit_re.sub("", cleaned)
    after_units_letters = re.sub(r"[0-9+\-.,\[\]\s()%$:/]", "", after_units)
    if not after_units_letters:
        return
    if _DATE_LIKE_RE.search(cleaned):
        return
    if cleaned.startswith("[") and cleaned.endswith("]"):
        ok = True
        for part in (p.strip() for p in cleaned[1:-1].split(",")):
            if not part:
                ok = False; break
            if re.fullmatch(r"[0-9+\-.\s()%$]+", part):
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z\-]{0,29}", part):
                continue
            ok = False; break
        if ok:
            return
    if " " not in cleaned and len(cleaned) <= 30 and re.fullmatch(r"[A-Za-z][A-Za-z\-/]{0,29}", cleaned):
        return
    raise ValueError("answer contains unsupported characters or prose")


def cmd_finalize(argv: list[str]) -> int:
    if not argv:
        print("usage: tools.py finalize \"<value>\"", file=sys.stderr)
        return 1
    raw = " ".join(argv).strip().replace("\u2212", "-")
    try:
        _validate(raw)
    except ValueError as exc:
        print(f"finalize REJECTED: {exc}", file=sys.stderr)
        return 1
    try:
        ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANSWER_PATH.write_text(raw + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"finalize WRITE_ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        _CALL_LOG.write_text("0", encoding="utf-8")
    except OSError:
        pass
    print(raw)
    return 0


HELP = """OfficeQA shell helper. Subcommands:

  search "<query>" [--year-start Y] [--year-end Y] [--max N]
        Multi-term search across the corpus. Returns top hits as
        FILE:LINE  (score, matched=...)  <line text>

  read <file> [--row "label"] [--col "header"] [--around L]
        Parse markdown tables in <file> and print the row(s) whose label
        contains "label". If --col is supplied, only that column's value
        is printed. --around N restricts to tables near line N.

  table <file> <line>
        Print the markdown table at or before <line> as TSV with
        flattened multi-row headers.

  calc "<expression>" [var=value ...]
        Safe arithmetic. Functions: abs, round, min, max, sum, mean,
        median, stdev, pstdev, variance, pvariance, sqrt, log, exp,
        pct_change(old, new), abs_pct_change(old, new), cagr(start, end, years).

  finalize "<value>"
        Validate and write the final answer to /app/answer.txt. Same
        validator as the MCP finalize_answer tool. Use this instead of
        raw shell redirection so the answer format is checked.

  help
        Show this message.
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(HELP)
        return 0
    if argv[0] != "finalize":
        _nag(_bump())
    cmd, rest = argv[0], argv[1:]
    if cmd == "search":   return cmd_search(rest)
    if cmd == "read":     return cmd_read(rest)
    if cmd == "table":    return cmd_table(rest)
    if cmd == "calc":     return cmd_calc(rest)
    if cmd == "finalize": return cmd_finalize(rest)
    print(f"unknown subcommand: {cmd}\n", file=sys.stderr)
    print(HELP, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
