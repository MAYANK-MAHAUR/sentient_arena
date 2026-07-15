from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/app/corpus")

STOPWORDS = {
    "about",
    "according",
    "all",
    "amount",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "behalf",
    "both",
    "by",
    "calendar",
    "category",
    "comma",
    "comparable",
    "containing",
    "corresponding",
    "decimal",
    "dollar",
    "dollars",
    "enclosed",
    "expressed",
    "federal",
    "fiscal",
    "for",
    "from",
    "government",
    "highest",
    "in",
    "inclusive",
    "individual",
    "just",
    "million",
    "millions",
    "nominal",
    "number",
    "numbers",
    "only",
    "order",
    "output",
    "percent",
    "place",
    "question",
    "reported",
    "return",
    "rounded",
    "same",
    "separated",
    "specifically",
    "states",
    "subquestions",
    "the",
    "these",
    "this",
    "to",
    "total",
    "treasury",
    "united",
    "us",
    "using",
    "value",
    "values",
    "was",
    "were",
    "what",
    "which",
    "within",
    "with",
    "year",
    "years",
}

KNOWN_PHRASES = [
    "associated activities",
    "budget expenditures",
    "budget receipts",
    "cash income",
    "cash outgo",
    "debt outstanding",
    "expenditures by agencies",
    "expenditures by functions",
    "foreign and international",
    "geometric mean",
    "gross public debt",
    "individual income",
    "individual income taxes",
    "interest-bearing debt",
    "marketable securities",
    "means of financing",
    "national defense",
    "national defense and associated activities",
    "net of refunds",
    "net interest",
    "noncash rollover",
    "non-domestic investors",
    "ordinary least squares",
    "public debt",
    "receipts by source",
    "social insurance",
    "tax and loan",
    "treasury bills",
    "treasury bonds",
    "treasury notes",
]

SKIP_TABLE_MARKERS = [
    "cumulative index",
    "cumulative table of contents",
    "table of contents",
    "page number",
    "issue and page number",
]

MONTHS = {
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

MONTH_RE = (
    r"Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|May|"
    r"Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|Sep(?:t|tember)?\.?|"
    r"Oct(?:ober)?\.?|Nov(?:ember)?\.?|Dec(?:ember)?\.?"
)

DATE_RE = rf"(?:{MONTH_RE})\s+\d{{1,2}},\s+(?:19|20)\d{{2}}"
MONTH_ONLY_RE = re.compile(rf"(?:{MONTH_RE})", re.IGNORECASE)


def resolve_root(root: str | None = None) -> Path:
    for candidate in [Path(root)] if root else []:
        if candidate.exists() and candidate.is_dir():
            return candidate
    for candidate in [DEFAULT_ROOT, Path("corpus"), Path("/workspace/corpus")]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return DEFAULT_ROOT


def iter_files(root: Path, year_start: int | None = None, year_end: int | None = None) -> Iterable[Path]:
    for path in sorted(root.glob("treasury_bulletin_*.txt")):
        match = re.search(r"(\d{4})[_-](\d{2})", path.name)
        if match:
            year = int(match.group(1))
            if year_start is not None and year < year_start:
                continue
            if year_end is not None and year > year_end:
                continue
        yield path


def safe_file(root: Path, file_name: str) -> Path:
    path = Path(file_name)
    if not path.is_absolute():
        path = root / file_name
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("file must stay inside corpus root")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


@lru_cache(maxsize=1024)
def read_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="replace")


def lines(path: Path) -> list[str]:
    return read_text(str(path.resolve())).splitlines()


def file_year(path: Path) -> int | None:
    match = re.search(r"(\d{4})[_-](\d{2})", path.name)
    return int(match.group(1)) if match else None


def unique(items: Iterable[str], limit: int | None = None) -> list[str]:
    seen = set()
    out = []
    for item in items:
        cleaned = item.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if limit is not None and len(out) >= limit:
            break
    return out


def question_terms(question: str) -> tuple[list[str], list[str], list[int]]:
    q = question.lower()
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    words = re.findall(r"[a-z][a-z0-9-]*", q)
    terms = []
    for word in words:
        if len(word) < 3 or word in STOPWORDS:
            continue
        terms.append(word)
        if word.endswith("s") and len(word) > 4:
            terms.append(word[:-1])
    phrases = [phrase for phrase in KNOWN_PHRASES if phrase in q]
    for size in (4, 3, 2):
        for idx in range(0, max(0, len(words) - size + 1)):
            chunk = words[idx : idx + size]
            if any(word in STOPWORDS or len(word) < 3 for word in chunk):
                continue
            phrase = " ".join(chunk)
            if len(phrase) >= 9:
                phrases.append(phrase)
    return unique(terms, 28), unique(phrases, 20), years


def split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def separator_row(line: str) -> bool:
    cells = split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_bounds(all_lines: list[str], idx: int) -> tuple[int, int]:
    start = idx
    while start > 0 and all_lines[start - 1].lstrip().startswith("|"):
        start -= 1
    end = idx
    while end + 1 < len(all_lines) and all_lines[end + 1].lstrip().startswith("|"):
        end += 1
    return start, end


def title_context(all_lines: list[str], table_start: int, scan: int = 10) -> list[str]:
    start = max(0, table_start - scan)
    return [all_lines[i] for i in range(start, table_start) if all_lines[i].strip()][-scan:]


def parse_table(all_lines: list[str], start: int, end: int) -> dict[str, object]:
    table_lines = all_lines[start : end + 1]
    sep = next((i for i, line in enumerate(table_lines) if separator_row(line)), None)
    if sep is None:
        sep = 1 if len(table_lines) > 1 else 0
    header_lines = table_lines[:sep]
    headers = split_row(header_lines[-1]) if header_lines else []
    rows = []
    for line_no, line in enumerate(table_lines[sep + 1 :], start=start + sep + 2):
        cells = split_row(line)
        if not cells:
            continue
        rows.append(
            {
                "line": line_no,
                "label": cells[0],
                "cells": [
                    {"column": headers[i] if i < len(headers) else f"column_{i}", "value": cell}
                    for i, cell in enumerate(cells)
                ],
            }
        )
    return {"headers": headers, "rows": rows}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_money_million(value: str | None) -> int | None:
    if not value:
        return None
    cleaned = value.replace(",", "")
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", cleaned)
    if not match:
        return None
    return int(Decimal(match.group(1)).to_integral_value(rounding=ROUND_HALF_UP))


def money_to_dollars(value_million: int | None) -> int | None:
    if value_million is None:
        return None
    return value_million * 1_000_000


def parse_date(text: str) -> dict[str, int | str] | None:
    match = re.search(rf"\b({MONTH_RE})\s+(\d{{1,2}}),\s+((?:19|20)\d{{2}})\b", text, re.IGNORECASE)
    if not match:
        return None
    month_name = match.group(1).rstrip(".").lower()
    month = MONTHS.get(month_name)
    if month is None and month_name.startswith("sept"):
        month = 9
    if month is None:
        return None
    day = int(match.group(2))
    year = int(match.group(3))
    return {
        "text": match.group(0),
        "year": year,
        "month": month,
        "day": day,
        "iso": f"{year:04d}-{month:02d}-{day:02d}",
    }


def question_security_kind(question: str) -> str | None:
    q = question.lower()
    if "bond" in q:
        return "bond"
    if "bill" in q:
        return "bill"
    if "note" in q:
        return "note"
    return None


def question_term_year(question: str) -> int | None:
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    return years[-1] if years else None


def question_target_month(question: str) -> int | None:
    q = question.lower()
    if "end of " in q:
        tail = q.split("end of ", 1)[1]
        match = MONTH_ONLY_RE.search(tail)
        if match:
            return MONTHS.get(match.group(0).rstrip(".").lower())
    matches = MONTH_ONLY_RE.findall(q)
    if matches:
        return MONTHS.get(matches[-1].rstrip(".").lower())
    return None


def question_target_date(question: str) -> dict[str, int | str] | None:
    target = parse_date(question)
    if target:
        return target
    year = question_term_year(question)
    month = question_target_month(question)
    if year is None or month is None:
        return None
    # Treasury questions often say "maturing at the end of <month> <year>".
    # The matching auction paragraph normally spells this as the last day.
    last_day_by_month = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    day = last_day_by_month[month]
    if month == 2 and year % 4:
        day = 28
    return {
        "text": f"{year:04d}-{month:02d}-{day:02d}",
        "year": year,
        "month": month,
        "day": day,
        "iso": f"{year:04d}-{month:02d}-{day:02d}",
    }


def percent(part: int | None, whole: int | None) -> str | None:
    if part is None or whole in (None, 0):
        return None
    value = Decimal(part) / Decimal(whole) * Decimal(100)
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def auction_windows(
    root: Path,
    question: str,
    year_start: int | None = None,
    year_end: int | None = None,
    window_lines: int = 16,
) -> list[dict[str, object]]:
    q = question.lower()
    kind = question_security_kind(question)
    target_date = question_target_date(question)
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    if years:
        offset = 0
        if "30-year" in q:
            offset = 31
        elif "10-year" in q:
            offset = 11
        elif "5-year" in q:
            offset = 6
        elif "2-year" in q:
            offset = 3
        elif "bond" in q or "note" in q:
            offset = 31

        calc_start = max(1939, min(years) - offset - 3)
        calc_end = max(years) + 3
        if year_start is None:
            year_start = calc_start
        else:
            year_start = min(year_start, calc_start)
        if year_end is None:
            year_end = calc_end
        else:
            year_end = max(year_end, calc_end)

    candidates = []
    reserve_banks = r"Federal Reserve(?:\s+[A-Z]{2,})?\s+[Bb]anks"
    for path in iter_files(root, year_start, year_end):
        all_lines = lines(path)
        for idx, line in enumerate(all_lines):
            if "auction of" not in line.lower():
                continue
            title = normalize_text(line)
            title_l = title.lower()
            if kind == "note" and "note" not in title_l:
                continue
            if kind == "bill" and "bill" not in title_l:
                continue
            if kind == "bond" and "bond" not in title_l:
                continue
            end = min(len(all_lines), idx + window_lines)
            raw_window = all_lines[idx:end]
            text = normalize_text(" ".join(raw_window))
            text_l = text.lower()
            if "tenders" not in text_l and "bids" not in text_l:
                continue

            offered_amount = None
            offered_match = re.search(
                r"auction \$?([\d,]+(?:\.\d+)?)\s+million of",
                text,
                re.IGNORECASE,
            )
            if offered_match:
                offered_amount = parse_money_million(offered_match.group(1))

            refund_amount = None
            refund_maturity = None
            refund_match = re.search(
                rf"refund \$?([\d,]+(?:\.\d+)?)\s+million of .*?maturing\s+({DATE_RE})",
                text,
                re.IGNORECASE,
            )
            if refund_match:
                refund_amount = parse_money_million(refund_match.group(1))
                refund_maturity = parse_date(refund_match.group(2))

            issue_date = None
            issue_match = re.search(rf"dated\s+({DATE_RE})", text, re.IGNORECASE)
            if issue_match:
                issue_date = parse_date(issue_match.group(1))

            due_date = None
            due_match = re.search(rf"\bdue\s+({DATE_RE})", text, re.IGNORECASE)
            if due_match:
                due_date = parse_date(due_match.group(1))

            total_tenders = None
            accepted_auction = None
            total_match = re.search(
                r"(?:tenders|bids).*?totaled \$?([\d,]+(?:\.\d+)?)\s+million,\s+of which \$?([\d,]+(?:\.\d+)?)\s+million w(?:as|ere) accepted",
                text,
                re.IGNORECASE,
            )
            if total_match:
                total_tenders = parse_money_million(total_match.group(1))
                accepted_auction = parse_money_million(total_match.group(2))

            noncompetitive = None
            noncomp_match = re.search(
                r"Noncompetitive tenders.*?These totaled \$?([\d,]+(?:\.\d+)?)\s+million",
                text,
                re.IGNORECASE,
            )
            if noncomp_match:
                noncompetitive = parse_money_million(noncomp_match.group(1))

            competitive_private = None
            private_match = re.search(
                r"Competitive tenders accepted from private investors totaled \$?([\d,]+(?:\.\d+)?)\s+million",
                text,
                re.IGNORECASE,
            )
            if private_match:
                competitive_private = parse_money_million(private_match.group(1))

            foreign_international = None
            foreign_match = re.search(
                rf"\$?([\d,]+(?:\.\d+)?)\s+million(?:\s+of tenders)?(?:\s+w(?:as|ere))?(?:\s+accepted)?\s+at the average price from {reserve_banks},?\s+as agents for foreign and international monetary authorities",
                text,
                re.IGNORECASE,
            )
            if foreign_match:
                foreign_international = parse_money_million(foreign_match.group(1))

            government_accounts = None
            gov_match = re.search(
                rf"\$?([\d,]+(?:\.\d+)?)\s+million(?:\s+of tenders)?(?:\s+w(?:as|ere))?(?:\s+accepted)?\s+at the average price from Government accounts and {reserve_banks} for (?:their own account|themselves) in exchange for maturing securities",
                text,
                re.IGNORECASE,
            )
            if gov_match:
                government_accounts = parse_money_million(gov_match.group(1))

            score = 0
            matched = []
            if target_date:
                for label, date in (("refunded_maturity", refund_maturity), ("offered_due", due_date), ("issue", issue_date)):
                    if isinstance(date, dict) and date.get("iso") == target_date.get("iso"):
                        add = 90 if label == "offered_due" else 25 if label == "issue" else 18
                        score += add
                        matched.append(f"{label}={date['iso']}")
                    elif isinstance(date, dict) and date.get("year") == target_date.get("year") and date.get("month") == target_date.get("month"):
                        add = 34 if label == "offered_due" else 10 if label == "issue" else 8
                        score += add
                        matched.append(f"{label}_month={date['iso']}")
            if kind and kind in title_l:
                score += 10
                matched.append(kind)
            if "2-year" in q and "2-year" in text_l:
                score += 14
                matched.append("2-year")
            for term in ("foreign and international", "noncash", "rollover", "government accounts", "private investors"):
                if term in q and term in text_l:
                    score += 8
                    matched.append(term)
            if total_tenders is not None:
                score += 4

            fields = {
                "offered_amount_million": offered_amount,
                "offered_amount_dollars": money_to_dollars(offered_amount),
                "refund_amount_million": refund_amount,
                "refund_amount_dollars": money_to_dollars(refund_amount),
                "refunded_maturity_date": refund_maturity,
                "issue_date": issue_date,
                "offered_due_date": due_date,
                "total_submitted_tenders_million": total_tenders,
                "total_submitted_tenders_dollars": money_to_dollars(total_tenders),
                "accepted_in_auction_million": accepted_auction,
                "noncompetitive_accepted_million": noncompetitive,
                "competitive_private_accepted_million": competitive_private,
                "foreign_international_rollover_accepted_million": foreign_international,
                "government_and_fed_own_rollover_accepted_million": government_accounts,
            }
            fields["candidate_percentages"] = {
                "foreign_international_rollover_of_refund_amount": percent(foreign_international, refund_amount),
                "foreign_international_rollover_of_total_submitted_tenders": percent(foreign_international, total_tenders),
                "foreign_international_rollover_of_auction_accepted": percent(foreign_international, accepted_auction),
                "foreign_international_rollover_of_total_rollover_accepted": percent(
                    foreign_international,
                    (foreign_international or 0) + (government_accounts or 0) if foreign_international is not None or government_accounts is not None else None,
                ),
            }
            candidates.append(
                {
                    "score": score,
                    "matched": unique(matched, 12),
                    "file": path.name,
                    "line": idx + 1,
                    "title": title,
                    "fields": fields,
                    "evidence": [
                        {"line": idx + offset + 1, "text": raw_line}
                        for offset, raw_line in enumerate(raw_window)
                        if raw_line.strip()
                    ],
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def cmd_rank(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    terms = [term.lower() for term in args.terms if term.strip()]
    out = []
    for path in iter_files(root, args.year_start, args.year_end):
        text = read_text(str(path.resolve())).lower()
        counts = {term: text.count(term) for term in terms}
        matched = sum(1 for count in counts.values() if count)
        total = sum(counts.values())
        if total:
            out.append({"file": path.name, "matched_terms": matched, "total_hits": total, "counts": counts})
    out.sort(key=lambda item: (item["matched_terms"], item["total_hits"]), reverse=True)
    print(json.dumps(out[: args.max_files], indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    flags = 0 if args.case_sensitive else re.IGNORECASE
    pattern = re.compile(args.query if args.regex else re.escape(args.query), flags)
    hits = []
    for path in iter_files(root, args.year_start, args.year_end):
        all_lines = lines(path)
        for idx, line in enumerate(all_lines):
            if not pattern.search(line):
                continue
            start = max(0, idx - args.context)
            end = min(len(all_lines), idx + args.context + 1)
            hits.append(
                {
                    "file": path.name,
                    "line": idx + 1,
                    "context": [{"line": i + 1, "text": all_lines[i]} for i in range(start, end)],
                }
            )
            if len(hits) >= args.max_results:
                print(json.dumps(hits, indent=2))
                return
    print(json.dumps(hits, indent=2))


def cmd_rows(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    row_query = args.row.lower()
    title_terms = [term.lower() for term in args.title]
    paths = [safe_file(root, args.file)] if args.file else list(iter_files(root, args.year_start, args.year_end))
    results = []
    for path in paths:
        all_lines = lines(path)
        for idx, line in enumerate(all_lines):
            if not line.lstrip().startswith("|") or row_query not in line.lower():
                continue
            start, end = table_bounds(all_lines, idx)
            title = title_context(all_lines, start)
            parsed = parse_table(all_lines, start, end)
            haystack = "\n".join(title + [json.dumps(parsed["headers"])]).lower()
            if title_terms and not all(term in haystack for term in title_terms):
                continue
            row = next((row for row in parsed["rows"] if row["line"] == idx + 1), None)
            if row is None:
                continue
            results.append(
                {
                    "file": path.name,
                    "table_start_line": start + 1,
                    "title_context": title,
                    "headers": parsed["headers"],
                    "row": row,
                }
            )
            if len(results) >= args.max_results:
                print(json.dumps(results, indent=2))
                return
    print(json.dumps(results, indent=2))


def cmd_table(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    path = safe_file(root, args.file)
    all_lines = lines(path)
    idx = min(max(args.line - 1, 0), len(all_lines) - 1)
    while idx > 0 and not all_lines[idx].lstrip().startswith("|"):
        idx -= 1
    if not all_lines[idx].lstrip().startswith("|"):
        raise ValueError("no table found at or before line")
    start, end = table_bounds(all_lines, idx)
    parsed = parse_table(all_lines, start, end)
    rows = parsed["rows"]
    if args.row_filter:
        needle = args.row_filter.lower()
        rows = [row for row in rows if needle in row["label"].lower()]
    print(
        json.dumps(
            {
                "file": path.name,
                "table_start_line": start + 1,
                "table_end_line": end + 1,
                "title_context": title_context(all_lines, start),
                "headers": parsed["headers"],
                "rows": rows[: args.max_rows],
                "truncated": len(rows) > args.max_rows,
            },
            indent=2,
        )
    )


def row_score(label: str, row_text: str, table_text: str, terms: list[str], phrases: list[str]) -> tuple[int, list[str]]:
    label_l = label.lower()
    row_l = row_text.lower()
    table_l = table_text.lower()
    if any(marker in table_l for marker in SKIP_TABLE_MARKERS):
        return 0, []
    score = 0
    matched = []
    priority_terms = {
        "accepted",
        "bid",
        "bids",
        "investor",
        "investors",
        "maturing",
        "non-domestic",
        "noncash",
        "rollover",
        "submitted",
        "tender",
        "tenders",
    }
    for phrase in phrases:
        if phrase in label_l:
            score += 22 if any(term in phrase for term in priority_terms) else 14
            matched.append(phrase)
        elif phrase in row_l:
            score += 18 if any(term in phrase for term in priority_terms) else 10
            matched.append(phrase)
        elif phrase in table_l:
            score += 11 if any(term in phrase for term in priority_terms) else 5
            matched.append(phrase)
    for term in terms:
        weight = 3 if term in priority_terms else 1
        if term in label_l:
            score += 4 * weight
            matched.append(term)
        elif term in row_l:
            score += 2 * weight
            matched.append(term)
        elif term in table_l:
            score += 1 * weight
            matched.append(term)
    return score, unique(matched, 16)


def table_score(table_text: str, terms: list[str], phrases: list[str], years: list[int], path: Path) -> tuple[int, list[str]]:
    text_l = table_text.lower()
    if any(marker in text_l for marker in SKIP_TABLE_MARKERS):
        return 0, []
    score = 0
    matched = []
    priority_terms = {
        "accepted",
        "bid",
        "bids",
        "investor",
        "investors",
        "maturing",
        "non-domestic",
        "noncash",
        "rollover",
        "submitted",
        "tender",
        "tenders",
    }
    for phrase in phrases:
        if phrase in text_l:
            score += 20 if any(term in phrase for term in priority_terms) else 10
            matched.append(phrase)
    for term in terms:
        if term in text_l:
            score += 7 if term in priority_terms else 2
            matched.append(term)
    fy = file_year(path)
    for year in years:
        year_s = str(year)
        if year_s in text_l:
            score += 4
            matched.append(year_s)
        if fy is not None and abs(fy - year) <= 2:
            score += 1
    return score, unique(matched, 18)


def text_score(text: str, terms: list[str], phrases: list[str], years: list[int], path: Path) -> tuple[int, list[str]]:
    text_l = text.lower()
    if any(marker in text_l for marker in SKIP_TABLE_MARKERS):
        return 0, []
    priority_terms = {
        "accepted",
        "bid",
        "bids",
        "maturing",
        "non-domestic",
        "noncash",
        "refund",
        "rollover",
        "tender",
        "tenders",
    }
    score = 0
    matched = []
    for phrase in phrases:
        if phrase in text_l:
            score += 22 if any(term in phrase for term in priority_terms) else 12
            matched.append(phrase)
    for term in terms:
        if term in text_l:
            score += 8 if term in priority_terms else 2
            matched.append(term)
    fy = file_year(path)
    for year in years:
        if str(year) in text_l:
            score += 8
            matched.append(str(year))
        if fy is not None and abs(fy - year) <= 1:
            score += 1
    return score, unique(matched, 18)


def cmd_candidates(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    terms = unique(args.terms or [], 30)
    phrases = unique(args.phrases or [], 20)
    years: list[int] = []
    if args.question:
        q_terms, q_phrases, years = question_terms(args.question)
        terms = unique(list(terms) + q_terms, 30)
        phrases = unique(list(phrases) + q_phrases, 22)
    year_start = args.year_start
    year_end = args.year_end
    if years and year_start is None:
        year_start = max(1939, min(years) - 3)
    if years and year_end is None:
        year_end = max(years) + 2

    
    import time as _time

    deadline = _time.monotonic() + float(getattr(args, "time_budget_seconds", 20.0) or 20.0)
    truncated_at: str | None = None

    sweep_files = list(iter_files(root, year_start, year_end))
    if years:
        # Scan in proximity-to-question-year order so the time budget is
        # spent on the most likely files first. ~85% of answers live in the
        # bulletin of the data year or year+1; a chronological sweep of a
        # 1978-1983 window burns the whole budget on 1978-79 before ever
        # touching 1981.
        anchor = max(years)

        def _proximity(path: Path) -> tuple:
            fy = file_year(path)
            if fy is None:
                return (99, 0)
            d = fy - anchor
            # 0 and +1 first, then +2, then -1, then by distance.
            order = {0: 0, 1: 0, 2: 1, -1: 2}.get(d, 3 + abs(d))
            return (order, abs(d))

        sweep_files.sort(key=_proximity)
    elif year_start is None and year_end is None and len(sweep_files) > 60:
        # No year bounds: a sorted sweep + time budget would only ever see
        # the 1939-1945 files before the deadline. Interleave the corpus
        # (every Nth file round-robin) so partial results span all decades.
        stride = max(1, len(sweep_files) // 60)
        sweep_files = [
            sweep_files[offset + i * stride]
            for offset in range(stride)
            for i in range((len(sweep_files) - offset + stride - 1) // stride)
            if offset + i * stride < len(sweep_files)
        ]

    top_rows = []
    top_tables = []
    top_text = []
    for path in sweep_files:
        if _time.monotonic() > deadline:
            truncated_at = path.name
            break
        all_lines = lines(path)
        for line_idx, line in enumerate(all_lines):
            if not line.strip() or line.lstrip().startswith("|"):
                continue
            start_ctx = max(0, line_idx - args.context_lines)
            end_ctx = min(len(all_lines), line_idx + args.context_lines + 1)
            context_lines = [
                {"line": idx + 1, "text": all_lines[idx]}
                for idx in range(start_ctx, end_ctx)
                if all_lines[idx].strip()
            ]
            context_text = "\n".join(item["text"] for item in context_lines)
            score, matches = text_score(context_text, terms, phrases, years, path)
            if score <= 0:
                continue
            top_text.append(
                {
                    "score": score,
                    "matched": matches,
                    "file": path.name,
                    "line": line_idx + 1,
                    "context": context_lines,
                    "read_command": f"search --query {json.dumps(line.strip()[:80])} --file {path.name}",
                }
            )
        idx = 0
        while idx < len(all_lines):
            if not all_lines[idx].lstrip().startswith("|"):
                idx += 1
                continue
            start, end = table_bounds(all_lines, idx)
            title = title_context(all_lines, start)
            parsed = parse_table(all_lines, start, end)
            row_labels = [str(row.get("label", "")) for row in parsed["rows"][: args.sample_rows]]
            table_text = "\n".join(title + parsed["headers"] + row_labels + all_lines[start : end + 1])
            t_score, t_matches = table_score(table_text, terms, phrases, years, path)
            if t_score > 0:
                top_tables.append(
                    {
                        "score": t_score,
                        "matched": t_matches,
                        "file": path.name,
                        "table_start_line": start + 1,
                        "title_context": title,
                        "headers": parsed["headers"][: args.max_headers],
                        "sample_row_labels": row_labels,
                        "read_command": f"table --file {path.name} --line {start + 1}",
                    }
                )
            for row in parsed["rows"]:
                label = str(row.get("label", ""))
                row_text = json.dumps(row, ensure_ascii=False)
                score, matches = row_score(label, row_text, table_text, terms, phrases)
                if score <= 0:
                    continue
                top_rows.append(
                    {
                        "score": score,
                        "matched": matches,
                        "file": path.name,
                        "table_start_line": start + 1,
                        "title_context": title,
                        "headers": parsed["headers"][: args.max_headers],
                        "row": row,
                        "read_command": f"table --file {path.name} --line {start + 1} --row-filter {label}",
                    }
                )
            idx = end + 1

    top_rows.sort(key=lambda item: item["score"], reverse=True)
    top_tables.sort(key=lambda item: item["score"], reverse=True)
    top_text.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "question_terms": terms,
        "phrases": phrases,
        "year_start": year_start,
        "year_end": year_end,
        "top_text_hits": top_text[: args.max_text],
        "top_rows": top_rows[: args.max_rows],
        "top_tables": top_tables[: args.max_tables],
    }
    if truncated_at:
        payload["truncated"] = True
        payload["truncated_warning"] = (
            f"Time budget hit at {truncated_at}; files from there onward were "
            "NOT scanned. Results are partial. Re-call with year_start/year_end "
            "to narrow the window"
            + ("" if year_start else " — this query had NO year bounds, which forces a full 697-file scan")
            + "."
        )
    print(json.dumps(payload, indent=2))


def cmd_auctions(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    candidates = auction_windows(
        root=root,
        question=args.question,
        year_start=args.year_start,
        year_end=args.year_end,
        window_lines=args.window_lines,
    )
    print(json.dumps(candidates[: args.max_results], indent=2))


def cmd_quick(args: argparse.Namespace) -> None:
    q = args.question.lower()
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    year_start = max(1939, min(years) - 3) if years else None
    year_end = (max(years) + 2) if years else None
    row = None
    title = []
    if "national defense" in q:
        row = "national defense"
        title = ["expenditure"]
    elif "net interest" in q:
        row = "net interest"
        title = ["outlay"] if "outlay" in q else ["expenditure"]
    elif "individual income" in q:
        row = "individual income"
        title = ["receipt"]
    elif "treasury notes" in q or "tenders" in q or "bids" in q:
        row = None
        title = ["tender", "bids", "accepted", "offering", "maturing"]
    terms = [term for term in [row, "budget" if "budget" in q else "", "expenditure" if "expenditure" in q else "", "receipt" if "receipt" in q else ""] if term]
    phrases = [row] if row else []
    ns = argparse.Namespace(
        root=args.root,
        question=args.question,
        terms=terms + title,
        phrases=phrases,
        year_start=year_start,
        year_end=year_end,
        max_rows=args.max_results,
        max_tables=args.max_results,
        max_text=args.max_results,
        max_headers=24,
        sample_rows=10,
        context_lines=3,
    )
    cmd_candidates(ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact OfficeQA corpus retrieval")
    parser.add_argument("--root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rank")
    p.add_argument("--terms", nargs="+", required=True)
    p.add_argument("--year-start", type=int)
    p.add_argument("--year-end", type=int)
    p.add_argument("--max-files", type=int, default=12)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("search")
    p.add_argument("--query", required=True)
    p.add_argument("--year-start", type=int)
    p.add_argument("--year-end", type=int)
    p.add_argument("--regex", action="store_true")
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--max-results", type=int, default=20)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("rows")
    p.add_argument("--row", required=True)
    p.add_argument("--title", action="append", default=[])
    p.add_argument("--file")
    p.add_argument("--year-start", type=int)
    p.add_argument("--year-end", type=int)
    p.add_argument("--max-results", type=int, default=10)
    p.set_defaults(func=cmd_rows)

    p = sub.add_parser("table")
    p.add_argument("--file", required=True)
    p.add_argument("--line", type=int, required=True)
    p.add_argument("--row-filter")
    p.add_argument("--max-rows", type=int, default=80)
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("candidates")
    p.add_argument("--question")
    p.add_argument("--terms", nargs="+", default=[])
    p.add_argument("--phrases", nargs="+", default=[])
    p.add_argument("--year-start", type=int)
    p.add_argument("--year-end", type=int)
    p.add_argument("--max-rows", type=int, default=10)
    p.add_argument("--max-tables", type=int, default=10)
    p.add_argument("--max-text", type=int, default=10)
    p.add_argument("--max-headers", type=int, default=24)
    p.add_argument("--sample-rows", type=int, default=10)
    p.add_argument("--context-lines", type=int, default=3)
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("auctions")
    p.add_argument("--question", required=True)
    p.add_argument("--year-start", type=int)
    p.add_argument("--year-end", type=int)
    p.add_argument("--max-results", type=int, default=8)
    p.add_argument("--window-lines", type=int, default=16)
    p.set_defaults(func=cmd_auctions)

    p = sub.add_parser("quick")
    p.add_argument("--question", required=True)
    p.add_argument("--max-results", type=int, default=12)
    p.set_defaults(func=cmd_quick)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
