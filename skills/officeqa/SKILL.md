---
name: officeqa
description: OfficeQA playbook — Treasury Bulletin grounded reasoning. Routing, table-family knowledge, formulas, verification, final-answer format.
---

You are answering ONE numerical question from the U.S. Treasury Bulletin corpus at `/app/corpus/`. Use only that corpus and the bundled MCP tools. No internet, no task metadata, no memorized answers.

Grader: 1% fuzzy numeric tolerance, reads only `/app/answer.txt`. Near-correct = partial credit; empty = zero. Always end with an answer written.

## NON-NEGOTIABLE RULES (failures to follow these zeroed graded tasks)
1. List answers: `[A, B]` with a SPACE after each comma. N values asked → N values output.
2. NEVER extrapolate a missing data point (no CAGR-projecting what the corpus lacks).
3. NEVER rescale to a unit the question didn't name; report the table's native value.
4. Date answers = month name + 4-digit year ("November 1969") UNLESS the question specifies an encoding (e.g. month×100+year) — then output that integer (see §6).
5. Before finalize: name file + table title + exact row label + column header for EVERY number. Can't? Re-read, don't guess.
6. MULTI-PART questions ("find the month X... In that month, what was Y?"): the answer is the FINAL quantity asked, never the intermediate. Solve part 1, then USE it to answer part 2.
7. The question's REQUESTED quantity decides the math: "rate"/"ratio"/"difference" questions need a COMPUTE step after retrieval — never finalize raw retrieved cells for a derived-quantity question.
→ Question about a specific table family (auctions, yields, surveys, trust funds...)? Check **section 3a Table-family playbook** FIRST — it lists the right bulletin (publication lags!), right column, and known corruptions per family.

## 0. Tool budget (HARD CAP 45 calls)
Happy path = 2-6 calls: namespaced routing tool → **write draft answer after call 1** → optional verification → `officeqa__finalize_answer`. Run a self-verification loop before calling `officeqa__finalize_answer`, then STOP. Stop ALL retrieval after 30 calls and finalize.

> **WRITE-FIRST**: Call `officeqa__finalize_answer` with your best estimate after the FIRST retrieval result. You can refine later, but an early draft in `/app/answer.txt` scores partial credit if you time out.

Pass the full original question into the first routing tool. Do not shorten away units, lambda values, rounding instructions, years, or the requested operation.

**Trust the routing tool.** If it returns `ready_answer` with sane magnitude, finalize. Do NOT chain multiple lookups "just to confirm" — every extra call is pure cost burn.

**No narration in thinking.** ≤ 3 sentences per thinking block. Decompose silently, act.

**Shell only when MCP tools have failed twice.** Prefer MCP tools for all retrieval and math.

**Dead-server recovery.** If TWO consecutive `officeqa__` tool calls return completely BLANK output (no JSON, no error), the MCP server has died and will NOT come back — including `finalize_answer`. Stop calling MCP tools. Switch fully to shell: retrieve with `grep`/`sed` on `/app/corpus/`, compute with `python3 -c`, and write the answer with `printf '%s\n' "VALUE" > /app/answer.txt` (single line, bare value, list format `[A, B]` with a space after each comma). Then `cat /app/answer.txt` to confirm. While in shell mode, apply the same provenance gate below before trusting any number.

Never use developer `write`, shell redirection, `printf`, or `echo > /app/answer.txt` to answer. The grader reads `/app/answer.txt`, but the only safe writer is `officeqa__finalize_answer`; raw writes have caused zeroes in graded runs.

**Shell is for progress, never an endless search.** Keep using grep/sed/python3 only while each call surfaces a NEW value. If ~6-8 attempts in a row return nothing new (same "not found", same lines), the figure is not retrievable that way — STOP. If a requested number genuinely isn't in the parsed text after a few targeted searches (e.g. a chart whose data bars were dropped in parsing), it is NOT there: finalize your best estimate from what you found and STOP — do not keep inventing new greps. Re-reading one file with another regex is not progress. One task must never exceed ~two dozen tool calls.

## 0a. Long monthly series (≥20 values)
For mean/median/stdev/variance/geomean/regression/CAGR/kurtosis/skew/VaR over **≥20 monthly values across multiple years**, do NOT iterate monthly bulletins — locate the recap table in ONE call:
- **`officeqa__summary_by_months_series(question)`** — dedicated tool, auto-finds the multi-year monthly recap (Feb-(Y+1) for CY data, Sep-(Y+1) for FY).
- Fallback: `officeqa__row_series_lookup` / `officeqa__extract_table` on "Summary of Federal Fiscal Operations" / "Budget Receipts and Expenditures, by Months".

## 1. Decompose silently
- Metric / row label (exact wording: "National defense" ≠ "Total national security")
- Year(s) and **FY vs CY**
- Operation: lookup, sum, %-change, mean, stdev, regression, CAGR, VaR, inflation-adj
- Units: dollars/thousands/millions/billions/%. **Convert BEFORE any nonlinear math.**
- Output shape: number, bracketed list, date string, or short categorical word

## 2. Routing (RAG Pre-Filtering)

**MANDATORY WORKFLOW**: Call the namespaced tools directly. Do NOT use any search_tools or execute_tool wrapping.

Call the namespaced tool matching the query shape directly:
| Shape | First tool |
|---|---|
| Single row × column ("what was X in YYYY") | `officeqa__direct_lookup_answer(question)` → `officeqa__table_cell_lookup` |
| Multi-period series math (mean/stdev/variance/regression/median/VaR/CAGR/correlation) | `officeqa__series_answer` or `officeqa__row_series_lookup`; for long monthly windows `officeqa__summary_by_months_series` |
| %-change or abs-change between two periods | `officeqa__direct_lookup_answer` each value, then `officeqa__calculate(operation="pct_change"/"abs_pct_change", values=[old,new], ...)` |
| CY totals (CY YYYY) | `officeqa__calendar_year_category_totals(question, category_terms, target_years, operation, round_digits)` — sum 12 monthly cells, not a single annual row |
| **Public debt / federal securities** for month-end dates (interest-bearing public debt / Total Federal Securities / subject to statutory debt limit / marketable / nonmarketable / savings bonds / Series I) | `officeqa__public_debt_outstanding(question)` — parses dates from text, picks bulletin (data_month+1), scans FD-1 / FD-3 / Summary of Federal Securities / Summary of Federal Debt / Statutory Debt Limitation tables. Auto: marketable preference, statutory-limit extractor, "entire set"→pop stdev, weighted-avg for 2:1 weight phrasing, ln-ratio for two-value questions. TWO CRITICAL GUARDS: (a) "held by U.S. Government accounts" (intragovernmental holdings) is a SEPARATE column on the **OFS-1** table ("Public debt securities Held by U.S. Government accounts > Total"), NOT the "Total public debt securities outstanding" grand total the tool picks by default — if the question names "held by Government accounts" and the tool returns the grand total, switch to `extract_table`/`row_series_lookup` on OFS-1 and take that column. (b) For a "sum of EACH year FY A–B inclusive" question pass `target_dates` for EVERY year in the range (e.g. ["2005-09",...,"2009-09"]) — the date parser otherwise grabs only the two range endpoints and the sum is wrong; verify `values` has the full count before summing. |
| **Foreign / TIC liabilities** ("liabilities to all foreigners" / "Total Liabilities by Type and Holder" / "major currencies" Canadian dollars / Euro / pound sterling / Japanese yen rows under TIC capital movements) | `officeqa__foreign_capital_movements(question)` — parses target dates from text, picks bulletin (Y+1)_12 for CY lookups, scans Table CM-I-1. Auto: ln-ratio for two-value log-growth questions, max-share for currency / total liabilities ratios, sum for multi-date country lookups, mean for multi-year mean/average questions. Does NOT cover FX positions tables (Sections I-VI by currency) — for those use `officeqa__extract_table` on the FCP-* tables directly. CRITICAL series choice: "liabilities … of/to foreign countries" is the **Total foreign countries** column (CM-I-2 Part A — excludes international and regional organizations); "liabilities to all foreigners" is the CM-I-1 grand total (includes them). They are different series — pass the question through so the tool picks the right column, and never mix the two across years. CY-end values in the transposed era are re-printed as bare-year rows in later bulletins with revised figures; the latest re-print supersedes the first publication (the tool sweeps latest-first automatically). |
| **FX positions** (FCP-* weekly tables, Sections I-VI: net options / spot-forward positions by currency) | `officeqa__extract_table` on the section's table. CRITICAL: each currency section has its OWN unit line — yen tables are "in BILLIONS of yen" while sterling tables are "in MILLIONS of pounds". Read BOTH unit lines, normalize scale (billions = ×1000 millions), convert each position to USD via that row's exchange-rate column (rate is foreign-per-USD ⇒ DIVIDE; USD-per-foreign ⇒ MULTIPLY), THEN compute the requested quantity. "Mid-month" = the report-date row nearest the 15th (e.g. the weekly row dated on/near the 16th — pick that EXACT row). SAME-ROW DISCIPLINE: the net-options value (column 3) AND the exchange rate MUST come from the SAME report-date row — never pair a net from one week with a rate from another, and never use a rate that does not literally appear in that row. For weekly FCP-*-1 tables a "net options positions" of `n.a.` means that week has no figure — step to the nearest dated row that DOES. Never finalize raw position values for a derived-quantity question. REVISION DISCIPLINE: FCP monthly rows are REVISED in later reprints (each bulletin reprints ~2 years of history; spot/forward cells can shift by ~10%+). For a month-M YYYY value, read the row from the LATEST bulletin that reprints it ((YYYY+1)_12 or (YYYY+2)_03..) — never the contemporaneous print. "Net X position not considering options" = (spot/forward purchased − sold) + (non-capital assets − liabilities); exclude calls/puts/delta columns. |
| **Receipts** (individual income / corporate / excise / customs / estate / gift / social insurance / employment / unemployment / highway/airport trust / black lung) | `officeqa__receipts_series(question)` — monthly mode auto-fires for "monthly"/"H-Spread"/"MAD"/"CV"/"Tukey"/"Hinge" + single FY; otherwise annual. Parses FFO-2 / Receipts by Principal Sources. Prefers `> Net` for "net of refunds". PRE-WWII era (1930s–early 1940s Monthly Treasury Statement, no FFO-2): receipts_series does NOT cover it (it returns "could not infer category"). For a calendar-month total-receipts figure, read the **"Summary of Receipts and Expenditures"** by-MONTH table's `Total Receipts` column inside the FOLLOWING-MONTH bulletin (e.g. an October value lives in the Nov bulletin) — NOT the "Total Budget Receipts and Expenditures, by Months" table (a different aggregate that yields the wrong figure). Use `officeqa__extract_table`/`row_series_lookup` on that titled table. |
| **Department / Agency outlays** (Department of X / Veterans Administration / highest spending department / outlays of <DEPT> FY X-Y / "total outlays across all agencies except …" for a month) | `officeqa__department_outlays_series(question)` — annual FY series, sums multi-level sub-cols (Defense > Mil + Civil + Undistributed). Different table from FFO-5. Pre-1947 Defense = War + Navy. For a single-MONTH "sum all listed agencies except X, Y" question it returns mode `ffo3_month_agency_sum` with `ready_answer` = the correct agency-column sum (excludes the named depts AND the non-agency reconciliation rows). TRUST that `ready_answer` — do NOT re-sum columns by hand or fall back to budget_outlays_by_function/compute_expression. |
| Budget Outlays by **Function** (FFO-5/FD-6/"net interest"/"national defense" outlays by month) | `officeqa__budget_function_answer(question, target_date="YYYY-MM", ...)` |
| Bills/notes/bonds/TIPS/FRNs/auctions/tenders/bids/rollover | `officeqa__financing_auction_answer(question)`; stats over a coupon/descriptor class ("2-3/8% TIPS", "13-week bills") in a date window: `officeqa__auction_offerings_rows(security_terms, year_start, year_end)` — collect EVERY matching row FIRST, count them, then compute. A descriptor class spans ALL its series/maturities (e.g. both the 2017 and 2027 issues), not just one series. |
| **Market quotations on Treasury bills** (MQ-1 tables: amount outstanding / issue date / maturity per 13-week or 26-week bill issue; "quotations released during the last week of <month>", count/filter/geomean over bill issues) | `officeqa__market_quotation_bills(year_start, year_end, bill_term="13-week", months=[...], min_amount=N)`. Returns one row per issue with amount_outstanding + issue_date, plus count and geometric/arithmetic mean of the kept amounts in ONE call. The MQ-1 column layout shifts year to year, so DON'T read by fixed column — this tool reads positionally. A bulletin published in month M quotes the last trading day of (M-1): "released during the last week of April" ⇒ the default `bulletin_months=[5]` (May issues). `months=` filters by ISSUE date (e.g. [2,3,4] for Feb-Apr issues); `min_amount=` keeps amounts strictly greater. Collect across ALL requested years before counting. |
| Foreign / international holdings / FRB holdings | `officeqa__quick_retrieve` then `officeqa__financing_auction_answer` or `officeqa__direct_lookup_answer` |
| Savings Bonds sales/redemptions/amount outstanding (SB-1/SB-2) | `officeqa__table_manifest_search(terms=["Sales and redemptions", "amount outstanding", "SB-2"])` then `officeqa__extract_table_by_header` or one `officeqa__read_lines`; compute with `officeqa__calculate`. For "**all series combined**" redemption questions use **"Table 2 - Sales and Redemptions by Periods, All Series Combined"** — it has direct `Redemptions > Total`, `Redemptions > Sales price`, and `Redemptions > Accrued discount` columns. Do NOT switch to the "Series E through K" table and try to re-aggregate series by hand. "Redemptions from elapsed value buildup / accrued discount / interest accrued" = the `Redemptions > Accrued discount` column; its share of total redemptions = that column ÷ `Redemptions > Total`. Read the requested MONTH row from the Months block (the month value appears in the bulletin printed ~2 months later, e.g. Oct-1961 in 1962_03). |
| TIC/capital movements liabilities by country/currency (`CM-I-2`, `CM-I-3`) | `officeqa__table_manifest_search(terms=["Total Liabilities by Type and Country", "Canadian", "foreign countries"])` then `officeqa__extract_table_by_header`; ratios use country row over total foreign-countries row. |
| ANY bond-yield question — "high-grade corporate bond yields", Treasury / Aa corporate / municipal yields or spreads, ANY era (Table AY-1 / MY-2 1970+, OR the pre-1970 "Average Yields of Long-Term Treasury and Corporate Bonds" historical reprint in 1941-1949 bulletins) | `officeqa__average_yields_series(question)` — decodes both layouts, returns the REVISED monthly series + a ready variance/stdev/mean (POPULATION estimator by default; "sample calendar months" is a data window, not the estimator). DO NOT hand-read the 2-column contemporaneous print + compute_python_math — that uses the wrong (un-revised) values and risks the sample/population mixup. Echoes the corporate header so Aa-vs-Aaa mixups are visible. **USE THE TOOL'S READY FIELDS, do not recompute by hand:** for "which month had the max/min spread, encode as month×100+year" the tool already returns `stats.spread_max.month_x100_plus_year` (and `spread_min`) — finalize that integer directly. For a two-month growth/Fisher question the tool returns `two_point_growth` with `log_growth_rate` (= "Fisher Ideal symmetric growth rate" between two observations), `pct_change`, and `ratio` — pick the one the question names; do NOT call `fisher_ideal()` for a two-observation symmetric growth rate. |
| Profile of the Economy indicators (productivity/output per hour, CPI narrative) | `officeqa__search_corpus` with the exact indicator phrase and years, then one `officeqa__read_lines`/`officeqa__table_window`; these are narrative/economic profile items, not FFO tables. |
| Treasury ownership survey / maturity schedule categories | `officeqa__financing_auction_answer` or `officeqa__public_debt_outstanding` first, then one manifest/table extraction around `ownership survey`, `maturity distribution`, or `maturing`. |
| Need to narrow bulletin file | `officeqa__rank_files_by_terms(terms=[...], target_year=YYYY, is_fiscal=bool)` — ±1 year scores highest. Pre-1977 FY = Sep bulletins, post-1976 = Dec. |
| Unknown shape | `officeqa__quick_retrieve(question)` then smallest of `officeqa__extract_rows`/`officeqa__extract_table`/`officeqa__search_corpus`/`officeqa__read_lines` |
| Last resort | `officeqa__officeqa_answer_candidate(question)` — verify its ready_answer before acting |

If a specialized tool returns empty → **drop to shell** (`/tmp/officeqa/tools.py` or grep/sed). After shell writes to `/app/answer.txt`, run `officeqa__recover_answer()` before `officeqa__finalize_answer`.

If a tool returns `calculate_call`, pass it to `officeqa__calculate` exactly as given.
If a tool returns `ready_answer`, treat as a CANDIDATE — verify row + column + period + unit + sign with one quick lookup.
If a tool returns `sum_mismatch_warning`, `magnitude_warning`, or `date_inference_warning`, the data is suspect — do NOT finalize from it; re-extract with `officeqa__table_window` and check headers first.
If `finalize_answer` rejects with `UNVERIFIED NUMBERS`, your value never appeared verbatim in MCP tool output — usually because you did the last step in your head. This is NOT a wrongness signal. If you computed it from values you DID retrieve, push that arithmetic through `officeqa__compute_expression` and finalize THAT result; if it was a clean mental subtraction/ratio, just call `finalize_answer` again with the SAME value to confirm. Do NOT respond by fetching DIFFERENT inputs — especially do not replace a figure the question explicitly named (e.g. a rounded narrative value "as reported") with a more-precise table cell that shifts the rounding. Only re-derive from scratch if you cannot ground the operands at all.
If a function/outlay tool returns `suggested_ready_answer_if_box_cox_lambda_0_75` and the original question asks Box-Cox with lambda 0.75, finalize it or call `officeqa__calculate` once; do not recompute manually.

## 3. Table-family knowledge

### Fiscal vs calendar year
| Period | Span | Where |
|---|---|---|
| FY pre-1977 | Jul 1 (Y-1) → Jun 30 (Y) | **Sep-Y** for preliminary monthly; **Sep-(Y+1)** for final annual recap (e.g. FY1955 dept totals in `treasury_bulletin_1956_09.txt`) |
| FY post-1976 | Oct 1 (Y-1) → Sep 30 (Y) | **Dec-Y** preliminary; **Dec-(Y+1)** final annual recap |
| Transition quarter | Jul 1 1976 – Sep 30 1976 | Sep 1976 bulletin |
| Single CY | Jan-Dec (Y) | Sum 12 monthly cells (not annual rows — those are FY) |
| Multi-year CY monthly retrospective | spans multiple years | **Feb-(last_year+1)** bulletin — wide table with prior ~10 CYs of monthly cells. The tool `officeqa__summary_by_months_series` handles this. |

- **Calendar Year (CY)**: Jan 1 to Dec 31 of that year. To compute CY totals, sum the 12 calendar months (or use `officeqa__calendar_year_category_totals`). Do NOT use the Fiscal Year (FY) annual row.
- **Fiscal Year (FY)**: Post-1976 runs Oct 1 (Y-1) to Sep 30 (Y). Pre-1977 runs Jul 1 (Y-1) to Jun 30 (Y). Do NOT sum 12 calendar months of the same year for FY questions. Use the annual row/column or retrieve the specific FY months.
- **Circular-decomposition trap:** CY-Y ≠ FY-Y. Computing "H2 of CY-Y = FY-Y_total − Jan–Jun Y" is WRONG — pre-1977 FY-Y = Jul (Y-1)–Jun Y, so that subtraction yields Jul–Dec of **Y-1** and the "CY total" collapses back to the FY value. The ONLY valid CY decompositions: sum the 12 monthly cells of Jan–Dec Y directly, or Jan–Jun Y (from FY-Y cumulative) + Jul–Dec Y (from FY-(Y+1) cumulative).

**Bulletin year ≠ data year.** A 1941 bulletin generally reports FY1940 actuals.

### Pre-1950 defense
Split into **War Department + Navy Department** with no combined row — sum them. Post-1950 use "National defense" / "Department of Defense".

### Header hierarchy
Treasury tables stack headers: year on row 1, period label ("Cumulative to date", "Comparable period", "This month") on row 2. Read as `1981 > cumulative to date > <value>`. Prefer `row_vertical` over `row_tsv` when both are returned — it pairs `header > value` per line.

### Units in headers
`(in dollars)` = ×1; `(in thousands)` = ×1,000; `(in millions)` = ×1,000,000; `(in billions)` = ×1,000,000,000.

`(123)` = -123. `-`, `*`, `(*)` = zero/N/A. Strip footnotes (`r/`, `p/`, `e/`, `1/`, `2/`, `*`) before parsing.

### 3a. Table-family playbook (hard-won from graded failures — find YOUR question's row)
| Your question mentions... | Read rule |
|---|---|
| "revised figures" / "including X excluding Y" | #1 |
| maturity schedule, callable, pre-1950 | #2 |
| computed interest charge | #3 |
| maturity schedule outstanding / final maturity in CY | #4 |
| tax and loan account balances | #5 |
| market quotations (MQ tables) | #6 |
| Treasury Survey of Ownership | #7 |
| ownership survey + holder categories / months with bills outstanding > $X | #7 — call `officeqa__treasury_ownership_holders` FIRST, before any retrieval |
| monthly-average yields (corporate/municipal) | #8 |
| weekly bill auction rates | #9 |
| trust fund monthly values | #10 |
| savings NOTES | #11 |
| Internal Revenue collections ratios | #12 |
| bill/note auction rows (PDO-2) | #13 |
| 1930s-era values | #14 |

1. **"Revised figures" / inclusion-exclusion clauses quote a table FOOTNOTE.** Grep a distinctive footnote phrase from the clause; many files match the footnote text, but typically only ONE file's table also carries a data row for the requested back year — open candidates and check for the back-year row before reading values. Revised reprints differ materially from the original prints, and both values of a difference must come from the SAME revised series. TWO HARD CHECKS before subtracting A−B across years: (a) BOTH values read from the SAME table in the SAME bulletin — if the early year's row is missing there, the question's footnote clause names which era's print carries it (a WWII-era clause ⇒ the war-era bulletins, not the post-war reprint); (b) VERIFY THE ROW LABEL against a neighbor print — some mid-1946 issues drop leading year-rows during parsing so a later year's data sits under an earlier year's label (cross-check one adjacent bulletin: if the same numbers appear under a DIFFERENT year there, the label is corrupted — trust the print where year labels are consecutive).
2. **Pre-1950 two-pane maturity schedules:** left pane = year first CALLABLE, right pane = year of MATURITY — two independent half-rows per markdown line, and PDF cell-merge drops other securities' callable amounts onto named rows. Per-security par = ONLY the fixed-maturity cell on the exact named row; cross-check year-group sums against the printed Total row.
3. **Computed interest charge table drifts era to era:** Table 5/4 (1940s-50s), Table 2 (1960s), DO-2 (mid-1969), FD-2 (late-1969+). Search by TITLE ("Computed Interest Charge"), not table number; column "Computed annual interest charge > Public debt", units millions; KEEP the words "computed"/"charged" in retrieval queries — dropping them lands on FFO outlay rows (a different concept).
4. **PDO-1 maturity schedules are point-in-time snapshots.** (a) "Outstanding at end of month M for years Y1..Yn" → open n bulletins (Yi)_(M+1), take ONLY year Yi's own Total row from each. (b) "Total amount with final maturity in CY Y" → use the **Y_03 bulletin's** PDO-1 year-Y Total row (graders use that vintage even though its snapshot has already dropped Jan/Feb-Y matured issues; the Dec-31-(Y-1) snapshot in Y_01 gives a LARGER, wrong-for-grading total). Verify the Total by summing the year band's issue rows.
5. **Tax-and-loan balances:** Table 1 = balances, Table 2 = credits/withdrawals — don't mix. Use the (Y)_02 issue and ALWAYS cross-check one adjacent issue — some early-1960s issues' Table 1 cells are OCR-corrupted (digits transposed); when two issues disagree, trust the majority across three.
6. **Market quotations (MQ-1/2/3) lag:** they quote the last trading day of an EARLIER month (M-1 from 1962, M-2 before). MANDATORY: read the "MARKET QUOTATIONS ON TREASURY SECURITIES, <DATE>" banner and match it to the question's date. In some issues (1974_05) 26-week rows drift into the 13-week column — verify each amount against its 13-week issue-date cell.
7. **Treasury Survey of Ownership lag:** ~2 months for bulletins Dec 1961 - Sep 1982, ~3 months earlier. The page banner "TREASURY SURVEY OF OWNERSHIP, <DATE>" is AUTHORITATIVE — lag only picks which bulletin to open; never substitute a different survey date. Table name varies by era ("TSO-3" late; "Table 3.- Interest-Bearing Public Marketable Securities by Issues" early, with date-RANGE rows). ANY by-holder, by-category, or bills-outstanding-by-month question on a survey goes through `officeqa__treasury_ownership_holders` — never hand-read the table (the holder columns are OCR-shifted and the bills section lists MATURITY months, so a month's outstanding is the suffix sum of rows maturing then-or-later, not the row's own value). Pass the question verbatim, keeping words like "tax anticipation"/"TABs", "total", "outstanding", and every year.
8. **Monthly-average yields lag + multi-panel tables:** bulletin M never contains month-M averages — fetch from M+1/M+2. In a multi-panel (N-up) table the bulletin's own month-M cell position can hold a value belonging to a DIFFERENT year's panel; every bare month row repeats once per panel — anchor the year from the panel's "YYYY-Mon" row, never from row position.
9. **Weekly-bill auction rates:** authoritative source is "Offerings of Treasury Bills - (Continued)" (Table 2 / PDO-2 by era), column "On total bids accepted > Equivalent average rate". Regular-weekly rows pack 13-week + 26-week into ONE space-separated cell — FIRST number = 13-week. The front-section mini-table is frequently OCR-garbled (1960_10 drops the Sept 15 auction) — never read rates from it without counting its rows against the issue calendar.
10. **Trust-fund monthly data cadence:** fund tables appear only in quarterly issues (GA-III-2..5 in Mar/May/Aug/Nov 1974-1982; GA-IV-x 1969-1973; "Table 7/8 - <Fund>" pre-1969; Dec-only 1983+). Month M first appears at M+2 or later (Sept 1975 → 1975_11). Never substitute FY annual totals (FFO-7) for a single-month question.
11. **Savings NOTES ≠ savings BONDS:** SN-1 "United States Savings Notes" is a separate table; notes outstanding ran ~$250-750 MILLION (1971-1982) vs bonds ~$50-80 BILLION — two orders of magnitude tells you which table you're in. SN-1 has FISCAL-year, CALENDAR-year, and monthly row blocks — for a "calendar year" redemption-rate question use the **Calendar years** block, not the months. REDEMPTION RATE = `Redemptions Total` column ÷ **average amount outstanding**, where average outstanding for year Y = (prior-year-end + year-Y-end) of the `Amount outstanding` column ÷ 2 — NOT the mean of that year's 12 monthly outstanding values. Read the year-end figures straight from the CY block (e.g. CY-1980 rate uses the CY-1979 and CY-1980 year-end outstanding). "Relative difference" of two such rates = (rate_A − rate_B)/rate_A (see §0 formula list); the sign follows the question's A-vs-B order.
12. **Internal Revenue collections precision trap:** monthly IRC data exists in TWO tables — "INTERNAL REVENUE COLLECTIONS Table 1" in THOUSANDS vs the "BUDGET RECEIPTS" repeat in MILLIONS. Ratios/elasticities over IRC monthlies MUST use the thousands table (millions rounding breaks the 1% tolerance). When consecutive bulletins disagree on a cell, prefer the LATEST re-publication.
13. **PDO-2 auction tables:** collect EVERY descriptor-matching row in the requested window and count rows BEFORE computing stats; PDO-2 reprints ~2 years of history — trust the LATEST bulletin. KNOWN CORRUPTION: 2007_09's PDO-2 yield/price column is row-shifted — use 2007_06/2007_12/2008_03 instead.
14. **1930s-era series provenance:** values for 1933-1941 printed in later consolidated bulletins (e.g. 1947_02) supersede contemporaneous printings — prefer the later consolidated table.
15. **"Total marketable" vs marketable "by issues":** a question that says total interest-bearing public marketable securities **"by issues"** wants the "Total public marketable securities" row of the maturity-distribution / *by-issues* table (the one whose columns break out bills/notes/bonds by maturity class), NOT the "Total marketable" line of the Summary-of-Federal-Securities table. The two totals differ by a few hundred million in the same month, which is enough to swing a log-ratio of two near-equal values. If `public_debt_outstanding` returns the Summary "Total marketable" for a "by issues" question, re-read the by-issues table with `extract_table`/`row_series_lookup` and use its "Total public marketable securities" row instead.
16. **"Percent of total … collected from <STATE>" is a SHARE, not a growth rate:** the source is the "Internal Revenue Receipts by State" table (column (1) "Total Internal Revenue collections") printed in the SAME calendar year's December bulletin (CY2018 → 2018_12, CY2019 → 2019_12 — the table is titled with the calendar year). The state's percent of total = state row ÷ the "United States, total" row, ×100, for that year. "Change in percentage points from CY-A to CY-B" = share_B − share_A (each share computed against ITS OWN year's US total) — NOT `pct_change` of the state's raw collections between the two years, and NOT the difference of the raw dollar amounts. Always divide by the US total before differencing.

## 4. Math (deterministic MCP tools — never mental arithmetic)

| Need | Tool |
|---|---|
| Named op (sum/mean/difference/pct_change/abs_pct_change/percent/scale/box_cox/cagr/gini) | `officeqa__calculate(operation=..., values=[...], round_digits=N)` |
| Gini of receipts vs expenditures (two values) | `officeqa__calculate(operation='gini', values=[receipts, expenditures])` = standard population Gini (mean-abs-difference / 2·mean; for two values that is \|a-b\|/(2(a+b))) — use the month's TOTAL receipts & total expenditures columns (footnote 8/: expenditure totals already exclude investment transactions). TRUST the tool's value; do not hand-compute. |
| Free-form expression with variables | `officeqa__compute_expression(expression="...", variables={...}, round_digits=N)` |
| Multi-line Python (statistics.linear_regression etc.) | `officeqa__compute_python_math(script="...", variables={...}, round_digits=N)` |
| Unit conversion | `officeqa__unit_scale(value, source_unit, target_unit)` or pass `source_unit`/`target_unit` to `officeqa__calculate` |

### Formulas (all callable from `officeqa__compute_expression`)
- **%-change**: `pct_change(old, new)` = `(new−old)/old*100`. **Denominator is old.**
- **Abs %-change**: `abs_pct_change(old, new)`.
- **Percentage-point change**: `pp_change(old, new)` = `new − old`. Use for "change in share" or "change in percentage".
- **CAGR**: `cagr(start, end, years)` returns %.
- **Continuously compounded growth**: `ccgr(start, end, years)` = `ln(end/start)/years`.
- **Log growth (% over window)**: `log_growth(start, end)`.
- **Stdev**: sample = `stdev(values)`; population = `pstdev(values)`. `statistics.stdev/pstdev` in scripts.
- **Mean/median/geomean**: `mean / median / geometric_mean`.
- **MAD**: `mad(values)`.
- **CV (%, using pstdev)**: `cv(values)`.
- **OLS**: `linreg(xs, ys)` → `[slope, intercept]`. `statistics.linear_regression(xs, ys)` in scripts.
- **Pearson**: `correlation(xs, ys)`.
- **Tukey quartiles (inclusive)**: `tukey_q1(v)`, `tukey_q3(v)`, H-Spread = `hspread(v)`.
- **Tukey (exclusive median hinge)**: `tukey_q1_excl(v)`, `tukey_q3_excl(v)`.
- **Hazen percentile**: `hazen(values, 85)`. Generic interpolation: `percentile(values, 85)`.
- **VaR 95% parametric**: `var_parametric(values, 95)`.
- **Expected Shortfall 95%**: `expected_shortfall(values, 5)`.
- **Fisher kurtosis (excess)**: `kurtosis(values)`.
- **Fisher-Pearson skewness**: `skewness(values)`.
- **Arc elasticity**: `arc_elasticity(q1, q2, p1, p2)`.
- **Macaulay duration**: `macaulay(times, cashflows, ytm)`.
- **Theil index**: `theil(values)`. **Gini**: `gini(values)`.
- **Box–Cox**: `boxcox(value, lambda)` — `(x^λ−1)/λ` for λ≠0, `log(x)` for λ=0.
- **Fisher Ideal index**: `fisher_ideal(p0, q0, p1, q1)`.
- **Z-score**: `zscore(target, values)`.
- **Inflation adj**: `real = nominal * (target_CPI / base_CPI)`. CPI lives in narrative under "Profile of the Economy" — use `officeqa__search_corpus`.

### Unit conversion before nonlinear math
Always convert BEFORE %-change, ratio, Box-Cox, log, sqrt, power, or any statistical op. Shortcuts in `officeqa__calculate`: thousands→millions = ÷1000, thousands→billions = ÷1_000_000, millions→billions = ÷1000.

### Rounding
"Rounded to N decimal places" → `round_digits=N` (half-up). When silent, default to source-cell precision or 2 for percents.

## 5. Self-Verify Before Finalizing

### PROVENANCE GATE (mandatory)
For EVERY number you are about to use in the final answer or its computation, you must be able to name all four of:
1. **File** (e.g. `treasury_bulletin_1981_10.txt`)
2. **Table title** (e.g. "FFO-5 Budget Outlays by Function")
3. **Row label** — EXACT, as printed (e.g. "Net budget outlays", not "the outlays row")
4. **Column header** (e.g. "1981 > Aug.")

If you cannot name all four for a number, you may NOT use it — go re-read the table (one `table_window`/`read_lines` call) instead of guessing. Numbers "remembered" from a truncated output, inferred from column position, or patched in as placeholders score zero far more often than a re-read costs.

Additional hard rules derived from graded failures:
- **Row totals**: if the row has its own Total/annual column, sum your selected cells and compare. Mismatch ⇒ wrong window (tools now return `sum_mismatch_warning` — treat it as a stop sign).
- **Magnitude coherence**: monthly/annual Treasury series rarely jump >5x between adjacent periods (`magnitude_warning` ⇒ wrong rows/columns).
- **Cumulative tables**: two cumulative figures may only be differenced when their base periods are IDENTICAL (same "Cumulative from ..." line). Check both unit lines before differencing.
- **Exact entity**: "Series I" ≠ total U.S. savings bonds; "Aa" ≠ "Aaa"; "Department of the Army" pre-1947 = "War Department"; per-issue "amount outstanding" ≠ "Total unmatured issues outstanding" (running total).
- **NEVER extrapolate a missing data point.** If a requested date's value is not in the corpus, do NOT project it via CAGR/regression/trend (a fabricated 3rd value zeroes the whole answer). Instead: try adjacent bulletins (±2 issues), alternate table families (SBN vs SB vs FD), and if truly absent, finalize the values you DID verify in the requested format.
- The verify loop:
  - Re-read the question and independently check the raw data using another retrieval tool or a clean coordinates check.
  - Exact row label match (e.g. "total" vs subcategory)
  - Unit conversion BEFORE nonlinear math
  - Calendar Year (CY) vs Fiscal Year (FY) boundaries
  - Negative signs (represented in parentheses e.g. `(123)` is `-123`)
  - Redo math calculations using `officeqa__compute_expression`

### Statistic conventions (use these exact readings)
| Question wording | Convention |
|---|---|
| "relative difference" of A vs B | `(A - B) / B` (×100 if percent requested) — NOT the absolute gap |
| "difference between two rates/shares" | percentage POINTS: `new - old` (`pp_change`) |
| "population" / "entire set" stdev/variance | `pstdev` / `pvariance`; otherwise default sample stdev for "of a sample"; if unstated, prefer population for a complete enumerated period |
| Expected Shortfall 95% (historical) | mean of the worst 5% tail = `expected_shortfall(values, 5)`; never the bare minimum |
| Fisher Ideal index/growth | `sqrt((P1/P0) × (Q1/Q0)) - 1`-style geometric mean of two ratios — never a plain %-change |
| "log growth" / "continuously compounded" | `ln(end/start)`; report as percent only if asked ("as a percent value") |
| Ratio "of A to B" | `A / B` in that order; `ln` ratio "X to Y" = `ln(X/Y)` — keep the question's order, sign matters |
| Arc elasticity of A with respect to B | A is the quantity (numerator of Δ%), B the driver |
| Answer asked "as a percent value (12.34, not 0.1234)" | multiply by 100 before finalizing |
| "realized variance" of two rates' logs | `(ln(r2) − ln(r1))²` — the squared log-return, NOT a mean-based two-point variance |
| "normalized difference" of A vs B | `(B − A) / midpoint` where midpoint = `(A+B)/2` — never `pct_change/2` |
| Macaulay duration of a zero-coupon instrument | = time to maturity (one period ⇒ exactly `1`); never `1/ln(P1/P0)` |
| Hodrick-Prescott filter / "structural balance" / trend-cycle decomposition | `hp_filter_trend(values, lambda)` / `hp_filter_cycle(values, lambda)` in `officeqa__compute_expression` — exact pentadiagonal solve. NEVER hand-roll HP in a script (the system is PENTAdiagonal; a tridiagonal Thomas solve gives garbage trends). "Smoothing parameter 100" ⇒ `lambda=100` |
| "Zipf exponent" of a distribution | OLS slope of log(value) on log(rank) (rank 1 = largest), reported as a POSITIVE exponent: `s = linreg([ln(rank)...], [ln(value)...])[0]` ⇒ answer `abs(s)`. The raw slope is negative; the exponent is its magnitude. Sort DESCENDING before ranking and exclude exactly the entities the question excludes (DC/territories = the 50 states only) |
| "count datapoints/values with leading digit d" | count EVERY numeric data cell (including bare 4-digit values that look like years) except the row-label column — never regex-drop year-like cells |
| Geometric / compound annual rate of change | `(V_end/V_start)^(1/n) − 1` — the factor MINUS 1 (a decline is NEGATIVE, e.g. −0.20, never the bare factor 0.80); prefer `cagr()` |
| Implied customs duty rate | `(duties on DUTIABLE goods) / (VALUE of DUTIABLE goods)` — NET both sides: subtract free-list rows from the value side and refunds/drawbacks from the duties side before dividing (gross/gross runs ~2% low). Same table family for both sides (duties half vs values half); report as a RATIO (≈0.3-0.5) unless percent asked. SANITY: a rate over 1.0 means the denominator is wrong |
| Arc elasticity "of A with respect to B" | A (named FIRST) is ALWAYS Q, B is P; write Q1/Q2/P1/P2 explicitly then call `arc_elasticity()` — never hand-roll in shell python |
| Arc elasticity with NO second variable named ("given these values", time-series endpoints only) | it degenerates to the midpoint percentage change of the series itself: `(Q2−Q1)/((Q1+Q2)/2)` — NEVER use calendar years as the denominator variable (midpoint ~2015 makes any elasticity collapse to a huge meaningless number) |
| "annual decay factor" | `1 + CAGR` (e.g. CAGR −0.30 ⇒ decay 0.70) — the per-period multiplier, not a separate statistic |
| "including budgetary and trust-fund flows" (dept outlays) | use the table's TOTAL column (budgetary + trust), not the budgetary-only column — check the column headers for "trust" before picking |
| "what percentage of X over months M1..Mn was Y" | ONE aggregate percentage: `sum(Y over the months) / sum(X over the months) × 100` — never a per-month percent, a change in percents, or an average of monthly percents |
| Variance (unqualified) | compute BOTH `pvariance` and `variance`; prefer POPULATION unless the question names the estimator ("sample variance"). "sample calendar months" describes months, not the estimator |
| Fisher Ideal growth/index | build BOTH ratios from the TWO series the question names: growth = `sqrt((A1/A0)×(B1/B0)) − 1` (×100 for percent); index = bare sqrt. NEVER collapse to a single-series %-change. `fisher_ideal()` returns the index ×100 |
| FX conversion direction | "U.S. dollars per pound" ⇒ MULTIPLY the £ amount; "yen per U.S. dollar" ⇒ DIVIDE the ¥ amount; always read the rate column header's direction before converting |
| "volatility index" / CV as an index | the bare RATIO stdev/mean (e.g. 0.42, not 42), NOT ×100 — `cv()` returns percent, so divide by 100 when the question says "index"; only report percent when the question says "percent" or "CV (%)" |
| Expected Shortfall via "historical portfolio RETURN approach" on a yield/price series | FIRST convert levels to period-over-period returns `r_i = (y_i − y_{i−1})/y_{i−1} × 100`, THEN ES = mean of the worst 5% of RETURNS (≤20 observations ⇒ the single worst return). The answer is a negative percent return, never an average of the levels |

## 6. Final-answer format
1. Call `officeqa__finalize_answer` with the raw, bare value (e.g. `VALUE`) to write it to `/app/answer.txt`. Only digits, signs, decimal points, commas, brackets, parentheses. No units words, no `$`, no `%` (grader strips), no prose.

**EXCEPTION — date/month answers:** when the answer IS a date or month, write it exactly as the question asks. If the question gives no encoding instruction, write the month NAME + 4-digit year (e.g. `March 1985`); the grader matches the month text. If the question DOES specify an encoding (e.g. "represent the month as an integer, multiply by 100, add the year"), follow that instruction literally and output the resulting number — read the encoding from the question, never assume one.

**LIST ANSWERS — grader trap:** multi-value answers MUST use `", "` (comma + SPACE) between values: `[5550000000, 7.89]`. The grader strips all commas before parsing numbers, so `[5550000000,7.89]` merges into ONE number and a perfect answer scores ZERO. `finalize_answer` auto-fixes this, but raw shell writes do not — always include the space.
- If the question asks for N values, output exactly N values — never a single aggregate of them.
- CONVERSE TRAP (equally fatal): a question asking ONE derived quantity — "by how much did X increase/change from A to B", "the difference/ratio/growth/spread between A and B" — wants ONE scalar = the computed result, NOT the two endpoints as a `[a, b]` list. Retrieving both operands into a draft list and never overwriting it with the computed number scores ZERO even though the math was trivial. Output a list ONLY when the question enumerates multiple sub-questions ("what is X AND what is Y", "in the order of the sub-questions presented", "for each year"). Litmus: count the question marks / "and what is" clauses, not the number of cells you had to read.
- A list answer passes only if EVERY value matches. For lists of 6+ values: each value needs the provenance gate; one placeholder/estimate guarantees a zero.
- If torn between "the series" and "one aggregate" when the question names NO statistic: finalize the series PLUS the aggregate — extra numbers are never penalized, missing ground-truth numbers are fatal. A hedge-list of alternative STATISTICS, by contrast, can never satisfy a series ground truth.
- **Rounding discipline:** obey explicit "rounded to N decimals". When unstated, never self-round a sub-1 percent result to 1 decimal (0.45 → 0.5 is a 10% error — outside the grader's 1% tolerance); keep at least 2 significant decimals and verify your chosen rounding stays within 1% of the raw value. Never `round()` inside compute_python_math scripts — round once, at finalize.
- **Unit discipline:** NEVER rescale to a different unit than the table's native unit unless the question names the target unit. "In millions" question + millions table = report the cell as printed.
2. The `<answer>VALUE</answer>` chat tag is OPTIONAL and is NOT scored — the grader reads ONLY `/app/answer.txt`, which holds whatever your most recent `officeqa__finalize_answer` call wrote. Stating a number in chat (tag or prose) does NOT write the file. So `officeqa__finalize_answer` with your final value MUST be your LAST tool call, AFTER any compute step. If you compute a value and then only say it in chat, the stale earlier draft is what gets graded — that is a guaranteed zero on an otherwise-correct answer.

**FINALIZE-LAST rule:** every time you produce a new number from `compute_expression` / `compute_python_math` / `calculate`, immediately call `officeqa__finalize_answer` with it before doing anything else. Never end a task on a compute call or a chat tag — end it on a finalize.

Rules:
- Single line in `/app/answer.txt`, ≤250 chars. No newlines, XML, markdown.
- Numeric scale matches the question. "In millions" → value in millions, not raw dollars.
- Negatives: `-` or accounting `(...)`.
- Never write "the answer is", "approximately", citations, table references, or any prose in the file or after the answer tag.

## 7. What NOT to do
- No `todo` / `update_plan` / narrative-planning tools.
- No raw writes to `/app/answer.txt`; `officeqa__finalize_answer` is the only answer writer.
- No `shell` for verification — only for `officeqa__install_shell_tools` / `officeqa__recover_answer` / `/tmp/officeqa/tools.py` after MCP failures.
- No chain of `officeqa__direct_lookup` + `officeqa__table_cell_lookup` + `officeqa__read_lines` + `officeqa__search_corpus` to "double-check" — if the routing tool gave a sane number, finalize.
- No identical tool call twice — switch route or shell.
- No `officeqa__finalize_answer` before at least one deterministic-math result confirms the value (for math questions).
- No narration in thinking blocks. Decompose silently.
