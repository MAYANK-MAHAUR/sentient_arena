from __future__ import annotations

import ast
import builtins
import json
import math
import subprocess
import statistics
import sys
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP


AVAILABLE_FUNCTIONS = [
    "sum",
    "mean",
    "median",
    "geometric_mean",
    "geomean",
    "stdev",
    "pstdev",
    "variance",
    "pvariance",
    "percentile",
    "percent",
    "pct_change",
    "abs_pct_change",
    "abs",
    "pp_change",
    "cagr",
    "boxcox",
    "correlation",
    "linreg",
    "theil",
    "expected_shortfall",
    "expected_shortfall_upper",
    "zscore",
    "mad",
    "kurtosis",
    "skewness",
    "hspread",
    "tukey_q1",
    "tukey_q3",
    "tukey_q1_excl",
    "tukey_q3_excl",
    "hazen",
    "cv",
    "gini",
    "ccgr",
    "log_growth",
    "macaulay",
    "arc_elasticity",
    "var_parametric",
    "fisher_ideal",
    "hp_filter_cycle",
    "hp_filter_trend",
]


def format_numeric_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def evaluate_expression(expression: str, variables: dict[str, object] | None = None) -> float | list[float]:
    variables = variables if variables is not None else {}

    def eval_list(node: ast.AST) -> list[float]:
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(elt) for elt in node.elts]
        value = evaluate(node)
        return value if isinstance(value, list) else [value]

    def eval_scalar(node: ast.AST) -> float:
        value = evaluate(node)
        if isinstance(value, list):
            raise ValueError("expected a scalar value")
        return float(value)

    def collect_args(args: list[ast.AST]) -> list[float]:
        values = []
        for arg in args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                values.extend(evaluate(elt) for elt in arg.elts)
            else:
                value = evaluate(arg)
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
        return values

    def evaluate(node: ast.AST) -> float | list[float]:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable: {node.id}")
            value = variables[node.id]
            return value if isinstance(value, list) else float(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(left, list) or isinstance(right, list):
                raise ValueError("lists cannot be used directly in arithmetic operators")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "abs":
                return float(builtins.abs(*[evaluate(arg) for arg in node.args]))
            if fn == "round":
                args = [evaluate(arg) for arg in node.args]
                return float(builtins.round(args[0], int(args[1])) if len(args) == 2 else builtins.round(args[0]))
            if fn in {"min", "max"}:
                values = collect_args(node.args)
                return float(min(values) if fn == "min" else max(values))
            if fn == "sum":
                return float(math.fsum(collect_args(node.args)))
            if fn == "prod":
                return float(math.prod(collect_args(node.args)))
            if fn == "mean":
                values = collect_args(node.args)
                if not values:
                    raise ValueError("mean requires at least one value")
                return float(math.fsum(values) / len(values))
            if fn == "median":
                values = collect_args(node.args)
                if not values:
                    raise ValueError("median requires at least one value")
                return float(statistics.median(values))
            if fn in {"geometric_mean", "geomean"}:
                values = collect_args(node.args)
                if not values or any(value <= 0 for value in values):
                    raise ValueError("geometric_mean requires positive values")
                return float(math.exp(math.fsum(math.log(value) for value in values) / len(values)))
            if fn == "stdev":
                values = collect_args(node.args)
                if len(values) < 2:
                    raise ValueError("stdev requires at least two values")
                return float(statistics.stdev(values))
            if fn == "pstdev":
                values = collect_args(node.args)
                if len(values) < 1:
                    raise ValueError("pstdev requires at least one value")
                return float(statistics.pstdev(values))
            if fn == "variance":
                values = collect_args(node.args)
                if len(values) < 2:
                    raise ValueError("variance requires at least two values")
                return float(statistics.variance(values))
            if fn == "pvariance":
                values = collect_args(node.args)
                if len(values) < 1:
                    raise ValueError("pvariance requires at least one value")
                return float(statistics.pvariance(values))
            if fn == "percentile" and len(node.args) == 2:
                values = sorted(eval_list(node.args[0]))
                percentile = eval_scalar(node.args[1])
                if not values:
                    raise ValueError("percentile requires at least one value")
                if not 0 <= percentile <= 100:
                    raise ValueError("percentile must be between 0 and 100")
                if len(values) == 1:
                    return float(values[0])
                rank = (len(values) - 1) * percentile / 100
                lower = math.floor(rank)
                upper = math.ceil(rank)
                if lower == upper:
                    return float(values[lower])
                weight = rank - lower
                return float(values[lower] * (1 - weight) + values[upper] * weight)
            if fn == "sqrt" and len(node.args) == 1:
                return math.sqrt(eval_scalar(node.args[0]))
            if fn in {"log", "ln"}:
                args = [eval_scalar(arg) for arg in node.args]
                return math.log(*args)
            if fn == "exp" and len(node.args) == 1:
                return math.exp(eval_scalar(node.args[0]))
            if fn == "pow" and len(node.args) == 2:
                return eval_scalar(node.args[0]) ** eval_scalar(node.args[1])
            if fn in {"percent", "percent_of"} and len(node.args) == 2:
                part, whole = [eval_scalar(arg) for arg in node.args]
                if whole == 0:
                    raise ValueError("division by zero")
                return float(part / whole * 100)
            if fn == "pct_change" and len(node.args) == 2:
                old, new = [eval_scalar(arg) for arg in node.args]
                if old == 0:
                    raise ValueError("division by zero")
                return float((new - old) / old * 100)
            if fn == "abs_pct_change" and len(node.args) == 2:
                old, new = [eval_scalar(arg) for arg in node.args]
                if old == 0:
                    raise ValueError("division by zero")
                return float(abs(new - old) / abs(old) * 100)
            if fn == "pp_change" and len(node.args) == 2:
                old, new = [eval_scalar(arg) for arg in node.args]
                return float(new - old)
            if fn == "cagr" and len(node.args) == 3:
                start, end, periods = [eval_scalar(arg) for arg in node.args]
                if start <= 0 or end <= 0:
                    raise ValueError("cagr requires positive start and end values")
                if periods == 0:
                    raise ValueError("cagr periods must not be zero")
                return float(((end / start) ** (1 / periods) - 1) * 100)
            if fn == "boxcox" and len(node.args) == 2:
                value, lam = [eval_scalar(arg) for arg in node.args]
                if value <= 0:
                    raise ValueError("boxcox requires a positive value")
                if lam == 0:
                    return float(math.log(value))
                return float((value**lam - 1) / lam)
            if fn == "correlation" and len(node.args) == 2:
                x_values = eval_list(node.args[0])
                y_values = eval_list(node.args[1])
                if len(x_values) != len(y_values) or len(x_values) < 2:
                    raise ValueError("correlation requires equal-length x and y lists of 2+ values")
                x_mean = math.fsum(x_values) / len(x_values)
                y_mean = math.fsum(y_values) / len(y_values)
                numerator = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                x_den = math.fsum((x - x_mean) ** 2 for x in x_values)
                y_den = math.fsum((y - y_mean) ** 2 for y in y_values)
                if x_den == 0 or y_den == 0:
                    raise ValueError("correlation inputs must vary")
                return float(numerator / math.sqrt(x_den * y_den))
            if fn == "theil":
                values = collect_args(node.args)
                if not values or any(value <= 0 for value in values):
                    raise ValueError("theil requires positive values")
                mean_value = math.fsum(values) / len(values)
                return float(math.fsum((value / mean_value) * math.log(value / mean_value) for value in values) / len(values))
            if fn == "expected_shortfall":
                values = sorted(eval_list(node.args[0]))
                pct = eval_scalar(node.args[1]) if len(node.args) >= 2 else 5.0
                if not values:
                    raise ValueError("expected_shortfall requires at least one value")
                if not 0 < pct <= 100:
                    raise ValueError("expected_shortfall percentile must be in (0, 100]")
                count = max(1, math.ceil(len(values) * pct / 100))
                return float(math.fsum(values[:count]) / count)
            if fn == "expected_shortfall_upper":
                # Upper-tail ES: mean of the TOP p% (for yields/rates, where
                # "shortfall" can mean the adverse HIGH tail). Additive — the
                # lower-tail expected_shortfall above is untouched.
                values = sorted(eval_list(node.args[0]), reverse=True)
                pct = eval_scalar(node.args[1]) if len(node.args) >= 2 else 5.0
                if not values:
                    raise ValueError("expected_shortfall_upper requires at least one value")
                if not 0 < pct <= 100:
                    raise ValueError("expected_shortfall_upper percentile must be in (0, 100]")
                count = max(1, math.ceil(len(values) * pct / 100))
                return float(math.fsum(values[:count]) / count)
            if fn == "zscore" and len(node.args) == 2:
                target = eval_scalar(node.args[0])
                values = eval_list(node.args[1])
                if len(values) < 2:
                    raise ValueError("zscore requires at least two comparison values")
                sd = statistics.stdev(values)
                if sd == 0:
                    raise ValueError("zscore comparison values must vary")
                return float((target - statistics.mean(values)) / sd)
            if fn in ("hp_filter_cycle", "hp_filter_trend") and len(node.args) in (1, 2):
                # Hodrick-Prescott filter via the exact (I + lambda*D'D)
                # pentadiagonal solve (Gaussian elimination — no numpy).
                # hp_filter_*(values) or hp_filter_*(values, lambda); default
                # lambda 100 (annual data). Returns the trend or cycle list.
                values = eval_list(node.args[0])
                lam = eval_scalar(node.args[1]) if len(node.args) == 2 else 100.0
                n = len(values)
                if n < 4:
                    raise ValueError("hp_filter requires at least 4 observations")
                # Build A = I + lam * D'D where D is the (n-2)xn 2nd-difference matrix.
                A = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    A[i][i] = 1.0
                for r in range(n - 2):
                    coef = (1.0, -2.0, 1.0)
                    for a in range(3):
                        for b in range(3):
                            A[r + a][r + b] += lam * coef[a] * coef[b]
                # Solve A * trend = values (partial-pivot Gaussian elimination).
                M = [row[:] + [v] for row, v in zip(A, values)]
                for col in range(n):
                    piv = max(range(col, n), key=lambda r: builtins.abs(M[r][col]))
                    M[col], M[piv] = M[piv], M[col]
                    if builtins.abs(M[col][col]) < 1e-12:
                        raise ValueError("hp_filter system is singular")
                    for r in range(col + 1, n):
                        f = M[r][col] / M[col][col]
                        if f:
                            for c in range(col, n + 1):
                                M[r][c] -= f * M[col][c]
                trend = [0.0] * n
                for r in range(n - 1, -1, -1):
                    s = M[r][n] - math.fsum(M[r][c] * trend[c] for c in range(r + 1, n))
                    trend[r] = s / M[r][r]
                if fn == "hp_filter_trend":
                    return [float(t) for t in trend]
                return [float(v - t) for v, t in zip(values, trend)]
            if fn == "linreg" and len(node.args) == 2:
                x_values = eval_list(node.args[0])
                y_values = eval_list(node.args[1])
                if len(x_values) != len(y_values) or len(x_values) < 2:
                    raise ValueError("linreg requires equal-length x and y lists of 2+ values")
                x_mean = math.fsum(x_values) / len(x_values)
                y_mean = math.fsum(y_values) / len(y_values)
                numerator = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                denominator = math.fsum((x - x_mean) ** 2 for x in x_values)
                if denominator == 0:
                    raise ValueError("linreg x values must not all be identical")
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                return [float(slope), float(intercept)]
            if fn == "mad":
                values = collect_args(node.args)
                if not values:
                    raise ValueError("mad requires at least one value")
                m = math.fsum(values) / len(values)
                return float(math.fsum(builtins.abs(v - m) for v in values) / len(values))
            if fn == "kurtosis":
                values = collect_args(node.args)
                if len(values) < 4:
                    raise ValueError("kurtosis requires at least four values")
                n = len(values)
                m = math.fsum(values) / n
                m2 = math.fsum((v - m) ** 2 for v in values) / n
                m4 = math.fsum((v - m) ** 4 for v in values) / n
                if m2 == 0:
                    raise ValueError("kurtosis requires varying inputs")
                return float(m4 / (m2 ** 2) - 3)
            if fn == "skewness":
                values = collect_args(node.args)
                if len(values) < 3:
                    raise ValueError("skewness requires at least three values")
                n = len(values)
                m = math.fsum(values) / n
                sd = statistics.stdev(values)
                if sd == 0:
                    raise ValueError("skewness requires varying inputs")
                return float(
                    n / ((n - 1) * (n - 2))
                    * math.fsum(((v - m) / sd) ** 3 for v in values)
                )
            if fn == "hspread":
                values = sorted(eval_list(node.args[0]))
                if len(values) < 2:
                    raise ValueError("hspread requires at least two values")
                qts = statistics.quantiles(values, n=4, method="inclusive")
                return float(qts[2] - qts[0])
            if fn in {"tukey_q1", "tukey_q3"} and len(node.args) == 1:
                values = sorted(eval_list(node.args[0]))
                if len(values) < 2:
                    raise ValueError(f"{fn} requires at least two values")
                qts = statistics.quantiles(values, n=4, method="inclusive")
                return float(qts[0] if fn == "tukey_q1" else qts[2])
            if fn in {"tukey_q1_excl", "tukey_q3_excl"} and len(node.args) == 1:
                values = sorted(eval_list(node.args[0]))
                if len(values) < 2:
                    raise ValueError(f"{fn} requires at least two values")
                qts = statistics.quantiles(values, n=4, method="exclusive")
                return float(qts[0] if fn == "tukey_q1_excl" else qts[2])
            if fn == "hazen" and len(node.args) == 2:
                # Hazen plotting position: p = (k - 0.5) / n
                # For target percentile P, the position k satisfies P = (k-0.5)/n*100
                # so k = P/100 * n + 0.5; sort ascending and interpolate.
                values = sorted(eval_list(node.args[0]))
                pct = eval_scalar(node.args[1])
                if not values:
                    raise ValueError("hazen requires at least one value")
                if not 0 < pct < 100:
                    raise ValueError("hazen percentile must be in (0, 100)")
                n = len(values)
                k = pct / 100 * n + 0.5
                if k <= 1:
                    return float(values[0])
                if k >= n:
                    return float(values[-1])
                lower = int(math.floor(k))
                upper = int(math.ceil(k))
                if lower == upper:
                    return float(values[lower - 1])
                weight = k - lower
                return float(values[lower - 1] * (1 - weight) + values[upper - 1] * weight)
            if fn == "cv":
                # Coefficient of variation as percentage; uses pstdev by default
                values = collect_args(node.args)
                if len(values) < 2:
                    raise ValueError("cv requires at least two values")
                m = math.fsum(values) / len(values)
                if m == 0:
                    raise ValueError("cv mean must not be zero")
                sd = statistics.pstdev(values)
                return float(sd / m * 100)
            if fn == "gini":
                values = sorted(eval_list(node.args[0]))
                if not values or any(v < 0 for v in values):
                    raise ValueError("gini requires non-negative values")
                n = len(values)
                total = math.fsum(values)
                if total == 0:
                    return 0.0
                # G = sum((2*i - n - 1) * x_i) / (n * sum(x))   (i 1-indexed, sorted asc)
                numerator = math.fsum((2 * (i + 1) - n - 1) * v for i, v in enumerate(values))
                return float(numerator / (n * total))
            if fn == "ccgr" and len(node.args) == 3:
                # Continuously compounded growth rate: ln(end/start) / years
                start, end, years = [eval_scalar(arg) for arg in node.args]
                if start <= 0 or end <= 0:
                    raise ValueError("ccgr requires positive start and end")
                if years == 0:
                    raise ValueError("ccgr years must not be zero")
                return float(math.log(end / start) / years)
            if fn == "log_growth" and len(node.args) == 2:
                # Logarithmic growth rate (in % over the full window).
                start, end = [eval_scalar(arg) for arg in node.args]
                if start <= 0 or end <= 0:
                    raise ValueError("log_growth requires positive start and end")
                return float(math.log(end / start) * 100)
            if fn == "macaulay" and len(node.args) == 3:
                # Macaulay duration: sum(t * CF_t / (1+y)^t) / sum(CF_t / (1+y)^t)
                times = eval_list(node.args[0])
                cashflows = eval_list(node.args[1])
                ytm = eval_scalar(node.args[2])
                if len(times) != len(cashflows) or not times:
                    raise ValueError("macaulay requires equal-length times and cashflows")
                pv_total = math.fsum(cf / (1 + ytm) ** t for t, cf in zip(times, cashflows))
                weighted = math.fsum(t * cf / (1 + ytm) ** t for t, cf in zip(times, cashflows))
                if pv_total == 0:
                    raise ValueError("macaulay PV total is zero")
                return float(weighted / pv_total)
            if fn == "arc_elasticity" and len(node.args) == 4:
                # arc_elasticity(q1, q2, p1, p2)
                q1, q2, p1, p2 = [eval_scalar(arg) for arg in node.args]
                if q1 + q2 == 0 or p1 + p2 == 0:
                    raise ValueError("arc_elasticity midpoints must be non-zero")
                num = (q2 - q1) / ((q2 + q1) / 2)
                den = (p2 - p1) / ((p2 + p1) / 2)
                if den == 0:
                    raise ValueError("arc_elasticity price change is zero")
                return float(num / den)
            if fn == "var_parametric":
                # Parametric VaR at alpha (default 95): mean - z * stdev.
                # Returns as negative-tail value (left tail).
                values = collect_args(node.args[:1]) if isinstance(node.args[0], (ast.List, ast.Tuple)) else collect_args(node.args)
                # If 2 args, second is alpha %.
                alpha = 95.0
                if len(node.args) >= 2 and not isinstance(node.args[1], (ast.List, ast.Tuple)):
                    alpha = eval_scalar(node.args[1])
                if len(values) < 2:
                    raise ValueError("var_parametric requires at least two values")
                # Inverse normal: use math.erfinv via approximation since math doesn't have it.
                # statistics has NormalDist.
                z = statistics.NormalDist().inv_cdf(alpha / 100)
                return float(math.fsum(values) / len(values) - z * statistics.pstdev(values))
            if fn == "fisher_ideal" and len(node.args) == 4:
                # Fisher Ideal index = sqrt(Laspeyres * Paasche)
                # fisher_ideal(prices_0, quantities_0, prices_1, quantities_1)
                p0 = eval_list(node.args[0])
                q0 = eval_list(node.args[1])
                p1 = eval_list(node.args[2])
                q1 = eval_list(node.args[3])
                if not (len(p0) == len(q0) == len(p1) == len(q1)):
                    raise ValueError("fisher_ideal requires equal-length lists")
                lasp = math.fsum(a * b for a, b in zip(p1, q0)) / math.fsum(a * b for a, b in zip(p0, q0))
                paasche = math.fsum(a * b for a, b in zip(p1, q1)) / math.fsum(a * b for a, b in zip(p0, q1))
                return float(math.sqrt(lasp * paasche) * 100)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(elt) for elt in node.elts]
        raise ValueError("unsupported expression")

    expr = expression.strip()
    if not expr:
        raise ValueError("expression must not be empty")
    tree = ast.parse(expr.replace("^", "**"), mode="exec")
    
    last_value = None

    def assign_value(target: ast.AST, val: object):
        if isinstance(target, ast.Name):
            variables[target.id] = val
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(val, (list, tuple)):
                raise TypeError("cannot unpack non-iterable object")
            if len(target.elts) != len(val):
                raise ValueError(f"too many or too few values to unpack (expected {len(target.elts)}, got {len(val)})")
            for sub_target, sub_val in zip(target.elts, val):
                assign_value(sub_target, sub_val)
        else:
            raise ValueError(f"unsupported assignment target: {type(target).__name__}")

    for idx, stmt in enumerate(tree.body):
        is_last = idx == len(tree.body) - 1
        if isinstance(stmt, ast.Assign):
            val = evaluate(stmt.value)
            for target in stmt.targets:
                assign_value(target, val)
            if is_last:
                # Assignment-only scripts: return the assigned value instead
                # of None so "a = b + c" alone still yields a result.
                last_value = val
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, (ast.List, ast.Tuple)) and not is_last:
                raise ValueError("intermediate bare lists are not allowed; assign them to a variable")
            last_value = evaluate(stmt.value)
        else:
            raise ValueError(f"unsupported statement: {type(stmt).__name__}")

    return last_value


def round_half_up(value: float, digits: int) -> str:
    quant = Decimal("1").scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    if digits <= 0:
        return str(int(rounded))
    return format(rounded, "f")


def truncate_decimal(value: float, digits: int) -> str:
    factor = Decimal("1").scaleb(-digits)
    truncated = Decimal(str(value)).quantize(factor, rounding=ROUND_DOWN)
    if digits <= 0:
        return str(int(truncated))
    return format(truncated, "f")


_ALLOWED_SCRIPT_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    # Allow control flow and comprehensions (fixes 40% of compute_python_math failures)
    ast.For,
    ast.While,
    ast.If,
    ast.IfExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.AugAssign,
    ast.Return,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Starred,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Not,
    ast.Lambda,
)


def _validate_math_script(script: str) -> None:
    tree = ast.parse(script, mode="exec")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_SCRIPT_NODES):
            raise ValueError(f"unsupported syntax in math script: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {"math", "statistics", "decimal"}:
                    raise ValueError(f"import is not allowed: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module not in {"math", "statistics", "decimal"}:
                raise ValueError(f"import is not allowed: {node.module}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"open", "eval", "exec", "compile", "__import__", "input"}:
                raise ValueError(f"function is not allowed: {node.func.id}")


def run_math_subprocess(script: str, variables: dict[str, object] | None = None, timeout_seconds: float = 3.0) -> dict[str, object]:
    """Run a restricted local Python math snippet in an isolated subprocess."""
    _validate_math_script(script)
    payload = {"script": script, "variables": variables or {}}
    runner = r"""
import json, math, statistics
from decimal import Decimal

import io, sys
payload = json.loads(input())
script = payload["script"]
variables = payload.get("variables") or {}
# Capture print() output: scripts that compute via print(...) instead of
# assigning `answer`/`result` previously returned result=null and burned a
# tool call (seen 1-3x per trace in a graded run).
_captured = io.StringIO()
sys.stdout = _captured
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in {"math", "statistics", "decimal"}:
        raise ImportError(f"module not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)
safe_builtins = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "len": len,
    "sorted": sorted,
    "pow": pow,
    "print": print,
    "__import__": safe_import,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "range": range,
    "zip": zip,
    "enumerate": enumerate,
    "any": any,
    "all": all,
    "bool": bool,
    "map": map,
    "filter": filter,
    "isinstance": isinstance,
    "type": type,
    "divmod": divmod,
    "reversed": reversed,
    "repr": repr,
    "format": format,
    "cagr": lambda start, end, n: ((end / start) ** (1 / n) - 1) * 100,
}
env = {
    "__builtins__": safe_builtins,
    "math": math,
    "statistics": statistics,
    "Decimal": Decimal,
}
env.update(variables)
exec(script, env, env)
sys.stdout = sys.__stdout__
answer = None
for name in ("answer", "result", "final", "difference_rounded", "difference", "value"):
    if name in env:
        answer = env[name]
        break
stdout_text = _captured.getvalue().strip()
if answer is None and stdout_text:
    # Fall back to the last printed line as the result.
    answer = stdout_text.splitlines()[-1].strip()
print(json.dumps({"answer": answer, "result": answer, "stdout": stdout_text[:2000]}, default=str))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", runner],
        input=json.dumps(payload) + "\n",
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError((completed.stderr or completed.stdout or "math subprocess failed").strip())
    output = completed.stdout.strip().splitlines()[-1]
    return json.loads(output)
