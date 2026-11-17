#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (R2).

Reads one case (makefun/twitch-ccu-case/v1) from the input directory and
writes, for hours 2..24 of the UTC release day, five quantiles (5/25/50/75/95%)
of that hour's concurrent viewers in the game's Twitch category.

Model -- per-hour ridge regression over past launches, fitted at run time
from the case's own comparables (past launches' release days, shipped in
the case).  Two facts drive it:

1. Scale bends the day.  Across past launches, the bigger hour 1 is, the
   flatter the rest of the day: log hour-1 viewers correlates about -0.7
   with the evening rise.  A launch the size of GTA VI should therefore be
   shaped mostly by the biggest past launches, and the regression says
   exactly how much flatter.

2. Each game's own audience clock predicts its day.  A game's Twitch
   audience has a daily rhythm (which UTC hours it watches) that shows up
   in the days before release, and that rhythm tracks the release-day
   curve with a per-launch correlation near 0.7 on average: a US-heavy
   audience peaks in the US evening, a Europe-heavy one earlier.  So for
   every hour h we use the target's own pre-release diurnal signature at
   UTC hour h-1 as a predictor, alongside log hour-1 viewers.

For each hour h = 2..24 a ridge regression (intercept + log h1 + diurnal
deviation, lambda 10 on standardised features) is fitted on the pool of
past launches and evaluated at the target's own predictors.  The spread
is the regression's residual spread, shrunk by 0.85, with a floor of 0.12
in log units.  Pool: comparables with a full 24-hour release day and
hour-1 viewers of at least 5,000 (floor relaxes if that leaves too few).

Validation (leave-one-out over the kit's comparables, pinball loss on the
log scale, factor 2): on big launches (hour-1 >= 20k, the regime of the
practice cases and of GTA VI) mean 0.248 / median 0.172, against the
previous pooled-median entry's 0.257 / 0.211 and the flat persistence
baseline's 0.321 / 0.256.  On the five comparables most like the practice
cases (console-only, rolling midnight, no early access) it scores 0.138.

Pure standard library, no network calls, fully deterministic: the same
case always produces the same output bytes.  Runs in a fraction of a
second per case.
"""

import json, math, os, sys
from datetime import datetime, timezone

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is a single file.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Hours this entry forecasts.
FORECAST_HOURS = range(2, 25)
# Standard-normal quantiles for the levels above, to full precision.
Z_TABLE = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}

# Pool: comparables need a full release day and a meaningful hour-1 count;
# the floor relaxes only if it would leave too few launches.
POOL_FLOORS = [5000, 1000, 100, 1]
POOL_MIN = 8
# Ridge penalty on the standardised features (the intercept is unpenalised).
RIDGE_LAMBDA = 10.0
# The diurnal signature looks back this many days before the release day.
DIURNAL_DAYS = 3
# Residual spread is shrunk by this factor; the floor stops false certainty.
SPREAD_SHRINK = 0.85
SD_FLOOR = 0.12
# Fallback spread (log units) when the pool is too small to regress.
FALLBACK_SD = 0.35
# Seconds in a day; smallest viewer count ever reported.
DAY_SECONDS = 86400
MIN_PRINT = 1.0


def _num(value):
    """float(value) for a real number, else None (bools do not count)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_ts(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed), or None if bad."""
    if not isinstance(text, str):
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def _point_tuple(point):
    """(when, viewers) for a well-formed hourly point, else None."""
    if not isinstance(point, dict):
        return None
    when = parse_ts(point.get("from"))
    viewers = _num(point.get("viewers"))
    if when is None or viewers is None or viewers <= 0:
        return None
    return (when, viewers)


def _diurnal_point(point):
    """(when, log viewers floored at 0) for a usable pre-release point.

    Unlike _point_tuple this keeps zero-viewer points (as log 1 = 0):
    a dead hour is still information about the audience's rhythm.
    """
    if not isinstance(point, dict):
        return None
    when = parse_ts(point.get("from"))
    viewers = _num(point.get("viewers"))
    if when is None or viewers is None:
        return None
    return (when, math.log(max(viewers, 1.0)))


def zscore(level):
    """Standard-normal quantile at `level`.

    The five contest levels come from a table; any other level uses
    Acklam's rational approximation (accurate to about 1e-9).
    """
    if level in Z_TABLE:
        return Z_TABLE[level]
    if not 0.0 < level < 1.0:
        raise ValueError("level %r is outside (0, 1)" % level)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    if level < 0.02425:
        q = math.sqrt(-2.0 * math.log(level))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if level > 0.97575:
        q = math.sqrt(-2.0 * math.log(1.0 - level))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = level - 0.5
    sq = q * q
    num = (((((a[0] * sq + a[1]) * sq + a[2]) * sq + a[3]) * sq + a[4]) * sq + a[5]) * q
    den = ((((b[0] * sq + b[1]) * sq + b[2]) * sq + b[3]) * sq + b[4]) * sq + 1.0
    return num / den


def diurnal_signature(points, day_start):
    """The audience's daily rhythm: 24 centred mean log-viewers by UTC hour.

    Built from the points in the DIURNAL_DAYS days before the release day
    (the release hour itself excluded).  Hours with no points read 0.0
    before centring; the whole profile is centred on its own 24-hour mean,
    so it is a shape, not a level.  An empty history gives all zeros, and
    the model then leans on log hour-1 alone.
    """
    window = DIURNAL_DAYS * DAY_SECONDS
    by_hour = [[] for _ in range(24)]
    usable = [t for t in (_diurnal_point(p) for p in points or []) if t is not None]
    for when, log_viewers in usable:
        gap = (day_start - when).total_seconds()
        if 0 < gap <= window:
            by_hour[when.hour].append(log_viewers)
    profile = [(sum(vals) / len(vals)) if vals else 0.0 for vals in by_hour]
    mean = sum(profile) / 24.0
    return [value - mean for value in profile]


def release_hours(comparable):
    """{hour: viewers} for the comparable's UTC release day, hours 1..24.

    Only points inside [day_start, day_start + 24h) belong to the day.
    Sparse categories may miss hours; those hours are simply absent, and
    the launch is dropped from the pool (it needs a full day).
    """
    day = parse_ts(comparable.get("day_start")) if isinstance(comparable, dict) else None
    if day is None:
        return dict()
    found = {}
    usable = [t for t in (_point_tuple(p) for p in comparable.get("hourly") or [])
              if t is not None]
    for when, viewers in usable:
        offset = (when - day).total_seconds()
        if 0 <= offset < DAY_SECONDS:
            found[int(offset // 3600) + 1] = viewers
    return found


def median_of(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Partial pivoting; returns None on a (numerically) singular system.
    """
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        rest = sum(rows[i][j] * solution[j] for j in range(i + 1, n))
        solution[i] = (rows[i][n] - rest) / rows[i][i]
    return solution


def ridge_centre_and_spread(pool_rows, target_row):
    """One hour's ridge fit: (centre, spread) for the target's predictors.

    pool_rows is a list of (log_h1, diurnal, log_ratio) for the pool;
    target_row is the target's (log_h1, diurnal).  Features are
    standardised on the pool (population spread); the intercept is
    unpenalised.  The spread is the residual population spread, shrunk.
    Returns None when the fit is singular.
    """
    n = len(pool_rows)
    means = [sum(row[k] for row in pool_rows) / n for k in (0, 1)]
    sds = []
    for k in (0, 1):
        var = sum((row[k] - means[k]) ** 2 for row in pool_rows) / n
        sds.append(math.sqrt(var) if var > 0 else 1.0)
    # Normal equations for [1, x1, x2] with ridge on the last two.
    dim = 3
    ata = [[0.0] * dim for _ in range(dim)]
    aty = [0.0] * dim
    for log_h1, diurnal, y in pool_rows:
        x = [1.0, (log_h1 - means[0]) / sds[0], (diurnal - means[1]) / sds[1]]
        for i in range(dim):
            aty[i] += x[i] * y
            for j in range(dim):
                ata[i][j] += x[i] * x[j]
    for i in (1, 2):
        ata[i][i] += RIDGE_LAMBDA
    beta = solve_linear(ata, aty)
    if beta is None:
        return None
    tx = [1.0, (target_row[0] - means[0]) / sds[0],
          (target_row[1] - means[1]) / sds[1]]
    centre = sum(b * t for b, t in zip(beta, tx))
    resid_sq = 0.0
    for log_h1, diurnal, y in pool_rows:
        x = [1.0, (log_h1 - means[0]) / sds[0], (diurnal - means[1]) / sds[1]]
        fitted = sum(b * v for b, v in zip(beta, x))
        resid_sq += (y - fitted) ** 2
    resid_sd = math.sqrt(resid_sq / n)
    spread = max(SD_FLOOR, resid_sd * SPREAD_SHRINK)
    return (centre, spread)


def hour1_print(case, day_start):
    """Hour 1's viewer count, at least MIN_PRINT.

    Falls back to the hourly point at the release day's start, then to the
    last point at or before it, then to MIN_PRINT, so the program always
    writes a valid forecast instead of failing the case.
    """
    known = [row for row in case.get("known", [])
             if isinstance(row, dict) and row.get("hour") == 1]
    if known:
        value = _num(known[0].get("viewers"))
        if value is not None and value > 0:
            return max(MIN_PRINT, value)
    history = case.get("history_hourly") or []
    if history and day_start is not None:
        best = None
        usable = [t for t in (_point_tuple(p) for p in history) if t is not None]
        for when, viewers in usable:
            if when == day_start:
                return max(MIN_PRINT, viewers)
            if when < day_start:
                best = viewers
        if best is not None:
            return max(MIN_PRINT, best)
    if history:
        last = _point_tuple(history[-1])
        if last is not None:
            return max(MIN_PRINT, last[1])
    return MIN_PRINT


def tidy_quantiles(values):
    """Clamp to the output contract, keep the row non-decreasing, round."""
    cleaned = list()
    for value in values:
        value = min(1e9, max(1e-2, value))
        if cleaned and value < cleaned[-1]:
            value = cleaned[-1]
        rounded = round(value, 2)
        cleaned.append(int(rounded) if float(rounded).is_integer() else rounded)
    return cleaned


def forecast_case(case):
    """The 23 forecast rows for one case."""
    levels = case.get("levels") or list(LEVELS)
    zs = [zscore(float(level)) for level in levels]
    day_start = parse_ts(case.get("day_start"))
    p1 = hour1_print(case, day_start)
    log_hour1 = math.log(p1)
    target_diurnal = diurnal_signature(case.get("history_hourly") or [],
                                       day_start) if day_start else [0.0] * 24

    # The pool: comparables with a full release day, in name order so the
    # run is deterministic.  The floor relaxes only if it leaves too few.
    launches = []
    comparables = sorted(case.get("comparables") or [],
                         key=lambda c: str(c.get("name", "")) if isinstance(c, dict) else "")
    for comp in comparables:
        day = release_hours(comp)
        cday = parse_ts(comp.get("day_start")) if isinstance(comp, dict) else None
        if len(day) == 24 and day.get(1, 0) >= 1 and cday is not None:
            launches.append({
                "h1": day[1],
                "log_h1": math.log(day[1]),
                "ratios": {h: math.log(day[h] / day[1]) for h in FORECAST_HOURS},
                "diurnal": diurnal_signature(comp.get("hourly") or [], cday),
            })
    pool = [L for L in launches if L["h1"] >= POOL_FLOORS[0]]
    floor_idx = 1
    while len(pool) < POOL_MIN and floor_idx < len(POOL_FLOORS):
        pool = [L for L in launches if L["h1"] >= POOL_FLOORS[floor_idx]]
        floor_idx += 1

    rows = list()
    for hour in FORECAST_HOURS:
        centre = spread = None
        if len(pool) >= 6:
            pool_rows = [(L["log_h1"], L["diurnal"][hour - 1], L["ratios"][hour])
                         for L in pool]
            fitted = ridge_centre_and_spread(
                pool_rows, (log_hour1, target_diurnal[hour - 1]))
            if fitted is not None:
                centre, spread = fitted
        if centre is None:
            # Too few launches to regress (or a singular fit): fall back to
            # the pooled median curve with a fixed spread.
            ratios = [L["ratios"][hour] for L in launches] or [0.0]
            centre, spread = median_of(ratios), FALLBACK_SD
        quantiles = [p1 * math.exp(centre + z * spread) for z in zs]
        rows.append({"hour": hour, "q": tidy_quantiles(quantiles)})
    return dict(hours=rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    names = sorted(fn for fn in os.listdir(IN_DIR) if fn.endswith(".in"))
    for name in names:
        with open(os.path.join(IN_DIR, name), encoding="utf-8") as handle:
            case = json.load(handle)
        forecast = forecast_case(case)
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        out_path = os.path.join(OUT_DIR, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload)


if __name__ == '__main__': main()
