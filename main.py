"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case (makefun/twitch-ccu-case/v1) from the input directory and writes,
for hours 2..24 of the UTC release day, five quantiles (5/25/50/75/95%) of that
hour's concurrent viewers in the game's Twitch category.

Method — pooled launch-shape forecaster, fitted at run time from the case's own
comparables (past big launches' release days, shipped inside the case):

* Anchor: hour 1's print p1.  Every forecast is p1 times a ratio.
* Shape: for each hour h, the distribution of log(v_h / v_1) across the
  comparables.  Big launches behave alike, small ones do not, so the pool is
  restricted to comparables whose hour-1 count is at least 5,000 (relaxed
  only if that leaves too few), and each comparable is weighted by how close
  its hour-1 scale is to p1: w = exp(-|log h1 - log p1| / 1.5).  A launch the
  size of GTA VI is therefore shaped mostly by the biggest past launches.
* Centre: the weighted median log-ratio per hour.  It captures the real
  launch-day shape the flat persistence baseline misses: a dip in the small
  hours after midnight UTC, flat midday, then a strong climb into the US
  afternoon/evening (peak around hour 21) and a fall into the night.
* Quantiles: the centre three levels take the empirical weighted
  quartiles directly; the 5/95% tails use centre + z_tau * sd in log space,
  z from the standard normal.

Validation (leave-one-out over the kit's 38 comparables, pinball loss on the
log scale, factor 2): big launches (hour-1 >= 20k, the regime of the practice
cases and of GTA VI) score mean 0.258 / median 0.217 against the persistence
baseline's 0.321 / 0.256, winning 23 of 27.  The flat baseline scores 0.93
mean over all 37 launches; this model 0.74.

Standard library only, no network, deterministic: same case, same bytes.
Runs in well under a second per case.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Pool: comparables need a full release day and a meaningful hour-1 count.
POOL_FLOORS = (5000, 1000, 100, 1)
POOL_MIN_SIZE = 8
# Scale kernel: how fast a comparable's weight falls with |log h1 - log p1|.
SCALE_BANDWIDTH = 1.5
# Spread floor, in natural-log units, so no hour claims false certainty.
SD_FLOOR = 0.15
# The persistence baseline's spread, used only when the pool is empty.
BASELINE_SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1.15e-9 absolute; used only to turn a case's levels into
    z-scores, so the program also works if a case ever asks for other levels.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"level {p} is outside (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def release_day_hours(comparable: dict) -> dict:
    """{hour: viewers} for the comparable's UTC release day, hours 1..24.

    A comparable's `hourly` list covers the fortnight around its release;
    only points inside [day_start, day_start + 24h) belong to the day.
    Sparse categories may miss hours; those hours are simply absent.
    """
    try:
        day = _parse_time(comparable["day_start"])
        hourly = comparable["hourly"]
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for point in hourly:
        try:
            when = _parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        delta = (when - day).total_seconds()
        if 0 <= delta < 24 * 3600:
            out[int(delta // 3600) + 1] = float(viewers)
    return out


def _parse_time(text: str):
    """Parse an ISO-8601 UTC timestamp (with a trailing Z) to a datetime."""
    from datetime import datetime, timezone
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def weighted_quantile(pairs: list, p: float) -> float:
    """Weighted p-quantile of [(value, weight), ...]; 0.0 when empty."""
    if not pairs:
        return 0.0
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return 0.0
    threshold = p * total
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= threshold:
            return value
    return ordered[-1][0]


def hour_one_print(case: dict) -> float:
    """Hour 1's viewer count, at least 1.

    Falls back to the last pre-checkpoint hourly point, then to 1, so the
    program always writes a valid forecast instead of failing the case.
    """
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                return max(1.0, float(row["viewers"]))
            except (KeyError, TypeError, ValueError):
                pass
    hourly = case.get("history_hourly") or []
    if hourly:
        try:
            return max(1.0, float(hourly[-1]["viewers"]))
        except (KeyError, TypeError, ValueError):
            pass
    return 1.0


def forecast_case(case: dict, name: str) -> dict:
    """The 23 forecast rows for one case."""
    levels = case.get("levels") or LEVELS
    z_scores = [norm_ppf(float(level)) for level in levels]
    p1 = hour_one_print(case)

    # The pool: comparables with a full release day, in name order so the
    # run is deterministic.  The floor relaxes only if it leaves too few.
    days = []
    for comparable in sorted(case.get("comparables") or [], key=lambda c: str(c.get("name", ""))):
        hours = release_day_hours(comparable)
        if len(hours) == 24 and hours.get(1, 0) >= 1:
            days.append((comparable, hours))
    pool = []
    for floor in POOL_FLOORS:
        pool = [(c, h) for c, h in days if h[1] >= floor]
        if len(pool) >= POOL_MIN_SIZE or floor == POOL_FLOORS[-1]:
            break

    log_p1 = math.log(p1)
    rows = []
    for hour in range(2, 25):
        pairs = []
        for _, hours in pool:
            weight = math.exp(-abs(math.log(hours[1]) - log_p1) / SCALE_BANDWIDTH)
            pairs.append((math.log(hours[hour] / hours[1]), weight))
        if pairs:
            centre = weighted_quantile(pairs, 0.5)
            lower = weighted_quantile(pairs, 0.25)
            upper = weighted_quantile(pairs, 0.75)
            spread = max(SD_FLOOR, (upper - lower) / 1.349)
            # Hybrid quantiles: the centre three levels take the empirical
            # weighted quartiles directly, while the 5/95% tails use the
            # normal form centre + z*spread — with only ~27 launches the
            # raw 5/95% percentiles are noisy and the IQR-based tail is
            # steadier.  (LOO on big launches: 0.2577 mean vs 0.2590 for
            # the all-normal form.)  Any other level a case might ask for
            # also takes the normal form.
            empirical = {0.25: lower, 0.5: centre, 0.75: upper}
            quantiles = [
                p1 * math.exp(empirical.get(level, centre + z * spread))
                for level, z in zip(levels, z_scores)
            ]
        else:
            # No usable comparable: the flat persistence baseline.
            quantiles = [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(hour - 1)) for z in z_scores]
        # Clamp to the output contract and keep the row non-decreasing.
        cleaned = []
        previous = 0.01
        for value in quantiles:
            value = min(1e9, max(0.01, value))
            value = max(previous, value)
            cleaned.append(round2(value))
            previous = cleaned[-1]
        rows.append({"hour": hour, "q": cleaned})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
