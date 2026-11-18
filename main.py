"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecast model: comparable-launch quantiles. GTA VI launches midnight-local rolling (like Call of Duty,
Elden Ring, Hogwarts Legacy, Ghost of Yōtei, ...), so hour h's count relative to hour 1's print is best estimated
from comparable releases that share that unlock pattern. For each such comparable we compute ln(viewers_h / viewers_1)
on its release day; across the comp set we keep mean m_h and std-dev s_h per hour h = 2..24, then forecast

    q_tau(h) = p1 * exp(m_h + z_tau * s_h),  rounded to 2 decimals (half up),

with z_tau the standard normal quantile at levels [0.05, 0.25, 0.5, 0.75, 0.95] = [-1.645, -0.674, 0, 0.674, 1.645],
p1 = hour 1's print (min 1). If too few rolling-midnight comparables are usable, fall back to the persistence shape
q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).

This keeps each quantile non-decreasing in tau, every value > 0 and bounded below 1e9, runs in well under
8 seconds with no network and no pip-installed modules.
"""

import json
import math
import os
import sys
from pathlib import Path

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence fallback spread, in natural-log units per square-root hour (the kit starter's value).
SIGMA_BASE = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_hourly(comparable: dict) -> dict:
    """hour -> viewers for release day hours 1..24 of one comparable launch."""
    from datetime import datetime

    ds = datetime.fromisoformat(comparable["day_start"].replace("Z", "+00:00"))
    d = {}
    for row in comparable.get("hourly", []):
        v = row.get("viewers")
        if v is None:
            continue
        ts = datetime.fromisoformat(row["from"].replace("Z", "+00:00"))
        h = int((ts - ds).total_seconds() // 3600)
        if 0 <= h < 25:
            d[h] = v
    return d


def build_ratios(comps) -> dict:
    """hour -> list of ln(viewers_h / viewers_1) across comparables."""
    ratios = {}
    for c in comps:
        rh = release_hourly(c)
        v1 = rh.get(1)
        if not v1 or v1 <= 0:
            continue
        for h in range(2, 25):
            v = rh.get(h)
            if v and v > 0:
                ratios.setdefault(h, []).append(math.log(v / v1))
    return ratios


def hour_quantiles_gaussian(p1: float, h: int, ratios: dict) -> list[float]:
    """q_tau = p1 * exp(m_h + z_tau * s_h) from comparable log-ratios."""
    vals = ratios.get(h, [])  # default to empty list if hour missing
    n = len(vals)
    if n == 0:
        return [round2(p1 * math.exp(z * SIGMA_BASE * math.sqrt(h - 1))) for z in Z]
    m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / n)
    return [round2(p1 * math.exp(m + z * sd)) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    """Return {'hours': [{'hour': h, 'q': [q05, q25, q50, q75, q95]} ...]} for hours 2..24."""
    known = next((k for k in case.get("known", []) if k["hour"] == 1), None)
    if known is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, known["viewers"])
    comps = case.get("comparables", [])
    unlock_lower = lambda c: c.get("unlock", "").lower()
    rolling = [c for c in comps if ("midnight" in unlock_lower(c) and "local" in unlock_lower(c)) or "rolling" in unlock_lower(c)]
    source = rolling if len(rolling) >= 4 else (comps if len(comps) >= 4 else [])
    ratios = build_ratios(source)
    hours = []
    for h in range(2, 25):
        hours.append({"hour": h, "q": hour_quantiles_gaussian(p1, h, ratios)})
    return {"hours": hours}


def main() -> None:
    INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
