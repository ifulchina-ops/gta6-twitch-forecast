[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ifulchina-ops/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ifulchina-ops/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast (R2)

An entry for the make.fun bounty [Forecast GTA VI's launch-day Twitch
viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

The entry is a single Python file, `main.py` (Python 3.13, standard library
only). Given one case in `/input`, it writes five quantiles (5/25/50/75/95%)
of concurrent viewers for each of hours 2–24 of the UTC release day in the
Grand Theft Auto VI Twitch category.

## How it forecasts

A per-hour ridge regression over past launches, fitted at run time from the
case's own comparables (past big launches' release days, shipped inside the
case):

- **Anchor:** the hour-1 print p1. Each forecast is p1 times a modelled ratio.
- **Scale bends the day:** log hour-1 viewers correlates about −0.7 with the
  evening rise — bigger launches climb less through the day. The regression
  uses log hour-1 as a predictor, so a launch GTA VI's size is shaped mostly
  by the biggest past launches, with the right amount of flattening.
- **Each game's own audience clock:** a game's Twitch audience has a daily
  rhythm by UTC hour, visible in the days before release, and that rhythm
  tracks its release-day curve (per-launch correlation ≈ 0.7 on average).
  The target's own 3-day pre-release diurnal signature at UTC hour h−1 is
  the second predictor for hour h.
- **Fit:** for every hour, ridge regression (intercept + log h1 + diurnal
  deviation, λ = 10 on standardised features) on comparables with a full
  24-hour release day and hour-1 viewers ≥ 5,000. Spread is the residual
  spread, shrunk ×0.85, floored at 0.12 in log units.

## Backtest

Leave-one-out over the kit's comparables, scored by pinball loss on the log
scale (factor 2). On big launches — hour-1 ≥ 20k, the size class of the
practice cases and of GTA VI — the model scores mean **0.248** / median
**0.172**, against the previous pooled-median entry's 0.257 / 0.211 and the
flat persistence baseline's 0.321 / 0.256. On the five comparables closest
to the practice cases (console-only, rolling midnight, no early access) it
scores **0.138**.

The model was picked by a leave-one-out tournament over 20+ candidates:
pooled-median variants, scale kernels, kNN, full ridge models, shape-family
mixtures, and diurnal-profile regressions at several windows and spreads.
The diurnal regression won on every slice (big launches, evening-peak
launches, and the five most practice-like comparables). Hard family
assignment by unlock flags was tried and rejected — misclassification is
catastrophic when a launch's shape surprises you.

## Running it

Run `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py` from the
kit directory (the output folder is created if needed). Same case, same
bytes: the program is deterministic, makes no network calls, and finishes
each case in a fraction of a second.

## Licence

Released under the 0BSD licence — see the LICENSE file.
