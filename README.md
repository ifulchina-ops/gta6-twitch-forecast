[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ifulchina-ops/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ifulchina-ops/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on make.fun.

`main.py` (Python 3.13, standard library only) reads one case from `/input`
and writes, for hours 2–24 of the UTC release day, five quantiles (5/25/50/75/95%)
of that hour's concurrent viewers in the Grand Theft Auto VI Twitch category.

## Method

Pooled launch-shape forecaster, fitted at run time from the case's own
comparables (past big launches' release days, shipped inside the case):

- **Anchor:** hour 1's print. Every forecast is hour 1 times a ratio.
- **Shape:** for each hour, the distribution of log(viewers_h / viewers_1)
  across comparables. The pool keeps launches with hour-1 counts of at least
  5,000, each weighted by how close its hour-1 scale is to hour 1
  (`exp(-|log h1 - log p1| / 1.5)`), so a launch GTA VI's size is shaped
  mostly by the biggest past launches.
- **Centre:** the weighted median log-ratio per hour — the real launch-day
  shape the flat baseline misses: a dip in the small hours after midnight UTC,
  flat midday, then a strong climb into the US afternoon/evening (peak around
  hour 21) and a fall into the night.
- **Spread:** `max(0.15, weighted IQR / 1.349)` per hour. The centre three
  quantiles take the empirical weighted quartiles; the 5/95% tails use the
  normal form, which is steadier with ~27 launches.

## Validation

Leave-one-out over the kit's 38 comparables (pinball loss on the log scale,
factor 2): on big launches (hour-1 ≥ 20k — the regime of the practice cases
and of GTA VI) the model scores mean **0.258** / median **0.217** against the
persistence baseline's 0.321 / 0.256, winning 23 of 27. Over all 37 launches
it scores 0.74 mean against the baseline's 0.93.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: the same case always writes the same bytes. No network, no
model calls, well under a second per case.

## License

0BSD — see [LICENSE](LICENSE).
