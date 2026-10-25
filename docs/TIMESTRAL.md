# Timestral parallel forecast (dev only)

Mistral's [Timestral](https://github.com/mistralai/timestral) (released 15 Oct 2026) is a
pretrained time-series foundation model for zero-shot forecasting. It comes in 5M, 28M, 123M
and 209M sizes, takes up to 32,768 context steps, accepts past and known-future covariates,
and outputs 21 quantiles. It is not trained on our data; it is only conditioned on it at
inference time.

## How it runs

- `prices/timestral.py` builds a 30-minute UTC grid that covers `TIMESTRAL_CONTEXT_DAYS`
  (default 365) of context plus the horizon of the latest stored run.
  - **Target:** `PriceHistory.day_ahead`, with GB60 appended where it has been published.
  - **Known covariates** are `solar, emb_wind, bm_wind, demand, dispatchable_capacity,
    temp_2m, wind_10m`. Each slot takes its value from the most recent stored run issued
    before it: the shortest lead over the context, and the latest run over the horizon.
  - **Calendar covariates:** hour sin/cos, a peak (16–19h) indicator, and a weekend/bank
    holiday indicator.
- Output is the median, with p10 and p90 as low and high. It is converted to Agile G and
  stored as `ExternalForecast` source `timestral`. That gives it the v2 chart overlay
  (`ts=1`), the v2 history comparison (`compare_timestral=1`), and the same retention as
  AgileForecast and X2R.
- The update worker runs `manage.py timestral_forecast` in a separate process after each
  completed update when `TIMESTRAL_ENABLED=1`, which is set in dev's `.env`. Running it
  separately means torch is released from memory afterwards. The output goes to
  `logs/timestral.log`.
- Dependencies are in `requirements-timestral.txt`: CPU torch 2.14.1 and the pinned
  timestral commit. Production installs neither, and the toggles are hidden there.

## Resources (dev CT: 2 vCPU, 1 GB RAM)

The benchmark used one year of context, 15 covariate rows and a 700-step horizon.

| model | forecast time | peak RSS |
|---|---|---|
| 5M | 13 s | 600 MB |
| 28M | 49 s | 880 MB |
| 209M | not tried: the 837 MB of weights alone exceed the CT's RAM | |

A live run with the 5M model takes about 15 s end to end.

## Backtest: 5M, 2026-10-25

`manage.py timestral_backtest` takes each stored daily run (the one nearest 16:15) and
rebuilds what was known at the time:

- covariates from runs created at or before that run;
- prices only before the first slot that the run itself forecast.

It then compares Timestral with that run's own ensemble forecast and with the outturn. There
were 54 runs (26 Aug – 23 Oct) and 30.6k scored slots. Errors are Agile G in p/kWh.

| | MAE | RMSE | sd of forecast |
|---|---|---|---|
| Timestral 5M, lead ≥ 2 d | 7.82 | 10.31 | 8.97 |
| Ensemble, lead ≥ 2 d | 7.41 | 9.91 | 10.99 |
| 50/50 blend, lead ≥ 2 d | 7.35 | | |
| Actual | | | 13.69 |

- **Timestral 5M is slightly worse overall.** It is level or better only at leads of 10–13
  days. Lead 0–1 favours the ensemble because its day 0–1 includes GB60 (see the GB60
  caveat), so those leads aren't a fair comparison.
- **It doesn't fix tail compression; it is more compressed.** Its sd is 9.0 against 11.0
  for the ensemble and 13.7 actual.
  - In the bottom decile of actuals (mean −0.2p), Timestral forecasts 20.4p and the
    ensemble 16.3p.
  - In the top decile (mean 51.1p), the figures are 44.4p and 48.0p.
- **Errors are highly correlated with the ensemble's (r = 0.84)**, so blending gains only
  about 0.06p.
- **More covariate history doesn't help.** Week by week, the gap doesn't shrink as the
  covariate context grows from 0 to 8 weeks.
- **Caveat:** before about 5 Oct the benchmark ensemble is dev's trial algorithm with the old
  day_ahead units (see MODEL_DYNAMIC_RANGE.md). The clean window is runs from 5 Oct, where
  the ensemble leads by about 1p a week: Timestral 7.0 / 9.5 / 14.1 against ensemble
  6.2 / 8.2 / 12.4.
