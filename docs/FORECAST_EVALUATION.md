# Forecast evaluation on user-facing metrics

`manage.py evaluate_forecasts` (prices/forecast_eval.py) scores stored runs on Agile G in
p/kWh. It reports MAE, RMSE and bias, plus metrics closer to how people use the site:

- **sd ratio:** forecast spread divided by actual spread. Below 1 means compressed.
- **MAE bottom/top 10%:** error on slots where the outturn was in its lowest or highest
  decile.
- **neg recall / precision:** whether negative prices were forecast as negative.
- **cheap 4h regret:** for each forecast day, the actual average price in the 4-hour
  window the forecast picked as cheapest, minus the actual cheapest window. **cheap 4h hit**
  is how often the pick was within 1 hour of the true one.
- **peak 3h regret:** the same comparison for the priciest 3-hour window.
- **day rank corr:** within-day Spearman correlation between the forecast and the actual.

Slots that a run copied from already-published prices are excluded, so they can't score as
perfect forecasts.

## 1. Out-of-sample spread calibration (2026-10-26)

`--calibrate stretch,qmap` remaps each run's forecast per lead bucket (0–1, 1–2, 2–4, 4–7
and 7+ days). It is fitted only on pairs from earlier runs whose outturn had already
settled, within a trailing window. Two methods were tried:

- **stretch:** rescale around the mean so the forecast's sd matches the actual sd.
- **qmap:** empirical quantile mapping.

The fit is done in day-ahead space, so the peak adder, shift and VAT aren't stretched.

**Data caveat.** Production's 221 clean runs since 28 Aug weren't used: the export was
blocked. Dev's runs before 5 Oct were the dynamic-range trial algorithm. So the clean
evaluation is only runs from 5 Oct (history), scored from 12 Oct. That is about 2 weeks of
lead ≥ 2 d outturn.

Clean window, lead ≥ 2 d, with a 14-day calibration window:

| | MAE | RMSE | bias | sd ratio | MAE bottom 10% | MAE top 10% | neg recall | cheap 4h regret |
|---|---|---|---|---|---|---|---|---|
| ensemble | **9.49** | **12.47** | 5.70 | 0.55 | 22.3 | 5.30 | 0.00 | **2.51** |
| + stretch | 10.45 | 13.16 | 6.78 | 0.73 | 20.9 | 5.17 | 0.00 | 2.56 |
| + qmap | 10.09 | 13.03 | 6.60 | 0.70 | **20.6** | **3.87** | 0.02 | 2.66 |

A 7-day window gave the same picture. Over all 245 dev runs, which mixes in trial-period
history, stretch overshot (sd ratio 1.2) and MAE rose from 7.2 to 9.3.

Findings:

- **Calibration can't improve timing.** A monotone remap preserves the order of slots
  within a day. The cheapest/priciest-window and rank metrics only move slightly, where a
  day straddles two lead buckets. Picking the right slots needs a better model, not
  post-processing.
- **It trades average error for tails, and not cheaply.** MAE and RMSE got worse while
  bottom-decile error improved by only about 2p.
- **Negative prices are essentially never forecast at lead ≥ 2 d.** Recall was 0 from
  5 Oct across 1,458 negative slots, and calibration barely moves it.
- **The ensemble's recent error is mostly bias, not spread.** It ran +5.7p high over the
  clean window. Calibration fitted on the preceding fortnight doesn't track a level shift.
- **Verdict:** not worth deploying on this evidence. It could be re-checked on production's
  clean history if that export is allowed.
