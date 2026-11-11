# Forecast review: extremes, 14 Oct – 11 Nov 2026

Question from the owner: wholesale prices swung hard over the last four weeks; could the model
have predicted the extremes better?

Scripts in `analysis/forecast_review_2026_11/` (data in its gitignored `data/`). Run on the dev box.
Forecasts scored are the **dev box's** stored runs. Prod's run history isn't reachable from here.
Dev ran the **default feature set (prod-equivalent)** until 30 Oct 17:33 UTC, and the
**`full_fr_eu` candidate** (+ continental wind) after that. All horizons quoted are ≥2 d unless
stated, because 0–1 d is the GB60 day-ahead passthrough, not model output.

## 1. What happened

Settled day-ahead (£/MWh). Over the window the 10th percentile was £90, the 90th £250, and prices ran from −£20 to £410.

| period | what | actual | model 2–4 d | why it was hard |
|---|---|---|---|---|
| 24–25 Oct | windy weekend, negatives to −£20 | daily mean £22–42 | £108–128 | covered in [the October review](FORECAST_REVIEW_2026_10.md): continental wind |
| 28 Oct – 1 Nov | calm GB and continent, TTF €95–104 | £189–214 | £123–170 (−£40 to −£66) | **no day in the prior 60 had a mean this high** |
| 6 Nov | wind 2 GW at peak | max £317 | max £250 | only 5 slots that high in the prior 60 d |
| 9 Nov | wind 1.6 GW, demand 39.9 GW at peak | max **£410**, mean £255 | max £257, mean £214 | **above the 60-day max (£373); daily mean a new high** |
| 10 Nov | | max £330 | max £252 | |

Our forecast daily maximum at 2–4 d sat at **£250–260 every November day**, while the actual
ranged £246–410.

By horizon (all target days): high-10% bias −£23 at 1–2 d, −£40 at 2–3 d, −£51 at 3–5 d,
−£73 at 8–15 d; low-10% bias +£47 / +£64 / +£77 / +£107.

## 2. The inputs were right. The model's range was the limit

The 2–4 d inputs matched the <1 d inputs on every extreme day. For example, on 9 Nov BM wind was
2.5 GW forecast against 1.3 GW nearer the day, and demand 32.3 against 32.8 GW. The 28 Oct – 1 Nov
calm was also forecast correctly (`eu_wind_cf` 0.04–0.06 at both horizons). **The weather inputs
weren't the problem.**

The 60-day training window didn't contain these states:

* 9 Nov evening residual demand (demand − wind − solar, as forecast) was 36–38 GW. The 60 d before
  had **13 slots ≥35 GW**, all in the preceding week. September–October never exceeded 31 GW.
* The production ensemble (CatBoost + LightGBM + ExtraTrees) is all trees, and trees can't predict
  above the largest training target. The £410 was unreachable. So was 29 Oct's £213 daily mean,
  because no day in the window was that high.
* The dev DB only keeps forecast inputs from 12 Sep 2026, so I couldn't test whether last
  winter's tight days would have taught the model these states.

## 3. Other forecasters (Agile region G, p/kWh, pairwise on common slots)

| 2–4 d | high-10% bias | high-10% MAE | low-10% MAE | mid MAE |
|---|---|---|---|---|
| **AgilePredict** | −8.8 | 9.9 | 14.4 | 5.6 |
| AgileForecast | **+1.8** | **8.2** | 16.9 | 6.4 |
| X2R | −11.9 | 12.8 | **10.0** | 6.7 |
| AgileLab (since 7 Nov only; ours on the same slots: −13.9 / 14.5 / — / 4.4) | −12.9 | 12.9 | — | 3.8 |

AgileForecast was the only forecaster that caught the November peaks: its 9 Nov max at 2–4 d was
73.9p against 94.8p actual, and ours was 56.9p. It pays for that in the mid-range. X2R was best on
the low tail. AgilePredict had the best mid-range accuracy, but was the flattest at the top.

## 4. Offline test of model forms (`counterfactual.py`)

Rolling setup: train on the prior 60 d (22–46 h rows), then predict each day's 04:15 run at 2–4 d. All
variants get the same inputs, including `eu_wind_cf`. **Compare the variants with each other, not with
production.** Offline replays haven't reproduced production before ([`MODEL_DYNAMIC_RANGE.md`](MODEL_DYNAMIC_RANGE.md)).
The ET (price) baseline's tail biases are close to production's, which is reassuring.

| variant | MAE | RMSE | high-10% bias | low-10% bias | mid bias |
|---|---|---|---|---|---|
| ExtraTrees, target price (≈ production form) | 26.5 | 34.4 | −42.5 | +44.0 | −8.6 |
| **ExtraTrees, target price / TTF** | **24.4** | **33.0** | −38.3 | +48.7 | **+1.5** |
| LightGBM `linear_tree` | 34.5 | 68.1 | −35.9 | +56.7 | −3.0 |
| Ridge, hinges in residual demand (price / TTF) | 30.5 | 38.0 | **+6.5** | +39.3 | +12.6 |

* **The gas-normalised target is the one clear win.** It cuts MAE by 8%, removes the mid-range bias,
  and follows the 28 Oct – 1 Nov step up better (daily means £195–214 against £173–185). This is
  recommendation 2 of the October review, which hasn't been built yet.
* **Models that extrapolate can reach the spikes, but they also call spikes that never happen.** The
  hinge model forecast £405 for 9 Nov (actual £410) and £414 for 6 Nov (£317). It also forecast £438
  for 31 Oct (£256) and £430 for 8 Nov (£265). With about 13 tight slots to learn from, it can't tell
  a real scarcity evening from a merely cold one. Its overall error is worse.
* **None of the variants fixes the 24–25 Oct lows** (+£40–57 low-tail bias, even with continental
  wind). This matches §6.5 of the October review: an extreme combination of conditions, flattened
  by the trees towards the training average.

## 5. Conclusions

1. **The lows and the level shift could have been better.** The October review's two
   recommendations are still the right ones:
   * The continental-wind candidate is live on dev. Its effect can't be separated from this
     window's regime change, so keep scoring it live.
   * The gas-normalised target hasn't been built. It is the strongest result here.
2. **The November peaks were mostly not predictable 2–4 d ahead from this model's 60 days of
   history.**
   * The inputs signalled scarcity correctly, but the window had almost no precedent for it.
   * Getting there needs either more history from tight conditions (last winter's actual
     drivers, gas-normalised so the 2025/26 TTF level doesn't mislead) or a component that
     extrapolates in residual demand.
   * Either one has to be judged on how many false spikes it calls, not just on catching 9 Nov.
3. **At 1–2 d the high tail is still −£23 low.** AgileForecast's peak handling shows that sharper
   peaks are achievable without wrecking the mid-range. It's worth checking how close a
   peak-hour-only uplift tied to residual demand would get.

### Caveats

* Dev-box forecasts stand in for prod. Until 30 Oct dev ran prod's default set on the same data,
  but they are separate runs.
* Four weeks is a short sample, and the model change on 30 Oct overlaps the regime change, so the
  default-versus-candidate split in `score.py` is not a model comparison.
* The external snapshots are pruned (newest plus first of each day), so their run counts differ
  from ours. The comparisons use common slots only.
