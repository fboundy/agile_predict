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

> Superseded by §13–14. Kept as written on the first pass; §8 and §12 correct two of its claims.

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

---

# Part 2: deeper review (11 Nov, after the owner asked for more)

Part 1 looked only at four weeks of stored forecasts and one small offline test. Part 2 adds three
years of actuals, so the questions can be asked against three winters, not 13 half-hours. Each
experiment below was written up when it finished.

Data built for this part (all in the gitignored `data/`):

* `panel.py`, `panel_gas.py`, `panel_tail.py` → `panel.pkl`: half-hourly settled price, NESO
  generation mix, TTF and continental wind, **1 Jul 2023 – 11 Nov 2026** (59,010 rows). The NESO mix
  lags a week, so 5–11 Nov comes from Elexon FUELHH (checked on a 3-week overlap: residual demand
  corr 0.988, mean difference 0.1 GW).
* `opmr_hist.py` → `opmr.pkl`: NESO's OPMR margin forecast for every target day **as published 2, 3,
  4 and 7 days before** (history back to 2021 is on the NESO API).
* "Residual demand" in Part 2 = generation − wind − solar − nuclear, i.e. what gas, imports,
  biomass, hydro and storage had to supply. It runs about 5 GW below the Part 1 figure, which
  didn't subtract nuclear or solar.

**Correction to Part 1, §2 and conclusion 2.** Part 1 said last winter's tight days couldn't be
tested because the dev DB keeps forecast inputs only from 12 Sep. That was wrong: the actuals are
public, and the `History` table already holds them to Aug 2026.

## 6. E1: the price curve against residual demand, three winters (`stack.py`)

Median price ÷ TTF by residual-demand band (GW):

| winter | mean TTF | 15–20 | 20–25 | 25–30 | 30–33 | 33–36 | ≥36 | slots ≥36 |
|---|---|---|---|---|---|---|---|---|
| 2023/24 | €36 | 2.24 | 2.39 | 2.65 | 2.97 | 3.31 | 3.51 | 45 |
| 2024/25 | €45 | 2.07 | 2.19 | 2.42 | 2.72 | 3.09 | 5.71 | 136 |
| 2025/26 | €35 | 2.49 | 2.68 | 2.95 | 3.45 | 4.20 | 8.04 | 24 |
| 2026/27 so far | €88 | 2.00 | 2.17 | 2.48 | 3.28 | 3.69 | 4.98 | 3 |

* **Below about 33 GW the curve is the same shape every winter once gas is divided out**, within
  ±12%. So history from earlier winters is usable for the bulk of the curve.
* **The ratio is not quite the right normalisation.** Within a band, price ≈ £23 + b × TTF, and the
  £23 doesn't scale with gas (carbon cost and other fixed adders). Dividing by **(TTF + 12)** removes
  most of the remaining year-to-year spread (20–25 GW band: 1.65–1.93 against 2.17–2.68 for the
  plain ratio). Part 1's "price ÷ TTF" candidate should be "price ÷ (TTF + c)".
* **Above about 33 GW the curve is not stable.** The same residual demand gave £101 in 2023/24, £270
  in 2024/25, £223 in 2025/26 and £410 on 9 Nov. Residual demand alone doesn't set the scarcity
  price: what matters is how much plant is available on the day.
* 24 Jun 2026 shows the same thing from the other side: £381 at only 27 GW residual demand
  (price ÷ TTF = 9.3, the second-highest in three years), in the summer maintenance season.
* 9 Nov in context: residual demand peaked at 37.1 GW. Each of the last three winters went higher
  (39.4–39.5 GW). **It was tight for November, but not unprecedented.**

## 7. E2: NESO's own margin forecast flags the spike days (`margin.py`)

NESO publishes a daily "National Surplus" for each of the next 14 days (available plant, including
forecast wind, minus peak demand and reserve). The app already downloads it and stores it on every
forecast row as `opmr_national_surplus`. **No feature set uses it.** The model gets
`dispatchable_capacity` instead, which is the same figure without demand subtracted.

NESO's surplus for 9 Nov, by publication date: 11.0 GW (26 Oct), 8.0 (1 Nov), 6.4 (4 Nov),
5.3 (5 Nov), **3.4 GW (7 Nov, two days ahead)**. For 6 Nov it was 3.9 GW two days ahead. Those were
the two lowest values of the four weeks. The model's `dispatchable_capacity` for the same days was
49.9 and 52.9 GW, in the middle of its range.

Across 1,222 days since July 2023, taking a "spike day" as the top 5% of daily peak ÷ (TTF + 12):

| signal known in advance | AUC for spike days |
|---|---|
| **NESO surplus, 2 days ahead** | **0.91** |
| NESO surplus, 3 / 4 / 7 days ahead | 0.88 / 0.88 / 0.80 |
| `dispatchable_capacity` (what the model has) | 0.60 |
| *actual* peak residual demand (not knowable in advance) | 0.83 |

Winter days by 2-day-ahead surplus:

| surplus | days | median peak ÷ (TTF+12) | 90th pct | share setting a new 60-day high |
|---|---|---|---|---|
| under 2 GW | 55 | 2.98 | 5.49 | 15% |
| 2–4 GW | 59 | 2.65 | 3.49 | 3% |
| 4–6 GW | 90 | 2.46 | 2.94 | 1% |
| 6–8 GW | 89 | 2.33 | 2.69 | 0% |
| 8–12 GW | 175 | 2.26 | 2.50 | 0% |
| over 12 GW | 115 | 2.10 | 2.41 | 0% |

* **A published two-day-ahead number ranks spike days better than the actual residual demand
  does**, because it includes plant availability. It is the scarcity signal the model is missing.
* It flags the day, not the size. 9 Nov's peak (4.24 normalised) was above the 90th percentile for
  its surplus band (3.49).
* The app computes `dispatchable_capacity` as availability + interconnector import − reserve −
  constrained plant. NESO's own surplus is availability − peak demand − reserve (checked on five
  records): interconnectors and constrained plant aren't in it. The note in the data-sources memory
  had this wrong.

**Incidental:** Yahoo's TTF history endpoint returned only the latest quote on 11 Nov, for any date
range. The live run only needs the latest value, so forecasts aren't affected, but
`get_gas_ttf_history` can't backfill at the moment. I haven't checked how long this has been so.

## 8. E3: what the stored forecasts got wrong, and how (`forensics.py`)

Stored dev forecasts for target dates from 14 Oct, scored against outturn (£/MWh). Dev ran the
production feature set until 30 Oct 17:33 UTC and `full_fr_eu` (adds continental wind) after.

**(a) The forecast is flat, but it is not mis-scaled.** Regress actual on forecast; a slope above 1
means stretching the forecast would help.

| horizon | slope | sd(forecast) ÷ sd(actual) | corr | RMSE | RMSE after the best in-sample stretch |
|---|---|---|---|---|---|
| 1–2 d | 1.07 | 0.81 | 0.87 | 33.5 | 33.2 |
| 2–4 d | 1.09 | 0.70 | 0.76 | 44.3 | 43.3 |
| 4–7 d | 1.04 | 0.64 | 0.67 | 49.2 | 46.9 |
| 7–15 d | 1.18 | 0.50 | 0.59 | 57.4 | 53.0 |

The forecast has 50–80% of the actual spread, yet the slope is close to 1. That is what an honest
forecast with limited information looks like: it hedges towards the middle by about the right
amount. Stretching it (the affine fix from the September trial) buys 1 £/MWh at 2–4 d even with the
coefficient fitted on the same data. **Part 1's "tail bias" tables overstate the fault**: any
imperfect forecast is biased low on the highest actuals. The fix has to be more information about
which days are extreme, not more amplitude.

**(b) Three-quarters of the error is the day's level, not its shape.** 2–4 d MSE 1966 = level 1448
(74%) + within-day shape 518. The shape error is concentrated at 07:00 (−21) and 15:00–19:00
(−19 to −25): the peaks are under-called on average, not only on spike days.

**(c) Errors persist for a day, not for three.** Autocorrelation of the daily level error: 0.57 at
1 day, 0.25 at 2, 0.11 at 3, negative after. Carrying the last known error forward cuts daily RMSE
34.1 → 27.4 at a 1-day lag, but only 34.4 → 33.0 at 2 days and 34.8 → 34.4 at 3 (in-sample, so
upper bounds). A 2–4 d forecast can only use errors that are 2+ days old, so **an error-correction
term is not worth building**.

**(d) What the errors line up with.** Correlation of the 2–4 d error with stored inputs
(all slots / daily mean / evening peak):

| input | all slots | daily | evening peak |
|---|---|---|---|
| continental wind (`eu_wind_cf`) | 0.57 | 0.68 | 0.46 |
| embedded wind | 0.33 | 0.43 | 0.25 |
| NESO surplus (stored, unused) | 0.33 | 0.41 | 0.34 |
| residual demand, derived | −0.33 | −0.36 | −0.30 |
| `bm_wind` (stored, unused) | 0.28 | 0.37 | 0.27 |
| demand | −0.23 | −0.22 | −0.26 |
| gas price | −0.15 | −0.19 | −0.05 |
| `dispatchable_capacity` | −0.03 | −0.03 | −0.10 |

Positive = forecast too high when the input is high. The model under-reacts to wind in both
directions, and continental wind is the strongest single line.

Split by era (daily means):

| era | days | continental wind | embedded wind | NESO surplus |
|---|---|---|---|---|
| production set (to 30 Oct) | 21 | +0.83 | +0.52 | +0.56 |
| `full_fr_eu` (from 30 Oct) | 11 | +0.21 | −0.01 | −0.02 |

* Adding continental wind removed most of the wind-shaped error, on 11 days. That is the first
  live evidence for the October recommendation, and it is thin.
* **Caution for the NESO surplus:** its correlation with the daily error also vanished once
  continental wind was in. Generator Availability includes forecast wind, so part of what the
  surplus knows is wind the model can already see. E4 has to test it *on top of* the wind features.

**(e) But the peaks on tight days were still missed.** Evening-peak (16–19) error at 2–4 d by NESO
surplus:

| surplus | slots | days | mean error | actual | forecast |
|---|---|---|---|---|---|
| under 4 GW | 13 | 1 | −60 | 309 | 249 |
| 4–6 GW | 181 | 6 | −58 | 315 | 258 |
| 6–8 GW | 275 | 12 | −9 | 257 | 248 |
| 8–10 GW | 556 | 19 | −24 | 239 | 214 |
| over 10 GW | 521 | 15 | −13 | 189 | 176 |

Seven days under 6 GW, and the forecast barely moved between the 6–8 and 4–6 GW bands while the
outturn rose £58. These include 6 and 9 Nov, which fall in the `full_fr_eu` era: continental wind
fixed the daily level and not the scarcity peak.

## 9. E5: the two tails fail for different reasons (`tails.py`)

**(a) High tail: is the spike outside what a 60-day window has seen?** Days whose peak beat the
highest price of the previous N days (Jul 2023 – Nov 2026):

| window | days tested | above the window's max | of which winter | median excess |
|---|---|---|---|---|
| 60 d | 1,169 | 29 (2.5%) | 11 | 7% |
| 365 d | 864 | 6 (0.7%) | 5 | 11% |
| 730 d | 499 | 2 (0.4%) | 1 | 8% |

* Gas-normalising the target doesn't change these counts (26 / 6 / 2). Normalising fixes the
  *level* between years, not the range within a window.
* Of the 62 spike days (top 5% of gas-normalised peak), only 17 beat the prior 60-day maximum; the
  median spike day peaked at 0.91 of it. So a tree model *could* have reached most spikes. What a
  60-day window lacks is not one high price but enough rows like the spike day to learn from:
  before 9 Nov it held 16 slots at £300+, 6 with residual demand over 33 GW and none over 36 GW.
* A one-year window cuts out-of-range days by four-fifths.

**(b) Low tail: a cliff in residual demand.** 148 of 1,229 days had a zero or negative slot. How
well each daily figure ranks them (AUC):

| signal | AUC |
|---|---|
| the day's minimum residual demand (actual) | 0.96 |
| NESO minimum demand forecast, 2 d ahead | 0.84 |
| GB wind, daily mean | 0.84 |
| NESO surplus, 2 d ahead | 0.79 |
| continental wind, daily mean | 0.79 |
| weekend | 0.63 |
| interconnector export capacity, 2 d ahead | 0.56 |

Share of days with a slot at or below zero, by the day's minimum residual demand (the 981 days
since Mar 2024 that have continental wind data):

| min residual demand | share | days |
|---|---|---|
| under 5 GW | 85% | 46 |
| 5–8 GW | 31% | 197 |
| 8–11 GW | 2% | 313 |
| over 11 GW | 0% | 425 |

* Within a residual-demand band, continental wind adds nothing visible (5–8 GW: 38% / 15% / 38%
  across its terciles). Its value to the model is as a stand-in for what the model can't see.
* The model can't see residual demand. It has embedded wind and demand as separate inputs, and
  **transmission wind (`bm_wind`) is not a feature at all**. `bm_wind` as a raw feature lost its
  live trial; residual demand as one explicit feature has not been tried.

**(c) On 24–25 Oct the model ranked the low slots and missed their level.** 380 slots settled at
£20 or less (mean £3). At 2–4 d the forecast ranked them almost perfectly (AUC 0.95) but averaged
£88, and was under £40 on 1% of them. The training window was not short of examples: the 60 days
before 24 Oct held 173 slots at £20 or less and 307 with residual demand under 8 GW. All of this was
under the production feature set; there has been no low day since continental wind went in.

**Reading the two together:** the high tail is short of *examples* (early-winter tight days aren't in
a 60-day window), the low tail is short of a *feature* (the examples are there, the model can't
place the cliff).

## 10. E6: the published band misses the same slots (`band.py`)

Region G low/high band against outturn, target dates from 14 Oct. The band is the 10th–90th
percentile of holdout errors by horizon, widened by the weather ensemble, so it should hold at
least 80%.

| horizon | inside | outturn below | outturn above | mean width (p/kWh) |
|---|---|---|---|---|
| 1–2 d | 49% | 20% | 30% | 7.0 |
| 2–4 d | 65% | 9% | 26% | 19.2 |
| 4–7 d | 60% | 7% | 33% | 19.5 |
| 7–15 d | 50% | 8% | 41% | 18.6 |

At 2–4 d:

| slots | inside | below | above | missed by (p/kWh) |
|---|---|---|---|---|
| evening peak, NESO surplus under 6 GW (6 days) | 48% | 0% | 52% | 11.8 |
| evening peak, surplus 6–8 GW | 77% | 0% | 23% | 2.1 |
| outturn at or under 5p | 19% | 81% | 0% | 6.0 |
| outturn 15–35p | 84% | 4% | 12% | — |

* The band is about right in the middle and fails at both ends, because its width depends on the
  horizon and not on the state of the system. Its width on tight evenings (16.9p) is *narrower*
  than average.
* Even if the central forecast can't call the size of a spike (§7), the band could say "this
  evening is at risk". At present it doesn't.
* Four weeks, one regime: the overall lean (above more often than below) is partly this window's
  level shift.

## 11. E7: a scarcity floor, scored on the forecasts actually issued (`uplift.py`)

The cheapest way to use §7, and the one test here that isn't an offline replay. Fit the typical
gas-normalised daily peak for a given NESO surplus on the 556 winter days *before* 14 Oct. Then, for
every stored 2–4 d forecast since, lift the 16:00–19:00 slots so the day's peak is at least that
figure for the surplus stored with the forecast. No retraining; forecast-time inputs only.

| floor | all-slot RMSE | evening RMSE | evening bias | peak error, surplus < 6 GW (20 run-days) | peak error, the rest (132) |
|---|---|---|---|---|---|
| none (as issued) | 44.3 | 49.1 | −22 | −68 | −20 |
| mean peak for the surplus | 44.0 | 46.3 | 0 | −54 | +5 |
| median | 43.9 | 46.1 | −1 | −58 | +4 |
| 75th percentile | 44.5 | 50.6 | +11 | −46 | +17 |

* **It works, but not as a spike catcher.** It lifted 84% of ordinary days: the issued evening
  peaks were below what three winters say is typical for the margin, by about £20. Removing that
  takes evening RMSE from 49 to 46 and the ordinary-day peak error from 38 to 28 (MAE).
* **On the tight days it recovers a fifth of the miss** (−68 → −54). Pushing the floor to the 75th
  percentile gets a third and costs more elsewhere than it gains.
* **The surplus for 9 Nov only collapsed at two days out.** The 2–4 d runs carried 5.5 GW on
  average, not the 3.4 GW published on 7 Nov, so the floor for 9 Nov was £262 against a £410
  outturn. §7's AUC of 0.91 is for the 2-day figure; at 3–4 days the flag is weaker (0.88) and for
  this event it was late.
* So the evening under-call has two parts: a general one of about £20 that history can remove, and
  the spike size itself, which the margin flags but does not size.

## 12. E4: which change to the model form is worth most? (`modelform.py`)

Walk-forward over two years (3 Oct 2024 – 11 Nov 2026, 36,930 slots, 110 weekly refits). Train on
everything before T, predict days T+2 to T+9. 25 variants = training window × target × features.

**Read this first.** The inputs are *actual* drivers (perfect weather and demand foresight) and
the model is LightGBM alone. Earlier offline replays did not reproduce production
(`MODEL_DYNAMIC_RANGE.md`), so the absolute numbers mean nothing and a win here is only a reason to
test live. Gains from anything that has to be forecast (transmission wind above all) will shrink.

RMSE, £/MWh. "norm" = target is price ÷ (TTF + 12). Long windows carry a recency weight (a
year-old row counts half).

| features | 60 d raw | 60 d norm | 365 d raw | 365 d norm | all raw | all norm |
|---|---|---|---|---|---|---|
| base (≈ production + continental wind) | **21.1** | 21.1 | 19.4 | 19.0 | 19.0 | 18.7 |
| + NESO surplus (2 d lead) | 20.6 | 20.6 | 19.1 | 18.3 | 18.5 | 17.9 |
| + residual demand, one feature | 19.2 | | | | | 17.7 |
| + transmission wind, raw feature | 20.2 | | | | | 17.8 |
| + surplus, residual demand, headroom | 18.8 | 19.2 | 17.3 | 17.1 | 17.2 | **16.9** |
| base without continental wind | 23.1 | | | | | |

The tails, for the variants that matter (reference = 60 d raw base):

| variant | RMSE | weeks won vs reference | low slots caught¹ | spike-day peak error² | spike days caught / false³ | 9 Nov peak (actual 410) |
|---|---|---|---|---|---|---|
| 60 d raw base | 21.1 | | 24% | −76 | 44% / 26 | 251 |
| without continental wind | 23.1 | 24% | 16% | −84 | 31% / 22 | 251 |
| + NESO surplus | 20.6 | 57% | 18% | −67 | 46% / 22 | 259 |
| + residual demand | 19.2 | 78% | 31% | −58 | 56% / 23 | 253 |
| + transmission wind raw | 20.2 | 75% | 24% | −66 | 49% / 24 | 258 |
| all-history raw base | 19.0 | 78% | 37% | −72 | 51% / 13 | 237 |
| all-history norm base | 18.7 | 77% | 40% | −73 | 49% / 19 | 293 |
| all-history norm + residual demand | 17.7 | 84% | 51% | −60 | 56% / 20 | 355 |
| all-history norm + all three | 16.9 | 86% | 45% | −61 | 49% / 24 | 310 |
| 365 d raw + all three | 17.3 | 86% | 40% | −57 | 59% / 13 | 251 |
| average of 60 d and all-history (norm, all three) | 17.3 | 85% | 35% | −58 | 51% / 21 | 267 |

¹ share of slots at £10 or less forecast at £10 or less (1,475 slots). ² mean error of the day's
peak on the 39 spike days (top 5% of gas-normalised peak). ³ share of spike days forecast above the
spike threshold / ordinary days forecast above it (of 730).

* **A longer training window is the largest single gain: −8% (one year), −10% (all history)**, and
  it wins about four weeks in five. It is the best thing here for the low tail (low slots caught
  24% → 37–40%). This is not what the September trial tested: that widened the *horizons* trained
  on (`TRAIN_HORIZON_DAYS`), not the calendar window (`max_days`).
* **Residual demand as one explicit feature: −9% on a 60-day window.** The same information as a
  raw `bm_wind` column gets −4%. That matches `bm_wind` losing its live trial: 60 days is too little
  for trees to learn demand − wind − nuclear from separate columns. With all history the two forms
  are equal (17.7 vs 17.8).
* **NESO surplus as a feature: −2.5% (60 d) to −4.5% (all history).** It works where §7 said it
  would (spike-day peak error −76 → −67; 8 Jan 2025 peak 229 → 278 against 318) but far less than
  an AUC of 0.91 suggests, because a model that sees the drivers already knows most of it. A 4-day
  lead scores the same overall (20.6) and worse on spikes.
* **The gas-normalised target is neutral on its own (21.1 → 21.1)** and matters only with a long
  window: this autumn's RMSE 35.4 → 31.7, 9 Nov peak 237 → 293. Part 1 called it "the strongest
  result here"; over two years it is the companion to a long window, not a fix by itself.
* **Averaging a short and a long window is no better than the long one.** Dropped.
* **Continental wind is worth −9% over two years.** The first multi-season evidence for the
  candidate now on dev.
* **Spike size stays out of reach.** No variant catches more than 59% of spike days, and the best
  cuts the spike-day peak miss from £76 to £55–60. No variant got above £167 for 24 Jun 2026 (£381).
* The ordering among the top four or five variants is within noise (25 variants, one history).
  The factor effects above are not.

**The catch for a longer window:** `update.py` deletes stored forecasts older than `max_days` on
every run, so production holds 60 days and nothing earlier exists to train on. `max_days` is
already a run option, but raising it only lets the window grow from today.

## 13. Answer: could the model have predicted these better?

| what went wrong | could it have been better? | evidence |
|---|---|---|
| **The lows of 24–25 Oct** (forecast £88 on slots that settled at £3) | **Yes, and this is the most fixable.** The model ranked the slots correctly and had 173 comparable slots in its window. It can't see residual demand, where the cliff is. | §9b–c, §12 |
| **Evening peaks about £20 low on ordinary days** | **Yes.** Three winters of history say what an evening peak looks like for a given margin; a 60-day window in autumn doesn't. | §11, §12 |
| **6 Nov (£317, forecast £250)** | **Partly.** NESO's surplus flagged it from four days out (3.6–4.2 GW). | §7, §11 |
| **9 Nov (£410, forecast £257)** | **About half.** The surplus only collapsed two days out (5.3 → 3.4 GW). With perfect-foresight drivers the better model forms reach £310–360. | §11, §12 |
| **The size of the largest spikes** | **No.** Above about 36 GW of residual demand the price has ranged from £101 to £410 between winters. Nothing tested catches more than 59% of spike days. | §6, §12 |
| **The band** | **Yes.** It held 48% of tight evenings and 19% of the low slots, against a nominal 80%. | §10 |

Two corrections to Part 1:

* Its tail-bias tables overstated the fault. The forecast is flat but close to calibrated (§8a), so
  the cure is information, not amplitude.
* The gas-normalised target is not a fix on its own (§12). It is what makes a long window usable.

## 14. Solutions to test, in order

Ranked by evidence ÷ cost. Offline figures are from §12 and will shrink live.

| # | candidate | offline | cost | main risk |
|---|---|---|---|---|
| 1 | **Residual demand as one feature**: `demand − bm_wind − nuclear` (`demand` is NESO's ND, already net of embedded generation) | −9% RMSE on a 60-day window; low slots caught 24% → 31%; spike-day peak miss £76 → £58 | One derived column; every input is already stored on every row, so it trains from the existing 60 days at once | `bm_wind` forecast error at 2–4 d. The raw feature lost its live trial; §12 says the explicit form is worth twice as much on 60 days |
| 2 | **NESO surplus as a feature** (`opmr_national_surplus`, already stored) | −2.5% RMSE; evening peaks on tight days; +£50 on 8 Jan 2025 | Add to a feature set | The stored column was null 18 Sep – 5 Oct, so run `backfill_opmr` first. Trained at 1–2 d lead, applied at up to 14 d, where NESO revises by ±2 GW |
| 4 | **Make the band depend on the state of the system**: error quantiles by surplus / residual-demand band as well as horizon | Band held 48% of tight evenings, 19% of low slots | Change to `compute_horizon_quantiles` | None to the central forecast. Few events per band in 60 days of holdout |
| 5 | **Evening floor from history** (§11) | Scored on issued forecasts: evening RMSE 49 → 46, evening bias −22 → 0 | A lookup table refreshed from NESO history | A patch; use it as the benchmark any candidate must beat |
| 6 | **Continental wind** (on dev since 30 Oct) | −9% over two years; the wind-shaped error is gone on dev (§8) | None | Already being scored; decide on its own rule |

Not worth testing, with the reason:

* **A longer training window** (was candidate 3: keep history, train on a year with price ÷ (TTF + 12)):
  ruled out by the owner on 11 Nov 2026 as impractical. It was the best offline result for the low
  tail (§12), so the tails now rest on candidates 1 and 2 alone. Numbering above is unchanged.
* **Stretching the forecast** (affine recalibration, heavier tail weights): slope is 1.04–1.18, best
  case 1 £/MWh at 2–4 d (§8a).
* **Carrying the last error forward**: gone by the time a 2–4 d forecast can use it (§8c).
* **Averaging short and long windows**: no better than the long one (§12).
* **Using the surplus to size a spike**: it flags the day, not the size (§7, §11).
* **Curtailment, Scotland/England split**: October review.

### How to test them

The September trial failed to separate candidates because each ran on its own schedule against a
moving market. So:

1. **Shadow-score in one run.** Have the dev update fit the baseline and each candidate on the same
   training rows and the same forecast inputs, and store each candidate's prediction beside the
   baseline's. Differences are then model form and nothing else. Candidates 1, 2 and 1+2 start
   together (§15).
2. **Fix the scoring rule before the first run.** 2–4 d horizon only (0–1 d is GB60 passthrough).
   * Primary: all-slot RMSE, paired by target day.
   * High tail: evening-peak (16–19) MAE on days whose stored surplus is under 6 GW.
   * Low tail: mean forecast on slots that settle at £20 or less.
   * False alarms: ordinary days forecast above £300.
3. **Stop on an event budget, not a date.** At least 10 tight evenings and 5 days with a slot at £20
   or less. The last four weeks produced 7 and 2, so expect six to ten weeks.
4. **Pass** = primary no worse than baseline, both tail measures better on more than half the
   event days, false alarms no higher. Anything else is a fail; no re-scoring on a new metric.
5. Candidate 4 is scored separately, on coverage of the band in the same event sets.

### Caveats on Part 2

* §12 uses actual drivers and LightGBM alone. §11 is the only test on issued forecasts.
* Stored forecasts cover 14 Oct – 11 Nov: seven tight evenings, two low days.
* French wind, radiation and nuclear are in production's feature set and not in the §12 panel.
* The fixed offset of 12 in price ÷ (TTF + 12) was fitted on the same three winters it is tested on.
