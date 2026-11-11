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
