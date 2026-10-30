# Forecast review: 14–30 Oct 2026 (wide-weather fortnight)

Branch `analysis/curtailment-imbalance` (from `fix/audit-1-5`). Scripts in
`analysis/forecast_review_2026_10/`; raw data dumps go to its `data/` folder (gitignored).
Forecasts scored are the **dev** box's stored runs (feature set `full_fr`).

Question from the owner: what would improve the model, and in particular should
**wind curtailment** and **regional (Scotland vs England) generation/demand imbalance**
be model inputs?

## 1. What happened (2–4 d horizon, i.e. genuine model output — not GB60 passthrough)

Settled day-ahead daily mean ranged from **£25 (Sun 25 Oct)** to **£213 (Thu 29 Oct)**.

| GB day | actual £ | 2–4 d pred £ | bias | note |
|---|---|---|---|---|
| 20 Oct | 174 | 181 | +6 | |
| 22 Oct | 153 | 174 | +21 | |
| 24 Oct (Sat) | 41 | 129 | **+88** | windy weekend |
| 25 Oct (Sun) | 25 | 100 | **+75** | windy weekend, negatives to −£20 |
| 26 Oct | 134 | 137 | +3 | still windy (~18 GW BM wind) |
| 27 Oct | 113 | 102 | −11 | windiest day (20 GW) yet £113 |
| 28 Oct | 190 | 126 | **−64** | |
| 29 Oct | 213 | 159 | **−54** | calm, cold |
| 30 Oct | 204 | 164 | **−40** | |

By horizon (all target days 14–30 Oct): sd(pred)/sd(actual) 0.72 at 1–2 d, 0.60 at 2–3 d,
0.53 at 3–5 d — the familiar compression ([`MODEL_DYNAMIC_RANGE.md`](MODEL_DYNAMIC_RANGE.md)).

## 2. The inputs were right; the model's response was wrong

Daily means of the features as forecast 2–4 d ahead vs the <1 d forecast were close on
every big-miss day (24 Oct: BM wind 17.0 vs 16.6 GW, embedded 3.8 vs 3.9, demand 25.6 vs
26.6; 29 Oct: BM wind 6.1 vs 4.9). These are **model failures, not weather-forecast failures**.

Two observations that matter for what follows:

* **The same GB wind produced wildly different prices.** 24–25 Oct (~17–18 GW BM wind) settled
  at £25–41; 26–27 Oct (~17–20 GW) settled at £113–134. Weekend demand explains part of it,
  not all of it.
* **Gas rose ~15 % over the fortnight** (TTF feature 86 → 99) while the model under-predicted the
  calm days by £40–64 — tree ensembles cannot extrapolate above the price levels in their 60-day
  training window.

## 3. Curtailment and the Scotland/England split — measured

Derived from Elexon (`fetch_bm.py`): for every wind BMU, curtailment = max(PN − accepted BOA
level, 0) at 5-min resolution. Region is assigned from the unit's **transmission loss factor**,
which is zonal (−0.0305 = North Scotland, −0.0099 = South Scotland; Seagreen, Moray, Beatrice
fall in N Scotland; Dogger Bank/Sofia, Walney, Hornsea in England zones).

**Validation:** PN − derived curtailment vs FUELHH metered wind, half-hourly corr **0.966**
(PN alone 0.905); daily means agree to ~0.3 GW except the 25 Oct clock change.

| GB day | Scot PN GW | Scot curtailed GW | E&W PN GW | E&W curtailed |
|---|---|---|---|---|
| 14 Oct | 9.1 | 6.2 | 7.6 | 0.1 |
| 24 Oct | 7.5 | 3.6 | 10.5 | 0.0 |
| 25 Oct | 8.7 | 4.2 | 9.0 | 0.0 |
| 27 Oct | 10.5 | **7.0** | 10.2 | 0.0 |
| 29 Oct | 3.2 | 0.7 | 1.0 | 0.0 |

Curtailment is **entirely Scottish** (E&W < 0.7 GW on every day) and on windy days takes
**40–67 %** of Scottish wind's notified output.

**Our `bm_wind` feature (NESO WINDFOR) tracks the unconstrained PN, not metered output** — e.g.
27 Oct forecast 20.4 GW vs PN 20.7 GW vs metered 14.0 GW. That is the right quantity for a
day-ahead price model: the day-ahead auction is a single national zone, curtailed Scottish wind
has already sold its output day-ahead and is paid to turn down afterwards in the BM (cost lands
in BSUoS, not in the day-ahead price).

## 4. Continental coupling explains the residual

Daily means (energy-charts day-ahead, FUELHH interconnector net flow, + = GB import):

| day | GB £ | DE € | FR € | NL € | GB net import GW |
|---|---|---|---|---|---|
| 22 Oct | 152 | 153 | 157 | 154 | +5.7 |
| 24 Oct | 39 | 28 | 47 | 32 | +3.1 |
| 25 Oct | 25 | 14 | 27 | 26 | +2.6 |
| 27 Oct | 113 | 159 | 166 | 162 | −0.8 |
| 28 Oct | 190 | 250 | 253 | 251 | −0.3 |
| 29 Oct | 213 | 315 | 267 | 312 | −0.6 |

* 24–25 Oct: the **whole of NW Europe** was windy — the continent collapsed first, GB still
  imported. GB price follows continental surplus, not just GB surplus.
* 27 Oct: GB's windiest day, but the continent was firm (~€160) so GB exported and held £113.
* 28–30 Oct: a continental scarcity spike (DE €250–315) dragged GB up to £190–213 on only
  moderate GB tightness.

The model's only continental signal is `fr_wind`/`fr_rad`: 10 m wind and radiation at a
**single point in central France** (47 N 2 E, `config/utils.py:197`) — not where the
continental wind fleet is (north Germany, NL, Denmark, the North Sea).

## 5. What the dev model's errors line up with

`residual_vs_drivers.py`: dev model 2–4 d bias (pred − actual) against actual drivers, 14–30 Oct.
`eu_wind_cf` is a crude capacity-factor proxy — a cubic power curve on Open-Meteo 100 m wind
averaged over seven points in the NW-Europe wind belt (N Germany ×2, German Bight, NL, DK,
Belgian coast, N France).

| correlation with bias | daily (n=17) | half-hourly |
|---|---|---|
| **continental wind (`eu_wind_cf`)** | **+0.79** | **+0.69** |
| GB unconstrained wind (model already sees it) | +0.49 | +0.38 |
| central-France 10 m wind (`fr_wind`, in the model) | — | +0.45 |
| Scottish curtailment | +0.13 | +0.09 |
| Scottish share of GB wind | −0.25 | −0.18 |

The model is too high when the continent is windy and too low when it is calm (24–25 Oct:
`eu_wind_cf` 0.78–0.90, bias +£88/+£75; 28–30 Oct: 0.05–0.30, bias −£40 to −£64).
Curtailment and the regional split carry almost none of the remaining error.

## 6. Structural test: which information explains price? (Jun–Oct, actual drivers)

`structural_test.py`: LightGBM on **actual** half-hourly drivers (≈4,500 slots, 1 Jun–30 Oct), so
forecast error is taken out and only the information content is tested. Base = transmission
demand, actual solar (B1630), nuclear, time, weekend, TTF. Each candidate is added on its own.
`fwd` mimics production (train on the prior 60 days, predict the next 7, rolling from 1 Aug);
`lowo` is leave-one-week-out. Tails = bottom/top decile of actual price.

| feature set | fwd RMSE | fwd low-10 % RMSE | fwd high-10 % bias | lowo RMSE |
|---|---|---|---|---|
| GB + **unconstrained** wind (PN) | 29.71 | 52.5 | −22.7 | 31.73 |
| GB + **metered** (post-curtailment) wind | 30.48 | 54.9 | −19.6 | 32.60 |
| + Scottish curtailment | 29.32 | 54.5 | −22.1 | 31.60 |
| + Scottish share of wind | 29.72 | 54.0 | −22.6 | 31.40 |
| + central-France 10 m wind (today's `fr_wind`) | 29.15 | 52.2 | −21.3 | 31.02 |
| + **continental wind belt** (`eu_wind_cf`) | 28.13 | **46.9** | −18.0 | 29.04 |
| + continental + Scottish share + curtailment | 27.73 | 50.9 | −18.0 | 29.27 |
| GB + PN, target **price / gas** | 28.67 | 51.8 | −14.6 | 26.77 |
| **continental wind + price/gas target** | **26.66** | 48.6 | **−7.0** | **22.97** |

Findings:

1. **Curtailment: do not add.** Scottish curtailment moves RMSE by ≤0.4 and makes the low tail
   *worse*. It is also not knowable in advance (it is decided in the BM after the day-ahead
   auction) — any forecastable version would be a proxy for "lots of Scottish wind", which the
   wind feature already carries.
2. **Regional split: do not add.** Scottish share of wind is inert (29.71 → 29.72). This is what the
   market design predicts: the day-ahead auction is one national zone (zonal pricing was
   rejected in the July 2025 REMA decision), so it prices *unconstrained* national wind and
   resolves the Scotland/England imbalance afterwards at NESO's cost. Using **metered** wind is
   measurably worse than PN — so the one thing to protect is that the wind feature stays the
   unconstrained forecast (WINDFOR/PN), never an outturn net of curtailment.
3. **Continental wind: add.** It is the single best new input (−5 % RMSE, −11 % low-tail RMSE) and
   is the variable the dev model's error tracks (§5). The existing `fr_wind` single point captures
   about a third of that gain.
4. **Gas-normalised target: try.** Predicting price/TTF and multiplying back is the largest single
   improvement on the high tail (bias −22.7 → −7.0) because trees cannot extrapolate price level
   when gas rises above the training window (TTF went 45 → 104 Jun → Oct; the fortnight's calm
   days were under-forecast by £40–64). Together with continental wind: fwd RMSE −10 %, lowo −28 %.
5. **What none of this fixes:** 24–25 Oct is still over-forecast by £45–85 even with *actual*
   drivers. The 60 days before contained 315 slots with GB wind >15 GW and `eu_wind_cf` >0.6
   (mean £66) — the regime was not absent, it was milder; that weekend (£31) was a more
   extreme joint state and the tree ensemble regresses to the leaf mean. This is the
   compression problem in [`MODEL_DYNAMIC_RANGE.md`](MODEL_DYNAMIC_RANGE.md), not a missing input.

## 7. Recommendations

In priority order. Per the standing caveat in `MODEL_DYNAMIC_RANGE.md` (offline retrain/replay
has not reproduced production), each should be validated by **live re-scoring of a
single-difference dev candidate** (dev = main + one change), with the scoring rule fixed in
advance and an event budget rather than a calendar stop.

1. **Continental wind feature.** Replace/extend `get_open_meteo_fr_weather` (`config/utils.py:197`)
   with a multi-point 100 m wind fetch over the NW-Europe wind belt (points in `drivers.py`),
   reduced to a power-curve capacity factor. Open-Meteo serves 16-day forecasts for all points in
   one call; backfill from its historical-forecast API.
2. **Gas-normalised target** (`day_ahead / gas_ttf`, rescale at predict time) — a small change in
   `build_training_data` / prediction; addresses rising-gas under-forecast directly.
3. **Do not add curtailment or Scotland/England imbalance features.** Keep the wind input on the
   unconstrained basis.
4. Not tested here, worth a look later: continental weekend/holiday calendar (24–25 Oct was a
   weekend on the continent too), and NO2 hydro (Norway priced well below GB on 27–30 Oct).

### Caveats

* §6 uses actual drivers; in production the continental wind is itself a forecast. At 2–4 d the
  GB wind inputs were accurate this fortnight (§2), so most of the gain should survive, but it
  must be measured live.
* `eu_wind_cf` is a crude proxy (one generic power curve, unweighted points). A capacity-weighted
  version should do better, not worse.
* 17 days of dev-model residuals (§5) is a small sample; §6 is the larger evidence base.

## 8. Can a continental wind feature line up with the 14-day forecast frame? (checked 2026-10-30)

* **Serving horizon:** Open-Meteo 100 m wind for all 7 points in one call reaches 15.3 d
  (`best_match`, `gfs_seamless`) / 14.4 d (`ecmwf_ifs025`). `best_match` switches ICON → GFS at
  ~7.3 d, a step change mid-horizon — **pin one model** (ECMWF, GFS-filled for the last hours, or
  ffill like other features).
* **Horizon-matched training history:** the Previous Runs API serves `wind_speed_100m_previous_dayN`
  for N ≤ 7 only (`previous_day14` is empty). That is sufficient for the current training window,
  which keeps only 22–46 h-ahead rows (`build_training_data`): backfill with `previous_day1/2`
  is a faithful reconstruction of what a run would have seen. (Better than `fr_wind`, whose
  backfill used archive actuals.)
* **Gap:** no 8–14 d vintages exist, so if the training window is ever widened (the reverted
  `TRAIN_HORIZON_DAYS` trial) the history must come from storing the feature per `ForecastData`
  row going forward.
