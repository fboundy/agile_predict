# Model dynamic range — under-prediction of price extremes

Investigation log. Started 2026-08-16 by Claude at the owner's request, after the
observation from the History page that the model appears to under-predict the
dynamic range of the data. Newest entries appended at the bottom.

Data: the dev SQLite database on the CT (`/srv/agile_predict/db.sqlite3`), which
holds 51 daily forecast runs (2026-06-17 → 2026-08-16) and half-hourly
`PriceHistory` actuals back to 2023.

---

# Claude's view — the observation is real, and it is a tails problem

Appended 2026-08-16 14:58 +01:00. For Codex's review.

## The observation is confirmed, and it is large

Pooling every stored forecast from the last 35 days against actual GB day-ahead
prices, filtered to the **≥2-day horizon** (below that, slots are filled from
GB60 day-ahead data rather than the model, so they measure Nord Pool, not us):

| Horizon | n | sd(pred)/sd(actual) | slope(actual~pred) | r | RMSE |
|---|---|---|---|---|---|
| 0–1d *(GB60, not the model)* | 6 229 | 0.989 | 0.998 | 0.986 | 6.30 |
| 1–2d | 6 003 | 0.756 | 1.186 | 0.897 | 17.29 |
| 2–3d | 5 763 | 0.600 | 1.414 | 0.848 | 22.38 |
| 3–5d | 10 879 | 0.571 | 1.429 | 0.815 | 26.87 |
| 5–7d | 9 942 | 0.590 | 1.348 | 0.795 | 26.06 |
| 7–14d | 26 014 | 0.535 | 1.359 | 0.727 | 30.77 |
| **≥2d combined** | **52 598** | **0.559** | **1.377** | **0.769** | **28.30** |

The 0–1d row is a useful control: where the pipeline is passing through GB60
prices, the dynamic range is essentially perfect (0.989). The compression is
specific to the model.

## It is not merely the shrinkage that regression is supposed to do

This is the distinction that decides whether there is anything to fix, so it is
worth being precise. A minimum-MSE conditional mean *should* be under-dispersed:
for an optimally calibrated predictor, `sd(pred)/sd(actual) = r` exactly, and the
regression `actual ~ pred` has slope exactly **1.000**. Shrinkage is not a defect;
predicting the mean when you are uncertain is correct.

Measured on the ≥2d set:

```text
observed sd ratio          0.559
MSE-optimal sd ratio (= r) 0.769
observed / optimal         0.726     ← shrinking ~27% harder than optimal
slope(actual ~ pred)       1.377     ← should be 1.000
```

A slope of 1.377 means the raw prediction is **not** the MSE-optimal linear
function of itself, and by construction an affine correction must reduce squared
error. So this is recoverable error, not the unavoidable cost of uncertainty.

## The range is missing almost entirely in the tails

The middle of the distribution is well calibrated. The damage is at the edges,
and it is asymmetric:

| Percentile | actual | predicted | miss |
|---|---|---|---|
| 1 | −21.65 | 46.53 | **+68.18** |
| 5 | 15.55 | 70.31 | **+54.76** |
| 10 | 65.60 | 82.66 | +17.06 |
| 25 | 98.20 | 97.53 | −0.67 |
| 50 | 115.15 | 111.43 | −3.72 |
| 75 | 132.65 | 125.55 | −7.10 |
| 90 | 151.30 | 141.07 | −10.23 |
| 95 | 165.15 | 150.62 | −14.53 |
| 99 | 213.77 | 165.02 | **−48.75** |

By outcome regime:

| Actual regime | n | mean actual | mean predicted | bias |
|---|---|---|---|---|
| < 0 | 1 901 | −17.76 | +71.50 | **+89.26** |
| 0–50 | 2 024 | 26.89 | 75.51 | **+48.62** |
| 50–150 | 42 845 | 112.10 | 110.05 | −2.05 |
| 150–250 | 5 620 | 168.62 | 144.42 | −24.20 |
| ≥ 250 | 208 | 291.79 | 165.53 | **−126.26** |

**The model effectively never predicts a negative price.** 1 901 slots — 3.6 % of
the evaluated month — settled below zero, and the model's average prediction for
them was +71.50 £/MWh. Given that negative and near-zero prices are precisely the
periods a user of this site most wants to be told about, I would treat this as the
headline defect rather than the aggregate sd ratio.

The mid-range bias of −2.05 over 42 845 slots says the model is well behaved where
most of the data is. Nothing here suggests a broken model; it suggests a model
fitted to the middle.

## Affine recalibration: real, modest, and a patch

Fitting slope/intercept on the first half of the window and applying it to the
second (so the correction is genuinely out of sample):

```text
fitted:  slope 1.338  intercept −35.65
test RMSE     30.20 → 28.70   (−5.0%)
test MAE      19.65 → 19.72   (+0.4%)
test sd ratio 0.538 → 0.720
```

A 5 % RMSE reduction for a two-parameter post-hoc correction is real, and it
restores most of the missing dispersion. But MAE gets marginally worse, which is
the signature of what it actually does: it buys the tails at a small cost to the
middle. I mention it mainly because it establishes an upper bound on what pure
recalibration can achieve, and because it is the fallback if the root-cause work
below does not pan out.

## What I tested, including what failed

Walk-forward CV mirroring the production harness (21-day train / 3-day test,
5 folds, the same three-model ensemble at reduced iteration counts), evaluated on
the 1–3 day horizon as production does. Weights as production sets them,
`max(1, |z|)`.

| Feature set | score | wRMSE | sd ratio | slope | tail RMSE |
|---|---|---|---|---|---|
| baseline (`_BASE`) | 15.726 | 18.155 | 0.895 | 1.049 | 27.636 |
| + `residual_load` | 16.028 | 18.498 | 0.893 | 1.050 | 28.133 |
| + `renew_share` | 16.216 | 18.735 | 0.886 | 1.057 | 28.819 |
| + `cap_margin` | 15.595 | 18.007 | 0.897 | 1.048 | 27.666 |
| + `residual_load` + `renew_share` | 16.204 | 18.738 | 0.892 | 1.049 | 28.661 |
| + all three | 15.818 | 18.275 | 0.899 | 1.042 | 27.708 |
| + `residual_load` × `peak` | 16.324 | 18.825 | 0.890 | 1.050 | 28.726 |
| **+ `bm_wind`** | **15.320** | **17.718** | **0.905** | **1.043** | **26.449** |

**My main hypothesis failed.** I expected `residual_load` (demand − solar −
emb_wind) and renewable share to help, because residual load is the standard
driver of negative prices in electricity price modelling and a tree has to spend
many splits to approximate a difference of three features. It did not help — it
consistently made things slightly worse. I am reporting that as measured rather
than arguing with it, but I would like Codex's read on whether the harness is
capable of detecting such an effect at all (see limitations).

**`bm_wind` came back as the best single addition**, which contradicts the
recorded finding that it was demoted because `emb_wind` alone beat both together.
That earlier result is from 2026-07-01 on a different window and a different
scoring configuration. I am not claiming the old finding was wrong — I am claiming
it no longer reproduces on the current two months of data and should be re-run
rather than inherited.

## Sample weighting is the real lever, and it is a genuine trade-off

Production uses `sample_weights = max(1.0, |z|)` where `z` is the training target's
z-score. Varying that exponent, on `_BASE` + `bm_wind`, 1–3d:

| weights | plain RMSE | plain MAE | wRMSE | wMAE | sd ratio | slope |
|---|---|---|---|---|---|---|
| none | **15.35** | **10.95** | 19.04 | 13.34 | 0.852 | 1.111 |
| `max(1,\|z\|)` *(current)* | 15.46 | 11.26 | 17.72 | 12.92 | 0.905 | 1.043 |
| `max(1,\|z\|)²` | 16.30 | 11.81 | **17.63** | 12.97 | 0.945 | **0.990** |
| `max(1,\|z\|)³` | 17.60 | 12.50 | 18.23 | 13.20 | 0.986 | 0.941 |
| `1+z²` | 16.43 | 12.03 | 17.89 | 13.24 | 0.945 | 0.990 |

Squaring the current weight is the sweet spot on every calibration measure —
slope moves 1.043 → **0.990**, sd ratio 0.905 → 0.945, tail RMSE (|z|>1.5)
26.45 → 23.43, a **11 % improvement on exactly the slots that are currently
worst**. Low-regime bias (actual < 50) improves monotonically with the exponent:
12.17 → 10.92 → 9.30.

**But it is not free, and I want to be explicit about that rather than quoting
only the metric that flatters it.** Plain RMSE degrades 15.46 → 16.30 (+5.4 %) and
MAE 11.26 → 11.81 (+4.9 %). Unweighted training has the best plain RMSE of all and
the worst calibration. There is no configuration here that wins on everything;
the choice is *what the forecast is for*.

My view is that tail accuracy is worth more than mid-range MAE for this product.
A user checking Agile prices is deciding when to run a load, and the value is
concentrated in correctly flagging the cheap and expensive periods. Being 5 %
better on a slot that was going to be 110 £/MWh either way is worth less than
being 11 % better on the slots that are actually unusual. That is a product
judgement rather than a statistical one, and it is the owner's call — but it is
the assumption behind my recommendation.

## Limitations, stated plainly

**My harness does not reproduce production's absolute compression, and I cannot
yet fully explain the gap.** Production shows sd ratio 0.535 at 7–14d; my harness
shows 1.021 at the same horizon with the same weights. The most likely cause is
that sd ratio is sensitive to the *actual* variance in the evaluation set: my
folds test a 3-day block of forecast runs, so the target slots span a narrow
calendar window with correspondingly narrow true variance, while the production
figure pools 35 days of targets. RMSE, which is not scale-sensitive in that way,
agrees closely between the two (33.64 vs 30.77 at 7–14d), which supports that
explanation.

The consequence: **the sd-ratio and slope columns in my experiment tables are
valid for comparing configurations against each other under an identical
evaluation design, but should not be read as estimates of the production figure.**
The production numbers in the first three sections are direct measurements and do
not have this problem.

I would rather flag this than quietly present two sets of numbers that do not
reconcile.

**Other caveats.** The evaluation window is 51 forecast days in June–August;
summer is exactly when high solar and low demand produce negative prices, so the
negative-price finding may be seasonally amplified. Five folds over 51 days
overlap substantially, so the fold-to-fold results are not independent. And I
have changed one thing inadvertently relative to production: my harness sets
`days_ago = 0` for all rows, whereas production has it varying in training and
fixed at 0 at inference.

## An incidental finding

`ForecastData.day_ahead_extra_trees` is **NULL for every row** in the database.
The column and its `plunge_probability` and `day_ahead_classified` siblings are
declared and stored but never populated, so the per-model comparison I wanted to
run (does averaging three models cause the compression?) is not possible from
stored data. Either they should be populated or they should be dropped; carrying
schema for a diagnostic that does not exist is worse than either.

## Proposals

Ordered by confidence, not by size.

**1. Square the extreme-price sample weight** — `max(1, |z|)` → `max(1, |z|)²`.
One line. Best wRMSE of any configuration tested, slope essentially 1.000, 11 %
better on extreme slots, monotonic improvement in the low-price regime. Costs
~5 % on plain RMSE/MAE. This is the change I would make first.

**2. Re-run the feature experiment rather than inheriting `bm_wind`'s demotion.**
`+bm_wind` was the best set tested, contradicting the stored result. `--force_experiment`
exists for this. I would not hard-code the change on one experiment of mine.

**3. Add tail metrics to the experiment's scoring output.** The current score is
`(wMAE + wRMSE)/2`, which is why a configuration can win while leaving the
negative-price regime 89 £/MWh out. Logging sd ratio, slope and a tail RMSE
alongside the score would have surfaced this without a special investigation, and
costs nothing.

**4. Do not adopt affine recalibration yet.** It works (−5 % RMSE out of sample)
but it is a correction applied on top of a model whose calibration we are about to
change. If proposal 1 lands and slope moves to ~1.0, the recalibration has nothing
left to correct. Revisit only if the root-cause work stalls.

**5. Not recommended: the engineered features.** `residual_load`, `renew_share`,
`cap_margin` and the peak interaction were all neutral or harmful. I would not add
them on theory alone given the measurement.

## What I would ask Codex

1. Is the harness capable of detecting a `residual_load` effect at all, or is the
   narrow-window problem that broke sd ratio also masking feature effects? If the
   latter, the feature conclusions need a wider evaluation design before they can
   be trusted.
2. Is the tails-over-middle trade-off in proposal 1 the right call for this
   product, or should the weighting stay and the tails be addressed another way —
   e.g. a quantile-loss ensemble member, or an explicit negative-price classifier
   feeding the regression?
3. Is there a causal-safe feature that would specifically predict negative prices
   that I have not considered? Interconnector export capability and wind
   curtailment volumes are the two I know of but have not checked for availability
   at forecast time.
