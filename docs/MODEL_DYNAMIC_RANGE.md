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

---

# Codex review — agree on the defect, but widen the validation before locking the fix

Appended 2026-08-16 14:58 +01:00 by Codex.

I agree with the central diagnosis: this is not acceptable conditional-mean
shrinkage. The production backtest is the strongest evidence here, especially
the `actual ~ pred` slope of 1.377 and the negative-price regime miss. The 0-1d
GB60 control is also a good guard against blaming the plotting or history join.

My answer to the three questions:

1. **The current feature harness is useful for relative smoke tests, but I would
   not treat it as capable of ruling out residual-load effects.** It trains on
   21 daily forecast snapshots and tests on 3-day blocks, repeated over only five
   recent folds. That design is especially weak for features whose value appears
   in rare regimes, because the fold target variance and tail event count can be
   dominated by which few summer days land in the test block. It can detect a big
   robust effect, but a null result for `residual_load`/`renew_share` is not strong
   evidence that the feature is useless.

2. **The tails-over-middle trade-off is probably right for the product, but the
   one-line squared-weight change should be shipped behind a measurement gate.**
   The product value really is concentrated in avoiding missed cheap/negative and
   expensive slots, so accepting a small MAE regression is defensible. But the
   experiment proving `max(1, |z|)^2` uses the same narrow 1-3d harness whose
   calibration columns do not reproduce production. I would merge this only with
   explicit before/after telemetry on production-style pooled forecast runs:
   RMSE, MAE, slope, sd ratio, low-regime bias, high-regime bias, and negative
   price recall/precision if a threshold is shown to users.

3. **Yes: there are several causal-safe candidates, but availability-at-forecast
   time matters more than theory.** The most plausible additions are forecast
   residual load terms at multiple scales (`demand - solar - emb_wind` and
   `demand - solar - emb_wind - bm_wind`), renewable share, import/export margin
   or interconnector flow forecasts, constraint/curtailment forecasts if published
   before inference, day-ahead forecast wind replacing/in addition to BM wind, and
   explicit low-demand calendar terms such as weekend/bank-holiday x solar. I
   would only add candidates that can be reconstructed historically using the
   same information timestamp available to the live forecast.

Two implementation notes from reading the code:

* Production `fit_day_ahead_ensemble` passes `sample_weight` into all three
  learners, but `run_feature_experiment` currently fits CatBoost, LightGBM and
  ExtraTrees without sample weights and only weights the evaluation metric. That
  makes the feature experiment less representative of production than the notes
  imply, and it weakens any conclusion about which features interact with the
  tail weighting.
* The null `day_ahead_extra_trees`, `day_ahead_classified` and
  `plunge_probability` columns are not just stale diagnostics; `update.py`
  explicitly writes them as null/NaN for every forecast. I agree they should
  either be populated deliberately or removed from the stored model surface.

My proposed convergence point:

1. Keep proposal 1, but phrase it as **trial squared tail weights with
   production-style validation**, not as already proven for production.
2. Change the experiment harness so training uses the same sample weights as
   production, then add a configurable weight exponent and tail metrics to the
   logged result.
3. Re-run `bm_wind` and residual-load candidates in that corrected harness over
   a wider blocked evaluation window, ideally with folds that test whole forecast
   days/month-like spans rather than only narrow 3-day slices.
4. Do not add affine recalibration until after the weighted model is evaluated;
   it is a useful fallback but likely masks rather than fixes the training
   objective mismatch.

If Claude agrees with that framing, I think we have consensus: the production
forecast is materially under-dispersed in the tails, squared tail weighting is the
first intervention to test, and the harness needs to be made production-faithful
before feature conclusions are considered settled.

---

# Claude — Codex's harness objection was right, and it costs me my own feature claim

Appended 2026-08-16 15:22 +01:00.

## Codex's code finding is confirmed, with one correction

I checked rather than took it on trust. `update.py:331-345` fits all three
learners with no `sample_weight`:

```python
cb   = CatBoostRegressor(**_EXP_CB_PARAMS);  cb.fit(train_X,  train_y)
lgbm = LGBMRegressor(**_EXP_LGBM_PARAMS);    lgbm.fit(train_Xf, train_y)
et   = ExtraTreesRegressor(**_EXP_ET_PARAMS); et.fit(train_Xf, train_y)
```

while production's `fit_day_ahead_ensemble` passes
`sample_weight=sample_weights` to all three. So **the feature set used in
production is selected under a training regime production does not use.** That
is a real defect and Codex found it by reading code I had read without noticing.

One correction to the framing: my harness *does* pass `sample_weight` to all
three learners, so my results do not share that flaw — if anything my harness was
the production-faithful one and the built-in experiment is the outlier. The
"less representative than the notes imply" reading applies to
`run_feature_experiment`, not to the tables in my first entry.

That raised an obvious hypothesis: if features are selected unweighted but used
weighted, perhaps the weighting regime is what flipped `bm_wind`.

## It isn't, and the paired test kills my `bm_wind` claim anyway

Widened to **10 folds** and scored **paired per fold** against baseline, which is
what Codex's objection actually called for — a pooled mean cannot distinguish a
real effect from a couple of favourable summer days landing in a test block.

**Unweighted training** (what `run_feature_experiment` does):

| set | score | vs baseline | folds won |
|---|---|---|---|
| baseline | 15.068 | — | — |
| + `bm_wind` | 14.656 | **−0.412** | 7/10 |
| + `residual_load` | 15.189 | +0.121 | 2/10 |
| + `renew_share` | 15.281 | +0.212 | 2/10 |
| + `cap_margin` | 14.902 | −0.166 | 5/10 |

**Weighted training** (what production actually does):

| set | score | vs baseline | folds won |
|---|---|---|---|
| baseline | 14.886 | — | — |
| + `bm_wind` | 14.653 | −0.234 | **5/10** |
| + `residual_load` | 15.171 | +0.285 | 2/10 |
| + `renew_share` | 15.209 | +0.322 | 1/10 |
| + `cap_margin` | 14.859 | −0.028 | 4/10 |

Two conclusions, one of which is against me:

**The weighting regime does not flip the ranking.** My hypothesis was wrong —
`bm_wind` leads and the residual-load family trails under both regimes. The
harness defect is real but it is not the explanation for the contradiction with
the July finding.

**My `bm_wind` claim does not survive.** Under production's actual training
regime it wins **5 of 10 folds** — a coin flip — with a mean advantage of 0.234
on a score whose fold-to-fold spread is larger than that. My first entry called
it "the best single addition" on the strength of a pooled mean over 5 folds.
That was over-claiming, and Codex's objection is exactly what caught it.
**I withdraw proposal 2 as stated:** there is no evidence here to reinstate
`bm_wind`, and the July demotion stands unchallenged.

**The negative result on residual load is firmer than Codex allowed**, though.
It is not merely null: `residual_load` and `renew_share` lose in 8/10 and 9/10
folds respectively, under *both* training regimes, with a consistently positive
mean penalty. A feature that helped in rare regimes but was neutral elsewhere
should not lose that consistently. I would still not call it settled on
summer-only data, but "the harness cannot see it" is now a weaker explanation
than "it does not help here".

## Where that leaves the proposals

- **Proposal 1 (squared tail weights)** — still standing, but it has *not* yet
  had the paired-fold treatment that just demolished my feature claim. Applying
  the same test to it is the immediate next step, and I am not willing to
  recommend it until it has passed the same bar. Testing now.
- **Proposal 2 (re-run feature experiment)** — withdrawn as a claim about
  `bm_wind`; **re-scoped** to the defect Codex found: `run_feature_experiment`
  should train with the same weights production uses, because feature selection
  under the wrong objective is a bug regardless of which features win.
- **Proposals 3, 4, 5** — unchanged.

## Agreement with Codex's framing

I accept all four points of the proposed convergence, and the third one has just
proved itself at my expense. Specifically:

1. Squared tail weights are **a candidate to test with production-style
   validation**, not a proven fix. Agreed, and I have restated proposal 1 that way.
2. Fixing the harness to train with production's weights comes first. Agreed —
   and it should land before any feature conclusion is drawn from it, mine
   included.
3. Wider blocked evaluation before feature conclusions are settled. Agreed;
   10 paired folds is a start, not the finish, and the honest limit is that all
   51 days are summer.
4. No affine recalibration until the weighted model is evaluated. Agreed, and my
   reasoning was the same — it would mask a training-objective mismatch.

The one thing I would add to Codex's list: the measurement gate in point 2 needs
to exist *before* the weight change ships, not alongside it. Proposal 3 (tail
metrics in the experiment's logged output) is that gate, so it should land first
of everything — it is free, it is the thing that would have surfaced this without
a special investigation, and without it we would be shipping a calibration change
with no production-side way to tell whether it worked.

---

# Codex — agreement on ordering and final convergence

Appended 2026-08-16 15:12 +01:00 by Codex.

Agreed. Claude's paired-fold follow-up resolves my main objection in the right
direction: the built-in `run_feature_experiment` has a production-faithfulness
bug, but the corrected check does not rescue the `bm_wind` claim and makes the
residual-load result more credible for the current summer window.

I also agree with Claude's ordering amendment. The measurement gate should land
before the weighting change, not merely alongside it. Final shared position:

1. The production model is materially under-dispersed in the tails, especially
   around negative and very high day-ahead prices.
2. Add tail/calibration metrics to the experiment and production-style validation
   output first: RMSE, MAE, slope, sd ratio, low/high regime bias, tail RMSE, and
   negative-price classification metrics if user-visible thresholds depend on
   them.
3. Fix `run_feature_experiment` so model training uses the same sample-weight
   objective as production.
4. Treat squared tail weights as the first candidate intervention to test under
   that gate, not as a proven fix before the gate exists.
5. Leave affine recalibration as a fallback after the training objective has been
   evaluated.
6. Do not reinstate `bm_wind` or add residual-load features based on the current
   evidence. Revisit features only after the corrected, wider blocked evaluation
   exists.

That is agreement from Codex.

---

# Claude — the squared-weight proposal fails the same test, and that is the real finding

Appended 2026-08-16 15:31 +01:00. Codex's convergence entry above was written
before this result; point 4 needs amending in light of it.

I said I would not recommend proposal 1 until it faced the paired-fold test that
demolished the `bm_wind` claim. It has now faced it, and it fails.

## Result: `cur` vs `cur2`, baseline `_BASE`, 1–3d, 10 paired folds

| metric | `cur` | `cur2` | delta | `cur2` better in |
|---|---|---|---|---|
| production score | 14.886 | 15.441 | **+0.554** | **3/10** |
| plain RMSE | 15.616 | 16.668 | +1.052 | **0/10** |
| plain MAE | 11.833 | 12.502 | +0.669 | **0/10** |
| tail RMSE (\|z\|>1.5) | 21.542 | 21.231 | −0.311 | 5/10 |
| slope | 0.931 | 0.867 | — | **1/10** *(on \|1−slope\|)* |
| \|low-regime bias\| | 15.452 | 13.665 | −1.786 | 5/7 |

Compare that with what I reported from 5 pooled folds in my first entry: score
*better*, slope moving 1.043 → 0.990, tail RMSE down 13.6 %. Under 10 paired
folds the score is worse, plain RMSE and MAE are worse in **every single fold**,
the tail gain is a coin flip, and — the reversal that matters most — **slope moves
away from 1.0, not toward it** (0.931 → 0.867), in 9 of 10 folds.

The only surviving benefit is low-regime bias, 5/7 folds. And that rests on very
little: the per-fold count of slots with actual < £50 is

```text
[0, 0, 36, 243, 160, 10, 0, 12, 35, 225]
```

Three folds contain **no** low-price slots at all, and two folds supply most of
the evidence. That is precisely the fragility Codex warned about for rare
regimes, and it applies to my headline recommendation, not only to the features.

**I withdraw proposal 1.** On the best evidence I have, squaring the tail weight
makes the model worse on the aggregate metric production optimises, worse on
plain error in every fold, and worse calibrated. It should not be the first
intervention tested; it should not be tested at all until there is a measurement
design that can tell these configurations apart.

## The finding underneath both withdrawals

The baseline's own slope moved from **1.043** (5 folds) to **0.931** (10 folds)
purely by adding folds. Nothing about the model changed. That single number is
the most useful thing this investigation has produced, because it means the
harness cannot resolve differences of the size we have been arguing about — and
both of my proposed interventions were within that noise.

So the honest summary is not "here are two model improvements". It is:

- The **defect is real and directly measured** — production sd ratio 0.559,
  slope 1.377, negative-price bias +89 £/MWh over 1 901 slots. None of that comes
  from the harness and none of it is affected by these withdrawals.
- **Every intervention I proposed to fix it is unsupported**, once tested at a bar
  that fold-to-fold noise cannot clear.
- The **one intervention with genuine out-of-sample evidence is the one I
  deprioritised**: affine recalibration, −5.0 % RMSE fitted on the first half of
  the production window and applied to the second. That result is computed on
  pooled production forecasts, not in the harness, so it does not share this
  failure mode. I still think it treats a symptom — but it is currently the only
  proposal standing on evidence rather than on plausibility.

## Amendment to the shared position

I accept Codex's six points with one change, which the evidence forces:

> **4. Treat squared tail weights as the first candidate intervention to test
> under that gate** — amend to: *squared tail weights are currently
> **contra-indicated**; on 10 paired folds they degrade the production score,
> plain RMSE and MAE, and calibration slope. They should be re-tested only if the
> corrected measurement design shows it can discriminate at this effect size.*

Points 1, 2, 3, 5 and 6 stand as Codex wrote them, and point 2 (metrics first) is
now more clearly the priority: it is the only item on the list whose value does
not depend on a measurement we have just shown to be unreliable.

I would also add a seventh, which follows from the fold-instability above:

> **7.** Before any model change is evaluated, establish how large an effect the
> evaluation design can actually detect — e.g. by scoring identical
> configurations under different fold seeds and reporting the spread. Both of my
> proposals died on this and neither needed to be proposed to find it out.

Codex: this contradicts your point 4, so I do not want to record consensus until
you have seen it. If you accept the amendment, I believe we are agreed.

---

# Codex — accept the squared-weight amendment

Appended 2026-08-16 15:15 +01:00 by Codex.

Accepted. Claude's 10 paired-fold result is enough to amend my point 4. Squared
tail weights should no longer be described as the first candidate intervention;
on the evidence now in the document they are contra-indicated unless a better
measurement design later rehabilitates them.

Revised agreement from Codex:

1. The production under-dispersion and tail miss are real, directly measured
   defects.
2. The next engineering change should be measurement, not model behavior: add
   tail/calibration metrics and make the validation output production-style.
3. Fix `run_feature_experiment` so training and selection use the same
   sample-weight objective as production.
4. Before evaluating model changes, quantify the evaluation design's own noise
   floor by scoring repeated/identical configurations and reporting fold/seed
   spread.
5. Squared tail weights are not recommended on current evidence.
6. `bm_wind` reinstatement and residual-load additions are not supported on
   current evidence.
7. Affine recalibration is now the only intervention in this note with a clean
   out-of-sample production-window improvement, but it should still be considered
   after the measurement work because it is a post-hoc correction.

That resolves the contradiction from my previous point 4. Codex agrees with
Claude's amendment and the resulting consensus.

---

# Claude — the noise floor, measured; and a correction to how my own withdrawal should be read

Appended 2026-08-16 15:44 +01:00. Consensus with Codex recorded below, plus a new
line of investigation opened at the owner's direction.

Codex's revised point 4 asks for the evaluation design's noise floor to be
quantified rather than asserted. I have now measured it, and it is worse than I
implied — but it does not mean everything in this document is noise, and the
distinction is important enough to state carefully.

## The measurement

Identical configuration throughout (`_BASE`, `w=cur`). Only the evaluation design
or the model seed varies, so every difference below is pure noise.

**A. Sliding the fold grid by whole days:**

| offset | score |
|---|---|
| 0 | 14.886 |
| 1 | 16.030 |
| 2 | 16.541 |
| 3 | 15.436 |
| 4 | 17.083 |

range **2.197**, sd 0.778

**B. Same folds, different model seeds:**

| seed | score |
|---|---|
| 42 | 14.886 |
| 7 | 14.872 |
| 2024 | 14.934 |
| 1 | 14.924 |
| 99 | 14.924 |

range **0.062**, sd 0.024

Set against the effects that have been argued about in this document:

```text
harness noise range (fold grid)   2.197
+bm_wind effect (weighted)        0.234    ← 9x smaller than the noise
cur2 effect (score)               0.554    ← 4x smaller than the noise
```

**The noise is essentially all evaluation design, not model stochasticity** —
a 35-fold difference between the two sources. Which days land in a test block
matters enormously; which seed the forest uses does not matter at all.

## The correction: paired comparison is not affected by this

I want to be careful not to over-swing. This result invalidates **pooled,
unpaired** comparisons — which is exactly what my original `bm_wind` claim was,
a mean over 5 folds. It does not invalidate **paired** comparisons, because
pairing evaluates both configurations on the *same* folds and therefore cancels
precisely the fold-selection variance measured above.

So, reading my two withdrawals correctly:

- **`bm_wind`**: withdrawn on strong grounds. The claim came from a pooled mean,
  the effect (0.234) is 9× below the fold noise, and the paired re-test showed
  5/10 folds. Nothing survives.
- **`cur2`**: the *score* evidence (3/10 folds, +0.554) is weaker than I implied,
  since it is of the same order as effects the design struggles with. But the
  plain RMSE and MAE results were **0/10 and 0/10** — worse in every single
  paired fold — and a clean sweep is not something fold-selection noise produces,
  because pairing removes it. That part of the withdrawal stands.

The practical rule this establishes for future work here: **never compare
configurations by pooled score in this harness; always pair on folds, and report
the fold-win count alongside the mean.**

## Consensus with Codex

Codex has accepted the amendment and posted a revised seven-point position. I
agree with all seven as written, and its point 4 is what this entry discharges.
**Recording consensus.**

The one thing worth adding: its point 4 should now be considered *done* for this
harness, with the answer being that the design cannot resolve effects below
roughly ±2 score points when compared unpaired, and that paired comparison is the
mitigation rather than more folds.

## New direction, at the owner's request

The owner's steer:

> Users are particularly interested in extreme highs or extreme lows so it would
> seem to make sense to over-weight outliers. The weighting may need to be
> considerable because of the relative scarcity of these data points in the
> training set.

I think this is right, and it reframes what I have been doing wrong. Two points
in its favour that the evidence already supports:

1. **The direction of travel is correct even in my failed test.** Low-regime bias
   improved monotonically with the weight exponent — 12.17 → 10.92 → 9.30 for
   `cur`/`cur2`/`cur3` — and `|low-regime bias|` improved in 5/7 folds for `cur2`.
   The thing that got worse was aggregate error. So heavier weighting *was* doing
   what it is supposed to do; I judged it against a metric that does not value it.

2. **Scarcity is the mechanism, and I have not been treating it as one.**
   `max(1, |z|)` gives a slot at z = −2.5 a weight of 2.5 against 1.0 for a
   typical slot. If negative-price slots are ~3.6 % of rows, they hold roughly
   8 % of the total weight mass even after weighting. Calling that
   "extreme-weighted" overstates it considerably.

**What I was getting wrong:** I have been scoring every candidate on
`(wMAE + wRMSE)/2`, an aggregate dominated by the 80 % of slots in the middle.
Under that metric any change that trades mid-range accuracy for tail accuracy
must look like a regression — which is precisely what I reported, twice. If the
product values extremes, the scoring function has to say so; otherwise the
experiment will keep selecting against the thing the owner is asking for.

So the next test changes two things together, and the second is the one that
matters:

1. Weighting schemes far heavier than exponent 2–3, including explicit
   **weight-mass targeting** (choose weights so extreme slots hold a specified
   share of total training weight, rather than hoping an exponent gets there).
2. **An extreme-focused primary metric** — tail RMSE, low/high regime bias, and
   negative-price detection — reported as first-class results rather than as
   diagnostics beside an aggregate that averages them away.

Paired folds throughout, per the rule established above. Running now; result to
follow in the next entry whether or not it supports the idea.

---

# Codex — consensus stands; extreme weighting needs a product metric

Appended 2026-08-16 15:26 +01:00 by Codex.

Agreed that consensus is recorded on the seven-point position above.

On the owner's new direction, I agree with the reframing: if the product values
extreme highs/lows more than mid-range accuracy, the primary score must encode
that preference rather than letting aggregate MAE/RMSE veto every tail-improving
change. The next test should therefore define the target metric first, then tune
weights against it.

My caveat is that "over-weight outliers" should not mean "optimize only observed
outliers." For a user-facing forecast the important behavior is earlier detection
of slots likely to be extreme, while keeping false alarms tolerable. So the
extreme-focused report should include, at minimum:

1. Negative/cheap-slot recall and precision at product-relevant thresholds.
2. High-price recall and precision at product-relevant thresholds.
3. Low-regime and high-regime bias.
4. Tail RMSE/MAE.
5. Whole-forecast MAE/RMSE as guardrail metrics, not the primary score.
6. Calibration slope/sd ratio on pooled production-style forecasts.

Weight-mass targeting is a sensible next experiment because it makes the intended
trade-off explicit. I would report results across a grid of target tail weight
mass values so the owner can choose the product trade-off, rather than hiding that
choice inside an exponent.

---

# Claude — heavy weighting does fix the tail bias; the owner's hypothesis holds

Appended 2026-08-16 15:57 +01:00.

Tested weighting far heavier than before, including explicit weight-mass
targeting, scored on extreme-focused metrics with paired folds throughout.
`pow1` is the current production scheme, `max(1, |z|)`.

## First, a correction to my own arithmetic

In the previous entry I said extreme slots "hold roughly 8 % of the total weight
mass". That was wrong — I computed it from the negative-price share (3.6 %) while
defining extremes as `|z| > 1.5`, which is 11.0 % of rows. Measured properly:

| scheme | share of rows | share of weight mass |
|---|---|---|
| `pow1` *(current)* | 11.0 % | **22.3 %** |
| `pow2` | 11.0 % | 42.0 % |
| `pow3` | 11.0 % | 67.0 % |

So the current scheme already doubles the influence of extremes, not the near-
nothing I implied. The owner's point still holds — 22 % is not much for the slots
the product exists to flag — but I overstated the starting position.

## Result

Paired against `pow1` on the same folds; "win" is folds where the scheme beats
`pow1` on that metric.

| scheme | tail RMSE | win | low bias | win | high bias | win | sd | RMSE | MAE |
|---|---|---|---|---|---|---|---|---|---|
| `pow1` *(current)* | 21.54 | — | **+9.04** | — | **−34.15** | — | 0.997 | 15.62 | 11.83 |
| `pow2` | 21.23 | 5/10 | +5.08 | 5/7 | −28.02 | **4/4** | 1.065 | 16.67 | 12.50 |
| `pow3` | 22.07 | 5/10 | **−0.48** | 5/7 | −21.24 | **4/4** | 1.136 | 18.14 | 13.49 |
| `pow4` | 23.31 | 4/10 | −4.07 | 4/7 | **−15.40** | **4/4** | 1.220 | 20.49 | 15.13 |
| `mass0.30` | 21.35 | 5/10 | +8.06 | 4/7 | −33.00 | 3/4 | 1.018 | 15.89 | 12.00 |
| `mass0.50` | 21.57 | 5/10 | +5.07 | 5/7 | −28.67 | **4/4** | 1.073 | 16.78 | 12.65 |
| `mass0.70` | 22.20 | 5/10 | +2.50 | 4/7 | −24.82 | **4/4** | 1.134 | 17.95 | 13.43 |

**The owner's hypothesis is supported.** Regime bias improves monotonically and
substantially with weight:

- **Low-price bias** runs +9.04 → +5.08 → **−0.48** → −4.07. At `pow3` the
  systematic tendency to over-predict cheap slots is *eliminated* rather than
  merely reduced.
- **High-price bias** runs −34.15 → −28.02 → −21.24 → −15.40, improving in
  **4 of 4 folds** at every level tested. That is the most consistent result
  anywhere in this document — unanimous across folds, monotone in the dial, and
  on exactly the slots the product cares about.

This is what I failed to see the first two times: I had the right dial and was
reading the wrong gauge.

**Tail RMSE barely moves** (21.54 → 21.23 → 22.07), a coin flip throughout. So
heavier weighting corrects *where the tail predictions sit* without making them
individually more accurate. For a user deciding when to run a load, being
systematically 34 £/MWh low on expensive slots is a worse failure than scatter
around the right level — but this is worth being explicit about rather than
letting "fixes the tails" imply more than it does.

**The cost is real and monotone**: RMSE 15.62 → 18.14 and MAE 11.83 → 13.49 at
`pow3`. There is no free setting; this is a dial with a price at every position,
which is why Codex is right that the owner should see the grid rather than have
a choice buried in an exponent.

Weight-mass targeting behaves as intended and is the more honest control:
`mass0.50` lands almost exactly on `pow2` for both benefit and cost, while
`mass0.30` is a genuinely cheaper option — most of the tail-bias gain for a third
of the aggregate penalty.

## The caveat that matters most for choosing a value

**This harness says `pow1` already achieves sd ratio 0.997. Production measures
0.559.** The harness therefore reports the current model as well-dispersed when
the directly-measured production behaviour is that it is badly under-dispersed,
and it scores heavier weighting as *over*-dispersing (1.065, 1.136, 1.220) when
production has 44 % of its spread missing.

This is the narrow-evaluation-window problem from my first entry, and its
consequence here is specific and important: **the harness systematically
understates how much weighting is needed.** A setting that looks like
over-correction in this table may be correct or still insufficient in production.
That is independent support for the owner's "the weighting may need to be
considerable" — and it means the value should not be chosen from this table
alone.

## Agreement with Codex's caveat, and what is next

Codex is right that "over-weight outliers" must not collapse into "optimise
observed outliers", and that the report needs recall/precision at
product-relevant thresholds rather than bias alone — a model that shifts every
prediction downward would improve low-regime bias while being useless. Bias
metrics cannot distinguish those cases; detection metrics can.

Its six-item metric list is the right specification and I accept it. Items 3, 4
and 5 are covered above. Next: items 1 and 2 — cheap-slot and expensive-slot
recall and precision across the weight grid, at thresholds that mean something to
a user, so the owner can see what each setting buys and costs in terms of
"slots correctly flagged" against "false alarms". Running now.

I am not recommending a value until that exists.

---

# Codex — regime-bias result is useful; wait for detection metrics

Appended 2026-08-16 15:34 +01:00 by Codex.

Agreed with the interpretation. The heavy-weighting grid is the first result here
that clearly moves the tail behaviour the owner cares about: low-regime bias and
high-regime bias improve monotonically, and the high-price result is especially
strong at 4/4 folds for every heavier setting tested.

I also agree with the restraint: this is not yet enough to pick a production
weight. Bias can improve because the model has become better at identifying
extreme slots, or because it has shifted broad regions of the forecast in a way
that creates too many false alarms. The next recall/precision table is therefore
not optional; it is the decision metric for a user-facing forecast.

Two specifics I would want in that next table:

1. Use thresholds that map to actual product decisions, not only statistical
   quantiles. For example: negative price, very cheap price, expensive peak, and
   extreme high price. The exact values should be stated in £/MWh.
2. Report both paired-fold metrics and pooled production-style metrics. The fold
   harness is useful for relative comparisons, but the production sd-ratio gap
   means any final setting needs to be checked on the same pooled forecast design
   that exposed the original 0.559 dispersion problem.

Until that detection report exists, my position is: heavier weighting is now a
credible intervention for tail bias, `mass0.30`/`mass0.50`/`pow2` look like the
interesting part of the cost-benefit frontier, and `pow3+` should be treated as
high-cost unless detection metrics show a clear user-facing gain.

---

# Claude — detection metrics: the case is now clear, and `pow3` is my recommendation

Appended 2026-08-16 16:12 +01:00. This answers Codex's caveat directly.

Pooled over 11 382 test rows from identical folds, so every scheme is compared
against the same actual vector — paired at row level, not pooled across
different fold grids.

## Codex's caveat is answered: this is not "shift everything down"

That was the right thing to check, and it is the single most important number
here. If heavier weighting were simply dragging predictions downward, negative
recall would rise and **precision would collapse**. It does not:

### Negative prices (actual < 0) — 231 of 11 382 rows, 2.0 % base rate

| scheme | recall | precision | F1 | slots flagged |
|---|---|---|---|---|
| `pow1` *(current)* | 0.303 | 0.864 | 0.449 | 81 |
| `pow2` | 0.455 | 0.868 | 0.597 | 121 |
| `pow3` | **0.550** | **0.864** | 0.672 | 147 |
| `pow4` | 0.641 | 0.818 | **0.718** | 181 |

**Recall more than doubles while precision is flat.** At `pow3` the model finds
55 % of negative-price slots instead of 30 %, and the ones it flags are correct
just as often as before (0.864 → 0.864). Those extra flags are real detections,
not a downward drift. This is the strongest result in the investigation.

### Spikes (actual > 250) — 40 rows, 0.4 % base rate

| scheme | recall | precision | F1 |
|---|---|---|---|
| `pow1` | 0.625 | 0.926 | 0.746 |
| `pow2` | 0.825 | 0.868 | 0.846 |
| `pow3` | **0.875** | 0.833 | **0.854** |
| `pow4` | 0.925 | 0.725 | 0.813 |

### Expensive (actual > 180) — 246 rows, 2.2 %

| scheme | recall | precision | F1 |
|---|---|---|---|
| `pow1` | 0.435 | 0.743 | 0.549 |
| `pow3` | 0.642 | 0.669 | 0.656 |
| `pow4` | 0.711 | 0.623 | **0.664** |

### Cheap (actual < 50) — 721 rows, 6.3 % — **the counter-example**

| scheme | recall | precision | F1 |
|---|---|---|---|
| `pow1` *(current)* | 0.904 | 0.738 | **0.812** |
| `pow2` | 0.933 | 0.646 | 0.763 |
| `pow3` | 0.949 | 0.605 | 0.739 |
| `pow4` | 0.968 | 0.518 | 0.675 |

Here the current scheme is **best**, and heavier weighting makes it worse. Cheap
slots at the < £50 threshold are already detected well (recall 0.904); the extra
weight buys 4 points of recall for 13 points of precision.

**So the benefit is concentrated where events are rare.** The rarer the event,
the more heavy weighting helps: negatives (2.0 %) and spikes (0.4 %) improve
dramatically, expensive (2.2 %) clearly, and the relatively common cheap band
(6.3 %) degrades. That is exactly the scarcity mechanism the owner identified,
and it is visible directly in the base rates.

## Weight-mass targeting is the worse control after all

I proposed mass targeting as the more honest dial, and Codex agreed. The
detection metrics show it is the wrong instrument for the rarest events:

| scheme | negative recall |
|---|---|
| `pow1` | 0.303 |
| `mass0.30` | 0.346 |
| `mass0.50` | 0.355 |
| `mass0.70` | **0.251** ← worse than doing nothing |

The mechanism is clear once seen: mass targeting applies a **flat** multiplier to
every `|z| > 1.5` slot, so as the target share rises, moderately-extreme slots
(which are many) absorb the added mass and the genuinely extreme slots (which are
few) get *relatively less* emphasis. `pow` keeps escalating with severity, which
is what rare-event detection needs. I withdraw the suggestion that mass targeting
is the better control — it is more interpretable and less effective, and for
`mass0.70` actively harmful on the metric that matters most.

## Recommendation: `pow3`, i.e. `max(1, |z|)³`

| | `pow1` (current) | `pow3` | change |
|---|---|---|---|
| negative recall | 0.303 | **0.550** | +81 % |
| negative precision | 0.864 | 0.864 | unchanged |
| spike recall | 0.625 | **0.875** | +40 % |
| spike F1 | 0.746 | **0.854** | best tested |
| expensive F1 | 0.549 | **0.656** | +19 % |
| low-regime bias | +9.04 | **−0.48** | eliminated |
| high-regime bias | −34.15 | −21.24 | −38 % |
| cheap F1 | **0.812** | 0.739 | −9 % |
| whole-set RMSE | **15.62** | 18.14 | +16 % |
| whole-set MAE | **11.83** | 13.49 | +14 % |

`pow3` sits at or near the F1 peak for spikes, gives the largest negative-price
gain available without precision loss, and eliminates low-regime bias. `pow4`
buys more negative recall but starts paying precision (0.818) and takes cheap F1
down to 0.675.

**The costs are honest and should be visible to the owner:** the aggregate error
rises ~15 %, and detection of ordinary cheap slots gets worse. My reading is that
this is the right trade for this product — a user who is told 55 % of negative
slots instead of 30 %, with the same reliability per flag, is materially better
served than one whose mid-range forecast is 2 £/MWh sharper. But it *is* a trade,
the owner's to make, and `pow2` is the defensible conservative choice
(negative recall 0.455 at precision 0.868, RMSE +6.7 %).

## What still stands, unchanged

The consensus seven points remain, with points 2, 3 and 4 unaffected — this is a
candidate intervention with evidence, not a merged change. In particular:

- **The measurement gate still comes first.** Everything above is harness
  evidence. The harness reports the current model at sd ratio 0.997 when
  production measures 0.559, so it understates how much weighting is needed and
  may equally be flattering `pow3`. Detection metrics on **pooled production
  forecasts** are what should decide the final value.
- The paired-comparison rule holds; these figures use identical folds and a
  shared actual vector, so they do not repeat the mistake that killed the
  `bm_wind` claim.
- Nothing here rehabilitates `bm_wind`, residual load, or affine recalibration.

Codex: this is the detection evidence you asked to wait for. My question is
whether you accept `pow3` as the recommended value to put in front of the owner,
or would argue for `pow2` on the strength of the cheap-band regression and the
aggregate cost.

---

# Codex — put `pow3` in front of the owner, with `pow2` as the conservative option

Appended 2026-08-16 15:42 +01:00 by Codex.

I accept `pow3` as the primary recommendation to put in front of the owner,
provided it is presented as a product trade-off and not as a universally better
forecast.

The reason is the detection table. `pow3` roughly doubles negative-price recall
without losing negative-price precision, improves spike recall and F1, improves
expensive-slot F1, eliminates low-regime bias, and materially reduces high-regime
bias. Those are the behaviours the owner explicitly prioritized. The cheap-band
regression and aggregate MAE/RMSE cost are real, but they are the price of moving
the model toward rare-event usefulness.

I would present the owner with this choice:

1. **`pow3` recommended for tail alerting/product value:** best tested balance for
   negative prices and spikes; accepts ~15 % aggregate RMSE/MAE cost and weaker
   ordinary-cheap detection.
2. **`pow2` conservative alternative:** meaningful negative/spike improvement
   with lower aggregate cost; less complete correction of low/high regime bias.
3. **`pow1` status quo:** best ordinary-cheap F1 and best aggregate error, but
   misses too many negative and spike events for a product focused on extremes.

One condition remains: do not ship the value from this harness alone. Add the
measurement gate first and run the same detection metrics on pooled
production-style forecasts, because the harness still disagrees sharply with
production dispersion. If production-pooled detection reproduces the `pow3`
pattern, `pow3` is the change I would make.

That is agreement with Claude's recommendation, with `pow2` retained as the
explicit conservative fallback for the owner's decision.

---

# Claude — consensus reached on `pow3`; running the production-style gate

Appended 2026-08-16 16:24 +01:00.

Agreement recorded. Codex accepts `pow3` as the primary recommendation with
`pow2` as the explicit conservative fallback and `pow1` as the documented status
quo, on the condition that the value is not shipped from harness evidence alone.
I agree with all of that, including the framing that this is a product trade-off
rather than a universally better forecast.

**Consensus position:**

1. The production model is materially under-dispersed in the tails — directly
   measured, sd ratio 0.559 and slope 1.377 at ≥2d, with negative-price bias of
   +89 £/MWh across 1 901 slots.
2. Measurement lands first: tail/calibration/detection metrics in the experiment
   output, production-style rather than narrow-window.
3. `run_feature_experiment` must train with production's sample-weight objective;
   selecting features under an objective production does not use is a defect
   regardless of which features win.
4. Evaluation designs must have their noise floor quantified before model changes
   are judged with them, and comparisons must be paired, never pooled across
   different fold grids.
5. **`pow3` recommended**, `pow2` conservative, `pow1` status quo — as a product
   choice, with costs stated: ~15 % aggregate RMSE/MAE and a weaker ordinary-cheap
   band.
6. `bm_wind`, residual-load features and affine recalibration are all unsupported
   on current evidence.
7. Nothing ships from this harness alone.

The remaining condition is Codex's and it is testable, so I am testing it rather
than leaving it as a caveat: **do the detection gains survive a production-style
evaluation?** The harness evaluates a narrow 1–3d window on 3-day test blocks,
which is exactly the design whose sd ratio (0.997) contradicts the production
measurement (0.559). If `pow3`'s advantage is an artefact of that window, it
should weaken or vanish when the same comparison is pooled across a wide calendar
span at ≥2d horizons, as the production measurement was.

Result to follow, whichever way it goes.

---

# Claude — production-style gate: the pattern survives, but my "precision unchanged" claim does not

Appended 2026-08-16 16:41 +01:00.

Re-ran the weight grid under a production-style design: 6 rolling splits, 21-day
train / 5-day test, evaluating **all horizons ≥2d** pooled across the full
calendar span — 50 412 rows, matching how the original production measurement was
taken rather than the narrow 1–3d window.

## Dispersion

| scheme | sd ratio | slope | RMSE | MAE |
|---|---|---|---|---|
| `pow1` *(current)* | 0.860 | 0.958 | 25.13 | 17.08 |
| `pow2` | 0.932 | 0.884 | 25.72 | 17.57 |
| `pow3` | 1.004 | 0.817 | 27.09 | 18.54 |
| `pow4` | 1.066 | 0.766 | 28.61 | 19.88 |

## Detection

| band (base rate) | scheme | recall | precision | F1 |
|---|---|---|---|---|
| **negative < 0** (3.6 %) | `pow1` | 0.284 | 0.644 | 0.394 |
| | `pow2` | 0.398 | 0.601 | 0.479 |
| | `pow3` | 0.494 | 0.535 | 0.513 |
| | `pow4` | 0.558 | 0.502 | **0.529** |
| **expensive > 180** (2.3 %) | `pow1` | **0.126** | 0.775 | 0.217 |
| | `pow2` | 0.267 | 0.693 | 0.385 |
| | `pow3` | 0.392 | 0.584 | 0.469 |
| | `pow4` | 0.495 | 0.507 | **0.501** |
| **spike > 250** (0.4 %) | `pow1` | **0.000** | — | 0.000 |
| | `pow2` | 0.034 | 1.000 | 0.065 |
| | `pow3` | 0.101 | 0.808 | 0.179 |
| | `pow4` | 0.173 | 0.667 | **0.275** |
| **cheap < 50** (7.3 %) | `pow1` | 0.806 | 0.616 | **0.698** |
| | `pow3` | 0.878 | 0.537 | 0.666 |

## Correction: precision is not flat under this design

In the previous entry I made a point of the fact that negative-price precision
was unchanged (0.864 → 0.864) while recall doubled, and I called that the
strongest result in the investigation. **Under production-style pooling it does
not hold.** Negative precision falls 0.644 → 0.535 from `pow1` to `pow3`.

So the honest statement is weaker than the one I made: heavier weighting buys
recall and **pays for it in precision**, with F1 improving overall (0.394 →
0.513 → 0.529). It is a good trade, not a free one. The narrow harness flattered
it, which is exactly the failure mode Codex attached the condition for — the
condition earned its place.

## What the gate reveals that the harness hid

The narrow window also *understated* the defect badly:

- **`pow1` detects 12.6 % of expensive slots** (>£180) at ≥2d. The narrow harness
  reported 43.5 %.
- **`pow1` detects 0 of 208 spikes** (>£250). Not a low rate — zero. The narrow
  harness reported 62.5 % recall.

A model that never predicts a spike at any horizon beyond two days is
indefensible for a product whose purpose is telling people when power is
expensive. That is a stronger argument for heavier weighting than anything in my
earlier entries, and it only became visible under the production-style design.

## Where this leaves the recommendation

The gate **passes** — the pattern holds and strengthens — but it shifts the
optimum. Under production-style evaluation, `pow4` is best on all three rare
bands (negative F1 0.529, expensive 0.501, spike 0.275), where under the narrow
harness `pow3` was the peak.

That is consistent with the owner's original intuition that the weighting "may
need to be considerable", and with the direction of the remaining harness/
production gap (below). I am not, however, going to swing the recommendation to
`pow4` on one run of a design I have just finished criticising other designs for.
My position:

- **`pow3` remains the recommendation.** It captures most of the rare-event gain
  (negative F1 +30 %, expensive F1 +116 %, spike recall 0 → 0.101) at a smaller
  precision and aggregate cost, and it is the value both reviewers converged on.
- **`pow4` is now a live option rather than an overshoot**, and should be on the
  owner's menu. If the priority is maximising rare-event detection, it is better
  on every rare band.
- **`pow2`** remains the conservative choice.
- `pow1` should not be retained on this evidence.

## The gap that remains open, and why it matters

My reconstruction still does not reproduce production. `pow1` here gives sd ratio
0.860 and slope **0.958**; the direct production measurement is 0.559 and
**1.377**. Those are errors in opposite directions — production is badly
*under*-dispersed, my reconstruction is close to calibrated and tips to
*over*-dispersed as weight rises.

So something in the production pipeline compresses further than anything I can
reproduce from stored `ForecastData` — candidates being a different effective
training window, feature availability differing at real forecast time, or a
post-processing step between the ensemble output and the stored `day_ahead`
value. **I have not identified it**, and until it is identified the exact
exponent cannot be tuned from any of my harnesses, only its direction.

That is a real limit on this work and I would rather end on it than imply the
number is settled. The direction is firmly established — more weight on extremes,
and `pow1` is not defensible. The precise value needs the measurement gate
(consensus point 2) running against real production forecasts.

Codex: this discharges your condition, with the correction that it also falsified
my precision claim. I do not think it changes the consensus, but it does mean the
`pow3` recommendation should be presented with the precision cost stated, and
with `pow4` visible as the option the production-style evidence actually favours.

---

# Codex — production-style gate preserves direction, not exact tuning

Appended 2026-08-16 15:47 +01:00 by Codex.

Agreed. The production-style gate answers the condition in the way that matters
for the product: heavier weighting materially improves rare-event detection, and
`pow1` is not defensible for a forecast expected to flag negative prices,
expensive periods and spikes.

I also agree with Claude's correction. The recommendation must no longer say
that negative-price precision is unchanged. The production-style result is a
recall/precision trade: negative recall rises 0.284 -> 0.494 at `pow3`, but
precision falls 0.644 -> 0.535. F1 improves, so the trade is still worthwhile,
but it is not free.

Revised owner-facing recommendation:

1. **`pow3` remains the primary recommendation** because it captures a large share
   of the rare-event gain at a lower aggregate and precision cost than `pow4`.
2. **`pow4` should be shown as the high-recall option** because the
   production-style gate favours it on negative, expensive and spike F1.
3. **`pow2` remains the conservative option** for a smaller move away from the
   status quo.
4. **`pow1` should not remain the default** if the product goal is reliable
   detection of extremes.

The unresolved issue is now the exact tuning mechanism, not the direction. The
remaining gap between direct production measurement (`sd ratio 0.559`,
`slope 1.377`) and Claude's production-style reconstruction (`sd ratio 0.860`,
`slope 0.958`) is too large to ignore. Until that compression source is found,
no harness can honestly settle whether `pow3` or `pow4` is the correct production
exponent.

So my final position is: ship the measurement gate and production-faithful
experiment fixes first; present `pow3`/`pow4`/`pow2` to the owner as explicit
product choices; then tune the exponent only against real production-style
metrics once the extra compression source is understood or directly measured.

---

# Claude — hunting the missing compression: three hypotheses, all rejected, defect confirmed harder

Appended 2026-08-16 17:03 +01:00.

Codex's closing point was that the gap between the direct production measurement
(sd 0.559 / slope 1.377) and my production-style reconstruction (0.860 / 0.958)
is too large to ignore, and that until it is understood no harness can settle the
exponent. Agreed, so I went looking for it. I did not find it, but three
candidates are now eliminated and the defect itself is on much firmer ground.

## 1. Post-processing — ruled out by reading the code

`update.py:1106` does blend the model output:

```python
fc["day_ahead"] = fc["day_ahead"] * scale_factors["mult"] \
                + scale_factors["day_ahead"] * (1 - scale_factors["mult"])
```

but the scale factors (`update.py:1068-1088`) are constructed in three bands:
`mult=0` over the known-Agile window, `mult=0` over the GB60 window, and
**`mult=1` for everything else**. At `mult=1` the expression reduces to the model
output unchanged. So beyond ~36 hours the stored `day_ahead` *is* the raw
ensemble prediction, and this step introduces no compression at the horizons
under discussion. It is also the mechanism behind the 0–1d control in my first
entry, which is now confirmed from the code rather than inferred.

## 2. The `days_ago` train/inference mismatch — tested and rejected

Production computes `days_ago = now − created_at` for training rows but hard-codes
it to **0** at inference. My harness set it to 0 in both, inadvertently removing
that mismatch — I flagged this in my first entry as a difference I had introduced.
It was my leading hypothesis for the gap.

| `days_ago` handling | sd ratio | slope | RMSE | MAE | neg recall |
|---|---|---|---|---|---|
| 0 everywhere *(my harness)* | 0.860 | 0.958 | 25.13 | 17.08 | 0.284 |
| production behaviour | 0.893 | 0.952 | **22.89** | **15.16** | 0.245 |
| feature removed | 0.857 | 0.960 | 25.18 | 17.17 | 0.253 |

**Rejected.** Reproducing production's handling moves sd ratio *up* (0.860 →
0.893), away from the 0.559 it needed to explain.

Incidental finding worth keeping: production's handling is also the **best**
configuration on aggregate error — RMSE 25.13 → 22.89, MAE 17.08 → 15.16. So
`days_ago` is earning its place after all, and my harness was quietly
handicapping itself by zeroing it. Relative comparisons between schemes are
unaffected, since the handicap was constant across them, but the absolute error
figures in my earlier entries are slightly pessimistic.

## 3. A pooling artefact in the production measurement — tested and rejected

The production figure pools 52 598 rows in which each target slot appears **34.8
times on average** (max 57) with an identical actual and a different prediction
each time. That is a real concern: it could inflate the apparent compression
without the model being at fault.

| measurement | sd ratio | slope |
|---|---|---|
| pooled across runs *(as first reported)* | 0.559 | 1.377 |
| **within each forecast run, averaged over 113 runs** | **0.565** | **1.477** |
| within runs, single 2–4d horizon band | 0.671 | — |

**Rejected, and the defect is firmer than when I reported it.** Each individual
forecast, judged on its own, shows the same compression — 0.565 — and the slope
is actually *worse* within runs (1.477) than pooled. Even holding the horizon
fixed at 2–4 days it is 0.671. This is not an artefact of how I aggregated; it is
a property of every forecast the model produces.

## Where the gap now stands

Still open. The leading remaining candidate is **training-set composition**, which
I have not tested: production trains on `ff_train` — the forecast closest to
16:15, i.e. roughly *one run per day* — filtered to `days_ago < 60`. My harness
trains on **every** run in the window, which is 2–4 per day covering heavily
overlapping slots. That is several times the row count with substantial
near-duplication, and it plausibly changes the effective regularisation of all
three learners. It is the obvious next thing to test and I have not tested it.

I want to be clear about what this does and does not change:

- **The defect is confirmed and strengthened.** Every forecast run individually
  reproduces it. Nothing in this entry weakens the case for acting.
- **The direction is unchanged.** `pow1` failing to detect any spike beyond two
  days stands, as does the whole detection table.
- **The exponent still cannot be tuned offline.** Codex's condition is
  unsatisfied, and three eliminations do not amount to an explanation. Until the
  reconstruction reproduces production's dispersion, `pow3` versus `pow4` cannot
  be settled from any harness in this document.

That last point is the one I would not want softened. This entry narrows the
search; it does not close it.

---

# Codex — next compression suspect is training-set composition

Appended 2026-08-16 15:52 +01:00 by Codex.

Agreed. The three rejected hypotheses are useful because they remove easy
explanations without weakening the original defect. The within-run measurement is
especially important: compression is present forecast-by-forecast, so this is not
a pooling artefact.

I agree that the next test should be training-set composition. Production does
not train on every stored forecast run; it uses `select_daily_training_forecasts`
to keep the run closest to 16:15 for each day, then builds training rows from
that daily subset. A harness trained on every run in the window is materially
different: it sees several near-duplicate forecasts per target period, changes
the weighting mass around repeated slots, and may make the learners look less
compressed than the actual production model.

The test I would run next is direct:

1. Rebuild the production-style reconstruction using exactly `ff_train =
   select_daily_training_forecasts(ff)` rather than every forecast run.
2. Keep production `days_ago` behaviour, since that was both faithful and better
   on aggregate error.
3. Recompute the direct comparison for `pow1` first. Do not spend time tuning
   `pow2`/`pow3`/`pow4` until `pow1` gets close to the observed production
   sd ratio/slope.
4. If daily training still fails to reproduce compression, compare the actual
   feature columns and null/imputation patterns at historical inference time
   against the reconstructed training frame.

I agree with Claude's boundary: direction is established, but exponent tuning is
still blocked. The immediate goal is no longer "which exponent wins"; it is
"make the offline reconstruction reproduce the current production model's
compression."

---

# Claude — found it: production trains only on the day-ahead window. The exponent was the wrong dial.

Appended 2026-08-16 by Claude.

Codex's last entry said the immediate goal was no longer "which exponent wins" but
"make the offline reconstruction reproduce production's compression". That is now
done, and the answer changes the recommendation rather than refining it.

It is not the training-set *selection* Codex proposed. That was tested and
rejected: restricting to one run per day (`select_daily_training_forecasts`)
gives sd ratio 0.885 against 0.893 for all runs — no effect. It is the training-set
*horizon window*, one line further on.

## The mechanism

[`forecast_features.py:185-186`](prices/forecast_features.py#L185-L186) defines an
"Agile day" window on each forecast run, and
[`build_training_data`](prices/forecast_features.py#L214) filters training rows to it:

```python
ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
ff["ag_end"]   = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)
...
train_X = train_X[(train_X.index >= train_X["ag_start"]) & (train_X.index < train_X["ag_end"])]
```

**Production trains exclusively on slots 22–46 hours ahead of the run, then
predicts 14 days.** The model has never seen a training row beyond ~2 days. Note
the asymmetry with the holdout: `build_holdout_data` applies only
`index > ag_start` with no upper bound, so production's own test set spans every
horizon the model is scored on but its training set does not.

The consequence is a train/serve feature-distribution mismatch, not a loss-function
problem. At 22–46 h the weather and demand inputs are sharp; at 7–14 d the same
columns are smoothed, mean-reverted NWP output. The model learns a mapping
calibrated to sharp inputs and is then fed blunt ones, so its output is blunt.

## It reproduces production

Reconstruction: 6 rolling splits, 21-day train / 5-day test, one run per day,
production `days_ago` handling, evaluated on **all horizons ≥2 d**, `_BASE`, `pow1`.
Only the training-row horizon window varies.

| training rows restricted to | train rows | sd ratio | slope | RMSE | MAE |
|---|---|---|---|---|---|
| **22–46 h  (production today)** | **1 008** | **0.563** | **1.325** | 29.46 | 20.29 |
| 22 h – 3 d | 3 024 | 0.679 | 1.129 | 27.74 | 19.25 |
| 22 h – 7 d | 7 048 | 0.824 | 0.961 | 27.12 | 18.48 |
| 22 h – 14 d | 12 849 | 0.884 | 0.948 | **24.05** | **16.15** |

Direct production measurement, for comparison: **sd 0.559, slope 1.377.**

The production window reproduces it to 0.563 / 1.325. **The gap Codex said was too
large to ignore is closed**, and dispersion recovers monotonically as the window
widens. Nothing else in this document has matched production to three significant
figures.

## Detection: the production window predicts no extreme, ever

Same runs, detection at the product thresholds:

| training window | neg recall | neg prec | exp recall | spike recall | cheap recall | cheap prec |
|---|---|---|---|---|---|---|
| **22–46 h  (production)** | **0.000** | — | **0.000** | **0.000** | 0.271 | 0.784 |
| 22 h – 3 d | 0.000 | — | 0.056 | 0.000 | 0.501 | 0.772 |
| 22 h – 7 d | 0.190 | 0.563 | 0.230 | 0.000 | 0.699 | 0.599 |
| 22 h – 14 d | **0.219** | 0.642 | **0.425** | 0.010 | **0.792** | 0.666 |

Beyond two days the production-window model predicts **zero** negative prices,
**zero** slots above £180 and **zero** spikes — not a low rate, none. Cheap-slot
recall is 0.271. This is the same "0 of 208 spikes" I found in the production-style
gate, and now it has a cause rather than a symptom.

## The control: horizon coverage, not row count

The wide window has 12.7× the rows, so the obvious objection is that this is just
more data. It is not. Subsampling the 22 h–14 d training set down to exactly the
production window's 1 008 rows, three seeds:

| config | train rows | sd ratio | slope | RMSE | neg recall |
|---|---|---|---|---|---|
| 22–46 h (production) | 1 008 | 0.563 | 1.325 | 29.46 | 0.000 |
| 22 h–14 d, subsampled, seed 0 | 1 008 | 0.843 | 0.969 | 25.74 | 0.060 |
| 22 h–14 d, subsampled, seed 1 | 1 008 | 0.794 | 1.021 | 25.82 | 0.000 |
| 22 h–14 d, subsampled, seed 2 | 1 008 | 0.854 | 0.942 | 26.30 | 0.089 |

At **identical row count** the wide window recovers most of the dispersion
(0.79–0.85 vs 0.563) and is 3.2–3.7 RMSE better. So it is the horizon coverage of
the training rows that matters, not their number. Row count then adds a further
1.7 RMSE on top (25.9 → 24.05).

## Why the weight exponent looked like the lever, and why it was not

The weight grid, run under each window:

| window | scheme | sd ratio | slope | RMSE | neg recall | exp recall | spike recall |
|---|---|---|---|---|---|---|---|
| 22–46 h | `pow1` | 0.563 | 1.325 | 29.46 | 0.000 | 0.000 | 0.000 |
| 22–46 h | `pow2` | 0.567 | 1.324 | 29.28 | 0.000 | 0.004 | 0.000 |
| 22–46 h | `pow3` | 0.585 | 1.287 | 29.03 | 0.000 | 0.010 | 0.000 |
| 22 h–14 d | `pow1` | 0.884 | 0.948 | **24.05** | 0.219 | 0.425 | 0.010 |
| 22 h–14 d | `pow2` | 0.957 | 0.873 | 24.88 | 0.287 | 0.512 | 0.111 |
| 22 h–14 d | `pow3` | 1.012 | 0.823 | 25.79 | 0.387 | 0.601 | 0.173 |

**Under production's actual training window the exponent is nearly inert.** From
`pow1` to `pow3` sd ratio moves 0.563 → 0.585 and negative recall stays at exactly
zero. Everything Codex and I agreed about `pow3` was measured on harnesses that had
already — accidentally — fixed the real defect by training on all horizons. We were
tuning the second-order dial on a model that had the first-order one wrong.

That also explains the contradiction I flagged three entries ago and could not
resolve: my harness reported the current model at sd ≈ 0.86–1.00 while production
measured 0.559. My harness was not modelling production. It was modelling the
fixed version.

## Consistency

Per-split RMSE, `pow1`, 6 splits:

```text
22–46 h    30.51  27.55  30.64  28.63  30.32  29.00     mean 29.44
22 h–14 d  17.82  26.09  22.30  29.57  22.47  17.52     mean 22.63
```

The wide window wins **5 of 6 splits** — paired, per the rule established earlier.
Not unanimous, and the one loss (28.63 vs 29.57) is worth noting rather than
rounding away, but the mean gap of 6.8 RMSE is an order of magnitude larger than
the ±2 score noise floor measured for this design.

## What this changes

**The primary recommendation is no longer the weight exponent. It is the training
window.**

This is also the first intervention in this investigation with no trade-off to
declare. Every previous candidate bought tails at the cost of aggregate error;
widening the window improves **both** — RMSE 29.46 → 24.05 (−18 %), MAE 20.29 →
16.15 (−20 %), sd ratio 0.563 → 0.884, and extreme detection from literally zero to
non-zero on all three bands. I have been suspicious of results this one-sided in
this document, and I would rather say plainly why I believe this one: it is not a
tuning gain, it is the removal of a train/serve mismatch, and the effect is
monotone in the amount of mismatch removed.

It is causal-safe. Training on a row at horizon `h` uses the features that run
actually published `h` ahead, paired with the price that subsequently settled. No
information from after the run is used. It needs no new data — those rows are
already in `ForecastData` and are already being discarded.

**Revised ordering:**

1. **Widen the training horizon window** in `build_training_data` (and in
   `run_feature_experiment`, which applies the same filter). First change.
2. Measurement gate — unchanged, and now more clearly required, since every
   comparison below depends on it.
3. **Re-tune the weight exponent afterwards, not before.** On the widened window
   `pow2`/`pow3` still buy detection at a real cost (RMSE 24.05 → 24.88 → 25.79),
   and that trade is now a cheap one worth making. But the `pow3` recommendation
   Codex and I converged on was derived on harnesses that had the window wrong, so
   the exponent decision should be re-taken, not inherited. I am not withdrawing it
   — under the widened window `pow2`/`pow3` look better than they did — but it is
   no longer the headline and its value is not settled.
4. `run_feature_experiment` sample-weight fix — unchanged.
5. **Every feature conclusion in this document was drawn on a window production
   does not use.** My harness trained on all horizons; production and
   `run_feature_experiment` train on 22–46 h. `bm_wind` and residual load stay
   unsupported, but they are now unsupported *and* untested under the regime that
   matters. They should be re-run after the window changes.

## Limitations

- One reconstruction, 6 splits, `_BASE` only, June–August. Matching production's
  dispersion to three figures is strong evidence the reconstruction is now faithful
  on *that* axis; it is not proof it is faithful on all of them.
- The wide training set contains heavily overlapping target slots across runs. The
  subsample control addresses the row-count objection but not the correlation
  structure completely.
- 14 days is the widest window the stored data supports and is not obviously the
  optimum — 7 d already recovers most of the dispersion at slightly worse RMSE. The
  window length is itself a parameter I have not tuned.
- Untested in the real pipeline. This is still my harness.

## For Codex

Two questions.

1. Is there a reason for the 22–46 h window that I am missing? It is coherent as a
   *day-ahead* price model — train where the inputs are sharp — and the mismatch
   only bites because the product then serves 14 days from the same model. If the
   intent was a day-ahead model, the alternative fix is a separate long-horizon
   model rather than one widened training set. I think widening is right because
   `days_ago` is already in the feature set and the ensemble can condition on
   horizon, but I would like that challenged.
2. The widened model has no explicit horizon feature — `days_ago` is the *age of
   the run*, not the distance to the slot, and those coincide only inside the
   22–46 h window. So the widened model cannot distinguish a 1-day-ahead row from a
   13-day-ahead one, and still beats production by 18 % RMSE. I expected adding
   `dt` (days ahead, known at inference, causal-safe) to help. It does not — see
   below. I would value a read on whether my explanation for that is right.

## Postscript: the horizon feature makes it worse

I ran it before posting, since it was the first objection I expected. `dt` = days
ahead, added to `_BASE`:

| config | sd ratio | slope | RMSE | MAE | neg recall | exp recall | cheap recall |
|---|---|---|---|---|---|---|---|
| 22 h–14 d, `_BASE` | **0.884** | **0.948** | **24.05** | **16.15** | **0.219** | **0.425** | **0.792** |
| 22 h–14 d, `_BASE` + `dt` | 0.848 | 0.968 | 25.17 | 17.05 | 0.156 | 0.250 | 0.722 |
| 22 h–14 d, `_BASE` + `dt`, `pow2` | 0.908 | 0.905 | 25.63 | 17.49 | 0.239 | 0.338 | 0.793 |
| 22 h–14 d, `_BASE` + `dt`, `pow3` | 0.952 | 0.866 | 26.04 | 17.98 | 0.269 | 0.420 | 0.853 |

**Worse on every axis**, including the aggregate error I expected it to improve:
RMSE 24.05 → 25.17, negative recall 0.219 → 0.156, expensive recall 0.425 → 0.250.
`pow3` on top only claws back to roughly where `_BASE` `pow1` already was, at
2 RMSE more.

My reading is that this is the compression mechanism reappearing by another route.
Given an explicit horizon, the ensemble can learn "at 10 days ahead, revert toward
the mean" — which is locally MSE-optimal per horizon and is precisely the
conditional-mean shrinkage that produces the defect. Without `dt` it cannot express
that, so it commits to what the (blunt) features say. For a product that needs
extremes flagged, not knowing the horizon is apparently an advantage. That the
plain RMSE also degrades is the part I am least confident I have explained, and I
would not want that hand-waved: the shrinkage story predicts better RMSE with `dt`,
not worse, so something else is contributing — most likely that a 21-day training
window gives too few distinct run-dates to learn a stable horizon effect, leaving
`dt` as an overfitting surface.

So: **do not add a horizon feature**. This was my suggestion and it is refuted;
recording it because the negative result is more useful than the proposal was.

---

# Codex — agree: the training window is the first-order fix

Appended 2026-08-16 16:08 +01:00 by Codex.

Agreed. This closes the reconstruction gap well enough to change the ordering.
The 22-46 h training window reproducing production at sd ratio 0.563 / slope
1.325 against the direct 0.559 / 1.377 measurement is the first result in this
document that explains the observed compression rather than only compensating for
it.

My answers to Claude's questions:

1. I do not see a good reason to keep the 22-46 h window for a model that is
   served to 14 days. It makes sense historically if the code began as an Agile
   day-ahead model, because that window matches the product's known next-day price
   use case and keeps training inputs sharp. But the current product surface uses
   the same ensemble for long horizons, so that original window has become a
   train/serve mismatch. A separate long-horizon model would be defensible if the
   application wanted distinct day-ahead and outlook semantics, but the simpler
   and better-supported first fix is to train the existing model on the horizons
   it is asked to predict.
2. I accept the `dt` result and would not add it. The empirical result is clear:
   `_BASE + dt` is worse on RMSE, MAE, dispersion and detection. Claude's
   explanation is plausible enough for a decision: with only a short rolling
   training window, explicit horizon gives the trees an easy way to learn a
   brittle horizon-specific reversion surface. More importantly, we do not need
   to explain every mechanism to reject a feature that loses this cleanly.

Revised implementation order:

1. **Widen `build_training_data` to the served horizon**. Use the broadest
   available horizon first, likely 22 h-14 d, because it wins on aggregate error
   and fixes most of the dispersion. If runtime or data quality forces a smaller
   window, 22 h-7 d is the next candidate, but it should be a measured compromise.
2. Apply the same horizon-window logic to `run_feature_experiment`; otherwise
   feature selection remains optimized for the wrong task.
3. Keep production `days_ago` behaviour.
4. Do **not** add `dt`.
5. After the widened-window model is in place, re-run the weight exponent grid.
   `pow2`/`pow3` still look useful for rare-event detection, but the exponent is
   now a second-order product trade-off rather than the root fix.
6. Re-run feature experiments after the window change. Previous `bm_wind`,
   residual-load and engineered-feature conclusions should be treated as stale,
   not because they are likely right or wrong, but because they were measured
   under the wrong training horizon.

This also changes the owner-facing message: the primary recommendation is no
longer "increase outlier weighting"; it is "train the model on the horizons it
serves." Outlier weighting remains a likely follow-up for rare-event recall, but
the training-window mismatch is the defect to fix first.

---

# Conclusions — agreed position (Claude + Codex)

Appended 2026-08-16 by Claude, following Codex's agreement above. This consolidates
the whole investigation for the owner; the entries above are the working record and
include everything that was tried and withdrawn.

## The defect

The production model materially under-predicts the dynamic range of GB day-ahead
prices at every horizon beyond two days. Directly measured on 52 598 stored
forecast/actual pairs at ≥2 d:

| | measured | should be |
|---|---|---|
| sd(pred)/sd(actual) | **0.559** | 0.769 (= r, for an optimal predictor) |
| slope(actual ~ pred) | **1.377** | 1.000 |
| mean bias, actual < £0 (1 901 slots) | **+89.26 £/MWh** | 0 |
| mean bias, actual ≥ £250 (208 slots) | **−126.26 £/MWh** | 0 |

This is not the shrinkage a conditional-mean predictor is supposed to show — it is
~27 % more shrinkage than optimal, and an affine correction provably reduces
squared error. It is present **within every individual forecast run** (sd 0.565,
slope 1.477 averaged over 113 runs), so it is not an artefact of pooling. The 0–1 d
horizon, where the pipeline passes GB60 prices through rather than model output,
shows sd ratio 0.989 — a clean control that localises the defect to the model.

## The cause

`build_training_data` ([forecast_features.py:214](prices/forecast_features.py#L214))
filters training rows to `ag_start ≤ t < ag_end`, defined at
[lines 185-186](prices/forecast_features.py#L185-L186) as **22 to 46 hours after
the run's midnight**. Production therefore trains exclusively on day-ahead rows and
then predicts 14 days. `build_holdout_data` applies no upper bound, so the holdout
spans horizons the training set never covers.

The result is a train/serve feature-distribution mismatch. At 22–46 h the weather
and demand inputs are sharp; at 7–14 d the same columns are smoothed, mean-reverted
NWP output. A model fitted to sharp inputs and served blunt ones produces blunt
output.

Reconstructing that window reproduces production for the first time in this
investigation — sd 0.563 / slope 1.325 against the direct 0.559 / 1.377 — and
widening it recovers dispersion monotonically:

| training window | train rows | sd ratio | slope | RMSE | MAE | neg recall | exp recall | spike recall |
|---|---|---|---|---|---|---|---|---|
| **22–46 h (production today)** | 1 008 | **0.563** | **1.325** | 29.46 | 20.29 | **0.000** | **0.000** | **0.000** |
| 22 h – 3 d | 3 024 | 0.679 | 1.129 | 27.74 | 19.25 | 0.000 | 0.056 | 0.000 |
| 22 h – 7 d | 7 048 | 0.824 | 0.961 | 27.12 | 18.48 | 0.190 | 0.230 | 0.000 |
| **22 h – 14 d** | 12 849 | **0.884** | **0.948** | **24.05** | **16.15** | **0.219** | **0.425** | 0.010 |

Beyond two days the current configuration predicts **zero** negative prices, zero
slots above £180 and zero spikes above £250. Not a low rate — none.

Two controls support the causal reading. Subsampling the wide window to the
production window's exact 1 008 rows still gives sd 0.79–0.85 and 3.2–3.7 better
RMSE, so it is horizon **coverage**, not data volume. And the wide window wins 5 of
6 paired splits, with a mean RMSE gap (6.8) an order of magnitude above this
design's measured ±2 noise floor.

## The agreed fix, in order

1. **Widen the training horizon window in `build_training_data`** to the horizons
   the model is served at. 22 h–14 d first: it is best on both aggregate error and
   dispersion. 22 h–7 d is the fallback if runtime or data quality forces it, as a
   measured compromise rather than a default.
2. **Apply the same widening to `run_feature_experiment`**, which carries the same
   filter — otherwise feature selection stays optimised for a task the product does
   not serve.
3. **Land the measurement gate**: tail/calibration/detection metrics (RMSE, MAE,
   slope, sd ratio, low/high regime bias, tail RMSE, and recall/precision at the
   product thresholds) in the experiment output, computed production-style rather
   than on a narrow window. Free, and it is what would have surfaced this without a
   special investigation.
4. **Fix `run_feature_experiment` to train with production's sample weights.** It
   currently fits all three learners unweighted and only weights the evaluation
   metric, so features are selected under an objective production does not use.
5. **Re-tune the weight exponent afterwards.** On the widened window `pow2`/`pow3`
   still buy rare-event detection at a real cost (RMSE 24.05 → 24.88 → 25.79) and
   that trade now looks worth making — but it is a second-order product choice, not
   the root fix, and the earlier `pow3` recommendation should be re-taken rather
   than inherited.
6. **Re-run the feature experiments** after the window changes. `bm_wind`,
   residual load and the engineered features are stale rather than refuted: they
   were measured under the wrong training horizon.

## Explicitly rejected

- **Squaring or cubing the sample weight as the primary fix.** Under production's
  actual training window the exponent is nearly inert — `pow1` → `pow3` moves sd
  ratio 0.563 → 0.585 and leaves negative-price recall at exactly 0.000. The
  earlier `pow3` consensus was reached on harnesses that had accidentally already
  fixed the real defect by training on all horizons.
- **An explicit horizon feature (`dt`).** Worse on every axis: RMSE 24.05 → 25.17,
  negative recall 0.219 → 0.156, expensive recall 0.425 → 0.250.
- **Weight-mass targeting.** More interpretable but less effective than an
  exponent, and actively harmful at `mass0.70`.
- **`bm_wind` reinstatement, `residual_load`, `renew_share`, `cap_margin`** — no
  support on current evidence, and now also untested under the corrected regime.
- **Affine recalibration** as a shipped fix. It works out of sample (−5.0 % RMSE)
  but it is a post-hoc correction to a training defect that is now identified.
  Retain only as a fallback.
- **One-run-per-day training selection** as the compression source — tested and
  rejected (0.885 vs 0.893 for all runs). Production does select this way; it just
  is not what causes the compression.

## Incidental findings worth acting on separately

- **`days_ago` earns its place.** Reproducing production's handling (varying in
  training, 0 at inference) rather than zeroing it improves RMSE 25.13 → 22.89 and
  MAE 17.08 → 15.16. Keep it as is.
- **`day_ahead_extra_trees`, `day_ahead_classified` and `plunge_probability` are
  NULL for every row**, written as null/NaN by `update.py` on every run. Populate
  them deliberately or drop them.
- **Never compare configurations by pooled score in the fold harness.** Its noise
  floor is ±2 score points from fold-grid placement alone (sd 0.778), against 0.062
  from model seed. Always pair on folds and report the fold-win count.

## What remains open

- **Untested in the real pipeline.** Everything above is an offline reconstruction.
  It now matches production's dispersion to three figures, which is the strongest
  validation any harness in this document has had, but that is one axis.
- **The window length is not tuned.** 14 d is the widest the stored data supports
  and is best on the metrics measured; 7 d already recovers most of the dispersion.
  Neither has been shown optimal.
- **Summer only.** 51 forecast days in June–August, which is exactly when high
  solar and low demand produce negative prices. The negative-price findings may be
  seasonally amplified.
- **The exponent value.** Deliberately left open until the window is fixed and the
  gate exists.

## Owner-facing summary

The headline has changed. It is **not** "increase the outlier weighting" — that was
the recommendation two rounds ago and it does almost nothing while the real defect
is present. It is:

> **Train the model on the horizons it is asked to predict.** It is currently
> fitted only on the next day and then used to forecast a fortnight, which is why
> the forecast flattens out beyond 48 hours and never calls a negative price or a
> spike.

Unusually for this investigation, that change costs nothing to trade off: aggregate
error improves ~18 % *and* the tails come back. Heavier outlier weighting — the
owner's original instinct — remains a sound follow-up for rare-event recall, and
becomes materially more effective once the window is fixed. It is a genuine
trade-off and the grid should be put in front of the owner at that point, not
before.

**Agreed by Claude and Codex.**

---

# Claude — implemented steps 1–4; for Codex's review

Appended 2026-08-16 by Claude. Commit `df87c2d`, pushed to `main`. Not deployed.

Steps 1–4 of the agreed order are implemented. Steps 5 and 6 (re-tune the
exponent, re-run the feature experiments) are deliberately not done — they depend
on this landing first.

## What changed

**1. Training horizon window** — `forecast_features.py`

```python
TRAIN_HORIZON_DAYS = 14

def training_horizon_mask(df, horizon_days=TRAIN_HORIZON_DAYS):
    window_end = df["ag_start"] + pd.Timedelta(days=horizon_days)
    return (df.index >= df["ag_start"]) & (df.index < window_end)
```

`build_training_data` takes `horizon_days=TRAIN_HORIZON_DAYS`. **`horizon_days=1`
reproduces the old `ag_start`..`ag_end` window exactly**, which is both the
rollback path and what the tests pin.

One deliberate difference from my harness: I measured "22 h – 14 d" as 22 h to
`midnight + 14 d`; the code uses `ag_start + 14 d`, i.e. 22 h to
`midnight + 14 d + 22 h`. The extra 22 hours sits beyond the upstream forecast
horizon, so in practice it admits no additional rows — but the shipped window is
nominally slightly wider than the measured one and I would rather say so than have
it found later. At 14 days this is effectively "no upper bound".

**2. Same window in the feature experiment** — `update.py`. The experiment now
calls `training_horizon_mask(train_df, horizon_days)` instead of carrying its own
copy of the filter, so the two cannot drift apart again. This was the mechanism by
which feature selection was optimised for a task production does not perform.

**3. The measurement gate** — new `prices/model_metrics.py`:

- `calibration_metrics` — sd ratio, slope, r, RMSE, MAE, tail RMSE (|z| > 1.5),
  low-regime and high-regime bias.
- `detection_metrics` — recall/precision/F1 for a one-sided band.
- `PRICE_BANDS` — negative (< £0), cheap (< £50), expensive (> £180), spike
  (> £250), on the model's native unscaled day-ahead scale.
- `forecast_report` / `format_report`.

Wired in two places. On the **holdout**, which spans every horizon the model is
served at, logged twice — all horizons and ≥2 d separately, because the defect is
horizon-dependent and is invisible if you only look at the aggregate. And per
**feature set** in the experiment, so `sd_ratio`, `slope`, `tail_rmse`, regime
bias, and negative/spike recall are stored alongside the score rather than only the
score that cannot see them.

One decision worth flagging for review: `detection_metrics` returns **recall 0.0
but precision NaN** when a model flags nothing. That is the exact state production
was in, and I wanted it to read as "found none of them", not as a missing
measurement — there is a test pinning it.

**4. Production's objective in the experiment** — `run_feature_experiment` fitted
CatBoost, LightGBM and ExtraTrees unweighted while weighting only the evaluation
metric. All three now fit with `np.maximum(1.0, |z|)`, as production does.

## Verification

Ten new tests, all passing, covering the default window, `horizon_days=1`
equivalence to the legacy window, an intermediate bound, and the metrics including
the never-flags-anything case.

Full suite: **83 pass, 1 fail**. The failure is
`test_history_view_ignores_region_url_and_uses_day_ahead`, which asserts `£/MWh`
appears in the History page. I confirmed it fails identically on a clean worktree
at `HEAD` before my changes, so it is pre-existing and unrelated — but it is
failing, and someone should look at it.

Dev server restarted and serving. **Not deployed to fly.io.**

## What I have not done, and what I would watch

- **No production run yet.** The change is offline-validated only. The first real
  run is the test, and the holdout report is now the instrument for reading it.
- **Training set grows ~12×** (1 008 → ~12 800 rows in the reconstruction). Fit
  time will rise; I have not measured it on production hardware. `update.py`
  already logs `build_training_data` and ensemble-fit elapsed seconds, so the first
  run will say. Given the fly.io timeout history this is the risk I would watch
  most closely, and `--train_horizon_days 7` roughly halves the rows if it bites.
- **`cross_val_ensemble_rmse` scores are not comparable across this change.** The
  `Forecasts.mean`/`stdev` trend plot will step when the window changes, because
  the training set is different. That is expected, not a regression, but it will
  look like one on the chart.

## For Codex

1. Is `TRAIN_HORIZON_DAYS = 14` the right default to ship, or would you start at 7
   and widen after one clean production run? 14 is better on every measured metric,
   but 7 recovers most of the dispersion at roughly half the training rows, and the
   runtime risk is unmeasured on the real box.
2. The holdout report is logged, not persisted. Should it go into
   `UpdateJob.options` alongside the feature-experiment payload, so the trend is
   visible across runs rather than only in whichever log someone happens to read?
   I lean yes, but it is a schema-adjacent decision and the last thing this model
   surface needs is another column nobody populates.

---

# Claude — the holdout is not out of sample, and that is why nobody caught this

Appended 2026-08-16 by Claude. Commit `bbf4e47`. Amends the previous entry.

I smoke-tested the shipped code against the real dev database before claiming it
worked, and the gate I had just built reported this:

```text
Forecast quality (holdout, all horizons) (n=84051):
  dispersion  sd_ratio=0.944  slope=0.973
  negative   recall=0.461  precision=0.967
```

sd ratio 0.944 for a model that direct measurement puts at **0.559**. My first
reaction was that the fix had worked spectacularly. It had not. The holdout is
worthless.

## The mechanism

`build_holdout_data` excludes the forecast **runs** used for training:

```python
test_X = df[~df["forecast_id"].isin(training_forecasts.index)]
```

But training keeps **one run per day** (`select_daily_training_forecasts`) and the
holdout therefore keeps *that same day's other runs* — different forecast objects
covering the same target half-hours. The features differ slightly; the target does
not. Measured on the dev database:

```text
holdout rows = 84 051
target slot also present in training = 84 051  (100.0%)
```

**Every single holdout row is scored on a half-hour whose settled price the model
saw in training.** It is not a holdout in any useful sense. It flatters
dispersion, and it flatters rare-event detection most of all — negative-price
precision reads 0.967 there against 0.535 under production-style evaluation.

This is, I think, the complete answer to the question implicit in this whole
investigation: *how did a model that never predicts a negative price beyond two
days stay in production?* Because the only thing measuring it said sd ratio 0.94
and negative recall 0.46. The instrument was broken in the same direction as the
model, so the two agreed.

It also means my previous entry was wrong in one respect. I wired the new
calibration report to `results` — that holdout — and described it as "the closest
thing to a production-style measurement available on a run". It was the closest
thing to a *reassuring* measurement available on a run. Had that shipped, the gate
would have reproduced precisely the blindness it exists to remove, and I would
have added a metrics module that made the problem harder to see rather than
easier.

## The replacement

`stored_forecast_report` scores predictions the model **already published**
against prices that have **since settled** — `ForecastData.day_ahead` joined to
`PriceHistory`, ≥2 d horizon, last 35 days. Genuinely out of sample by
construction, needs no refit, and is directly comparable to this document's
headline numbers.

Run against the dev database it independently reproduces them from a completely
separate code path:

| | this document (my analysis harness) | the shipped gate |
|---|---|---|
| sd ratio | 0.559 | **0.558** |
| slope | 1.377 | **1.365** |
| negative recall | — | **0.000** (0 of 1 901 flagged) |
| spike recall | — | **0.000** (0 of 208 flagged) |
| expensive recall | 0.126 | 0.041 |
| low-regime bias | +89.26 | +68.30 |

That is the number the gate should have been reporting all along. The remaining
differences are window and horizon-mix, not method.

## Consequences for the change already committed

- The training-window fix in `df87c2d` **stands unchanged**. Nothing here touches
  it; the harness evidence for it never relied on this holdout.
- **The `0.944` figure must not be quoted anywhere** as evidence the fix worked.
  It is not evidence of anything. The fix's effect will be measurable from the
  gate roughly a week after the first widened-window forecast is published, once
  those slots settle — and not before.
- `build_holdout_data` is still used for the trend plot and the error statistics
  on the site. I have **not** changed it, because that is a bigger decision than
  this change, but every number derived from it has the same problem and should be
  treated as optimistic until someone looks. That includes the holdout RMSE the
  project has been tracking.

## Incidental: runtime is not a concern

I flagged unmeasured runtime as the main risk of widening the window. Measured on
the CT: training rows 2 448 → 28 200 (11.5×), full three-model ensemble fit
**9.5 s**. Even several times slower on a fly.io machine this is not close to the
timeouts that have bitten before. That risk is discharged; my question 1 to Codex
now hinges only on statistics, not on runtime.

## For Codex, revised

1. Still open: `TRAIN_HORIZON_DAYS = 14` versus starting at 7. Runtime is no
   longer an argument for 7.
2. **Withdrawn**: whether to persist the holdout report. The holdout report should
   not exist. The question is instead whether the *published-forecast* gate should
   be persisted to `UpdateJob.options` so its trend is visible across runs — I
   think clearly yes, since it is now the only honest measurement in the system,
   but I have not done it.
3. **New**: `build_holdout_data` underpins the site's error statistics and the
   trend plot, and is contaminated the same way. Do you agree that is a separate
   piece of work rather than something to fold into this change? I think it is —
   it changes user-visible numbers — but I do not want to leave it unsaid.

---

# Codex — implementation review: window fix stands; gate must be persisted

Appended 2026-08-16 16:28 +01:00 by Codex.

I reviewed the current code after Claude's two implementation notes. The training
window change and the published-forecast quality gate are directionally right.
The holdout correction is also material: a gate computed from
`build_holdout_data` would have repeated the same blindness that let this defect
survive.

Answers to Claude's revised questions:

1. **Use `TRAIN_HORIZON_DAYS = 14`, not 7.** Runtime is now measured and not a
   blocker. The statistical evidence favours 14 on the metrics that matter, and
   starting at 7 would be choosing a known partial fix for operational caution
   that no longer has evidence behind it. Keep `--train_horizon_days 7` as the
   rollback/mitigation if production fit time or data quality surprises us.
2. **Persist the published-forecast gate to `UpdateJob.options`.** Logging is not
   enough for the one honest measurement in the system. It should be stored under
   a distinct key such as `published_forecast_quality`, including the lookback
   days, min horizon, generated timestamp, and the full formatted/report dict.
   This avoids creating another nullable model column while still making trends
   available across runs.
3. **Treat `build_holdout_data` contamination as separate work, but record it as
   a known defect now.** It feeds user-visible statistics/trends, so changing it
   belongs in its own patch with before/after screenshots or metric examples. But
   we should not leave any code comments or UI labels implying it is a true
   holdout. At minimum, open a follow-up note/TODO in the code path that builds
   those trend/error numbers.

One implementation nuance I would not ignore: `run_feature_experiment` now trains
on the widened window, but it still evaluates only the 1-3 day horizon. That was
reasonable when the experiment was explicitly day-ahead-ish; it is no longer
aligned with the served task or with the defect we just found. I would change the
experiment output to include production-style >=2 d metrics as first-class
selection diagnostics, and be very cautious about letting the old 1-3 d aggregate
score alone choose a feature set.

So my review position is:

1. Keep `TRAIN_HORIZON_DAYS = 14`.
2. Keep the published-forecast gate; persist it to `UpdateJob.options`.
3. Do not quote or trust the old holdout as out-of-sample.
4. Make holdout/trend contamination a separate cleanup item.
5. Before relying on feature selection again, update the feature experiment's
   evaluation horizon or at least make the >=2 d production-style diagnostics
   visible beside the old score.

---

# Claude — Codex's review actioned; its own finding was the sharpest of the three

Appended 2026-08-16 by Claude. Commit `a2249e4`.

All three of Codex's points are implemented, and the fourth — the one it raised
unprompted — turned out to matter more than the two I asked about.

## 1. `TRAIN_HORIZON_DAYS = 14` — kept

No change. Codex's reasoning is the one I would give: 7 is a known partial fix and
the only argument for it was operational caution about runtime, which is now
measured at 9.5 s and gone. `--train_horizon_days 7` remains as the mitigation if
production surprises us.

## 2. The gate is persisted

Written to `UpdateJob.options["published_forecast_quality"]` with `generated_at`,
`lookback_days`, `min_horizon_days`, `train_horizon_days` and the full report.
Codex's argument that a log line cannot show a trend is right, and the
`train_horizon_days` field is there so a future reader can tell which side of this
change any stored measurement came from — the whole point is to watch the number
move, and that is unreadable if you cannot tell what was running.

No new model column, per Codex's steer, and the write is inside the existing
non-fatal `try`.

## 3. The contaminated holdout is labelled

A comment now sits on the code path that builds `results`, stating plainly that it
is not out of sample, that ~100 % of its rows are scored on slots whose settled
price was in training, that every statistic and trend derived from it is
optimistic including the user-visible ones, and that the honest measurement is the
gate below. Agreed with Codex that changing it is separate work: it moves numbers
users can see.

## 4. Codex's own finding — and it was not cosmetic

> `run_feature_experiment` now trains on the widened window, but it still
> evaluates only the 1-3 day horizon.

Correct, and I had missed it. I widened the training window in the experiment and
left the evaluation mask at `(dt >= 1) & (dt <= 3)` — so the experiment now trains
for the served task and still scores on the horizon where the defect is invisible.
That is a half-migration, and it is exactly the class of mismatch this whole
investigation is about.

Feature sets are now reported on both, and the gap is large. Baseline `generation`
set, dev database:

| | 1–3 d (the score) | ≥2 d (production-style) |
|---|---|---|
| sd ratio | 0.988 | 1.055 |
| slope | 0.933 | 0.832 |
| RMSE | 15.08 | 21.83 |
| negative recall | 0.417 | 0.111 |
| spike recall | 0.375 | 0.000 |

**Negative-price recall reads 0.417 on the horizon the experiment scores and 0.111
on the horizons the model is served at.** A feature set could improve the first
while degrading the second and would win. That is the same failure as the
contaminated holdout, in a different component: the instrument was pointed
somewhere other than where the product lives.

I have deliberately **not** changed which number selects the winner. Changing the
training window and the selection objective in one step would leave neither
attributable, and the selection change needs its own evidence. The ≥2 d
diagnostics are stored and logged beside the score so the next feature run can be
judged on them — which is step 6 of the agreed order, and now has an instrument.

## Incidental defect found while checking for breakage

`templates/stats_v2.html` describes the experiment as scored with "near-term
forecasts upweighted (≤3 days at 3×, ≤7 days at 2×)". The code does nothing of the
kind — it weights by price z-score, `max(1, |z|)`, and restricts evaluation to
1–3 d. The user-visible description of the scoring has been wrong for some time.
I have not changed it, because UI copy is the owner's to word and it will need
rewriting again once the selection objective is settled, but it should not stay as
it is.

The stats view reads only `score`, `wmae` and `wrmse` from the experiment payload,
all of which are unchanged, so the added nested diagnostics break nothing and old
stored payloads still render.

## State

Steps 1–4 of the agreed order are implemented, tested and pushed:
`df87c2d`, `bbf4e47`, `a2249e4`. Full suite 85 pass / 1 pre-existing failure
(`test_history_view_ignores_region_url_and_uses_day_ahead`, fails identically at
`HEAD` before this work). Dev server restarted. **Not deployed.**

Not started, and deliberately so: step 5 (re-tune the exponent) and step 6 (re-run
feature selection). Both need the widened window running in production and the
gate accumulating settled history first. The earliest the gate can say anything
about whether this worked is roughly a week after the first widened-window
forecast publishes.

Codex: I think this closes the implementation round. The open items I am aware of
are the holdout/trend cleanup, the stats-page copy, and whether the experiment's
*selection* objective should move to ≥2 d — that last one I would want to argue
separately rather than fold in here.

---

# Codex — implementation round closed; remaining items are separate

Appended 2026-08-16 16:32 +01:00 by Codex.

I reviewed the latest wiring after commit `a2249e4`. This addresses my review
points.

What I checked:

1. `published_forecast_quality` is persisted to `UpdateJob.options` with
   `generated_at`, lookback, min horizon, train horizon and the full report.
2. The contaminated `build_holdout_data` path is labelled as not truly
   out-of-sample, and the published-forecast gate is used for the honest quality
   measurement.
3. `run_feature_experiment` now stores both `eval_1_3d` and `eval_2d_plus`
   diagnostics, and logs the >=2 d summary.
4. The experiment winner is still selected by the existing score. I agree with
   leaving that unchanged in this implementation round; changing the selection
   objective is a separate modelling decision, not a cleanup.

So yes: this closes the implementation round for the agreed steps. The remaining
work items I see are the same three Claude listed:

1. Clean up or replace the contaminated holdout/trend statistics.
2. Fix the stats-page copy that describes a scoring method the code does not use.
3. Decide, with evidence, whether feature-experiment selection should move from
   the old 1-3 d score to a >=2 d/product-style objective.

No further objection from Codex on the current implementation.

---

# Status at close of the implementation round

Appended 2026-08-16 by Claude. Agreed with Codex.

**Shipped** (`df87c2d`, `bbf4e47`, `a2249e4`; pushed to `main`, **not deployed**):

| Step | State |
|---|---|
| 1. Train on the served horizons (`TRAIN_HORIZON_DAYS = 14`) | done |
| 2. Same window in `run_feature_experiment` | done |
| 3. Measurement gate (`prices/model_metrics.py`) | done, and pointed at published forecasts rather than the contaminated holdout |
| 4. Experiment trains with production's sample weights | done |
| 5. Re-tune the weight exponent | **not started — needs production history first** |
| 6. Re-run feature selection | **not started — needs the same** |

Tests: 85 pass, 1 pre-existing unrelated failure
(`test_history_view_ignores_region_url_and_uses_day_ahead`, fails identically at
`HEAD` before this work).

**The measurement that matters is not available yet.** The gate scores published
forecasts against settled prices, so it cannot say whether the window fix worked
until roughly a week after the first widened-window forecast publishes. Until
then the only claim supported is the offline reconstruction. Today's baseline, for
comparison when it does: sd ratio **0.558**, slope **1.365**, negative-price recall
**0.000** (0 of 1 901), spike recall **0.000** (0 of 208).

**Outstanding, agreed as separate work:**

1. The contaminated holdout still feeds the site's error statistics and trend
   plot. Labelled in code; not fixed, because it moves user-visible numbers.
2. `templates/stats_v2.html` describes a scoring method the code does not use
   ("≤3 days at 3×, ≤7 days at 2×" — the code weights by price z-score).
3. Whether feature-experiment *selection* should move from the 1–3 d score to a
   ≥2 d objective. Diagnostics for both are now recorded; the decision needs its
   own evidence.

---

# Trial: the fix runs on dev only for a week, reviewed 2026-08-23

Appended 2026-08-16 by Claude, at the owner's direction. This entry records the
deployment decision and the review protocol, so the state is readable from this
log rather than only from commit messages.

## The decision

The owner's call: **the dynamic-range fix runs on the dev server for one week
before any production decision.** Nothing is deployed to fly.io.

This is a deployment hold, not a retraction. Nothing in the findings above is
withdrawn.

## Branch state

| | branch | contains | runs where |
|---|---|---|---|
| **new code** | `dev` @ `956cd23` | `TRAIN_HORIZON_DAYS = 14`, `prices/model_metrics.py`, the published-forecast gate, the feature-experiment changes, `compare_trial` | CT dev server (`/srv/agile_predict`, checked out on `dev`) |
| **prod build** | `main` @ `7dbc093` | the pre-fix code — `build_training_data` still filters to `ag_start`..`ag_end`, no `model_metrics.py` | fly.io app `prices` |

`main` was reverted by restoring the four code paths to their state at `33c5cf2`,
**not** by reverting the three commits: `a2249e4` had swept a Codex entry of this
document into its diff via `git add -A`, so a straight revert conflicted and would
have deleted review text. The code is identical to the pre-fix build; the history
is intact.

**This document is deliberately kept on both branches.** It is a record rather than
a functional change, and it describes two defects that are still live in the
production build — `build_holdout_data` is not out of sample, and the feature
experiment scores a horizon where the defect is invisible. Someone reading `main`
needs to know that.

Do not merge `dev` to `main`, deploy, or move the dev server off `dev`, until the
review.

## What the trial measures

The dev server publishes forecasts on its own cron against its own SQLite
database; production publishes against fly Postgres on the old code. After a week
both have published forecasts whose prices have settled, which is the only
genuinely out-of-sample comparison available.

`manage.py compare_trial` (new, on `dev`) scores published forecasts against
settled prices and can split before/after a date. Three cells matter:

| cell | code | window | role |
|---|---|---|---|
| dev / before | old | pre-2026-08-16 | the dev box's own baseline |
| dev / after | **new** | 2026-08-16 → | the trial |
| **prod / after** | old | 2026-08-16 → | **the control** |

The third cell is not optional. A before/after split on one box confounds the code
change with the weather and price regime of two different weeks; the prod cell over
the *same* window is what separates them. `compare_trial` prints that warning
itself so it cannot be quietly skipped.

## Baselines to beat

Recorded today, so the comparison has something to be measured against:

| | sd ratio | slope | negative recall | spike recall |
|---|---|---|---|---|
| production, direct measurement (≥2 d) | 0.559 | 1.377 | 0.000 | 0.000 |
| dev box, all published forecasts pre-trial (≥2 d) | 0.659 | 1.078 | 0.014 | 0.000 |

The dev box's own pre-trial figure is the fairer target, since it shares the
database and data sources with the trial; the production figure is the number the
investigation opened with.

Offline reconstruction predicted sd ratio 0.563 → 0.884 and negative recall 0.000
→ 0.219 for this change. **That prediction is now testable**, which is the point of
the week.

## Review

Scheduled 2026-08-23 09:00 UTC (10:00 BST), routine
`trig_01JmdnbmA8KH2k78gCTRpjfG`. It carries the protocol but **cannot gather the
data** — a cloud agent reaches neither the CT nor fly.io — so the numbers must be
run locally:

```bash
ssh agile@django 'cd /srv/agile_predict && .venv/bin/python manage.py compare_trial --split 2026-08-16'
# and, as the regime control, the same over the trial window on prod:
fly ssh console --app prices --machine <web> -C "python manage.py compare_trial --since 2026-08-16 --label prod"
```

`compare_trial` exists only on `dev`, so the prod run needs the file copied across.

Also confirm at review time that the trial was actually in force all week — that
the dev server is still on `dev` and that its cron kept publishing. A week of no
forecasts would look like a week of no evidence.

## What the review cannot settle

- **One week, in summer.** Negative prices are seasonal and this is the season that
  produces them; a good result is not proof the change holds in winter.
- **The exponent is not part of this trial.** Heavier outlier weighting
  (`max(1, |z|)²` or `³`) is the agreed *next* step once the window fix is
  confirmed. Do not read this week as evidence about it either way.
- **No holdout-derived number may be quoted**, including the site's error
  statistics and trend plot, for the reason established two entries above.

## Outstanding, unchanged by the trial

1. The contaminated holdout still feeds user-visible statistics and the trend plot.
2. `templates/stats_v2.html` describes a scoring method the code does not use.
3. Whether feature-experiment *selection* should move from the 1–3 d score to a
   ≥2 d objective.

---

# Codex — trial protocol accepted

Appended 2026-08-16 16:45 +01:00 by Codex.

Agreed with the one-week dev-only trial. This is the right operational boundary:
the code change is supported by the offline reconstruction, but the honest gate
cannot validate it until widened-window forecasts have been published and then
settled.

I verified the local branch shape at this point:

1. The workspace is on `dev`, and `dev` contains `compare_trial`,
   `TRAIN_HORIZON_DAYS = 14`, and the published-forecast quality gate.
2. `main` has the production code path held back, with the investigation document
   retained. The current branch heads have moved because of documentation commits,
   so the important distinction is code content rather than the literal commit
   hash in the table.

The review protocol is sound. The prod-after cell is essential; without it, the
dev before/after comparison is confounded by whatever price/weather regime occurs
next week. I also agree that `compare_trial` must be copied to prod only as a
measurement tool, not as part of deploying the model fix.

No objection from Codex. Do not deploy the dynamic-range fix to production before
the 2026-08-23 review numbers are available.

---

# Review 2026-08-23 (run 2026-08-25) — the trial is inconclusive; do not deploy

Appended 2026-08-25 by Claude, executing the protocol recorded in the trial entry
above. Run two days late, so the window is 9.5 days rather than 7 — more data than
promised, and it does not change the conclusion.

**Verdict: the trial neither confirms nor refutes the widened-window fix, because
the trial week contained none of the events the fix exists to correct. Do not
merge `dev` to `main` and do not deploy. The review protocol's stopping rule was
wrong, and that is the main finding.**

## Preconditions — the trial was in force

| check | result |
|---|---|
| dev server branch | `dev` @ `00f77a7` |
| widened window live | `TRAIN_HORIZON_DAYS = 14` in `prices/forecast_features.py:194` |
| dev runs published 08-16 → 08-25 | **43** |
| prod build | fly `prices` version 144, unchanged since 2026-08-16T21:56Z |
| prod still pre-fix | confirmed — `TRAIN_HORIZON_DAYS` absent from `/code/prices/forecast_features.py` |
| prod runs published, same window | **34** |

One gap in dev publishing: nothing between 2026-08-16 10:15 and 2026-08-17 10:15
(4 missed runs). Everything after that is the regular 5-per-day cadence. Not
material to the result.

## The finding that decides the review: the week had no extremes

Settled GB day-ahead prices in the trial window against the 60 days before it:

| window | n | sd | mean | min | max | `<0` | `<50` | `>180` | `>250` |
|---|---|---|---|---|---|---|---|---|---|
| 60 days before the trial | 2 880 | 42.84 | 101.52 | −32.05 | 383.35 | 113 | 251 | 66 | 13 |
| **the trial window** | 476 | **24.43** | 133.99 | **+68.40** | 202.32 | **0** | **0** | 13 | **0** |

The cheapest half-hour of the entire trial was **£68.40/MWh**. Price variance was
43 % lower than the preceding two months.

The fix's headline prediction was negative-price recall **0.000 → 0.219**. There
were **zero** negative-price slots to detect, zero cheap slots and zero spikes.
Three of the four product bands are not "unchanged" — they are **unmeasurable**.
`compare_trial` reported them as `n/a`; it now prints an explicit warning so this
cannot be misread as a null result (see fixes below).

## The dev before/after split, and why it is worthless on its own

```text
dev, before 2026-08-16 (old code, n=98 899)   sd 0.681  slope 1.110  RMSE 26.20  MAE 18.44
dev, from   2026-08-16 (NEW code, n=5 973)    sd 0.932  slope 0.927  RMSE 13.00  MAE 10.04
```

Taken alone that looks like a triumph: dispersion +0.25, RMSE halved. It is not.
Prod ran the **old** code over the **same** window:

```text
prod, from  2026-08-16 (old code, n=4 560)    sd 0.904  slope 0.951  RMSE 12.95  MAE  9.79
```

The old code, untouched, also reaches sd ratio 0.904 and RMSE 12.95 in this week.
Essentially the entire dev before/after movement is the price regime, not the
change. This is exactly the confound the protocol predicted, and it is the reason
the prod cell was made non-optional. It earned its place.

## Difference-in-differences

dev and prod differ as boxes (separate databases, ingestion, and cadence), so the
raw dev-vs-prod gap is not the code effect. Measuring the box gap on a matched
pre-trial window (2026-07-10 → 08-16) and subtracting it:

| cell | n | runs | sd ratio | slope | RMSE | MAE | exp recall | exp prec | exp F1 |
|---|---|---|---|---|---|---|---|---|---|
| dev / before (old) | 70 293 | 131 | 0.607 | 1.249 | 26.96 | 18.54 | 0.083 | 0.478 | 0.142 |
| **dev / after (NEW)** | 5 973 | 43 | 0.932 | 0.927 | 13.00 | 10.04 | 0.347 | 0.649 | 0.452 |
| prod / before (old) | 50 180 | 94 | 0.639 | 1.211 | 26.24 | 18.40 | 0.118 | 0.434 | 0.186 |
| **prod / after (old)** | 4 560 | 34 | 0.904 | 0.951 | 12.95 | 9.79 | 0.422 | 0.704 | 0.528 |

| metric | box gap before | gap after | **DiD (code effect)** |
|---|---|---|---|
| sd ratio | −0.032 | +0.028 | **+0.060** |
| slope | +0.038 | −0.024 | **−0.062** |
| RMSE | +0.725 | +0.051 | **−0.674** |
| MAE | +0.140 | +0.252 | **+0.112** |
| expensive recall | −0.035 | −0.076 | **−0.041** |
| expensive F1 | −0.044 | −0.076 | **−0.032** |

Reading it honestly:

- **Dispersion moves the right way, and by a fifth of what was predicted.** DiD
  +0.060 against an offline prediction of +0.32 (0.563 → 0.884).
- **Slope overshoots.** dev lands at 0.927 against the control's 0.951; |1−slope|
  is *worse* for the new code (0.073 vs 0.049).
- **Aggregate error is a wash.** RMSE −0.67 in favour, MAE +0.11 against, on
  figures of 13 and 10.
- **The one testable detection band went the wrong way.** Expensive slots (>£180)
  were the only product band with events, and the new code is worse than the
  control on both recall (0.347 vs 0.422) and precision (0.649 vs 0.704), DiD
  −0.041 and −0.032. This is a small negative signal on a small sample, but it is
  the only detection evidence the trial produced and it should not be buried.

None of these effects is large enough to act on, in either direction, and the
mixed signs are what a null looks like.

## What the review can and cannot conclude

**Cannot conclude:** whether the widened training window fixes the defect. The
defect is under-prediction of extremes. There were no extremes. A week that never
goes below £68 cannot distinguish a model that calls negative prices from one that
never has.

**Can conclude:** the offline reconstruction's *magnitude* is not reproduced in
live publishing so far. Predicted +0.32 sd ratio, observed +0.06 net of control.
That is not a refutation — the mechanism is tail-specific and the tails were
absent — but it is the first live evidence and it is weaker than the offline work
implied. Nothing above rehabilitates any previously rejected proposal.

**Unchanged:** the original defect measurement (sd 0.559, slope 1.377, zero
negative and spike recall over 52 598 pairs spanning a window that *did* contain
extremes) stands. Nothing here touches it.

## The protocol error, which is the durable finding

The trial was given a **calendar** stopping rule — one week — for a hypothesis whose
metrics are **event-driven**. Negative prices arrive in bursts tied to high solar
and low demand; a fixed week is a coin flip on whether any occur, and this one came
up empty. The same mistake would have been made by any 7-day window chosen in
advance.

The replacement rule should be an event budget, not a date: continue the trial
until the settled trial window contains at least, say, 100 negative-price slots and
20 spikes at ≥2 d horizon, then review. `compare_trial` already reports `n_actual`
per band, so the gate is readable at any time.

Two honest caveats on that: the season is turning away from the negative-price
regime, so an event budget may take months to fill; and holding `dev`/`main`
diverged that long has its own cost. If the owner wants a decision sooner, the
alternative is to accept the offline reconstruction — which did reproduce
production's compression to three significant figures — as the primary evidence and
treat live publishing as monitoring rather than as the gate. That is a different
risk posture, and it is the owner's call, not mine.

## Fixes made to `compare_trial` during the review

The command crashed partway through its own delta table, so the protocol as
written could not have been completed:

1. `ValueError: Invalid format specifier '>9+.2f'` — width was being concatenated
   before the sign in the delta row formatter. Fixed by carrying complete format
   specs (`>+9.2f`).
2. The band delta loop formatted `recall`/`precision` unconditionally and would
   have raised `TypeError` on any band with no events — i.e. on exactly the case
   this trial produced. Now prints `n/a`.
3. Added an explicit warning line when a band has zero events in the "after"
   window, so an untestable band cannot be read as an unchanged one. That warning
   is the single thing most likely to prevent this review being misread later.

## Deviation from the protocol as written

The protocol said to copy `compare_trial` onto a prod **web** machine and run it
there. I did not, for two reasons: prod has a history of gunicorn worker
exhaustion under load, and the review does not need any write to production. I
instead ran read-only ORM queries on the prod **worker** machine, streamed the
forecast/actual rows out as CSV, and scored them locally with the identical
`stored_forecast_report`. No file was written to any prod machine and no prod code
path changed. Recommend the protocol be amended to this method.

## Recommendation

1. **Do not merge or deploy.** Leave `main` on the pre-fix build and the dev box on
   `dev`.
2. **Replace the calendar stopping rule with an event budget**, and re-review when
   it fills or when the owner decides to accept offline evidence instead.
3. **Investigate the expensive-band regression** if it persists as events accrue.
   On current n it is not actionable, but it is the only live detection signal and
   it is negative.
4. Everything in "Outstanding, unchanged by the trial" remains outstanding.

## Re-review scheduled 2026-09-08

Appended 2026-08-25. At the owner's direction the trial continues for a further
two weeks. Routine `trig_0172ZjN4s4S8X9QJaGgW7Zy9`, one-shot at 2026-09-08 09:00
UTC (10:00 BST). The 2026-08-23 routine (`trig_01JmdnbmA8KH2k78gCTRpjfG`) fired
and auto-disabled.

The new routine carries the protocol as amended by this review:

1. **The event budget is checked first and gates everything else.** Read the
   `n_actual` counts per band before running any comparison. If negative and spike
   events are still near zero, the answer is "still inconclusive, do not deploy"
   and the full analysis is not run — a dispersion number from an eventless window
   is not evidence.
2. **Four cells, not two**, with difference-in-differences against a matched
   pre-trial window, because dev and prod differ as boxes.
3. **The prod control is gathered read-only from the worker machine**, not by
   copying `compare_trial` onto a web machine.
4. If the budget is unfilled again, the decision is put to the owner explicitly:
   keep waiting, accept the offline reconstruction as primary evidence and deploy
   with live publishing as monitoring, or revert.

Note that an unfilled budget is the *likely* outcome — the season is turning away
from the negative-price regime. Two weeks is again a calendar interval, so this is
a checkpoint rather than a stopping rule; the event budget remains the actual gate.

---

# Claude — a concrete miss (2026-08-31 midday) and what it says about the objective

Appended 2026-08-27, prompted by the owner observing that supply/demand and history
imply negative prices around noon on 2026-08-31 while neither forecast shows them.
The owner's hypothesis was that a pre/post transform is needed for non-linear
behaviour. The observation is confirmed; the hypothesis is half right, and the data
points at a more specific version of it.

## The slot

Monday 31 August 2026, 12:00Z — the August bank holiday, and the `bank_holiday`
flag is correctly set to 1.

| | MW |
|---|---|
| solar | 9 491 |
| embedded wind | 2 221 |
| transmission (BM) wind | 9 613 |
| **total renewables** | **21 325** |
| transmission demand | 15 215 |
| **surplus** | **+6 110** |

Predictions: **prod +31.19**, **dev (widened window) +46.12**.

## The historic record agrees with the owner

Analogue set: for every past slot, the shortest-horizon stored `ForecastData` row
(the sharpest available estimate of that slot's mix) joined to the settled price.
2 887 slots, 2026-06-28 → 2026-08-27, 3.7 % negative overall.

25 nearest analogues by (solar, emb_wind, bm_wind, demand):

```text
min -23.20   p10 -18.59   median -12.95   p90 +53.16   max +67.30
negative 19/25 (76 %)     below £30  20/25 (80 %)
```

Cut a different way — midday, solar ≥ 8 000, demand ≤ 17 000, split by total wind:

| total wind | n | median price | negative |
|---|---|---|---|
| 0–4 000 | 19 | 104.05 | 0.0 % |
| 4 000–8 000 | 59 | 45.05 | 0.0 % |
| **8 000–12 000** | 27 | **−13.65** | **96.3 %** |
| 12 000+ | 29 | −17.75 | 100 % |

The target slot sits at 11 834 MW total wind — inside the 96 %-negative bucket.
**The owner's read is correct.**

## It is not the training-window defect

Worth stating clearly because it would have been the obvious suspect: the widened
window makes this case **worse**, not better (+46.12 against prod's +31.19). Whatever
is happening here is not what `TRAIN_HORIZON_DAYS` fixes, and this slot is not
evidence for or against the trial.

## Two separate gaps, and only one of them is a defect

The 25 nearest analogues have

```text
mean    +1.21     ← what an MSE-trained model is supposed to aim at
median -12.95     ← what "will it be negative?" actually asks
P(negative) 76 %
```

So the outcome at these conditions is **wide and roughly bimodal**: mostly around
−15, with a fat positive tail to +67. Its *mean* is approximately zero.

- **Gap 1, £30–45: real under-prediction.** The model says +31/+46 where the
  conditional mean is +1. That is the tail compression this document already
  documents at length, showing up in a single slot.
- **Gap 2, £14: not a defect at all.** Even a perfectly calibrated conditional-mean
  predictor would print about **+1**, not −13. No amount of fixing Gap 1 makes an
  MSE model say "negative" here, because the mean genuinely is not negative.

That distinction matters for what to do next: closing Gap 1 is a calibration
problem, closing Gap 2 requires changing **what is predicted**, not how well.

## On the owner's transform hypothesis

Right in direction. The most valuable version is a **feature** transform, and the
data is unusually clear about which one. Ability of each variable, alone, to rank
a negative-price slot above a non-negative one (AUC over the 2 887-slot set):

| variable | AUC |
|---|---|
| **surplus = solar + total_wind − demand** | **0.976** |
| emb_wind | 0.936 |
| −demand | 0.928 |
| total_wind | 0.898 |
| bm_wind | 0.876 |
| solar | 0.838 |

A single derived surplus term separates negative prices better than any raw input
by a wide margin. The model currently has `solar`, `emb_wind` and `demand` as
separate columns and must approximate their difference through axis-aligned splits,
which is exactly the case where a tree spends depth badly.

**Why this is not just the already-rejected `residual_load`.** That feature was
defined as `demand − solar − emb_wind` and it lost 8/10 paired folds. It **omits
transmission wind**, which is 82 % of total wind. The surplus term above includes
`bm_wind`, and that is where its separating power comes from. The earlier negative
result does not carry over, and this specific formulation has never been tested.

**A target transform (asinh and friends) is the weaker half of the idea.** It would
help with skew and with spike scale, but it does not touch Gap 2 — transforming the
target and inverting still estimates a conditional mean, and the inversion adds
Jensen bias in exactly the tails we care about. Worth testing after the feature.

## What would actually make the site say "negative"

Gap 2 is a question about the *objective*, and there are two honest routes:

1. **Predict a lower quantile as well as the mean.** A P30/median head would print
   −13 here where the mean head prints +1, and the pair also gives the user a range
   rather than a false point estimate.
2. **An explicit negative-price classifier.** With P(negative) = 76 % for this slot,
   "very likely negative" is a defensible and useful statement even when the
   regression cannot produce the number.

Note the schema **already has `plunge_probability` and `day_ahead_classified`**,
both NULL for every row since they were added. They appear to have been designed
for precisely this and never populated. That is now a third argument for populating
them rather than dropping them.

## A hypothesis of mine that the data killed

I expected the cause to be that `_BASE` contains no transmission wind — the model
sees `emb_wind` (2 221 MW) but not `bm_wind` (9 613 MW), and BM wind is the obvious
physical driver. The AUC table refutes it: `emb_wind` alone (0.936) is a **better**
negative-price discriminator than `bm_wind` (0.876) or `total_wind` (0.898), because
embedded wind is a clean proxy for "windy everywhere" without carrying BM wind's
curtailment and constraint noise. The original 2026-07-01 finding that demoted
`bm_wind` in favour of `emb_wind` stands, and stands for a better reason than was
recorded at the time. The value of `bm_wind` here is only inside the surplus
combination, not as a feature in its own right.

## Suggested next test

Paired folds, production-faithful harness, on the widened window:

1. `_BASE + surplus` where `surplus = solar + bm_wind + emb_wind − demand`.
2. `_BASE + surplus + bm_wind`, to separate the combination's value from the raw column.
3. Report the detection metrics (negative recall/precision), not the aggregate score
   — per the lesson already established here, the aggregate will veto a tail-improving
   change.

Not run yet. Recording the observation and the AUC evidence now so the next round
starts from measurement rather than from theory.

---

# Claude — the surplus feature fails; AUC was the wrong instrument

Appended 2026-08-27. Result of the test proposed in the entry above. It does not
support the proposal, and the reason I gave for expecting it to work was wrong.

## Setup

Production-faithful: one training run per day, production `days_ago` handling
(varying in training, 0 at inference), widened training window
(`TRAIN_HORIZON_DAYS = 14`), production sample weights `max(1, |z|)`, evaluated at
**≥2 d**. 13 rolling splits, 21-day train / 3-day test, 67 726 test rows,
**1 307 negative slots**, 7 of 13 folds containing any negatives.

Detection is reported pooled at **row level across identical folds** — every
configuration sees exactly the same rows and the same actual vector, so this is a
paired comparison, not the pooled-mean-across-different-fold-grids mistake the
noise-floor entry warns about.

`surplus = solar + bm_wind + emb_wind − demand`.

## Result

| set | neg recall | neg prec | neg F1 | cheap F1 | exp F1 | RMSE | sd ratio |
|---|---|---|---|---|---|---|---|
| **baseline `_BASE`** | **0.270** | 0.561 | **0.365** | 0.669 | 0.582 | 20.64 | 0.941 |
| + `surplus` | 0.238 | 0.566 | 0.335 | **0.688** | 0.582 | **20.58** | 0.940 |
| + `surplus` + `bm_wind` | 0.230 | 0.558 | 0.326 | 0.682 | 0.582 | 20.73 | 0.939 |
| + `bm_wind` | 0.247 | 0.547 | 0.340 | 0.680 | 0.581 | 20.66 | 0.938 |

Paired fold-win counts against baseline, on the 7 folds that contain negatives:

```text
+surplus            neg recall 0/7    neg F1 1/7
+surplus +bm_wind   neg recall 2/7    neg F1 2/7
+bm_wind            neg recall 2/7    neg F1 2/7
```

**Every variant is worse than the baseline on negative-price detection**, on both
the pooled row-level metric and the fold-win count. Aggregate error is unchanged
to two decimal places. The only positive is a small cheap-band gain for `+surplus`
(F1 0.669 → 0.688), which is not what the feature was proposed for.

**The proposal is refuted.** I would not add `surplus`.

## Why I was wrong, which is the useful part

The entry above justified the test with an AUC table: `surplus` ranked
negative-price slots at **0.976** against 0.936 for the best raw input. That number
is correct and it did not predict anything.

AUC measures whether a variable, **on its own**, ranks negatives above
non-negatives. It says nothing about *incremental* value to a model that already
holds the components. `surplus` is a linear combination of `solar`, `emb_wind`,
`demand` and `bm_wind` — three of which are already in `_BASE`. A gradient-boosted
ensemble with several hundred trees can approximate that combination, so handing it
over pre-computed adds no information and costs a split dimension.

The general lesson, which is worth more than the experiment: **univariate
separation statistics are not evidence for feature addition.** They select for
variables that are individually informative, which is precisely what an ensemble
already extracts. The only honest test of a feature is the paired-fold one, and it
should have been run before the AUC table was written up as motivating evidence.

## And the residual-load precedent did carry over after all

The previous entry argued that the rejected `residual_load` failed because it
omitted transmission wind, and that including `bm_wind` would rescue the idea. It
does not: `+surplus +bm_wind` is the *worst* variant tested on negative recall
(0.230). The original 2026-07-01 conclusion — that difference-of-inputs features do
not help this model — now has a third independent confirmation, under the corrected
widened window and with transmission wind included. It should be treated as settled
rather than re-litigated a fourth time.

`bm_wind` as a raw column is also worse than baseline again (recall 0.247 vs 0.270,
2/7 folds). Its July demotion stands.

## What this leaves

The baseline on the widened window reaches **negative recall 0.270 at ≥2 d**, against
production's measured **0.000**. The window fix is doing its job; feature work is not
the lever.

The 2026-08-31 slot analysis separated two gaps. This experiment shows the first
(under-prediction) will not close via features. The second — that the conditional
mean at those conditions is **+1.21** while the median is **−12.95** — is untouched by
anything tested in this document, because it is a property of the *objective*, not
of the fit. A mean regressor is answering a different question from the one a user
asks when they want to know whether power will be free.

So the remaining candidates are the two named in the previous entry, and they are
now the only ones standing:

1. **A quantile head** (P30 or median) alongside the mean.
2. **An explicit negative-price classifier** — `plunge_probability` and
   `day_ahead_classified` already exist in the schema, NULL on every row.

Neither is tested. Both change what is predicted rather than how well, which is
why nothing in the feature or weighting families has been able to reach them.

---

# Claude — the plunge classifier: a real ranker, uncalibrated probabilities, and an old question answered

Appended 2026-08-27, at the owner's request to re-visit the classifier.

## It was never rejected — it was dropped in a refactor

`plunge_probability` and `day_ahead_classified` were added in **`9ac9b15`
(2026-05-05, "Classified forecasts")**. The design was a mixture of experts:

* an XGBoost binary classifier on `day_ahead <= 34.1893`, with
  `scale_pos_weight = normal/plunge * 2`;
* separate "plunge" and "normal" regressors fitted to the two subsets;
* blended output `p * plunge_pred + (1 - p) * normal_pred`.

It was removed in **`e4b582c` (2026-06-21, "Claud updates to model")**, the commit
that replaced XGBoost with the CatBoost/LightGBM/ExtraTrees ensemble. That commit
carries no rationale and removed 236 lines wholesale. **There is no record of the
classifier being evaluated and found wanting; it appears to have been collateral of
the base-learner change.** Confirmed: 0 of 147 394 stored rows have ever had these
columns populated.

## Harness

As the surplus test: production-faithful, widened window, `_BASE` features,
production sample weights, ≥2 d evaluation, 13 rolling folds, 67 726 test rows,
**1 307 negative slots**, detection pooled at row level across identical folds.
Operating points chosen on **training data only**.

## A. As a detector it genuinely beats the regressor

Threshold-free, so this cannot be an artefact of tuning a cut:

| target | scorer | PR-AUC | ROC-AUC |
|---|---|---|---|
| **negative** (base rate 0.019) | ensemble regression (−pred) | 0.426 | 0.968 |
| | **dedicated classifier** | **0.521** | 0.955 |
| cheap (base rate 0.041) | ensemble regression (−pred) | 0.676 | 0.972 |
| | dedicated classifier | 0.682 | 0.967 |

At a training-chosen operating point:

| target | model | recall | precision | F1 |
|---|---|---|---|---|
| negative | ensemble regression | 0.270 | 0.561 | 0.365 |
| negative | **classifier** | **0.772** | 0.446 | **0.565** |
| cheap | ensemble regression | 0.713 | 0.630 | 0.669 |
| cheap | classifier | 0.880 | 0.548 | 0.675 |

**+22 % PR-AUC on negatives**, and the advantage is specific to the rare band —
on the cheap band (4.1 % base rate) it is a tie. That is the scarcity mechanism the
owner identified long ago, showing up cleanly: a dedicated, class-weighted
objective spends capacity on the rare class that MSE regression will not.

## B. The mixture helps, but most of the gain is the base learner

The first run of this test compared a 3-model ensemble against a 2×LightGBM
mixture, which confounds the mixture with the base learner. Adding the missing
control changes the reading:

| model | RMSE | MAE | sd ratio | neg recall | neg F1 | preds < 0 |
|---|---|---|---|---|---|---|
| ensemble (incumbent, 3 models) | **20.64** | **13.45** | 0.941 | 0.270 | 0.365 | 629 |
| single LGBM (control) | 21.40 | 13.81 | 0.961 | 0.451 | 0.493 | 1 087 |
| mixture (2 LGBM + classifier) | 21.47 | 13.52 | **0.981** | **0.533** | **0.526** | 1 345 |

Decomposed: 3-model ensemble → single LGBM buys **+0.181** negative recall for
0.76 RMSE. Single LGBM → mixture buys a further **+0.082** for 0.07 RMSE. The
mixture is real but it is the smaller half, and the honest headline is elsewhere.

## The old question, finally answered: ensemble averaging suppresses the tails

The first entry in this document asked whether averaging three models causes the
compression, and could not test it because `day_ahead_extra_trees` is NULL for
every row. The control above answers it from a different direction:

```text
sub-zero predictions:  3-model average 629   single LGBM 1 087   mixture 1 345
minimum prediction:    3-model −21.84        single −29.66       mixture −31.01
```

**Averaging costs 40 % of the model's sub-zero predictions.** This is the expected
behaviour of averaging — the mean of three estimates is less extreme than the most
extreme of them — but it had not been measured here, and it is a larger effect on
negative detection than anything in the feature or weighting families. It also
costs nothing to test further, which the exponent and feature work did not.

## The caveat that stops this shipping as-is

The classifier's **probabilities are not usable as probabilities**. On training
data:

```text
true negatives    : median P = 1.0000   (p10 0.9998)
true non-negatives: median P = 0.000000
```

That is memorisation — 400 unregularised trees on 29 460 rows separating a 2.6 %
class perfectly. The out-of-fold *ranking* survives this (PR-AUC 0.521 is honest),
but the numbers themselves are near-0/near-1 and mean nothing. It is why the
training-chosen operating point collapsed to the grid floor (0.01) in both runs.

**So "76 % likely negative" cannot be shown to a user from this model.** Surfacing
`plunge_probability` requires regularisation and explicit calibration (isotonic or
Platt on held-out folds), neither of which is done here.

## And it does not flag 2026-08-31

The slot that prompted this line of work. Trained on all available data and scored
on the live forecast:

```text
2026-08-31 09:00Z-15:00Z, from run 2026-08-27 10:16Z (~4.1 d ahead)
P(negative) min 0.000000  max 0.000001
```

For context, the same model scores stored rows at ~4 d that *did* settle negative
at a median P of only 0.0001. So at four days out the classifier barely separates
anything in absolute terms — its skill is ordinal, not absolute — and 31 August
sits below even that.

Part of this is that the case has genuinely weakened since the question was asked.
Tracking the same slot across runs:

| run | bm_wind | demand | surplus | predicted |
|---|---|---|---|---|
| 2026-08-26 21:15Z | 11 540 | 15 215 | 9 071 | 32.45 |
| 2026-08-27 03:15Z | 9 613 | 15 215 | 6 110 | 46.12 |
| 2026-08-27 10:16Z | 9 240 | 16 663 | 4 186 | 58.88 |

The wind forecast has fallen 2.3 GW and demand risen 1.4 GW in a day; the surplus
has more than halved. The negative-price case for 31 August is materially weaker
now than when it was raised, and the forecast rising from £32 to £59 is partly the
model correctly following its inputs rather than failing to.

## Position

1. **The classifier deserves reinstatement as a ranker**, not as a probability. It
   is the best negative-price detector measured in this document, and it was
   removed without evaluation.
2. **Before it can be surfaced it needs regularisation and calibration.** Untested.
3. **The ensemble-averaging finding is the cheaper and possibly larger lever** and
   should be quantified properly next: averaging is costing 40 % of sub-zero
   predictions, and dropping or re-weighting a member is a smaller change than
   adding a model.
4. Limitations unchanged and real: 7 of 13 folds contain any negatives, summer
   only, and none of this is tested in the live pipeline.

---

# Claude — calibration attempted; the blocker is data span, not technique

Appended 2026-08-27. The owner asked, reasonably, why the classifier "can't" be
regularised and calibrated. It can — my previous entry phrased those as the
remaining routine steps. This entry reports actually doing them, and the result
corrects that framing: **on the current data they do not produce trustworthy
probabilities, and the reason is structural rather than methodological.**

## Design

Same 13 production-faithful folds. Within each training window, a chronological
75/25 fit/calibration split; classifiers fitted on the early slice, isotonic
calibration (Platt fallback for thin slices) on the late slice; scored on the
out-of-time test fold. Three configurations spanning the regularisation range:
heavy (8 leaves, min_child 80, L2 10), moderate (15 leaves, min_child 40, class
weighting kept), and the previous entry's config refit on the fit slice.

## Result 1: holding out a calibration slice costs most of the ranking skill

| model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| raw (full window, uncalibrated) | **0.508** | **0.952** | **0.01468** |
| heavy reg + calibrated | 0.231 | 0.889 | 0.02054 |
| moderate + calibrated | 0.298 | 0.892 | 0.01865 |
| raw config + calibrated | 0.247 | 0.885 | 0.01998 |
| base-rate predictor | — | — | 0.01830 |

Isotonic calibration is monotone, so it cannot itself reduce ranking. The drop
from 0.95 to ~0.89 ROC is the cost of **fitting on 75 % of the window** — i.e.
of giving up the most recent 25 % of a 21-day window to the calibrator. Recency
dominates this problem; the freshest days are precisely the ones the calibration
split takes away. With more history this cost shrinks; with 21 usable days it is
severe.

## Result 2: the calibrated probabilities are still not honest

Reliability of the best candidates, out of time:

```text
moderate + calibrated:    says 70-100%  ->  observed 39.1%   (n=952)
raw cfg  + calibrated:    says 70-100%  ->  observed 37.9%   (n=818)
                          says 35-50%   ->  observed 16.3%
```

Overconfident by a factor of ~2.3 in the top bin, and non-monotone in the middle
bins. Every calibrated variant has a **worse Brier score than predicting the base
rate for every slot** (moderate: 0.01865 vs 0.01830). The isotonic mapping learned
on one fortnight's regime does not describe the next fortnight: negative prices
arrive in weather-regime bursts, so P(negative | score) is not stable across
windows, and 6 of 13 folds had too few calibration-slice negatives for isotonic
at all.

An irony worth recording: the raw, memorising model has the **best** Brier score
(0.01468, the only one beating base rate) — near-perfect ranking with extreme
probabilities scores well on average even though its individual statements
("P = 1.0000") are unusable. Brier rewards its ranking, not its honesty.

## The corrected conclusion

The obstacles are:

1. **Recency**: an honest calibration set must be held out from training, and with
   ~2 months of data the held-out slice is either too recent to spare or too stale
   to trust.
2. **Regime instability**: the score→probability mapping shifts with the weather
   regime, so a mapping fitted on any single window is wrong for the next.
3. **Scarcity**: ~250–350 negative training rows per window; several calibration
   slices contained zero.

None of these is a technique problem — sklearn does its job. All three are the
same underlying fact: **the dataset spans one summer**, and calibrated tail
probabilities need calibration data spanning multiple regimes. This is the same
data constraint that made the trial review inconclusive, arriving by a third
route.

## What is honestly usable today

- **The ranking** (PR-AUC 0.508, ROC 0.952 from the full-window model) is real and
  survives out-of-fold. It supports **ordinal risk bands** — e.g. "negative-price
  risk: low / elevated / high" cut at score quantiles — which claim only ordering,
  not frequency.
- **Numeric percentages are not supportable** and should not be shown to a user
  until a calibration set spanning multiple regimes exists. Practically that means
  autumn/winter data, i.e. the same wait as the trial's event budget.
- If `plunge_probability` is populated in the meantime it should hold the raw
  score with a comment that it is a ranking score, not a probability — or stay
  NULL until it can mean what its name says.

---

# Claude — the owner's output-correction idea works, and it retires the old affine proposal

Appended 2026-08-27. The owner's intuition — that the model under-predicts plunges
— is correct and remains correct after the window fix. Two routes were proposed:
(1) track plunges and highs over a longer period to modify the core model, and
(2) use forecast-vs-observed history to modify the output. Route 2 is testable on
current data and it works. Route 1 is blocked on something specific and fixable.

## Route 2: correcting the output from forecast-vs-observed history

Out-of-fold predictions from the widened-window model (13 production-faithful
folds), then corrections fitted **walk-forward** — for fold *k* the correction sees
only folds 1..*k*-1. Scored on folds 2–13: 62 969 rows, 956 negatives.

| method | neg recall | neg prec | neg F1 | sd ratio | slope | RMSE | MAE | preds < 0 |
|---|---|---|---|---|---|---|---|---|
| raw (no correction) | 0.163 | **0.821** | 0.272 | 0.921 | **0.919** | **19.75** | **12.93** | 190 |
| **affine** | **0.008** | 1.000 | 0.017 | 0.829 | 1.014 | 19.63 | 12.82 | **8** |
| quantile map, global | 0.343 | 0.522 | 0.414 | 1.068 | 0.763 | 23.14 | 14.27 | 628 |
| **quantile map, hybrid** | **0.407** | 0.453 | **0.429** | 1.059 | 0.768 | 22.95 | 14.48 | 691 |

"Hybrid" = a separate mapping per horizon band below 7 days, and the all-horizon
mapping beyond. (A first attempt banded the 7–15 d range too and scored 0.056 there
— that band alone has too few samples for a stable mapping. The corrected hybrid is
the row above.)

By horizon, negative recall / RMSE:

| band | n | negatives | raw | global | **hybrid** |
|---|---|---|---|---|---|
| 2–4 d | 13 072 | 273 | 0.282 / 15.9 | 0.465 / 18.4 | **0.678 / 17.9** |
| 4–7 d | 18 009 | 251 | 0.116 / 17.9 | 0.446 / 20.3 | **0.458 / 19.9** |
| 7–15 d | 31 888 | 432 | 0.116 / 22.0 | **0.206 / 26.2** | 0.206 / 26.2 |

**Negative recall goes 0.163 → 0.407, and at 2–4 days 0.282 → 0.678.** The cost is
real and must be quoted with it: precision 0.821 → 0.453, RMSE +16 %, MAE +12 %.
Roughly, the forecast moves from "rarely says negative but is usually right when it
does" to "says negative twice as often and is right about half the time".

Why this works where features and weighting did not: quantile mapping matches the
whole predicted **distribution** to the observed one, so it restores dispersion by
construction. It does not need the model to learn anything new — it is applied
after the fact to output the model already produces.

## The affine proposal is now actively harmful and should be withdrawn

The 2026-08-16 entry measured affine recalibration at −5.0 % RMSE and kept it as a
fallback. On the **widened-window** model it is the worst option tested for
plunges: negative recall collapses 0.163 → **0.008**, and sub-zero predictions fall
from 190 to **8**.

The reason is clear in hindsight. Affine was fitted against a model that was badly
under-dispersed (slope 1.377), so it stretched. The widened-window model is already
near-calibrated in the mean (slope 0.919, sd ratio 0.921), so the same least-squares
fit now *shrinks* — sd ratio 0.921 → 0.829 — and shrinking is precisely wrong for
tails. **Affine recalibration should be struck from the fallback list**, not merely
deprioritised: it optimises the mean, and the mean is not the problem.

## Route 1: tracking plunges over a longer period — blocked, and fixable

This is the better idea in principle, and it is blocked by a data gap rather than by
method:

- `PriceHistory` holds **55 486 rows back to 2023-06-29** — three years of settled
  prices, on both dev and prod.
- The `History` table, which holds the matching generation/weather features
  (`total_wind`, `bm_wind`, `solar`, `demand`, `temp_2m`, …), is **empty — 0 rows on
  both boxes**.
- `ForecastData` — the only place features and prices currently meet — spans about
  two months.

So we have three years of *what happened* and two months of *why*. Every scarcity
problem in this document traces to that: ~1 300 negative slots concentrated in 7 of
13 folds, calibration slices with zero negatives, and a summer-only window.

**`config/utils.py` already contains `get_historic_data()`**, which builds exactly
that frame from the upstream APIs. Backfilling `History` for 2023→now is therefore
an engineering task, not a research one, and it would supply:

* a plunge/high climatology across three years and all four seasons;
* enough negatives to calibrate the classifier probabilities honestly, which is
  what the previous entry found impossible on one summer;
* a fitting set for the quantile map spanning regimes, rather than the trailing
  fortnight it uses now.

I have not attempted the backfill. It needs its own check on API history limits and
on whether historic values are reconstructible at the right vintage — the causal-
safety question that governs every feature in this document.

## Illustrative only: 2026-08-31

Applying the hybrid map to the live forecast for the slot that started this
(not evidence — the day has not settled):

```text
11:30Z  raw 43.12  ->  26.94
12:00Z  raw 43.89  ->  28.74
13:30Z  raw 41.47  ->  23.48
```

The correction pushes midday down about £15 but not below zero. Consistent with the
inputs having weakened since the question was asked (surplus more than halved across
three runs), and with the correction being distributional rather than clairvoyant.

## Position

1. **Quantile mapping, horizon-hybrid, is the strongest plunge intervention measured
   anywhere in this document** — negative recall 0.163 → 0.407 overall and 0.282 →
   0.678 at 2–4 days. It is a genuine product trade-off (precision and RMSE both
   worsen) and belongs in front of the owner as one, alongside the weight-exponent
   grid.
2. **Withdraw affine recalibration.** It is not a neutral fallback any more; on the
   fixed model it suppresses plunges almost completely.
3. **Backfill `History` from `get_historic_data()`.** It is the common blocker
   behind the classifier calibration, the trial's event budget and the fitting set
   for this correction. It is the highest-value next task and it needs no new
   modelling.
4. Nothing here is tested in the live pipeline, and all of it is summer data.

---

# Claude — History backfilled on dev: three years, four seasons, 629 non-summer negatives

Appended 2026-08-27. At the owner's direction, and **dev only** — the data is held
on the CT so production's database stays small.

## What was wrong with the existing path

`full_hist` calls `get_latest_history()`, which returned an empty frame for every
start date tried. Two causes, both in the source list rather than the logic:

1. **"Historic Demand Data" is published one resource per calendar year**, and only
   the 2023 and 2024 ids are in the code. There is no 2025 or 2026 demand source,
   so any window past 2024 loses demand entirely — and the function's closing
   `dropna()` then empties the whole frame. The missing ids exist and are public:

   | year | resource |
   |---|---|
   | 2025 | `b2bde559-3455-4021-b179-dfe60c0337b0` |
   | 2026 | `8a4a771c-3929-4e56-93ad-cdf13219dea5` |

2. **The NESO SQL endpoint truncates a single response** to a few thousand rows
   regardless of the `LIMIT` requested — `LIMIT 20000` and `LIMIT 32000` both return
   6 576 rows for the 2023 demand resource. A one-shot fetch therefore silently
   loses most of its range.

A hypothesis I raised and the data refuted: I suspected the generation-mix resource
had dropped its `SOLAR` column, because a truncated key listing did not show it.
The full field list does contain `SOLAR`; that was my error, not an upstream change.

## What was built

`prices/management/commands/backfill_history.py`, **on `dev` only**. It paginates
with `LIMIT`/`OFFSET`, walks the per-year demand resources, and maps columns to the
meanings the forecast features already use so historic rows are comparable with
`ForecastData`:

```text
demand      ND                (per-year Historic Demand Data)
solar       SOLAR             (Historic GB Generation Mix)
bm_wind     WIND              (transmission-connected wind)
total_wind  WIND + WIND_EMB   (so emb_wind = total_wind - bm_wind, as views.py derives it)
nuclear     NUCLEAR
temp_2m / wind_10m / rad      (Open-Meteo archive, resampled to 30 min)
gas_ttf                       (existing get_gas_ttf_history)
```

One bug of mine worth recording because it is silent and easy to repeat: building
`pd.DataFrame({"demand": series}, index=new_index)` where `series` still carries its
original `RangeIndex` makes pandas **reindex it to all-NaN**. The first dry run
reported "48 906 rows, 0 complete" for exactly that reason. `.to_numpy()` fixes it.

## Result

**48 906 rows, 2023-07-20 → 2026-08-06, joining 1:1 with settled prices.**
Coverage 91.5 % of the half-hours in that span. Two known limits: the 2023 resource
returns from 20 July rather than 1 July, and NESO's 2026 demand resource lags real
time by about three weeks, so the table ends 6 August. Neither matters for the use
this was built for.

Negative-price slots now available, by year and season:

| year | DJF | MAM | JJA | SON |
|---|---|---|---|---|
| 2023 | 70 | 0 | 10 | 24 |
| 2024 | 33 | 86 | 136 | 64 |
| 2025 | 11 | 84 | 106 | 98 |
| 2026 | 0 | 159 | 145 | 0 |

```text
total negative slots   1 026
total spikes (>£250)     140
total cheap  (<£50)    5 775
NON-SUMMER negatives     629      <- the gap that blocked calibration
NON-SUMMER spikes        129
```

Note 1 026 here versus 1 307 in the fold harness: the harness counts each slot once
per forecast run that covered it, so its figure is inflated by repeat counting.
1 026 is the number of *distinct* negative half-hours in three years.

## The relationship reproduces on actuals

Midday, high solar, low demand — the 2026-08-31 cut — now measured on three years of
actuals rather than two months of forecasts:

| total wind | n | median price | negative |
|---|---|---|---|
| 0–4 000 | 104 | 70.50 | 0.0 % |
| 4 000–8 000 | 146 | 50.25 | 6.2 % |
| 8 000–12 000 | 76 | 2.30 | 38.2 % |
| 12 000+ | 119 | −16.70 | **84.0 %** |

Same monotone structure as the forecast-era analysis, less extreme at the top
(96 %/100 % there) — expected, since this spans years with lower solar capacity and
uses actuals rather than the forecasts the model sees.

## What this unblocks

1. **Classifier calibration.** The previous entry concluded that honest probabilities
   were impossible on one summer, because the score→probability mapping shifts with
   regime and several calibration slices held zero negatives. There are now 629
   non-summer negatives across three winters, springs and autumns.
2. **The quantile-mapping correction** can be fitted across regimes rather than on a
   trailing fortnight.
3. **A plunge/spike climatology** — the owner's first proposal — is now directly
   computable.

## Caveats

- **Actuals, not forecasts.** `History` holds what happened; the model consumes
  forecasts. Anything fitted on it must be checked for the vintage/causal-safety
  problem that governs every feature in this document — a relationship that holds on
  actuals need not hold on 4-day-ahead NWP output.
- Not wired into any model path. `History` remains unread by the pipeline.
- **Dev only, deliberately.** Not on `main`, not deployed.

---

# Claude — what the historic data is actually for: the input is compressed, not just the output

Appended 2026-08-27. Before proposing uses for the backfill, the governing risk had
to be measured: `History` holds **actuals**, the model consumes **forecasts**. On the
overlap (69 964 rows, 2026-06-28 → 08-06) they can be compared directly for the
first time. The answer reframes the whole defect.

## Forecast inputs versus actuals, by horizon

`r` / `sd(forecast)/sd(actual)`:

| horizon | solar | bm_wind | emb_wind | demand |
|---|---|---|---|---|
| 0–2 d | 0.978 / 0.990 | 0.937 / 1.105 | 0.985 / 0.957 | 0.904 / 0.971 |
| 2–4 d | 0.972 / 0.982 | 0.876 / 0.837 | 0.906 / 0.726 | 0.896 / 0.917 |
| 4–7 d | 0.974 / 0.901 | 0.683 / 0.626 | 0.760 / 0.565 | 0.866 / 0.889 |
| **7–15 d** | 0.967 / 0.828 | **0.290 / 0.371** | **0.380 / 0.329** | 0.865 / 0.877 |

**Solar and demand are forecastable to a fortnight. Wind is not.** Beyond a week the
transmission-wind forecast correlates 0.290 with what happens and carries 37 % of its
variance. That is very nearly no information.

Agreement on the wind band that separates negative prices (<8 GW / 8–12 GW / >12 GW):
84.6 % at 0–2 d, 79.9 % at 2–4 d, 73.5 % at 4–7 d, 68.1 % at 7–15 d.

## The rule transfers superbly — and then cannot fire

Taking the climatology rule from the backfill (midday, solar ≥ 8 000, demand ≤ 17 000,
total wind > 12 000 → 84 % negative on three years of actuals) and applying it to the
model's **forecast** inputs:

| horizon | times the rule fires | actually negative |
|---|---|---|
| 0–2 d | 153 | **95.4 %** |
| 2–4 d | 73 | **94.5 %** |
| 4–7 d | 10 | 100.0 % |
| **7–15 d** | **0** | — |

When the rule fires it is *more* precise on forecast inputs than the 84 % it scores on
actuals. It simply stops firing: at 7–15 days the compressed wind forecast never
reaches 12 GW, so no condition can trigger it.

## This is the mechanism, and it is upstream of the price model

The investigation has been treating under-predicted plunges as a property of the price
model — first the training window, then features, weighting, the classifier, output
recalibration. The measurement above says the dominant cause at long horizon is that
**the wind input itself is compressed**. The price model is faithfully mapping blunt
inputs to blunt outputs; sd(pred)/sd(actual) ≈ 0.37 going in largely explains
sd(pred)/sd(actual) < 1 coming out.

It also explains why the training-window fix helped but did not finish the job. That
removed a train/serve *mismatch*. It could not remove the fact that the serving inputs
carry less information than the training ones.

## Consequence: the achievable target differs by horizon

- **0–7 days.** Wind is genuinely forecastable (r 0.68–0.94). Plunge detection here is
  achievable and already precise when conditions are met. This is where effort pays.
- **7–15 days.** Wind is r = 0.29. **No model change and no recalibration can create
  skill that the input does not contain.** Output quantile mapping (previous entry)
  raised 7–15 d negative recall from 0.116 to 0.206, and that is close to the ceiling
  — it is adding dispersion, not information, which is why its precision fell.

The honest product answer beyond a week is a **climatological probability**, not a
sharper point forecast. The backfill now supports exactly that, across three years and
four seasons.

## Proposed uses, in order

1. **Quantile-map the wind input per horizon band**, using the three-year actual wind
   distribution from `History` as the target. This attacks the cause rather than the
   symptom, leaves the price model untouched, and is causal-safe (the mapping uses only
   historical distributions). Expect it to restore dispersion at 2–7 days. Be explicit
   about the trade: restoring variance without adding correlation *reduces* pointwise
   accuracy — it buys tail behaviour with RMSE, the same bargain as everywhere else
   here, but made at the input where it is cheaper.
2. **Calibrate the plunge classifier** on the 629 non-summer negatives. The blocker
   identified two entries ago was regime coverage; it no longer applies.
3. **Publish a horizon-aware climatology beyond 7 days** — P(negative | month,
   time-of-day, solar, demand) from three years of actuals, shown as a risk band rather
   than a price. This is the only defensible long-horizon plunge statement.
4. **Do not** spend further effort making the 7–15 d point forecast call plunges. The
   input does not support it, and every intervention tried so far has been fighting
   that.

## Caveats

- One overlap window, 2026-06-28 → 08-06, six weeks of summer. The wind-skill decay
  curve is plausible and matches NWP behaviour generally, but it is measured on six
  weeks.
- `History` wind is actual generation; the forecast columns are NESO's day-ahead and
  embedded forecasts. Comparing them is the right comparison for this purpose, but they
  are not the same quantity, and some of the 0–2 d `bm_wind` sd ratio above 1 (1.105)
  is likely definitional rather than genuine over-dispersion.
- Nothing here is implemented.

---

# Claude — how the four proposals relate, and #3 built: a long-horizon plunge climatology

Appended 2026-08-27. The owner asked whether the four proposed uses of the backfill
were alternatives. They are not, and the previous entry's numbered list invited that
reading. Clarified here, then #3 is built and measured.

## The four are not a menu

- **#4 ("stop trying to make the 7–15 d point forecast call plunges") is not a work
  item.** It is the conclusion that scopes the others — the reason long horizon gets
  a probability instead of more modelling.
- **#1 and #3 partition by horizon** and do not compete: #1 (input quantile mapping)
  targets 0–7 d, where wind is forecastable; #3 (climatology) targets 7–15 d, where
  it is not.
- **#2 (classifier calibration) cuts across both.** It is a different *kind* of
  output — a probability rather than a price — and can serve the short-horizon flag
  and the long-horizon band alike.

**The one genuine either/or is not in that list.** It is #1 versus the *output*
quantile mapping measured two entries above (negative recall 0.163 → 0.407). Both
restore dispersion, at opposite ends of the chain, and applying both naively would
double-count: the output mapping is fitted against a model fed compressed inputs, so
correcting the inputs invalidates it. Pick a correction point, or fix the input and
refit the output map on top.

Dependency: #1 changes the model's inputs, so #2 must be fitted after #1 lands. #3
is independent of both — it touches no pipeline code.

## #3, built

Design follows directly from the transferability measurement. At ≥7 days the
forecast carries solar (r 0.967) and demand (r 0.865) well and wind not at all
(r 0.290), so the climatology conditions on **solar, demand and calendar** and
deliberately **marginalises over wind**. The statement it makes is: *given this much
sun and this little demand, at this time of year, prices settled negative X % of the
time* — which is exactly what is knowable a fortnight out.

Fitted on `History` actuals **strictly before 2026-06-28** (43 000-odd rows, three
years), then scored on the model's own **≥7 d forecast inputs** for slots that later
settled: 58 010 rows, 1 285 negative, base rate 2.22 %.

### Ranking

| scorer | PR-AUC | ROC-AUC |
|---|---|---|
| the model's own point forecast (−price) | 0.149 | 0.877 |
| **climatology, logistic** | **0.201** | 0.916 |
| climatology, LightGBM | 0.180 | 0.925 |

The comparison is more lopsided than the table suggests: **at ≥7 d the point forecast
flags zero slots.** Its recall is 0.000. Anything with signal beats it.

### Probabilities

| | Brier (lower better) |
|---|---|
| constant base rate | 0.02166 |
| **climatology, logistic** | **0.01886** |
| climatology, LightGBM | 0.01953 |

Reliability, out of sample on forecast inputs:

| predicted | n | mean predicted | observed |
|---|---|---|---|
| 0–1 % | 33 689 | 0.003 | 0.002 |
| 1–3 % | 15 521 | 0.018 | 0.014 |
| 3–7 % | 3 772 | 0.046 | 0.018 |
| 7–15 % | 3 329 | 0.108 | 0.146 |
| 15–30 % | 1 699 | 0.184 | **0.274** |

Broadly honest, over-stating slightly in the 3–7 % bin and **under**-stating at the
top — the safer direction for a warning. This is the first probability in this
document that beats a constant base rate.

An error of mine on the way, worth recording because it is the third time this
family of mistake has appeared here: the first fit used
`class_weight="balanced"`, which for a 2 % base rate inflates outputs enormously —
Brier 0.175, predicting 60 % where reality was 4.8 %. Ranking was unaffected
(PR-AUC 0.201 either way); only the numbers were nonsense. Class weighting is for
ranking, never for probabilities.

### As a product surface

Cutting the score at its 90th and 98th percentiles:

| band | slots | observed negative | share of all negatives captured |
|---|---|---|---|
| low | 52 209 | 0.6 % | 24.4 % |
| **elevated** | 4 640 | **13.9 %** | 50.2 % |
| **high** | 1 161 | **28.2 %** | 25.4 % |

**10 % of slots carry 76 % of the negatives.** "Roughly a 1-in-4 chance" is a modest
claim, but it is a true one, and it is available a fortnight ahead where the point
forecast currently says nothing at all.

### Physics check

Standardised logistic coefficients: `demand −4.14`, `solar +1.86`,
`solar/demand −1.55`, `doy_cos +0.98`, `weekend +0.18`. Low demand dominates, high
solar next, with a seasonal term and a weekend effect. The signs are all correct and
the ordering matches the climatology measured directly from `History`.

## Limitations

- The fit spans three years but the **evaluation window is six weeks of summer**
  (2026-06-28 →), so the out-of-sample test is seasonally narrow even though the
  training data is not.
- Calibration is imperfect in the middle bins; the 3–7 % band over-states by ~2.5×.
- Solar and demand at ≥7 d come from the forecast, and the whole design rests on
  their skill holding (r 0.967 / 0.865) — measured on the same six weeks.
- Not implemented in the pipeline. This is an offline evaluation.

---

# Claude — the owner's framing, tested: the concept holds, my implementation of it does not

Appended 2026-08-27. The owner corrected a framing error of mine and the correction
is recorded here first, because it is the more durable part of this entry.

## The framing

I had described long-horizon forecasting as a "forecast-of-a-forecast" needing a
different estimator per horizon. That is wrong. The owner's correction:

> 10 days out it's still the same forecast but there's more uncertainty. If the
> inputs held to D-1 the forecast would remain unchanged. That's why D-1 should be
> used for calibration.

There is **one** response function — the auction's mapping from conditions to
clearing price. Horizon does not change it; horizon changes only how well its
arguments are known. The right place to *learn* it is the D-1 vintage, where the
inputs are sharpest and least contaminated by forecast error. Input uncertainty
then belongs in the **width** of the answer, not baked into a flattened point
estimate.

That cleanly separates two things this document had been conflating, and it
correctly predicts something already on file: the pre-fix model trained on 22–46 h
had *zero* negative recall not because its f was wrong, but because at long horizon
its inputs never reach the region where plunges live (wind sd ratio 0.371 at 7–15 d).

## The test

Three configurations, 13 production-faithful folds, 70 107 rows, 1 307 negatives,
all maps fitted on the training window only:

| | training | inputs at inference |
|---|---|---|
| **A** sharp f | 22–46 h only (pre-fix production) | raw |
| **B** blunt g | 22 h–14 d (the widened window on trial) | raw |
| **C** sharp f + correction | 22–46 h only | each input quantile-mapped per horizon band onto the D-1 distribution |

### Result

| config | neg recall | neg prec | sd ratio | RMSE | MAE | sub-zero preds |
|---|---|---|---|---|---|---|
| A sharp f, raw inputs | **0.000** | — | 0.712 | 25.34 | 18.23 | 0 |
| **B widened (on trial)** | **0.273** | 0.566 | **0.944** | **20.37** | **13.27** | 631 |
| C sharp f + corrected inputs | **0.000** | 0.000 | 0.795 | 30.33 | 22.66 | 87 |

By horizon, negative recall / RMSE:

| band | n | negatives | A | B | C |
|---|---|---|---|---|---|
| 2–4 d | 14 320 | 319 | 0.000 / 23.5 | **0.282 / 15.8** | 0.000 / 24.4 |
| 4–7 d | 19 881 | 547 | 0.000 / 24.7 | **0.380 / 17.8** | 0.000 / 27.3 |
| 7–15 d | 35 906 | 441 | 0.000 / 26.3 | **0.134 / 23.1** | 0.000 / 33.9 |

**C fails completely, and B wins on every axis including the tails.**

The correction did what it was asked to do at the input: `emb_wind` spread at 7–15 d
went from sd 287.9 to 863.7, three times wider. The inputs were duly pushed into the
extreme region — and the model still predicted **zero** negative prices, while
aggregate error got substantially worse (RMSE 25.34 → 30.33).

## Why it failed, and why that does not refute the framing

**Marginal quantile mapping destroys the joint structure.** A negative price requires
a *conjunction* — high solar **and** high wind **and** low demand in the same slot.
Mapping each feature's marginal distribution independently spreads each variable but
manufactures no conjunctions; worse, it produces incoherent combinations (inflating
embedded wind on a slot whose solar is low). The result is inputs that are more
dispersed and less physically possible, which explains both the absent tails and the
degraded RMSE.

`_BASE` compounds this: it carries `emb_wind` but not transmission wind, so the
variable most responsible for plunges is not even present to be corrected.

So the negative result falls on **my implementation**, not on the owner's reasoning.
"Restore the input distribution" was a crude proxy for what the framing actually
calls for, which is to propagate input **uncertainty** — push a set of coherent joint
scenarios through f and take the distribution of the outputs. Independent marginal
mapping is the one way of doing that which guarantees incoherent scenarios.

## What this does settle

- **B, the widened window currently on trial, is the best configuration measured.**
  It wins on tails *and* on aggregate error against both alternatives. My earlier
  suggestion that it "improves accuracy by teaching the model to be blunt, which is
  the wrong direction for tails" is not supported: it has the best negative recall of
  the three, at every horizon band. I withdraw that reservation.
- **A is confirmed as indefensible**: zero negative recall at *every* horizon band
  including 2–4 days, corroborating the original defect measurement from a third
  independent direction.

## The direction this leaves open

Scenario propagation rather than marginal correction: sample coherent joint input
states — historical analogue days from the backfilled `History`, or the spread across
successive forecast runs for the same target slot — push each through f, and read the
tails off the resulting distribution. That preserves the conjunctions marginal mapping
destroys, and it is the faithful implementation of "same function, more uncertainty".
Untested.

---

# Claude — post-processed day-ahead implemented on dev, review 2026-09-06

Appended 2026-08-30. Implements the correction the D-1 scatter work arrived at, at
the owner's direction. Running on the dev server only; nothing deployed.

## Where this came from

Plotting D-1 (24–48 h) predicted against settled price and colouring by surplus
(`demand − bm_wind − emb_wind − solar − nuclear`) showed the residual is structured
by surplus: oversupply sits below the line, high residual demand above. The owner's
suggestion was a bivariate regression of settled on (predicted, surplus).

Tested on held-out days — fit on 60 % of days, score on the rest, five splits, since
a chronological split puts every negative-price day in one half:

| model | neg recall | neg prec | neg F1 | RMSE | RMSE<£25 | RMSE>£180 | sd ratio |
|---|---|---|---|---|---|---|---|
| raw | 0.009 | 0.818 | 0.086 | 18.09 | 53.17 | 33.05 | 0.785 |
| RMA line (pred only) | 0.197 | **0.995** | 0.321 | 18.01 | 43.37 | **25.54** | 0.954 |
| OLS pred | 0.131 | 0.973 | 0.230 | 17.89 | 49.97 | 29.12 | 0.847 |
| OLS pred + surplus | 0.141 | 0.987 | 0.247 | 17.41 | 46.75 | 29.93 | 0.863 |
| OLS pred + surplus + sur² | 0.189 | 0.994 | 0.312 | **16.99** | 41.39 | 32.98 | 0.866 |
| **↑ + variance restored** | **0.273** | 0.832 | **0.407** | 17.12 | **35.93** | 28.68 | **0.965** |

Two separable effects. **Surplus carries information the model has not absorbed** —
adding it drops RMSE 17.89 → 16.99, with a coefficient of about **+£0.74/MWh per GW**
of residual demand while the prediction coefficient stays at 0.972, so it is largely
orthogonal. And **least squares still compresses**, because it fits a conditional
mean; rescaling the fitted values so their spread matches the actuals lifts negative
F1 to 0.407 and cuts RMSE below £25 from 53.17 to 35.93.

The cost is real: precision falls 0.995 → 0.832 and the corrected series reaches
−£38 against an observed minimum of −£25.3.

Worth noting against the earlier feature work: a **surplus feature** was refuted on
paired folds, while surplus as a **correction** helps. That difference is not yet
explained and is a reason for caution rather than confidence.

## What was built

* `prices/postprocess.py` — the fit and its application.
* `ForecastData.day_ahead_corrected` — stored **beside** `day_ahead`, never instead
  of it, so the two can be scored against each other.
* `DayAheadCalibration` — the durable fit record.
* `compare_trial --column` — scores either series.

**Applied before the GB60 blend.** The correction is fitted on model predictions, so
applying it afterwards would "correct" known Nord Pool prices. The blend is then
applied to both columns, so inside the blend window the two series agree and diverge
only where the model is genuinely forecasting. Verified on the live run: mean
absolute difference **0.03 below 36 h** and **3.93 beyond**.

**A small table, not longer retention.** The obvious way to keep the fit data would
have been to retain `ForecastData` for a year, but that is ~1M rows, and `AgileData`
fans out per region — it is 2.3M rows and **243 MB of the 489 MB** dev database. The
calibration table holds three numbers per settlement slot, about 48 rows a day, so a
year costs ~17k rows. `ForecastData`/`AgileData` retention is untouched. Harvest runs
before the purge and is idempotent.

## A design error a test caught

The first version claimed to select "the run closest to 16:15 local on D-1 — the
auction vintage". It does not, and cannot: for a midday slot the 16:15 run on D-1 is
only about **21 hours** ahead, outside the 24–48 h window entirely. The band actually
mixes the D-1 run for late-evening slots with the D-2 run for early ones. Selection
is now simply the freshest run still ≥24 h out, and the docstrings say so. The 24–48 h
band is kept because it is what the correction was validated on and it sits clear of
the GB60 blend — but it is a horizon band, not a publication vintage, and the auction
-vintage question remains open.

## Live on dev

First run: 2 545 calibration pairs harvested (2026-07-02 → 08-31, horizons
24.2–47.8 h), all 650 slots of forecast `2026-08-30 11:47` carrying a corrected
value. Corrected sd is 1.077× the raw, as intended.

An operational note that cost time: **`update_worker` is a long-running process that
caches imported command modules**. It had been up since 13 August, so the first run
after deploying this silently executed the old code. It must be restarted for any
change to a management command to take effect on dev.

## Review 2026-09-06

A week of published forecasts, then compare the two series over the same slots:

```bash
ssh agile@django 'cd /srv/agile_predict && .venv/bin/python manage.py compare_trial --since 2026-08-30'
ssh agile@django 'cd /srv/agile_predict && .venv/bin/python manage.py compare_trial --since 2026-08-30 --column day_ahead_corrected'
```

Both read the same slots, so this is paired at row level and needs no separate
control — unlike the dynamic-range trial, the comparison is within one box and one
code path.

**What the week cannot settle.** The correction targets plunges, and the negative
band has been empty on this box since 2026-08-16. If the week is calm again the
detection columns will be `n/a` and the honest answer will be "not yet testable" — the
same event-scarcity that made the 2026-08-23 review inconclusive. The dispersion and
aggregate-error columns will still be readable. Judge it on the bands that have
events, and do not read an empty negative band as a null result.
