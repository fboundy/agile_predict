"""Pairwise: ours vs each external source on slots both forecast at the same lead bin (Agile G p/kWh)."""
import numpy as np
import pandas as pd

D = "analysis/forecast_review_2026_11/data"
pd.set_option("display.width", 250)
b = pd.read_pickle(f"{D}/vs_external.pkl")
b = b.drop(columns=[c for c in ("tail",) if c in b])
b["tail"] = np.select([b.actual <= 15.7, b.actual >= 55.2], ["low10", "high10"], "mid")
b["hbin"] = pd.cut(b.dt, [0, 1, 2, 4, 7])
rows = []
for src in ["agileforecast", "x2r", "agilelab"]:
    pair = b[b.source.isin([src, "agilepredict"])]
    m = pair.groupby(["hbin", "date_time"], observed=True).source.transform("nunique") == 2
    pair = pair[m]
    r = pair.groupby(["hbin", "tail", "source"], observed=True).err.agg(bias="mean", mae=lambda e: e.abs().mean()).unstack("source")
    r.columns = [f"{s[:6]}_{k}" for k, s in r.columns]
    r.insert(0, "slots", pair[pair.source == src].groupby(["hbin", "tail"], observed=True).date_time.nunique())
    rows.append(r.assign(vs=src).set_index("vs", append=True))
print(pd.concat(rows).round(1).to_string())

print("\n== Daily max (p/kWh), mean of each source's 2-4 d forecasts ==")
b["gb_date"] = b.date_time.dt.tz_convert("GB").dt.date
f = b[(b.dt >= 2) & (b.dt < 4)]
mx = f.groupby(["gb_date", "source", "dt"]).agile_pred.max()  # dt per run is unique-ish; take max per run
runmax = f.assign(run=(f.date_time - pd.to_timedelta(f.dt, unit="D")).dt.round("h")).groupby(["gb_date", "source", "run"]).agile_pred.max()
t = runmax.groupby(["gb_date", "source"]).mean().unstack("source")
t["actual"] = b.groupby("gb_date").actual.max()
print(t.loc[pd.Timestamp("2026-10-22").date():].round(1).to_string())
