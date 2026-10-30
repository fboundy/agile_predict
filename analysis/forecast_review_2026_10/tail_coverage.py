"""How much low-price history does a 60-day window contain before the 24 Oct fold?"""
import pandas as pd

D = "analysis/forecast_review_2026_10/data"
x = pd.read_csv(f"{D}/drivers_hh.csv", index_col=0, parse_dates=True)
w = pd.read_csv(f"{D}/wind_curtailment_hh_hist.csv", index_col=0, parse_dates=True)
x = x.join(w)
x["pn_wind"] = x.filter(like="pn_").sum(axis=1)
tr = x[(x.index >= "2026-08-25") & (x.index < "2026-10-24")]
print("60d window before 24 Oct: slots <£50:", (tr.price < 50).sum(), " <£0:", (tr.price < 0).sum(), " min:", tr.price.min())
print("Jun-Jul: slots <£50:", (x.loc[:"2026-07-31"].price < 50).sum(), " <£0:", (x.loc[:"2026-07-31"].price < 0).sum())
print("\nwindow slots with pn_wind>15GW & eu_wind_cf>0.6:", ((tr.pn_wind > 15000) & (tr.eu_wind_cf > 0.6)).sum(),
      " their mean price:", tr.price[(tr.pn_wind > 15000) & (tr.eu_wind_cf > 0.6)].mean().round(1))
ev = x["2026-10-24":"2026-10-25"]
print("24-25 Oct slots with same condition:", ((ev.pn_wind > 15000) & (ev.eu_wind_cf > 0.6)).sum(), " mean price:", ev.price[(ev.pn_wind > 15000) & (ev.eu_wind_cf > 0.6)].mean().round(1))
print("\nmonthly price & gas:")
print(x.groupby(x.index.month)[["price", "gas_ttf", "eu_wind_cf", "pn_wind"]].mean().round(1).to_string())
