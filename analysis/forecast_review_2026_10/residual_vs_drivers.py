"""Dev model 2-4 d daily bias (14-30 Oct) against actual drivers it does / doesn't see."""
import pandas as pd

D = "analysis/forecast_review_2026_10/data"
fd = pd.read_pickle(f"{D}/scored.pkl")
h = fd[(fd.dt >= 2) & (fd.dt < 4)]
bias = h.groupby("date_time")["err"].mean()
x = pd.read_csv(f"{D}/drivers_hh.csv", index_col=0, parse_dates=True)
w = pd.read_csv(f"{D}/wind_curtailment_hh.csv", index_col=0, parse_dates=True)
j = x.join(w).join(bias.rename("bias"), how="inner")
j["scot_share"] = j.pn_scotland / (j.pn_scotland + j.pn_england_wales).clip(lower=500)
j["gb_wind"] = j.pn_scotland + j.pn_england_wales
d = j.groupby(j.index.tz_convert("GB").date)[["bias", "eu_wind_cf", "gb_wind", "scot_share", "curt_scotland", "ic_net_import", "gas_ttf"]].mean()
print(d.round(2).to_string())
print("\nhalf-hourly corr with bias:\n", j[["eu_wind_cf", "gb_wind", "scot_share", "curt_scotland", "fr_central_ws10"]].corrwith(j.bias).round(2).to_string())
print("\ndaily corr with bias:\n", d.drop(columns="bias").corrwith(d.bias).round(2).to_string())
