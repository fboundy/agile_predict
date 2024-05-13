from typing import Any
from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.views.generic import TemplateView, FormView
from .models import Forecasts, PriceHistory, AgileData, ForecastData, Nordpool, History, UpdateErrors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from django.core.management import call_command
from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, update_if_required, get_nordpool
from .forms import RegionForm


regions = GLOBAL_SETTINGS["REGIONS"]


class GlossaryView(TemplateView):
    template_name = "base.html"


class ApiHowToView(TemplateView):
    template_name = "api_how_to.html"


class AboutView(TemplateView):
    template_name = "about.html"


class HomeAssistantView(TemplateView):
    template_name = "home_assistant.html"


class GraphFormView(FormView):
    form_class = RegionForm
    template_name = "graph.html"

    def get_form_kwargs(self):
        kwargs = super(GraphFormView, self).get_form_kwargs()
        kwargs["region"] = self.kwargs.get("region", "X").upper()
        return kwargs

    def update_chart(self, context, forecasts_to_plot):
        region = context["region"]
        print(forecasts_to_plot)

        first_forecast = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
        print(f"First Forecast: {first_forecast}")
        first_forecast_data = ForecastData.objects.filter(forecast=first_forecast).order_by("date_time")
        forecast_start = first_forecast_data[0].date_time
        print(f"Forecast Start: {forecast_start}")
        if len(first_forecast_data) >= 48 * 7:
            forecast_end = first_forecast_data[48 * 7].date_time
        else:
            forecast_end = [d.date_time for d in first_forecast_data][-1]

        print(f"Forecast End: {forecast_end}")
        price_start = PriceHistory.objects.all().order_by("-date_time")[48 * 3].date_time
        print(f"Price Start: {price_start}")

        start = min(forecast_start, price_start)

        data = []
        p = PriceHistory.objects.filter(date_time__gte=start).order_by("-date_time")

        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        agile = day_ahead_to_agile(day_ahead, region=region).sort_index()

        data = data + [
            go.Scatter(
                x=agile.loc[:forecast_end].index.tz_convert("GB"),
                y=agile.loc[:forecast_end],
                marker={"symbol": 104, "size": 1, "color": "white"},
                mode="lines",
                name="Actual",
            )
        ]

        limit = None
        width = 3
        for f in Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at"):
            d = AgileData.objects.filter(forecast=f, region=region)
            if len(d) > 0:
                if limit is None:
                    d = d[: (48 * 7)]
                    limit = d[-1].date_time
                    # print(limit)
                else:
                    d = d.filter(date_time__lte=limit)
                # d = AgileData.objects.filter(forecast=f, region=region)

                x = [a.date_time for a in d]
                y = [a.agile_pred for a in d]

                df = pd.Series(index=pd.to_datetime(x), data=y).sort_index()
                df.index = df.index.tz_convert("GB")
                df = df.loc[agile.index[0] :]

                data = data + [
                    go.Scatter(
                        x=df.index,
                        y=df,
                        marker={"symbol": 104, "size": 10},
                        mode="lines",
                        line=dict(width=width),
                        name=f.name,
                    )
                ]

                if (width == 3) and (d[0].agile_high != d[0].agile_low):
                    data = data + [
                        go.Scatter(
                            x=df.index,
                            y=[a.agile_low for a in d],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="Low",
                            showlegend=False,
                        ),
                        go.Scatter(
                            x=df.index,
                            y=[a.agile_high for a in d],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="High",
                            showlegend=False,
                            fill="tonexty",
                            fillcolor="rgba(255,127,127,0.5)",
                        ),
                    ]
                width = 1

        # nd = Nordpool.objects.filter(date_time__gte=d[-1].date_time)
        # nd = pd.DataFrame(
        #     index=pd.to_datetime([n.date_time for n in nd]).tz_localize("GB"),
        #     data={"day_ahead": [n.day_ahead for n in nd]},
        # )
        # print(agile.index[-1])
        # nd = pd.DataFrame(get_nordpool(start=agile.index[-1])).set_axis(["day_ahead"], axis=1)
        # nd["agile"] = day_ahead_to_agile(nd["day_ahead"], region=region)
        # print(nd)

        # data = data + [
        #     go.Scatter(
        #         x=nd.index,
        #         y=nd["agile"],
        #         marker={"symbol": 104, "size": 10},
        #         mode="lines",
        #         line=dict(width=2, color="blue"),
        #         name="Hourly Day Ahead Prediction",
        #     )
        # ]

        legend = dict(orientation="h", yanchor="top", y=-0.1, xanchor="right", x=1)

        layout = dict(
            yaxis={"title": "Agile Price [p/kWh]"},
            margin={
                "r": 5,
                "t": 50,
            },
            legend=legend,
            height=800,
            template="plotly_dark",
        )

        figure = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Agile Price", "Generation and Demand"),
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        for d in data:
            figure.append_trace(d, row=1, col=1)

        f = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
        print(forecast_end)
        d = ForecastData.objects.filter(forecast=f, date_time__lte=forecast_end).order_by("date_time")
        print([a.date_time for a in d][-1])
        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[a.demand / 1000 for a in d],
                line={"color": "cyan", "width": 3},
                name="Forecast National Demand",
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[a.bm_wind / 1000 for a in d],
                fill="tozeroy",
                line={"color": "rgba(63,127,63)"},
                fillcolor="rgba(127,255,127,0.8)",
                name="Forecast Metered Wind",
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[(a.emb_wind + a.bm_wind) / 1000 for a in d],
                fill="tonexty",
                line={"color": "blue", "width": 1},
                fillcolor="rgba(127,127,255,0.8)",
                name="Forecast Embedded Wind",
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[(a.solar + a.emb_wind + a.bm_wind) / 1000 for a in d],
                fill="tonexty",
                line={"color": "lightgray", "width": 3},
                fillcolor="rgba(255,255,127,0.8)",
                name="Forecast Solar",
            ),
            row=2,
            col=1,
        )

        h = History.objects.filter(date_time__gte=start, date_time__lte=forecast_end)

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in h],
                y=[a.demand / 1000 for a in h],
                line={"color": "#aaaa77", "width": 2},
                name="Historic Demand",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in h],
                y=[(a.total_wind + a.solar) / 1000 for a in h],
                line={"color": "red", "width": 2},
                name="Historic Solar + Wind",
            ),
            row=2,
            col=1,
        )

        figure.update_layout(**layout)
        figure.update_layout(
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
        )
        figure.update_xaxes(title_text="Date/Time (UTC)", row=2, col=1)
        figure.update_yaxes(title_text="Agile Price [p/kWh]", row=1, col=1)
        figure.update_yaxes(title_text="Power [MW]", row=2, col=1)

        context["graph"] = figure.to_html()
        for error_type in ["history", "forecast"]:
            context[f"{error_type}_errors"] = [
                {
                    "date_time": pd.Timestamp(x.date_time).tz_convert("GB"),
                    "dataset": GLOBAL_SETTINGS["DATASETS"][x.dataset]["name"],
                    "source": GLOBAL_SETTINGS["DATASETS"][x.dataset]["source"],
                }
                for x in list(UpdateErrors.objects.filter(type=error_type.title()))
            ]
        print(context["history_errors"])
        return context

    def get_context_data(self, **kwargs):
        # print(self.kwargs.get("region", "No region"))
        context = super().get_context_data(**kwargs)

        f = Forecasts.objects.latest("created_at")
        region = self.kwargs.get("region", "X").upper()
        context["region"] = region
        print(region)
        context["region_name"] = regions[region]["name"]
        # context = self.update_chart(context=context, region=region, forecasts_to_plot=[f.id])
        context = self.update_chart(context=context, forecasts_to_plot=[f.id])
        return context

    def form_valid(self, form):
        # update_if_required()
        context = self.get_context_data(form=form)
        # print(context["region"])
        # region = form.cleaned_data["region"]
        # print(region)
        forecasts_to_plot = form.cleaned_data["forecasts_to_plot"]

        # context = self.update_chart(context=context, region=region, forecasts_to_plot=forecasts_to_plot)
        context = self.update_chart(context=context, forecasts_to_plot=forecasts_to_plot)

        return self.render_to_response(context=context)
