from typing import Any
from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.views.generic import TemplateView, FormView
from .models import Forecasts, PriceHistory, AgileData, ForecastData, Nordpool, History
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from django.core.management import call_command
from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, update_if_required
from .forms import RegionForm


regions = GLOBAL_SETTINGS["REGIONS"]


class LatestAgileView(TemplateView):
    template_name = "base.html"

    def get_context_data(self, **kwargs):
        call_command("latest_agile")
        context = super().get_context_data(**kwargs)
        return context


class UpdateView(TemplateView):
    template_name = "base.html"

    def get_context_data(self, **kwargs):
        call_command("update")
        context = super().get_context_data(**kwargs)
        return context


class GraphFormView(FormView):
    form_class = RegionForm
    template_name = "graph.html"

    def get_form_kwargs(self):
        kwargs = super(GraphFormView, self).get_form_kwargs()
        kwargs["region"] = self.kwargs.get("region", "X").upper()
        return kwargs

    def update_chart(self, context, region, forecasts_to_plot):
        context["region"] = region

        first_forecast = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]

        forecast_start = ForecastData.objects.filter(forecast=first_forecast).order_by("date_time")[0].date_time
        print(f"Forecast Start: {forecast_start}")
        price_start = PriceHistory.objects.all().order_by("-date_time")[48 * 3].date_time
        print(f"Price Start: {price_start}")

        start = min(forecast_start, price_start)

        data = []
        p = PriceHistory.objects.filter(date_time__gte=start).order_by("-date_time")

        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        agile = day_ahead_to_agile(day_ahead, region=region).sort_index()

        data = data + [
            go.Scatter(
                x=agile.index.tz_convert("GB"),
                y=agile,
                marker={"symbol": 104, "size": 1, "color": "black"},
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

        legend = dict(orientation="h", yanchor="top", y=-0.1, xanchor="right", x=1)

        title = f"Agile Forecast - {regions[region]['name']}"
        if region != "X":
            title += f"| Region {region}"

        layout = dict(
            yaxis={"title": "Agile Price [p/kWh]"},
            margin={
                "r": 0,
            },
            legend=legend,
            height=800,
        )

        figure = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(title, "Generation and Demand"),
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        for d in data:
            figure.append_trace(d, row=1, col=1)

        f = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
        d = ForecastData.objects.filter(forecast=f, date_time__lte=limit)

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[a.demand / 1000 for a in d],
                line={"color": "red", "width": 3},
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
                fillcolor="rgba(127,255,127,0.5)",
                name="Forecast Metered Wind",
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[(a.solar + a.bm_wind) / 1000 for a in d],
                fill="tonexty",
                line={"color": "slategray", "width": 3},
                fillcolor="rgba(255,255,127,0.5)",
                name="Forecast Solar",
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in d],
                y=[(a.solar + a.emb_wind + a.bm_wind) / 1000 for a in d],
                fill="tonexty",
                line={"color": "blue", "width": 1},
                fillcolor="rgba(127,127,255,0.5)",
                name="Forecast Embedded Wind",
            ),
            row=2,
            col=1,
        )

        h = History.objects.filter(date_time__gte=start)

        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in h],
                y=[a.demand / 1000 for a in h],
                line={"color": "darkred", "width": 2},
                name="Historic Demand",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[a.date_time for a in h],
                y=[(a.bm_wind + a.solar) / 1000 for a in h],
                line={"color": "darkblue", "width": 2},
                name="Historic Solar + Metered Wind",
            ),
            row=2,
            col=1,
        )

        figure.update_layout(**layout)
        figure.update_xaxes(title_text="Date/Time (UTC)", row=2, col=1)
        figure.update_yaxes(title_text="Agile Price [p/kWh]", row=1, col=1)
        figure.update_yaxes(title_text="Power [MW]", row=2, col=1)
        # figure = go.Figure(
        #     data=data,
        #     layout=layout,
        # )

        # for trace in figure['data']:
        #     if trace['name'] in ['Low', 'High']:
        #         trace['showlegend'] = False

        context["graph"] = figure.to_html()
        return context

    def get_context_data(self, **kwargs):
        # print(self.kwargs.get("region", "No region"))
        context = super().get_context_data(**kwargs)

        f = Forecasts.objects.latest("created_at")
        region = self.kwargs.get("region", "X").upper()
        context = self.update_chart(context=context, region=region, forecasts_to_plot=[f.id])
        return context

    def form_valid(self, form):
        # update_if_required()
        context = self.get_context_data(form=form)
        # print(context["region"])
        region = form.cleaned_data["region"]
        # print(region)
        forecasts_to_plot = form.cleaned_data["forecasts_to_plot"]

        context = self.update_chart(context=context, region=region, forecasts_to_plot=forecasts_to_plot)

        return self.render_to_response(context=context)
