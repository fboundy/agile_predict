from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.views.generic import TemplateView
from .models import Forecasts, PriceHistory, AgileData
import plotly.graph_objects as go

from django.core.management import call_command
from config.settings import GLOBAL_SETTINGS
from .management.commands.update import day_ahead_to_agile


regions = GLOBAL_SETTINGS["REGIONS"]


class GraphView(TemplateView):
    template_name = "graph.html"

    def get_context_data(self, **kwargs):
        region = kwargs.get("region", "G")
        f = Forecasts.objects.latest("created_at")

        # hour_now = pd.Timestamp.now(tz="GB").hour
        # hour_updated = pd.Timestamp(f.created_at).hour
        # updated_today = pd.Timestamp(f.created_at).day == pd.Timestamp.now(tz="GB").day

        # if (hour_now >= 10 and not updated_today) or (hour_now >= 16 and hour_updated < 16):
        if (pd.Timestamp.now(tz="GB") - f.created_at).total_seconds() / 3600 > 1:
            call_command("update")

        data = []
        p = PriceHistory.objects.all().order_by("-date_time")[: 48 * 3]
        # p = PriceHistory.objects.all()

        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        agile = day_ahead_to_agile(day_ahead, region=region)

        data = data + [
            go.Scatter(
                x=agile.index,
                y=agile,
                marker={"symbol": 104, "size": 1, "color": "black"},
                mode="lines",
                name="Actual",
            )
        ]

        for f in Forecasts.objects.all().order_by("-created_at")[:3]:
            # f = Forecasts.objects.latest("created_at")
            d = AgileData.objects.filter(forecast=f, region=region)[: (48 * 7)]

            context = super(GraphView, self).get_context_data(**kwargs)

            x = [a.date_time for a in d]

            data = data + [
                go.Scatter(
                    x=x,
                    y=[a.agile_pred for a in d],
                    marker={"symbol": 104, "size": 10},
                    mode="lines",
                    name=f.name,
                )
            ]
        # trace2 = go.Scatter(
        #     x=x,
        #     y=[a.agile_actual for a in d],
        #     marker={"color": "blue", "symbol": 104, "size": 10},
        #     mode="lines",
        #     name="Actual",
        # )
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

        layout = go.Layout(
            title=f"Agile Forecast - {regions[region]['name']} | Region {region}",
            yaxis={"title": "Agile Price [p/kWh]"},
            legend=legend,
            width=1000,
        )
        figure = go.Figure(
            data=data,
            layout=layout,
        )

        context["graph"] = figure.to_html()
        context["region"] = region

        return context
