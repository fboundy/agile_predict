from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.views.generic import TemplateView
from .models import Forecasts, ForecastData, PriceHistory
import plotly.graph_objects as go

from django.core.management import call_command


class GraphView(TemplateView):
    template_name = "graph.html"

    def get_context_data(self, **kwargs):
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

        data = data + [
            go.Scatter(
                x=[a.date_time for a in p],
                y=[a.agile for a in p],
                marker={"symbol": 104, "size": 1, "color": "black"},
                mode="lines",
                name="Actual",
            )
        ]

        for f in Forecasts.objects.all().order_by("-created_at")[:3]:
            # f = Forecasts.objects.latest("created_at")
            d = ForecastData.objects.filter(forecast=f)[: (48 * 7)]

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
            title="Agile Forecast - NW England (Area G)",
            yaxis={"title": "Agile Price [p/kWh]"},
            legend=legend,
            width=1000,
        )
        figure = go.Figure(
            data=data,
            layout=layout,
        )

        context["graph"] = figure.to_html()

        return context
