from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.views.generic import TemplateView, FormView
from .models import Forecasts, PriceHistory, AgileData
import plotly.graph_objects as go

from django.core.management import call_command
from config.settings import GLOBAL_SETTINGS
from .management.commands.update import day_ahead_to_agile
from .forms import RegionForm


regions = GLOBAL_SETTINGS["REGIONS"]


class GraphFormView(FormView):
    form_class = RegionForm
    template_name = "graph.html"

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        context = self.get_context_data(form=form)
        region = form.cleaned_data["region"]

        # print(form.cleaned_data)
        context["region"] = region

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

        agile = day_ahead_to_agile(day_ahead, region=region).sort_index()
        # print(agile.iloc[-20:])
        data = data + [
            go.Scatter(
                x=agile.index.tz_convert("GB"),
                y=agile,
                marker={"symbol": 104, "size": 1, "color": "black"},
                mode="lines",
                name="Actual",
            )
        ]

        for f in Forecasts.objects.all().order_by("-created_at")[:3]:
            # f = Forecasts.objects.latest("created_at")
            d = AgileData.objects.filter(forecast=f, region=region)[: (48 * 7)]

            x = [a.date_time for a in d]
            y = [a.agile_pred for a in d]

            df = pd.Series(index=pd.to_datetime(x), data=y).sort_index()
            df.index = df.index.tz_convert("GB")

            data = data + [
                go.Scatter(
                    x=df.index,
                    y=y,
                    marker={"symbol": 104, "size": 10},
                    mode="lines",
                    name=f.name,
                )
            ]

        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

        layout = go.Layout(
            title=f"Agile Forecast - {regions[region]['name']} | Region {region}",
            yaxis={"title": "Agile Price [p/kWh]"},
            xaxis={
                "title": "Date/Time (UTC)",
                # "tickformat": "%d-%b %H:%M %Z",
            },
            legend=legend,
            width=1000,
        )
        figure = go.Figure(
            data=data,
            layout=layout,
        )

        context["graph"] = figure.to_html()
        context["region"] = region

        return self.render_to_response(context=context)
        # return context
