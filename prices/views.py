from typing import Any
from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.views.generic import TemplateView, FormView
from .models import Forecasts, PriceHistory, AgileData, ForecastData
import plotly.graph_objects as go

from django.core.management import call_command
from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, update_if_required
from .forms import RegionForm


regions = GLOBAL_SETTINGS["REGIONS"]


class GraphFormView(FormView):
    form_class = RegionForm
    template_name = "graph.html"

    # def get_initial(self):
    #     initial = super().get_initial()
    #     print(f"form.get_initial: {initial}")

    #     print(f"ctx: {self.kwargs}")
    #     return initial

    def get_form_kwargs(self):
        kwargs = super(GraphFormView, self).get_form_kwargs()

        # get users, note: you can access request using: self.request

        print(f"view kwargs: {kwargs}")
        print(f"view self.kwargs: {self.kwargs}")
        kwargs["region"] = self.kwargs.get("region", "X")
        return kwargs

    def update_chart(self, context, region, forecasts_to_plot):
        context["region"] = region

        data = []
        p = PriceHistory.objects.all().order_by("-date_time")[: 48 * 3]

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
        for f in Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at"):
            d = AgileData.objects.filter(forecast=f, region=region)
            if len(d) > 0:
                if limit is None:
                    d = d[: (48 * 7)]
                    limit = d[-1].date_time
                    print(limit)
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
                        name=f.name,
                    )
                ]

        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

        title = f"Agile Forecast - {regions[region]['name']}"
        if region != "X":
            title += f"| Region {region}"
        layout = go.Layout(
            title=title,
            yaxis={"title": "Agile Price [p/kWh]"},
            xaxis={
                "title": "Date/Time (UTC)",
                # "tickformat": "%d-%b %H:%M %Z",
            },
            legend=legend,
            # width=800,
        )
        figure = go.Figure(
            data=data,
            layout=layout,
        )

        context["graph"] = figure.to_html()
        return context

    def get_context_data(self, **kwargs):
        print(self.kwargs.get("region", "No region"))
        context = super().get_context_data(**kwargs)
        for f in Forecasts.objects.all():
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            print(f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()

        f = Forecasts.objects.latest("created_at")
        region = self.kwargs.get("region", "X")
        context = self.update_chart(context=context, region=region, forecasts_to_plot=[f.id])
        return context

    def form_valid(self, form):
        update_if_required()
        context = self.get_context_data(form=form)
        print(context["region"])
        region = form.cleaned_data["region"]
        print(region)
        forecasts_to_plot = form.cleaned_data["forecasts_to_plot"]

        context = self.update_chart(context=context, region=region, forecasts_to_plot=forecasts_to_plot)

        return self.render_to_response(context=context)
