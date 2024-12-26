import pandas as pd

# Create your views here.
from django.views.generic import TemplateView, FormView
from .models import Forecasts, PriceHistory, AgileData, ForecastData, History, UpdateErrors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile
from .forms import ForecastForm


regions = GLOBAL_SETTINGS["REGIONS"]
PRIOR_DAYS = 2


class GlossaryView(TemplateView):
    template_name = "base.html"


class ColorView(TemplateView):
    template_name = "color_mode.html"


class ApiHowToView(TemplateView):
    template_name = "api_how_to.html"


class AboutView(TemplateView):
    template_name = "about.html"


class HomeAssistantView(TemplateView):
    template_name = "home_assistant.html"


class StatsView(TemplateView):
    template_name = "stats.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        agile_actuals_end = pd.Timestamp(PriceHistory.objects.all().order_by("-date_time")[0].date_time)
        agile_actuals_start = agile_actuals_end - pd.Timedelta("7D")

        agile_actuals_objects = PriceHistory.objects.filter(date_time__gt=agile_actuals_start).order_by("date_time")
        df = pd.DataFrame(
            index=[obj.date_time for obj in agile_actuals_objects],
            data={"actuals": [obj.agile for obj in agile_actuals_objects]},
        )

        agile_forecast_data = AgileData.objects.filter(
            date_time__gt=agile_actuals_start, date_time__lte=agile_actuals_end
        )
        figure = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Agile Price", "Error HeatMap"),
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        for forecast in agile_forecast_data.values_list("forecast").distinct():
            forecast_created_at = pd.Timestamp(Forecasts.objects.filter(id=forecast[0])[0].created_at).tz_convert("GB")
            forecast_after = (
                pd.Timestamp.combine(forecast_created_at.date(), pd.Timestamp("22:00").time())
                .tz_localize("UTC")
                .tz_convert("GB")
            )

            if forecast_created_at.hour >= 16:
                forecast_after += pd.Timedelta("24h")

            agile_pred_objects = agile_forecast_data.filter(forecast=forecast[0])
            index = [
                obj.date_time
                for obj in agile_pred_objects
                if obj.date_time > forecast_after
                if obj.date_time < forecast_after + pd.Timedelta("7D")
            ]
            data = [
                obj.agile_pred
                for obj in agile_pred_objects
                if obj.date_time > forecast_after
                if obj.date_time < forecast_after + pd.Timedelta("7D")
            ]
            if len(data) > 0:
                df.loc[index, forecast_created_at] = data
                figure.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[forecast_created_at],
                        line={"color": "grey", "width": 0.5},
                        showlegend=False,
                        mode="lines",
                    ),
                )

        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["actuals"],
                line={"color": "yellow", "width": 3},
                showlegend=False,
            ),
        )

        layout = dict(
            yaxis={"title": "Agile Price [p/kWh]"},
            margin={
                "r": 5,
                "t": 50,
            },
            height=800,
            template="plotly_dark",
        )

        figure.update_layout(**layout)
        figure.update_layout(
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
        )

        for x in df.columns[1:]:
            df[x] = abs(df[x] - df["actuals"])
        df_to_plot = df.drop(["actuals"], axis=1).sort_index(axis=1).T
        df_to_plot = df_to_plot.loc[df_to_plot.index > agile_actuals_start - pd.Timedelta("3D")]
        x = df_to_plot.columns
        y = df_to_plot.index
        z = df_to_plot.to_numpy()

        figure.add_heatmap(x=x, y=y, z=z, row=2, col=1, colorbar={"title": "Error\n[p/kWh]"})

        context["stats"] = figure.to_html()
        return context


class GraphFormView(FormView):
    form_class = ForecastForm
    template_name = "graph.html"

    def get_form_kwargs(self):
        print(">>> views.py | GraphFormView | get_form_kwargs")
        kwargs = super(GraphFormView, self).get_form_kwargs()
        # kwargs["region"] = self.kwargs.get("region", "X").upper()
        kwargs["prefix"] = "test"
        print(kwargs)
        return kwargs

    def update_chart(self, context, **kwargs):
        region = context["region"]
        forecasts_to_plot = kwargs.get("forecasts_to_plot")
        days_to_plot = int(kwargs.get("days_to_plot", 7))
        show_generation_and_demand = kwargs.get("show_generation_and_demand", True)
        show_range = kwargs.get("show_range_on_most_recent_forecast", True)
        show_overlap = kwargs.get("show_forecast_overlap", False)
        print(">>> views.py | GraphFormView | update_chart")
        print(forecasts_to_plot)

        first_forecast = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
        print(f"First Forecast: {first_forecast}")
        first_forecast_data = ForecastData.objects.filter(forecast=first_forecast).order_by("date_time")
        forecast_start = first_forecast_data[0].date_time
        print(f"Forecast Start: {forecast_start}")
        if len(first_forecast_data) >= 48 * days_to_plot:
            forecast_end = first_forecast_data[48 * days_to_plot].date_time
        else:
            forecast_end = [d.date_time for d in first_forecast_data][-1]

        print(f"Forecast End: {forecast_end}")
        price_start = PriceHistory.objects.all().order_by("-date_time")[48 * PRIOR_DAYS].date_time
        print(f"Price Start: {price_start}")

        start = min(forecast_start, price_start)

        data = []
        p = PriceHistory.objects.filter(date_time__gte=start).order_by("-date_time")

        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        agile = day_ahead_to_agile(day_ahead, region=region).sort_index()

        hover_template_time_price = "%{x|%H:%M}: %{y:.2f}p/kWh"

        data = data + [
            go.Scatter(
                x=agile.loc[:forecast_end].index.tz_convert("GB"),
                y=agile.loc[:forecast_end],
                marker={"symbol": 104, "size": 1, "color": "white"},
                mode="lines",
                name="Actual",
                hovertemplate=hover_template_time_price
            )
        ]

        limit = None
        width = 3
        for f in Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at"):
            d = AgileData.objects.filter(forecast=f, region=region)
            if len(d) > 0:
                if limit is None:
                    d = d[: (48 * days_to_plot)]
                    limit = d[-1].date_time
                    # print(limit)
                else:
                    d = list(d.filter(date_time__lte=limit))

                x = [a.date_time for a in d if (a.date_time >= agile.index[-1] or show_overlap)]
                y = [a.agile_pred for a in d if (a.date_time >= agile.index[-1] or show_overlap)]

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
                        name="Prediction",
                        hovertemplate=hover_template_time_price
                    )
                ]

                if (width == 3) and (d[0].agile_high != d[0].agile_low and show_range):
                    data = data + [
                        go.Scatter(
                            x=df.index,
                            y=[a.agile_low for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="Low",
                            showlegend=False,
                        ),
                        go.Scatter(
                            x=df.index,
                            y=[a.agile_high for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
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

        if show_generation_and_demand:
            figure = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Agile Price", "Generation and Demand"),
                shared_xaxes=True,
                vertical_spacing=0.1,
            )

            height = 800
            legend = dict(orientation="h", yanchor="top", y=-0.075, xanchor="right", x=1)

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
            figure.update_xaxes(row=1, col=1, showticklabels=True)

        else:
            legend = dict(orientation="h", yanchor="top", y=-0.15, xanchor="right", x=1)
            height = 400
            figure = make_subplots(
                rows=1,
                cols=1,
            )

        for d in data:
            figure.append_trace(d, row=1, col=1)

        layout = dict(
            yaxis={"title": "Agile Price [p/kWh]"},
            margin={
                "r": 5,
                "t": 50,
            },
            legend=legend,
            height=height,
            template="plotly_dark",
            hovermode='x',
        )

        figure.update_layout(**layout)
        figure.update_layout(
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
        )
        figure.update_yaxes(
            title_text="Agile Price [p/kWh]",
            row=1,
            col=1,
            fixedrange=True,
        )
        figure.update_yaxes(
            title_text="Power [MW]",
            row=2,
            col=1,
            fixedrange=True,
        )
        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86000000], value="%H:%M\n%a %d %b"),
                dict(dtickrange=[86000000, None], value="%a\n%b %d"),
            ],
            # fixedrange=True,
        )

        context["graph"] = figure.to_html(
            config={
                "modeBarButtonsToRemove": [
                    "zoom",
                    "pan",
                    "select",
                    "zoomIn",
                    "zoomOut",
                    "autoScale",
                    "resetScale",
                ]
            }
        )
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
        print(">>>views.py | GraphFormView | get_context_data")
        context = super().get_context_data(**kwargs)
        # context["form2"] = OptionsForm()
        f = Forecasts.objects.latest("created_at")
        region = self.kwargs.get("region", "X").upper()
        context["region"] = region
        context["region_name"] = regions.get(region, {"name": ""})["name"]
        print(context)

        context = self.update_chart(context=context, forecasts_to_plot=[f.id])
        return context

    def form_valid(self, form):
        print(">>>views.py | GraphFormView | form_valid")
        print(form.cleaned_data)
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)

        return self.render_to_response(context=context)

    def form2_valid(self, form):
        print(">>>views.py | GraphFormView | form_valid")
        print(form.cleaned_data)
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)

        return self.render_to_response(context=context)
