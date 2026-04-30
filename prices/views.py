from hmac import compare_digest
import logging
from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

# Create your views here.
from django.views.generic import FormView, TemplateView
from plotly.subplots import make_subplots

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile

from .forms import ForecastForm
from .models import AgileData, ForecastData, Forecasts, History, PlotImage, PriceHistory, UpdateJob

regions = GLOBAL_SETTINGS["REGIONS"]
PRIOR_DAYS = 2
logger = logging.getLogger("prices.web")


def _truthy(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def _update_options(request):
    options = {}

    for key in ["debug", "no_day_of_week", "no_ranges", "skip_kde_plot"]:
        if key in request.POST or key in request.GET:
            options[key] = _truthy(request.POST.get(key, request.GET.get(key)))

    # The HTTP-triggered update skips the expensive diagnostic KDE plot unless explicitly requested.
    if "skip_kde_plot" not in request.POST and "skip_kde_plot" not in request.GET:
        options["skip_kde_plot"] = True

    for key in ["min_fd", "min_ad", "max_days", "train_frac", "drop_last"]:
        value = request.POST.get(key, request.GET.get(key))
        if value not in {None, ""}:
            options[key] = value

    ignore_forecast = request.POST.getlist("ignore_forecast") or request.GET.getlist("ignore_forecast")
    if ignore_forecast:
        options["ignore_forecast"] = ignore_forecast

    return options


def _job_payload(job):
    if job is None:
        return None
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "options": job.options,
        "error": job.error,
        "log_file": job.log_file,
        "requested_at": job.requested_at.isoformat() if job.requested_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def _forbidden_if_update_token_invalid(request):
    expected_token = getattr(settings, "UPDATE_TOKEN", "")
    provided_token = request.headers.get("X-Update-Token", "")
    if not expected_token or not compare_digest(provided_token, expected_token):
        return HttpResponseForbidden("Forbidden")
    return None


def _active_job():
    return UpdateJob.objects.filter(status__in=[UpdateJob.STATUS_PENDING, UpdateJob.STATUS_RUNNING]).first()


def _job_already_active_response(active_job):
    return JsonResponse(
        {
            "ok": False,
            "error": "Worker job already queued or running",
            "update": _job_payload(active_job),
        },
        status=409,
    )


@require_GET
def update_status(request):
    forbidden = _forbidden_if_update_token_invalid(request)
    if forbidden:
        return forbidden

    latest_job = UpdateJob.objects.order_by("-requested_at").first()
    return JsonResponse(
        {
            "ok": True,
            "update": _job_payload(latest_job),
            "running": latest_job is not None and latest_job.status == UpdateJob.STATUS_RUNNING,
        }
    )


@require_GET
def stats_plot(request, filename):
    plot = get_object_or_404(PlotImage, filename=filename)
    return HttpResponse(bytes(plot.content), content_type=plot.content_type)


@csrf_exempt
@require_POST
def run_update(request):
    forbidden = _forbidden_if_update_token_invalid(request)
    if forbidden:
        return forbidden

    active_job = _active_job()
    if active_job is not None:
        return _job_already_active_response(active_job)

    job = UpdateJob.objects.create(job_type=UpdateJob.JOB_UPDATE, options=_update_options(request))
    logger.info("Queued update job id=%s options=%s", job.id, job.options)
    return JsonResponse(
        {
            "ok": True,
            "status": "queued",
            "update": _job_payload(job),
        },
        status=202,
    )


@csrf_exempt
@require_POST
def run_latest_agile(request):
    forbidden = _forbidden_if_update_token_invalid(request)
    if forbidden:
        return forbidden

    active_job = _active_job()
    if active_job is not None:
        return _job_already_active_response(active_job)

    job = UpdateJob.objects.create(job_type=UpdateJob.JOB_LATEST_AGILE, options={})
    logger.info("Queued latest_agile job id=%s", job.id)
    return JsonResponse(
        {
            "ok": True,
            "status": "queued",
            "update": _job_payload(job),
        },
        status=202,
    )


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
        started = time.monotonic()
        context = super().get_context_data(**kwargs)
        cached_context = cache.get("stats_view_context")
        if cached_context is not None:
            context.update(cached_context)
            logger.debug("Stats context served from cache")
            return context

        logger.info("Building stats context")
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

            forecast_until = forecast_after + pd.Timedelta("7D")
            agile_pred_rows = list(
                agile_forecast_data.filter(
                    forecast=forecast[0],
                    date_time__gt=forecast_after,
                    date_time__lt=forecast_until,
                ).values_list("date_time", "agile_pred")
            )
            index = [row[0] for row in agile_pred_rows]
            data = [row[1] for row in agile_pred_rows]
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

        # HTML for the existing Plotly figure
        context["stats"] = figure.to_html(full_html=False, include_plotlyjs="cdn")

        # --- SECTION 2: Static Diagnostic PNG Plots ---
        descriptions = {
            "1_actual_vs_predicted_over_time.png": (
                "This plot shows the full training dataset used for the last forecast. Actual data are plotted as the black line."
                + " The fitted data from the trained mode are plotted in red and should generally overlay the black. Forecasts generated"
                + " from the model using prior data are plotted as the points with the colour indicating the lead time from forecast to actual pricing."
                + "All of the plots below other than the XGBoost Feature Importance show the same data in different ways.",
                "Actual vs Predicted Over Time",
            ),
            "2_scatters.png": (
                "Scatter plot of predicted vs actual prices. Color shows forecast lead time (in days).",
                "Prediction vs Actual Scatter",
            ),
            "3_residuals.png": (
                "Histogram of prediction errors (residuals) to visualize model bias and spread.",
                "Residuals Distribution",
            ),
            "4_kde_error_by_horizon.png": (
                "KDE heatmap showing how forecast error varies by lead time. Initially the data are biased towards shorted lead times but as the database"
                + "grows this bias should reduce. The distribution is, however, always expected to be tighter over short lead times.",
                "Forecast Error by Horizon (KDE)",
            ),
            "5_feature_importance.png": (
                "This plot is slightly different to the others in that it shows the relative importance of the various inputs in building the regression model. Details of each feauture can be found on the About page.",
                "XGBoost Feature Importance",
            ),
        }

        trend_plot = PlotImage.objects.filter(filename="trends/trend.png").first()
        context["trend_plot_url"] = (
            reverse("stats_plot", kwargs={"filename": trend_plot.filename}) if trend_plot is not None else None
        )

        plot_files = [
            {
                "filename": plot.filename,
                "url": reverse("stats_plot", kwargs={"filename": plot.filename}),
                "description": descriptions.get(Path(plot.filename).name, ("", ""))[0],
                "title": descriptions.get(Path(plot.filename).name, ("", ""))[1]
                or Path(plot.filename).name.replace("_", " ").title().replace(".Png", ""),
            }
            for plot in PlotImage.objects.filter(filename__startswith="stats_plots/").order_by("filename")
        ]

        context["plot_files"] = plot_files
        cache.set(
            "stats_view_context",
            {
                "stats": context["stats"],
                "trend_plot_url": context["trend_plot_url"],
                "plot_files": context["plot_files"],
            },
            timeout=60 * 15,
        )
        logger.info("Built stats context duration_seconds=%.2f plot_count=%s", time.monotonic() - started, len(plot_files))

        return context


class GraphFormView(FormView):
    form_class = ForecastForm
    template_name = "graph.html"

    def get_form_kwargs(self):
        kwargs = super(GraphFormView, self).get_form_kwargs()
        # kwargs["region"] = self.kwargs.get("region", "X").upper()
        kwargs["prefix"] = "test"
        # print(kwargs)
        return kwargs

    def update_chart(self, context, **kwargs):
        region = context["region"]
        if region not in regions:
            region = "X"
        forecasts_to_plot = kwargs.get("forecasts_to_plot")
        days_to_plot = int(kwargs.get("days_to_plot", 7))
        show_generation_and_demand = kwargs.get("show_generation_and_demand", True)
        show_range = kwargs.get("show_range_on_most_recent_forecast", True)
        show_overlap = kwargs.get("show_forecast_overlap", False)
        # print(">>> views.py | GraphFormView | update_chart")
        # print(forecasts_to_plot)

        first_forecast = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
        # print(f"First Forecast: {first_forecast}")
        first_forecast_data = ForecastData.objects.filter(forecast=first_forecast).order_by("date_time")
        forecast_start = first_forecast_data[0].date_time
        # print(f"Forecast Start: {forecast_start}")
        if len(first_forecast_data) >= 48 * days_to_plot:
            forecast_end = first_forecast_data[48 * days_to_plot].date_time
        else:
            forecast_end = [d.date_time for d in first_forecast_data][-1]

        # print(f"Forecast End: {forecast_end}")
        price_start = PriceHistory.objects.all().order_by("-date_time")[48 * PRIOR_DAYS].date_time
        # print(f"Price Start: {price_start}")

        start = min(forecast_start, price_start)

        data = []
        p = PriceHistory.objects.filter(date_time__gte=start).order_by("-date_time")

        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        agile = day_ahead_to_agile(day_ahead, region=region).sort_index()

        hover_template_time_price = "%{x|%H:%M}<br>%{y:.2f}p/kWh"
        hover_template_price = "%{y:.2f}p/kWh"

        data = data + [
            go.Scatter(
                x=agile.loc[:forecast_end].index.tz_convert("GB"),
                y=agile.loc[:forecast_end],
                marker={"symbol": 104, "size": 1, "color": "white"},
                mode="lines",
                name="Actual",
                hovertemplate=hover_template_price,
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
                try:
                    df.index = df.index.tz_convert("GB")
                except:
                    df.index = df.index.tz_localize("GB")

                df = df.loc[agile.index[0] :]

                data = data + [
                    go.Scatter(
                        x=df.index,
                        y=df,
                        marker={"symbol": 104, "size": 10},
                        mode="lines",
                        line=dict(width=width),
                        name=f"Prediction ({pd.to_datetime(f.name).tz_localize('GB').strftime('%d-%b %H:%M')})",
                        hovertemplate=hover_template_price,
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
                            hovertemplate=hover_template_price,
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
                            hovertemplate=hover_template_price,
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
            d = ForecastData.objects.filter(forecast=f, date_time__lte=forecast_end).order_by("date_time")
            logger.debug(
                "Graph generation/demand data forecast_id=%s forecast_end=%s rows=%s",
                f.id,
                forecast_end,
                d.count(),
            )
            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in d],
                    y=[(a.demand + a.solar + a.emb_wind) / 1000 for a in d],
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
                    y=[(a.demand + a.solar + (a.total_wind - a.bm_wind)) / 1000 for a in h],
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
            hovermode="x",
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
            title_text="Power [GW]",
            row=2,
            col=1,
            fixedrange=True,
        )
        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86000000], value="%H:%M<br>%a %d %b"),
                dict(dtickrange=[86000000, None], value="%H:%M<br>%a %d %b"),
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

        return context

    def get_context_data(self, **kwargs):
        # print(">>>views.py | GraphFormView | get_context_data")
        context = super().get_context_data(**kwargs)
        # context["form2"] = OptionsForm()
        f = Forecasts.objects.latest("created_at")
        region = self.kwargs.get("region", "X").upper()
        context["region"] = region
        context["region_name"] = regions.get(region, {"name": ""})["name"]
        # print(context)

        context = self.update_chart(context=context, forecasts_to_plot=[f.id])
        return context

    def form_valid(self, form):
        # print(">>>views.py | GraphFormView | form_valid")
        # print(form.cleaned_data)
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)

        return self.render_to_response(context=context)

    def form2_valid(self, form):
        # print(">>>views.py | GraphFormView | form_valid")
        # print(form.cleaned_data)
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)

        return self.render_to_response(context=context)
