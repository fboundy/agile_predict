from hmac import compare_digest
from datetime import datetime, time as datetime_time, timedelta
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
from django.utils.dateparse import parse_date
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

# Create your views here.
from django.views.generic import FormView, TemplateView
from plotly.subplots import make_subplots

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, import_agile_to_export_agile

from .forms import ForecastForm
from .models import AgileData, ExternalForecast, ForecastData, Forecasts, History, PlotImage, PriceHistory, UpdateJob

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

    for key in ["min_fd", "min_ad", "max_days", "train_frac", "drop_last", "feature_set", "features"]:
        value = request.POST.get(key, request.GET.get(key))
        if value not in {None, ""}:
            options[key] = value

    drop_feature = request.POST.getlist("drop_feature") or request.GET.getlist("drop_feature")
    if drop_feature:
        options["drop_feature"] = drop_feature

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


def _scheduled_duplicate_response(request, job_type):
    if not _truthy(request.POST.get("scheduled", request.GET.get("scheduled", ""))):
        return None

    since = timezone.now() - timedelta(minutes=20)
    recent_job = (
        UpdateJob.objects.filter(
            job_type=job_type,
            status__in=[
                UpdateJob.STATUS_PENDING,
                UpdateJob.STATUS_RUNNING,
                UpdateJob.STATUS_COMPLETED,
            ],
            requested_at__gte=since,
        )
        .order_by("-requested_at")
        .first()
    )
    if recent_job is None:
        return None

    return JsonResponse(
        {
            "ok": True,
            "status": "already_queued_recently",
            "update": _job_payload(recent_job),
        }
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

    duplicate = _scheduled_duplicate_response(request, UpdateJob.JOB_UPDATE)
    if duplicate is not None:
        return duplicate

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

    duplicate = _scheduled_duplicate_response(request, UpdateJob.JOB_LATEST_AGILE)
    if duplicate is not None:
        return duplicate

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


class HistoryView(TemplateView):
    template_name = "history.html"
    max_offset_days = 14
    window_options = {
        "last-week": {"label": "Last Week", "days": 7},
        "last-2-weeks": {"label": "Last 2 Weeks", "days": 14},
        "last-month": {"label": "Last Month", "days": 30},
        "custom": {"label": "Custom Dates", "days": None},
    }

    def format_offset_label(self, offset_days):
        if offset_days == 0:
            return "<1d"
        return f"{offset_days}d"

    def get_date_window(self):
        window_key = self.request.GET.get("window", "last-2-weeks")
        if window_key not in self.window_options:
            window_key = "last-2-weeks"

        end = timezone.now()
        start = end - timedelta(days=self.window_options[window_key]["days"] or 14)
        custom_start_date = start.date().isoformat()
        custom_end_date = end.date().isoformat()

        if window_key == "custom":
            start_date = parse_date(self.request.GET.get("start_date", ""))
            end_date = parse_date(self.request.GET.get("end_date", ""))
            if start_date and end_date and start_date <= end_date:
                current_timezone = timezone.get_current_timezone()
                start = timezone.make_aware(datetime.combine(start_date, datetime_time.min), current_timezone)
                end = timezone.make_aware(datetime.combine(end_date, datetime_time.max), current_timezone)
                custom_start_date = start_date.isoformat()
                custom_end_date = end_date.isoformat()
            else:
                window_key = "last-2-weeks"

        return window_key, start, end, custom_start_date, custom_end_date

    def add_prediction_traces(self, figure, predicted, offset_label, hover_template_price):
        if len(predicted) < 2:
            return 0

        segment_count = 0
        segment_x = []
        segment_y = []
        previous_timestamp = None

        for timestamp, value in predicted.items():
            if previous_timestamp is not None and timestamp - previous_timestamp != pd.Timedelta(minutes=30):
                if len(segment_x) >= 2:
                    segment_count += 1
                    figure.add_trace(
                        go.Scatter(
                            x=pd.DatetimeIndex(segment_x).tz_convert("GB"),
                            y=segment_y,
                            line={"color": "#ffc107", "width": 2},
                            mode="lines",
                            name=f"Prediction {offset_label} ahead" if segment_count == 1 else None,
                            showlegend=segment_count == 1,
                            hovertemplate=hover_template_price,
                        )
                    )
                segment_x = []
                segment_y = []

            segment_x.append(timestamp)
            segment_y.append(value)
            previous_timestamp = timestamp

        if len(segment_x) >= 2:
            segment_count += 1
            figure.add_trace(
                go.Scatter(
                    x=pd.DatetimeIndex(segment_x).tz_convert("GB"),
                    y=segment_y,
                    line={"color": "#ffc107", "width": 2},
                    mode="lines",
                    name=f"Prediction {offset_label} ahead" if segment_count == 1 else None,
                    showlegend=segment_count == 1,
                    hovertemplate=hover_template_price,
                )
            )

        return segment_count

    def calculate_error_metrics(self, actual, predicted):
        if len(actual) == 0 or len(predicted) == 0:
            return None

        actual_for_metrics = actual.copy()
        predicted_for_metrics = predicted.copy()
        actual_for_metrics.index = actual_for_metrics.index.tz_convert("UTC")
        predicted_for_metrics.index = predicted_for_metrics.index.tz_convert("UTC")

        common_index = predicted_for_metrics.index.intersection(actual_for_metrics.index)
        if len(common_index) == 0:
            return None

        errors = predicted_for_metrics.loc[common_index] - actual_for_metrics.loc[common_index]
        return {
            "n": len(errors),
            "mae": (errors.abs().mean()),
            "rmse": ((errors**2).mean() ** 0.5),
            "bias": errors.mean(),
        }

    def build_predicted_series(self, forecast_rows, start_hours, end_hours):
        predicted_by_date_time = {}
        for row in forecast_rows:
            lead_hours = (row.date_time - row.forecast.created_at).total_seconds() / 3600
            if lead_hours < start_hours or lead_hours >= end_hours:
                continue
            if row.date_time not in predicted_by_date_time:
                predicted_by_date_time[row.date_time] = row.agile_pred

        predicted = pd.Series(predicted_by_date_time).sort_index()
        if len(predicted) > 0:
            predicted.index = pd.to_datetime(predicted.index)
        return predicted

    def build_external_predicted_series(self, forecast_rows, start_hours, end_hours):
        predicted_by_date_time = {}
        for row in forecast_rows:
            lead_hours = (row.date_time - row.source_created_at).total_seconds() / 3600
            if lead_hours < start_hours or lead_hours >= end_hours:
                continue
            if row.date_time not in predicted_by_date_time:
                predicted_by_date_time[row.date_time] = row.agile_pred

        predicted = pd.Series(predicted_by_date_time).sort_index()
        if len(predicted) > 0:
            predicted.index = pd.to_datetime(predicted.index)
        return predicted

    def format_metric_values(self, metrics):
        if metrics is None:
            return {
                "n": "",
                "mae": "",
                "rmse": "",
                "bias": "",
            }

        return {
            "n": metrics["n"],
            "mae": f"{metrics['mae']:.2f}",
            "rmse": f"{metrics['rmse']:.2f}",
            "bias": f"{metrics['bias']:+.2f}",
        }

    def build_metrics_table(self, actual, forecast_rows, external_rows_by_label=None):
        external_rows_by_label = external_rows_by_label or {}
        columns_by_offset = {}
        metric_sets = {
            "AgilePredict": {},
            **{label: {} for label in external_rows_by_label},
        }

        for offset_days in range(self.max_offset_days + 1):
            predicted = self.build_predicted_series(
                forecast_rows,
                offset_days * 24,
                (offset_days + 1) * 24,
            )
            metrics = self.calculate_error_metrics(actual, predicted)
            if metrics is not None:
                columns_by_offset[offset_days] = {
                    "label": self.format_offset_label(offset_days),
                    "n": metrics["n"],
                }
                metric_sets["AgilePredict"][offset_days] = metrics

            for label, external_rows in external_rows_by_label.items():
                external_predicted = self.build_external_predicted_series(
                    external_rows,
                    offset_days * 24,
                    (offset_days + 1) * 24,
                )
                external_metrics = self.calculate_error_metrics(actual, external_predicted)
                if external_metrics is None:
                    continue

                if offset_days not in columns_by_offset:
                    columns_by_offset[offset_days] = {
                        "label": self.format_offset_label(offset_days),
                        "n": "",
                    }
                metric_sets[label][offset_days] = external_metrics

        columns = [
            {
                "offset": offset_days,
                **column,
            }
            for offset_days, column in sorted(columns_by_offset.items())
        ]

        rows = []
        mobile_rows = []

        for source_label, metrics_by_offset in metric_sets.items():
            if not metrics_by_offset:
                continue

            row_prefix = "" if source_label == "AgilePredict" else f"{source_label} "
            for metric_key, metric_label in [("mae", "MAE"), ("rmse", "RMSE"), ("bias", "Bias")]:
                rows.append(
                    {
                        "label": f"{row_prefix}{metric_label}",
                        "values": [
                            self.format_metric_values(metrics_by_offset.get(column["offset"]))[metric_key]
                            for column in columns
                        ],
                    }
                )

            for column in columns:
                values = self.format_metric_values(metrics_by_offset.get(column["offset"]))
                if not values["n"]:
                    continue

                mobile_rows.append(
                    {
                        "source": source_label,
                        "label": column["label"],
                        "n": values["n"],
                        "mae": values["mae"],
                        "rmse": values["rmse"],
                        "bias": values["bias"],
                    }
                )

        return {
            "columns": columns,
            "rows": rows,
            "mobile_rows": mobile_rows,
        }

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        region = self.kwargs.get("region", "X").upper()
        compare_external_region = region == "GX"
        data_region = "G" if compare_external_region else region
        if not compare_external_region and region not in regions:
            region = "X"
            data_region = "X"

        try:
            offset_days = int(self.request.GET.get("offset_days", 1))
        except ValueError:
            offset_days = 1
        offset_days = min(max(offset_days, 0), self.max_offset_days)
        start_hours = offset_days * 24
        end_hours = (offset_days + 1) * 24
        offset_label = self.format_offset_label(offset_days)

        compare_agileforecast = compare_external_region and _truthy(self.request.GET.get("compare_agileforecast"))
        compare_x2r = compare_external_region and _truthy(self.request.GET.get("compare_x2r"))

        context["region"] = region
        context["data_region"] = data_region
        context["compare_external_region"] = compare_external_region
        context["compare_agileforecast"] = compare_agileforecast
        context["compare_x2r"] = compare_x2r
        context["region_name"] = (
            "North Western England External Comparison"
            if compare_external_region
            else regions.get(region, {"name": ""})["name"]
        )
        context["offset_days"] = offset_days
        context["offset_label"] = offset_label
        context["offset_options"] = [
            {"value": days, "label": self.format_offset_label(days)} for days in range(self.max_offset_days + 1)
        ]

        window_key, start, end, custom_start_date, custom_end_date = self.get_date_window()
        context["window_key"] = window_key
        context["window_options"] = [
            {"key": key, "label": value["label"]} for key, value in self.window_options.items()
        ]
        context["custom_start_date"] = custom_start_date
        context["custom_end_date"] = custom_end_date

        actual_rows = list(
            PriceHistory.objects.filter(date_time__gte=start, date_time__lte=end)
            .order_by("date_time")
            .values_list("date_time", "day_ahead")
        )

        if actual_rows:
            day_ahead = pd.Series(data=[row[1] for row in actual_rows], index=[row[0] for row in actual_rows])
            actual = day_ahead_to_agile(day_ahead, region=data_region).sort_index()
        else:
            actual = pd.Series(dtype=float)

        forecast_rows = list(
            AgileData.objects.filter(region=data_region, date_time__gte=start, date_time__lte=end)
            .select_related("forecast")
            .order_by("date_time", "-forecast__created_at")
        )
        predicted = self.build_predicted_series(forecast_rows, start_hours, end_hours)

        external_sources = []
        if compare_external_region:
            if compare_agileforecast:
                external_sources.append(
                    (ExternalForecast.SOURCE_AGILEFORECAST, "AgileForecast", "#0dcaf0")
                )
            if compare_x2r:
                external_sources.append((ExternalForecast.SOURCE_X2R, "X2R", "#fd7e14"))

        external_rows_by_label = {}
        for source, label, _color in external_sources:
            external_rows_by_label[label] = list(
                ExternalForecast.objects.filter(
                    source=source,
                    region=data_region,
                    date_time__gte=start,
                    date_time__lte=end,
                ).order_by("date_time", "-source_created_at")
            )

        metrics_table = self.build_metrics_table(actual, forecast_rows, external_rows_by_label)

        figure = make_subplots(rows=1, cols=1)
        hover_template_price = "%{x|%d %b %H:%M}<br>%{y:.2f}p/kWh"

        if len(actual) > 0:
            figure.add_trace(
                go.Scatter(
                    x=actual.index.tz_convert("GB"),
                    y=actual,
                    line={"color": "white", "width": 2},
                    mode="lines",
                    name="Actual Agile",
                    hovertemplate=hover_template_price,
                )
            )

        prediction_segment_count = self.add_prediction_traces(figure, predicted, offset_label, hover_template_price)
        external_counts = []
        if compare_external_region:
            for source, label, color in external_sources:
                external_rows = external_rows_by_label[label]
                external_predicted = self.build_external_predicted_series(external_rows, start_hours, end_hours)
                external_counts.append({"label": label, "count": len(external_predicted)})
                if len(external_predicted) > 0:
                    figure.add_trace(
                        go.Scatter(
                            x=external_predicted.index.tz_convert("GB"),
                            y=external_predicted,
                            line={"color": color, "width": 2, "dash": "dot"},
                            mode="lines",
                            name=f"{label} {offset_label} ahead",
                            hovertemplate=hover_template_price,
                        )
                    )

        error_metrics = self.calculate_error_metrics(actual, predicted)
        if error_metrics is None:
            title = f"Agile Price History - {offset_label} ahead"
        else:
            title = (
                f"Agile Price History - {offset_label} ahead | "
                f"MAE {error_metrics['mae']:.2f}p/kWh | "
                f"RMSE {error_metrics['rmse']:.2f}p/kWh | "
                f"Bias {error_metrics['bias']:+.2f}p/kWh"
            )

        figure.update_layout(
            title={"text": title, "x": 0.5},
            yaxis={"title": "Agile Price [p/kWh]"},
            margin={"r": 5, "t": 50},
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="right", x=1),
            height=520,
            template="plotly_dark",
            hovermode="x",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
        )
        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86000000], value="%H:%M<br>%a %d %b"),
                dict(dtickrange=[86000000, None], value="%d %b"),
            ],
        )
        figure.update_yaxes(title_text="Agile Price [p/kWh]", fixedrange=True)

        context["graph"] = figure.to_html(
            full_html=False,
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
            },
        )
        context["prediction_count"] = len(predicted)
        context["external_counts"] = external_counts
        context["prediction_segment_count"] = prediction_segment_count
        context["error_metrics"] = error_metrics
        context["metrics_table"] = metrics_table
        context["actual_count"] = len(actual)
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
        show_export = kwargs.get("show_export_pricing", False)
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
        agile = day_ahead_to_agile(day_ahead, region=region, export=show_export).sort_index()
        price_label = "Agile Export Price" if show_export else "Agile Price"
        actual_label = "Actual Export" if show_export else "Actual"

        hover_template_time_price = "%{x|%H:%M}<br>%{y:.2f}p/kWh"
        hover_template_price = "%{y:.2f}p/kWh"

        data = data + [
            go.Scatter(
                x=agile.loc[:forecast_end].index.tz_convert("GB"),
                y=agile.loc[:forecast_end],
                marker={"symbol": 104, "size": 1, "color": "white"},
                mode="lines",
                name=actual_label,
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
                source = pd.Series(
                    index=pd.to_datetime([a.date_time for a in d]),
                    data=[a.agile_pred for a in d],
                )
                if show_export:
                    source = import_agile_to_export_agile(source, region=region)
                y = [source.loc[pd.Timestamp(a.date_time)] for a in d if (a.date_time >= agile.index[-1] or show_overlap)]

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
                    low_source = pd.Series(index=pd.to_datetime([a.date_time for a in d]), data=[a.agile_low for a in d])
                    high_source = pd.Series(index=pd.to_datetime([a.date_time for a in d]), data=[a.agile_high for a in d])
                    if show_export:
                        low_source = import_agile_to_export_agile(low_source, region=region)
                        high_source = import_agile_to_export_agile(high_source, region=region)

                    data = data + [
                        go.Scatter(
                            x=df.index,
                            y=[low_source.loc[pd.Timestamp(a.date_time)] for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="Low",
                            showlegend=False,
                            hovertemplate=hover_template_price,
                        ),
                        go.Scatter(
                            x=df.index,
                            y=[high_source.loc[pd.Timestamp(a.date_time)] for a in d if (a.date_time >= agile.index[-1] or show_overlap)],
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
                subplot_titles=(price_label, "Generation and Demand"),
                shared_xaxes=True,
                vertical_spacing=0.1,
            )

            height = 800
            legend = dict(orientation="h", yanchor="top", y=-0.075, xanchor="right", x=1)

            f = Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at")[0]
            d = ForecastData.objects.filter(forecast=f, date_time__lte=forecast_end).order_by("date_time")
            forecast_points = list(d)
            logger.debug(
                "Graph generation/demand data forecast_id=%s forecast_end=%s rows=%s",
                f.id,
                forecast_end,
                len(forecast_points),
            )
            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in forecast_points],
                    y=[(a.demand + a.solar + a.emb_wind) / 1000 for a in forecast_points],
                    line={"color": "cyan", "width": 3},
                    name="Forecast National Demand",
                ),
                row=2,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in forecast_points],
                    y=[a.nuclear / 1000 for a in forecast_points],
                    fill="tozeroy",
                    line={"color": "rgba(160,160,160,1)"},
                    fillcolor="rgba(180,180,180,0.8)",
                    name="Forecast Nuclear",
                ),
                row=2,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in forecast_points],
                    y=[(a.nuclear + a.bm_wind) / 1000 for a in forecast_points],
                    fill="tonexty",
                    line={"color": "rgba(63,127,63)"},
                    fillcolor="rgba(127,255,127,0.8)",
                    name="Forecast Metered Wind",
                ),
                row=2,
                col=1,
            )

            figure.add_trace(
                go.Scatter(
                    x=[a.date_time for a in forecast_points],
                    y=[(a.nuclear + a.emb_wind + a.bm_wind) / 1000 for a in forecast_points],
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
                    x=[a.date_time for a in forecast_points],
                    y=[(a.nuclear + a.solar + a.emb_wind + a.bm_wind) / 1000 for a in forecast_points],
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
            yaxis={"title": f"{price_label} [p/kWh]"},
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
            title_text=f"{price_label} [p/kWh]",
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
