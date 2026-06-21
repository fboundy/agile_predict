from hmac import compare_digest
from datetime import datetime, time as datetime_time, timedelta
import logging
from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import Group
from django.contrib.auth.views import LoginView
from django.db.models import Count, Sum
from django.core.cache import cache
from django.core.mail import send_mail
from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils.dateparse import parse_date
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

# Create your views here.
from django.views.generic import FormView, TemplateView
from plotly.subplots import make_subplots

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, import_agile_to_export_agile

from .external_forecasts import fetch_agileforecast, fetch_x2r
from .forms import ForecastForm, RegistrationForm
from .models import (
    AgileData,
    ExternalForecast,
    ForecastData,
    Forecasts,
    History,
    PlotImage,
    PriceHistory,
    RequestClientSeen,
    RequestMetric,
    UpdateJob,
)

regions = GLOBAL_SETTINGS["REGIONS"]
PRIOR_DAYS = 2
logger = logging.getLogger("prices.web")
USER_GROUP_NAME = "Users"
PRIVILEGED_GROUP_NAME = "Privileged Users"


def _is_raw_day_ahead_region(region):
    return regions.get(region, {}).get("raw_day_ahead", False)


def _price_display(region, show_export=False):
    if _is_raw_day_ahead_region(region):
        return {
            "label": "Day Ahead Price",
            "actual_label": "Actual Day Ahead",
            "unit": "£/MWh",
            "axis_title": "Day Ahead Price [£/MWh]",
            "hover_time_price": "%{x|%H:%M}<br>%{y:.2f} £/MWh",
            "hover_price": "%{y:.2f} £/MWh",
        }

    label = "Agile Export Price" if show_export else "Agile Price"
    return {
        "label": label,
        "actual_label": "Actual Export" if show_export else "Actual",
        "unit": "p/kWh",
        "axis_title": f"{label} [p/kWh]",
        "hover_time_price": "%{x|%H:%M}<br>%{y:.2f}p/kWh",
        "hover_price": "%{y:.2f}p/kWh",
    }


def _price_color(p):
    """Map a p/kWh Agile import price to a hex colour. Pass None for a neutral grey."""
    if p is None:
        return "#4a9eff"
    if p < 0:
        return "#6f42c1"
    if p < 5:
        return "#198754"
    if p < 15:
        return "#20c997"
    if p < 25:
        return "#ffc107"
    if p < 35:
        return "#fd7e14"
    return "#dc3545"


def _export_price_color(p):
    """Map a p/kWh Agile export price to a hex colour — high prices are desirable."""
    if p is None:
        return "#4a9eff"
    if p < 0:
        return "#dc3545"   # red  — paying to export
    if p < 5:
        return "#fd7e14"   # orange — poor
    if p < 15:
        return "#ffc107"   # amber — ok
    if p < 25:
        return "#20c997"   # teal — good
    if p < 35:
        return "#198754"   # green — very good
    return "#6f42c1"       # purple — excellent


def _price_badge(p):
    """Map a p/kWh Agile import price to a Bootstrap badge variant string."""
    if p is None:
        return "secondary"
    if p < 0:
        return "info"
    if p < 15:
        return "success"
    if p < 25:
        return "warning text-dark"
    return "danger"


def _truthy(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def _can_use_admin_only_features(request):
    user = getattr(request, "user", None)
    if not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_staff", False):
        return True
    return user.groups.filter(name=PRIVILEGED_GROUP_NAME).exists()


def _update_options(request):
    options = {}

    for key in ["debug", "no_day_of_week", "no_ranges", "skip_kde_plot"]:
        if key in request.POST or key in request.GET:
            options[key] = _truthy(request.POST.get(key, request.GET.get(key)))

    # The HTTP-triggered update skips the expensive diagnostic KDE plot unless explicitly requested.
    if "skip_kde_plot" not in request.POST and "skip_kde_plot" not in request.GET:
        options["skip_kde_plot"] = True

    for key in [
        "min_fd",
        "min_ad",
        "max_days",
        "train_frac",
        "drop_last",
        "feature_set",
        "features",
    ]:
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
        if not PriceHistory.objects.exists():
            context.update({"stats": "", "trend_plot_url": None, "plot_files": []})
            return context
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


class MetricsView(TemplateView):
    template_name = "metrics.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        days = 14
        start_date = timezone.localdate() - timedelta(days=days - 1)

        request_rows = (
            RequestMetric.objects.filter(date__gte=start_date)
            .values("date", "surface")
            .annotate(requests=Sum("request_count"))
            .order_by("-date", "surface")
        )
        unique_rows = {
            (row["date"], row["surface"]): row["unique_clients"]
            for row in RequestClientSeen.objects.filter(date__gte=start_date)
            .values("date", "surface")
            .annotate(unique_clients=Count("client_hash"))
        }

        daily_rows = []
        for row in request_rows:
            daily_rows.append(
                {
                    **row,
                    "unique_clients": unique_rows.get((row["date"], row["surface"]), 0),
                }
            )

        top_paths = (
            RequestMetric.objects.filter(date__gte=start_date)
            .values("surface", "path")
            .annotate(requests=Sum("request_count"))
            .order_by("-requests")[:25]
        )

        hourly_rows = (
            RequestMetric.objects.filter(date=timezone.localdate())
            .values("hour", "surface")
            .annotate(requests=Sum("request_count"))
            .order_by("hour", "surface")
        )

        context["daily_rows"] = daily_rows
        context["top_paths"] = top_paths
        context["hourly_rows"] = hourly_rows
        context["days"] = days
        return context


class SiteLoginView(LoginView):
    template_name = "registration/login.html"
    redirect_authenticated_user = True
    next_page = reverse_lazy("graph")


class RegisterView(FormView):
    template_name = "registration/register.html"
    form_class = RegistrationForm
    success_url = reverse_lazy("login")

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect("graph")
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        user = form.save(commit=False)
        user.email = form.cleaned_data["email"]
        user.is_active = False
        user.save()

        users_group, _created = Group.objects.get_or_create(name=USER_GROUP_NAME)
        user.groups.add(users_group)

        send_mail(
            subject="AgilePredict registration request",
            message=(
                "A new AgilePredict user has registered and is awaiting approval.\n\n"
                f"Username: {user.username}\n"
                f"Email: {user.email}\n"
            ),
            from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
            recipient_list=["foboundy@gmail.com"],
            fail_silently=False,
        )

        messages.success(
            self.request,
            "Registration submitted. Access will be approved on a case by case basis by the site administrators.",
        )
        return super().form_valid(form)


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

    _source_colors = {
        "AgilePredict": "#ffc107",
        "AgileForecast": "#0dcaf0",
        "X2R": "#fd7e14",
    }

    def _source_color(self, label):
        return self._source_colors.get(label, "#adb5bd")

    def _chart_title(self, title_str):
        """Return (title_dict_or_None, top_margin_px). Override in subclasses."""
        return {"text": title_str, "x": 0.5}, 50

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
                            line={"color": self._source_color("AgilePredict"), "width": 2},
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
                    line={"color": self._source_color("AgilePredict"), "width": 2},
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

    def build_predicted_series(self, forecast_rows, start_hours, end_hours, value_attr="agile_pred"):
        predicted_by_date_time = {}
        for row in forecast_rows:
            lead_hours = (row.date_time - row.forecast.created_at).total_seconds() / 3600
            if lead_hours < start_hours or lead_hours >= end_hours:
                continue
            if row.date_time not in predicted_by_date_time:
                predicted_by_date_time[row.date_time] = getattr(row, value_attr)

        predicted = pd.Series(predicted_by_date_time).sort_index()
        if len(predicted) > 0:
            predicted.index = pd.to_datetime(predicted.index)
        return predicted

    def build_external_predicted_series(self, forecast_rows, start_hours, end_hours, raw_day_ahead=False):
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
            if raw_day_ahead:
                predicted = day_ahead_to_agile(predicted, reverse=True, region="G")
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

    def build_metrics_table(
        self,
        actual,
        forecast_rows,
        external_rows_by_label=None,
        forecast_value_attr="agile_pred",
        external_raw_day_ahead=False,
    ):
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
                value_attr=forecast_value_attr,
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
                    raw_day_ahead=external_raw_day_ahead,
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

            for metric_key, metric_label in [("mae", "MAE"), ("rmse", "RMSE"), ("bias", "Bias")]:
                rows.append(
                    {
                        "model": source_label,
                        "parameter": metric_label,
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
        region = self.kwargs.get("region", "Z").upper()
        if region not in regions:
            region = "Z"
        data_region = region
        raw_day_ahead = _is_raw_day_ahead_region(region)
        can_compare_external = True
        price_display = _price_display(region)

        try:
            offset_days = int(self.request.GET.get("offset_days", 1))
        except ValueError:
            offset_days = 1
        offset_days = min(max(offset_days, 0), self.max_offset_days)
        start_hours = offset_days * 24
        end_hours = (offset_days + 1) * 24
        offset_label = self.format_offset_label(offset_days)

        compare_agileforecast = can_compare_external and _truthy(self.request.GET.get("compare_agileforecast"))
        compare_x2r = can_compare_external and _truthy(self.request.GET.get("compare_x2r"))

        context["region"] = region
        context["data_region"] = data_region
        context["can_compare_external"] = can_compare_external
        context["compare_agileforecast"] = compare_agileforecast
        context["compare_x2r"] = compare_x2r
        context["region_name"] = regions.get(region, {"name": ""})["name"]
        context["is_raw_day_ahead_region"] = raw_day_ahead
        context["price_label"] = price_display["label"]
        context["price_unit"] = price_display["unit"]
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
            day_ahead = pd.Series(dtype=float)
            actual = pd.Series(dtype=float)

        forecast_value_attr = "day_ahead" if raw_day_ahead else "agile_pred"
        if raw_day_ahead:
            forecast_rows = list(
                ForecastData.objects.filter(date_time__gte=start, date_time__lte=end)
                .select_related("forecast")
                .order_by("date_time", "-forecast__created_at")
            )
        else:
            forecast_rows = list(
                AgileData.objects.filter(region=data_region, date_time__gte=start, date_time__lte=end)
                .select_related("forecast")
                .order_by("date_time", "-forecast__created_at")
            )
        predicted = self.build_predicted_series(
            forecast_rows,
            start_hours,
            end_hours,
            value_attr=forecast_value_attr,
        )

        external_sources = []
        if can_compare_external:
            if compare_agileforecast:
                external_sources.append(
                    (ExternalForecast.SOURCE_AGILEFORECAST, "AgileForecast", self._source_color("AgileForecast"))
                )
            if compare_x2r:
                external_sources.append((ExternalForecast.SOURCE_X2R, "X2R", self._source_color("X2R")))

        external_rows_by_label = {}
        for source, label, _color in external_sources:
            external_rows_by_label[label] = list(
                ExternalForecast.objects.filter(
                    source=source,
                    region="G" if raw_day_ahead else data_region,
                    date_time__gte=start,
                    date_time__lte=end,
                ).order_by("date_time", "-source_created_at")
            )

        # Expose raw data as instance attrs so HistoryV2View can build alternate-unit metrics
        self._day_ahead_raw = day_ahead
        self._history_forecast_rows = forecast_rows
        self._history_external_rows = external_rows_by_label
        self._history_forecast_value_attr = forecast_value_attr
        self._history_data_region = data_region
        self._history_raw_day_ahead = raw_day_ahead

        metrics_table = self.build_metrics_table(
            actual,
            forecast_rows,
            external_rows_by_label,
            forecast_value_attr=forecast_value_attr,
            external_raw_day_ahead=raw_day_ahead,
        )

        figure = make_subplots(rows=1, cols=1)
        hover_template_price = f"%{{x|%d %b %H:%M}}<br>%{{y:.2f}} {price_display['unit']}"

        if len(actual) > 0:
            figure.add_trace(
                go.Scatter(
                    x=actual.index.tz_convert("GB"),
                    y=actual,
                    line={"color": "white", "width": 2},
                    mode="lines",
                    name=price_display["actual_label"],
                    hovertemplate=hover_template_price,
                )
            )

        prediction_segment_count = self.add_prediction_traces(figure, predicted, offset_label, hover_template_price)
        external_counts = []
        if can_compare_external:
            for source, label, color in external_sources:
                external_rows = external_rows_by_label[label]
                external_predicted = self.build_external_predicted_series(
                    external_rows,
                    start_hours,
                    end_hours,
                    raw_day_ahead=raw_day_ahead,
                )
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
            title = f"{price_display['label']} History - {offset_label} ahead"
        else:
            title = (
                f"{price_display['label']} History - {offset_label} ahead | "
                f"MAE {error_metrics['mae']:.2f}{price_display['unit']} | "
                f"RMSE {error_metrics['rmse']:.2f}{price_display['unit']} | "
                f"Bias {error_metrics['bias']:+.2f}{price_display['unit']}"
            )

        chart_title_dict, chart_top_margin = self._chart_title(title)
        layout_kw = {
            "yaxis": {"title": price_display["axis_title"]},
            "margin": {"r": 5, "t": chart_top_margin},
            "legend": dict(orientation="h", yanchor="top", y=-0.15, xanchor="right", x=1),
            "height": 520,
            "template": "plotly_dark",
            "hovermode": "x",
            "plot_bgcolor": "#212529",
            "paper_bgcolor": "#343a40",
        }
        if chart_title_dict:
            layout_kw["title"] = chart_title_dict
        figure.update_layout(**layout_kw)
        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86000000], value="%H:%M<br>%a %d %b"),
                dict(dtickrange=[86000000, None], value="%d %b"),
            ],
        )
        figure.update_yaxes(title_text=price_display["axis_title"], fixedrange=True)

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
        kwargs["region"] = self.kwargs.get("region", "X").upper()
        kwargs["prefix"] = "test"
        kwargs["local_realtime_external_forecasts"] = _can_use_admin_only_features(self.request)
        # print(kwargs)
        return kwargs

    def fetch_live_external_forecasts(self, region, show_agileforecast, show_x2r):
        if not _can_use_admin_only_features(getattr(self, "request", None)):
            return [], []

        forecasts = []
        errors = []
        sources = []
        if show_agileforecast:
            sources.append(("AgileForecast", "#0dcaf0", fetch_agileforecast))
        if show_x2r:
            sources.append(("X2R", "#fd7e14", fetch_x2r))

        for label, color, fetcher in sources:
            try:
                forecast = fetcher(region)
            except Exception as exc:
                logger.warning("Unable to fetch live %s forecast for region %s: %s", label, region, exc)
                errors.append(f"{label}: {exc}")
                continue

            forecasts.append(
                {
                    "label": label,
                    "color": color,
                    **forecast,
                }
            )

        return forecasts, errors

    def filter_forecast_rows_for_plot(self, rows, actual_end, plot_end, show_overlap):
        return [
            row
            for row in rows
            if row["date_time"] <= plot_end and (show_overlap or row["date_time"] >= actual_end)
        ]

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
        show_live_agileforecast = kwargs.get("show_live_agileforecast", False)
        show_live_x2r = kwargs.get("show_live_x2r", False)
        raw_day_ahead = _is_raw_day_ahead_region(region)
        if raw_day_ahead:
            show_export = False
            show_range = False
        price_display = _price_display(region, show_export=show_export)
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
        plot_end = forecast_end

        # print(f"Forecast End: {forecast_end}")
        ph_qs = PriceHistory.objects.order_by("-date_time")
        try:
            price_start = ph_qs[48 * PRIOR_DAYS].date_time
        except IndexError:
            last_ph = ph_qs.last()
            price_start = last_ph.date_time if last_ph is not None else forecast_start
        # print(f"Price Start: {price_start}")

        start = min(forecast_start, price_start)

        data = []
        p = PriceHistory.objects.filter(date_time__gte=start).order_by("-date_time")

        day_ahead = pd.Series(index=[a.date_time for a in p], data=[a.day_ahead for a in p])
        actual_price = day_ahead_to_agile(day_ahead, region=region, export=show_export).sort_index()
        price_label = price_display["label"]
        actual_label = price_display["actual_label"]

        hover_template_time_price = price_display["hover_time_price"]
        hover_template_price = price_display["hover_price"]

        data = data + [
            go.Scatter(
                x=actual_price.loc[:plot_end].index.tz_convert("GB"),
                y=actual_price.loc[:plot_end],
                marker={"symbol": 104, "size": 1, "color": "white"},
                mode="lines",
                name=actual_label,
                hovertemplate=hover_template_price,
            )
        ]

        limit = None
        width = 3
        for f in Forecasts.objects.filter(id__in=forecasts_to_plot).order_by("-created_at"):
            if raw_day_ahead:
                d = ForecastData.objects.filter(forecast=f).order_by("date_time")
            else:
                d = AgileData.objects.filter(forecast=f, region=region).order_by("date_time")
            if len(d) > 0:
                if limit is None:
                    d = d[: (48 * days_to_plot)]
                    limit = d[-1].date_time
                    # print(limit)
                else:
                    d = list(d.filter(date_time__lte=limit))
                d = [row for row in d if row.date_time <= plot_end]

                x = [
                    a.date_time
                    for a in d
                    if a.date_time <= plot_end and (a.date_time >= actual_price.index[-1] or show_overlap)
                ]
                source = pd.Series(
                    index=pd.to_datetime([a.date_time for a in d]),
                    data=[a.day_ahead if raw_day_ahead else a.agile_pred for a in d],
                )
                if show_export:
                    source = import_agile_to_export_agile(source, region=region)
                y = [
                    source.loc[pd.Timestamp(a.date_time)]
                    for a in d
                    if a.date_time <= plot_end and (a.date_time >= actual_price.index[-1] or show_overlap)
                ]

                df = pd.Series(index=pd.to_datetime(x), data=y).sort_index()
                try:
                    df.index = df.index.tz_convert("GB")
                except:
                    df.index = df.index.tz_localize("GB")

                df = df.loc[actual_price.index[0] :]

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

                if (not raw_day_ahead) and (width == 3) and (d[0].agile_high != d[0].agile_low and show_range):
                    low_source = pd.Series(index=pd.to_datetime([a.date_time for a in d]), data=[a.agile_low for a in d])
                    high_source = pd.Series(index=pd.to_datetime([a.date_time for a in d]), data=[a.agile_high for a in d])
                    if show_export:
                        low_source = import_agile_to_export_agile(low_source, region=region)
                        high_source = import_agile_to_export_agile(high_source, region=region)

                    data = data + [
                        go.Scatter(
                            x=df.index,
                            y=[
                                low_source.loc[pd.Timestamp(a.date_time)]
                                for a in d
                                if a.date_time <= plot_end and (a.date_time >= actual_price.index[-1] or show_overlap)
                            ],
                            marker={"symbol": 104, "size": 10},
                            mode="lines",
                            line=dict(width=1, color="red"),
                            name="Low",
                            showlegend=False,
                            hovertemplate=hover_template_price,
                        ),
                        go.Scatter(
                            x=df.index,
                            y=[
                                high_source.loc[pd.Timestamp(a.date_time)]
                                for a in d
                                if a.date_time <= plot_end and (a.date_time >= actual_price.index[-1] or show_overlap)
                            ],
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

        live_external_forecasts, live_external_errors = self.fetch_live_external_forecasts(
            "G" if raw_day_ahead else region,
            show_live_agileforecast,
            show_live_x2r,
        )
        live_external_counts = []
        for live_forecast in live_external_forecasts:
            rows = self.filter_forecast_rows_for_plot(
                live_forecast["rows"],
                actual_price.index[-1],
                plot_end,
                show_overlap,
            )
            if not rows:
                live_external_counts.append({"label": live_forecast["label"], "count": 0})
                continue

            source = pd.Series(
                index=pd.to_datetime([row["date_time"] for row in rows]),
                data=[row["agile_pred"] for row in rows],
            ).sort_index()
            if show_export:
                source = import_agile_to_export_agile(source, region=region)
            elif raw_day_ahead:
                source = day_ahead_to_agile(source, reverse=True, region="G")
            try:
                source.index = source.index.tz_convert("GB")
            except TypeError:
                source.index = source.index.tz_localize("GB")

            created_at = pd.Timestamp(live_forecast["source_created_at"]).tz_convert("GB")
            live_external_counts.append({"label": live_forecast["label"], "count": len(source)})
            data.append(
                go.Scatter(
                    x=source.index,
                    y=source,
                    marker={"symbol": 104, "size": 10},
                    mode="lines",
                    line={"color": live_forecast["color"], "width": 2, "dash": "dot"},
                    name=f"{live_forecast['label']} Live ({created_at.strftime('%d-%b %H:%M')})",
                    hovertemplate=hover_template_price,
                )
            )

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
            yaxis={"title": price_display["axis_title"]},
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
            title_text=price_display["axis_title"],
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
        context["live_external_forecasts_enabled"] = _can_use_admin_only_features(self.request)
        context["live_external_counts"] = live_external_counts
        context["live_external_errors"] = live_external_errors

        return context

    def get_context_data(self, **kwargs):
        # print(">>>views.py | GraphFormView | get_context_data")
        context = super().get_context_data(**kwargs)
        # context["form2"] = OptionsForm()
        f = Forecasts.objects.latest("created_at")
        region = self.kwargs.get("region", "X").upper()
        if region not in regions:
            region = "X"
        context["region"] = region
        context["region_name"] = regions.get(region, {"name": ""})["name"]
        context["is_raw_day_ahead_region"] = _is_raw_day_ahead_region(region)
        # print(context)

        context = self.update_chart(context=context, forecasts_to_plot=[f.id])
        return context

    def form_valid(self, form):
        # print(">>>views.py | GraphFormView | form_valid")
        # print(form.cleaned_data)
        context = self.get_context_data(form=form)
        context = self.update_chart(context=context, **form.cleaned_data)

        return self.render_to_response(context=context)


class V2NavMixin:
    """Injects v2 navigation context so all base.html navbar links stay within /v2/."""

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(
            {
                "region_link_prefix": "/v2/",
                "history_link": "/v2/history/",
                "stats_link": "/v2/stats/",
                "api_link": "/v2/api_how_to/",
                "about_link": "/v2/about/",
                "home_link": "/v2/X/",
                "is_v2": True,
                "classic_url": "/",
            }
        )
        return context


class ProductionLoginRequiredMixin:
    """Require authentication in production (DEBUG=False); transparent in dev."""

    def dispatch(self, request, *args, **kwargs):
        if not settings.DEBUG and not request.user.is_authenticated:
            from django.contrib.auth.views import redirect_to_login

            return redirect_to_login(request.get_full_path())
        return super().dispatch(request, *args, **kwargs)


class GraphV2View(V2NavMixin, TemplateView):
    """Colour-coded bar chart UI — alternative to the accordion-sidebar GraphFormView."""

    template_name = "graph_v2.html"

    _DAY_OPTIONS = [2, 3, 5, 7, 14]
    _BAR_WIDTH_MS = int(30 * 60 * 1000 * 0.92)
    _OLDER_COLORS = ["#adb5bd", "#6c757d", "#495057"]
    _CHARGE_SLOTS = 4  # 2-hour window

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        region = self.kwargs.get("region", "X").upper()
        if region not in regions:
            region = "X"
        raw = _is_raw_day_ahead_region(region)

        days = min(max(int(self.request.GET.get("days", 5)), 1), 14)
        show_band = self.request.GET.get("band", "1") != "0"
        show_export = self.request.GET.get("export", "0") == "1" and not raw
        show_gen = self.request.GET.get("gen", "1") == "1"
        color_fn = _export_price_color if show_export else _price_color

        fc_param = self.request.GET.getlist("fc")
        try:
            selected_ids = [int(x) for x in fc_param if str(x).strip()]
        except ValueError:
            selected_ids = []

        now_gb = pd.Timestamp.now(tz="GB")
        prior_gb = now_gb - pd.Timedelta(hours=12)
        end_gb = now_gb + pd.Timedelta(days=days)

        price_display = _price_display(region, show_export=show_export)
        unit = price_display["unit"]

        # --- Historical actuals ---
        ph_rows = list(
            PriceHistory.objects.filter(
                date_time__gte=prior_gb.tz_convert("UTC")
            ).order_by("date_time")
        )
        if ph_rows:
            day_ahead_s = pd.Series(
                index=pd.to_datetime([r.date_time for r in ph_rows]).tz_convert("GB"),
                data=[r.day_ahead for r in ph_rows],
            )
            actual = day_ahead_to_agile(day_ahead_s, region=region, export=show_export).sort_index()
        else:
            actual = pd.Series(dtype=float)

        actual_end = actual.index[-1] if not actual.empty else now_gb

        # --- Forecasts ---
        recent_forecasts = list(Forecasts.objects.order_by("-created_at")[:8])
        latest = recent_forecasts[0] if recent_forecasts else None

        if selected_ids:
            forecasts_to_plot = list(Forecasts.objects.filter(id__in=selected_ids).order_by("-created_at")[:5])
        else:
            forecasts_to_plot = [latest] if latest else []

        # Primary (most-recent) forecast — drives bars, band, and summary
        primary_s = pd.Series(dtype=float)
        low_s = pd.Series(dtype=float)
        high_s = pd.Series(dtype=float)

        if latest is not None:
            if raw:
                fd_rows = list(
                    ForecastData.objects.filter(
                        forecast=latest,
                        date_time__gte=actual_end.tz_convert("UTC"),
                        date_time__lte=end_gb.tz_convert("UTC"),
                    ).order_by("date_time")
                )
                if fd_rows:
                    primary_s = pd.Series(
                        index=pd.to_datetime([r.date_time for r in fd_rows]).tz_convert("GB"),
                        data=[r.day_ahead for r in fd_rows],
                    )
            else:
                ad_rows = list(
                    AgileData.objects.filter(
                        forecast=latest,
                        region=region,
                        date_time__gte=actual_end.tz_convert("UTC"),
                        date_time__lte=end_gb.tz_convert("UTC"),
                    ).order_by("date_time")
                )
                if ad_rows:
                    idx = pd.to_datetime([r.date_time for r in ad_rows]).tz_convert("GB")
                    primary_s = pd.Series(index=idx, data=[r.agile_pred for r in ad_rows])
                    low_s = pd.Series(index=idx, data=[r.agile_low for r in ad_rows])
                    high_s = pd.Series(index=idx, data=[r.agile_high for r in ad_rows])
                    if show_export:
                        # Apply export conversion to all three series together
                        primary_s = import_agile_to_export_agile(primary_s, region=region)
                        low_s = import_agile_to_export_agile(low_s, region=region)
                        high_s = import_agile_to_export_agile(high_s, region=region)

        # --- Summary statistics ---
        current_slot = now_gb.floor("30min")
        combined = pd.concat([actual, primary_s]).sort_index()
        upcoming = combined[combined.index >= current_slot]

        def fmt(p):
            return f"{p:.1f}"

        summary = {"unit": unit, "is_export": show_export}

        if not upcoming.empty:
            p = float(upcoming.iloc[0])
            summary["current_price"] = fmt(p)
            summary["current_time"] = upcoming.index[0].strftime("%H:%M")
            summary["current_badge"] = _price_badge(p if not raw else None)

        # Upcoming forecast for next-slot and window calculations
        upcoming_fc = primary_s[primary_s.index >= current_slot].iloc[:48]

        if not upcoming_fc.empty:
            if show_export:
                # For export, expensive slots are desirable
                best_single_idx = upcoming_fc.idxmax()
                best_single_p = float(upcoming_fc[best_single_idx])
                summary["best_slot_price"] = fmt(best_single_p)
                summary["best_slot_time"] = best_single_idx.strftime("%H:%M")
                summary["best_slot_badge"] = "danger" if best_single_p > 35 else "warning text-dark" if best_single_p > 25 else "secondary"
                summary["best_slot_label"] = "Best export slot (24h)"
            else:
                cheapest_idx = upcoming_fc.idxmin()
                cp = float(upcoming_fc[cheapest_idx])
                summary["best_slot_price"] = fmt(cp)
                summary["best_slot_time"] = cheapest_idx.strftime("%H:%M")
                summary["best_slot_badge"] = _price_badge(cp)
                summary["best_slot_label"] = "Cheapest slot (24h)"

            # Best contiguous 2h window (4 slots)
            if len(upcoming_fc) >= self._CHARGE_SLOTS:
                rolling_avg = upcoming_fc.rolling(self._CHARGE_SLOTS).mean().dropna()
                if show_export:
                    best_end = rolling_avg.idxmax()
                else:
                    best_end = rolling_avg.idxmin()
                end_loc = upcoming_fc.index.get_loc(best_end)
                start_loc = max(0, end_loc - self._CHARGE_SLOTS + 1)
                best_start = upcoming_fc.index[start_loc]
                window_close = best_end + pd.Timedelta("30min")
                summary["window_start"] = best_start.strftime("%H:%M")
                summary["window_end"] = window_close.strftime("%H:%M")
                summary["window_avg"] = fmt(float(rolling_avg[best_end]))
                summary["window_badge"] = _price_badge(float(rolling_avg[best_end]) if not raw and not show_export else None)
                summary["window_label"] = "Best 2h export window" if show_export else "Cheapest 2h window"

        if not primary_s.empty:
            upcoming_48h = primary_s[primary_s.index >= current_slot].iloc[:96]
            if not upcoming_48h.empty:
                summary["range_min"] = fmt(float(upcoming_48h.min()))
                summary["range_max"] = fmt(float(upcoming_48h.max()))
                cheap_count = int((upcoming_48h < 15).sum()) if not show_export else None
                summary["range_cheap"] = cheap_count
                summary["range_label"] = "Next 48h range"

        if latest is not None:
            try:
                summary["forecast_updated"] = (
                    pd.Timestamp(latest.created_at).tz_convert("GB").strftime("%d %b %H:%M")
                )
            except Exception:
                pass

        # --- API / data-source health status ---
        # Expected future forecast rows for a healthy 14-day horizon:
        _FULL_HORIZON_ROWS = 672   # 14 days × 48 half-hours
        _OK_THRESHOLD = 400        # ≥ 400 rows (~8 days) = green
        _WARN_THRESHOLD = 100      # ≥ 100 rows (~2 days) = amber

        def _row_health(rows, full=_FULL_HORIZON_ROWS, ok=_OK_THRESHOLD, warn=_WARN_THRESHOLD):
            if rows >= ok:
                return "ok"
            elif rows >= warn:
                return "warn"
            return "fail"

        # Read cached source-row counts written by the last update run
        cached = cache.get("api_source_status") or {}
        source_rows = cached.get("source_rows", {})

        api_sources = [
            {
                "name": "NESO",
                "rows": source_rows.get("neso", None),
                "health": _row_health(source_rows["neso"]) if "neso" in source_rows else "unknown",
                "detail": "wind, solar & demand forecasts",
            },
            {
                "name": "BMRS",
                "rows": source_rows.get("bmrs", None),
                "health": _row_health(source_rows["bmrs"]) if "bmrs" in source_rows else "unknown",
                "detail": "national demand forecast",
            },
            {
                "name": "Open-Meteo",
                "rows": source_rows.get("openmeteo", None),
                "health": _row_health(source_rows["openmeteo"]) if "openmeteo" in source_rows else "unknown",
                "detail": "weather forecast",
            },
        ]

        # Octopus: check freshness of PriceHistory
        latest_price = PriceHistory.objects.order_by("-date_time").first()
        octopus_health = "unknown"
        octopus_detail = "no data"
        if latest_price is not None:
            try:
                price_ts = pd.Timestamp(latest_price.date_time).tz_convert("UTC")
                hours_old = (pd.Timestamp.now(tz="UTC") - price_ts).total_seconds() / 3600
                # Agile prices publish by ~16:00 for next day; > 26h old = likely stale
                octopus_health = "ok" if hours_old <= 26 else "warn" if hours_old <= 36 else "fail"
                octopus_detail = pd.Timestamp(latest_price.date_time).tz_convert("GB").strftime("to %d %b %H:%M")
            except Exception:
                pass
        api_sources.append({
            "name": "Octopus",
            "rows": None,
            "health": octopus_health,
            "detail": octopus_detail,
        })

        # Overall health badge driven by worst individual source
        _rank = {"ok": 0, "warn": 1, "fail": 2, "unknown": 1}
        overall_health = max((s["health"] for s in api_sources), key=lambda h: _rank.get(h, 1))

        # Forecast horizon from latest ForecastData
        future_rows = 0
        if latest is not None:
            future_rows = ForecastData.objects.filter(
                forecast=latest, date_time__gt=now_gb.tz_convert("UTC"),
            ).count()

        api_status = {
            "sources": api_sources,
            "overall_health": overall_health,
            "horizon_days": f"{future_rows / 48:.1f}" if future_rows >= 24 else ("< 1" if future_rows > 0 else "—"),
            "updated": pd.Timestamp(latest.created_at).tz_convert("GB").strftime("%d %b %H:%M") if latest else None,
        }

        # --- Price heat strip (replaces daily table) ---
        # Shows a row of coloured half-hour slots per day — immediately scannable
        heat_strip = []
        combined_all = pd.concat([actual, primary_s]).sort_index()
        if not combined_all.empty:
            for day_start, group in combined_all.resample("D"):
                slots = [
                    {
                        "time": ts.strftime("%H:%M"),
                        "price": round(float(v), 1),
                        "color": color_fn(float(v) if not raw else None),
                    }
                    for ts, v in group.items()
                    if pd.notna(v)
                ]
                if slots:
                    heat_strip.append(
                        {
                            "date": pd.Timestamp(day_start).strftime("%a %d"),
                            "slots": slots,
                        }
                    )

        # --- Build chart ---
        if show_gen:
            figure = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(price_display["axis_title"], "Generation & Demand"),
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4],
            )

            def add_price(trace):
                figure.add_trace(trace, row=1, col=1)

            def add_gen(trace):
                figure.add_trace(trace, row=2, col=1)

            chart_height = 620
        else:
            figure = go.Figure()
            add_price = figure.add_trace
            chart_height = 420

        # Prepare error-bar arrays for uncertainty band (drawn on forecast bars)
        # This is the correct idiom for showing uncertainty on bar charts — whiskers extend
        # visibly above and below each bar regardless of bar colour.
        upper_err = lower_err = None
        if show_band and not low_s.empty and not high_s.empty and not primary_s.empty:
            aligned_lo = low_s.reindex(primary_s.index)
            aligned_hi = high_s.reindex(primary_s.index)
            upper_err = [
                max(0.0, float(h) - float(p)) if pd.notna(h) and pd.notna(p) else 0.0
                for h, p in zip(aligned_hi.values, primary_s.values)
            ]
            lower_err = [
                max(0.0, float(p) - float(l)) if pd.notna(l) and pd.notna(p) else 0.0
                for l, p in zip(aligned_lo.values, primary_s.values)
            ]

        # Confirmed actuals — two-layer: light colour bars for level context + white step-line
        # for unmistakable "this is real confirmed data" signal vs forecast bars.
        if not actual.empty:
            add_price(
                go.Bar(
                    x=actual.index,
                    y=actual.values,
                    marker_color=[color_fn(p if not raw else None) for p in actual.values],
                    marker_opacity=0.35,
                    marker_line_width=0,
                    showlegend=False,
                    hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>%{{y:.2f}} {unit}</b><extra>Confirmed</extra>",
                    width=self._BAR_WIDTH_MS,
                )
            )
            add_price(
                go.Scatter(
                    x=actual.index,
                    y=actual.values,
                    mode="lines",
                    line=dict(shape="hv", color="rgba(255,255,255,0.85)", width=2.0),
                    name="Confirmed Octopus price",
                    hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>%{{y:.2f}} {unit}</b><extra>Confirmed</extra>",
                )
            )

        # Primary forecast bars — full opacity; error bars carry the uncertainty band
        if not primary_s.empty:
            latest_label = (
                f"Forecast ({pd.Timestamp(latest.created_at).tz_convert('GB').strftime('%d %b %H:%M')})"
                if latest
                else "Forecast"
            )
            bar_kw = dict(
                x=primary_s.index,
                y=primary_s.values,
                marker_color=[color_fn(p if not raw else None) for p in primary_s.values],
                marker_opacity=1.0,
                name=latest_label,
                hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>%{{y:.2f}} {unit}</b><extra>Forecast</extra>",
                width=self._BAR_WIDTH_MS,
            )
            if upper_err is not None:
                bar_kw["error_y"] = dict(
                    type="data",
                    symmetric=False,
                    array=upper_err,
                    arrayminus=lower_err,
                    color="rgba(255,255,255,0.55)",
                    thickness=1.5,
                    width=0,
                )
            add_price(go.Bar(**bar_kw))

        # Older selected forecasts — dotted line traces
        older = [f for f in forecasts_to_plot if latest is None or f.id != latest.id]
        for i, fc_obj in enumerate(older[:3]):
            if raw:
                od_rows = list(
                    ForecastData.objects.filter(
                        forecast=fc_obj,
                        date_time__gte=actual_end.tz_convert("UTC"),
                        date_time__lte=end_gb.tz_convert("UTC"),
                    ).order_by("date_time")
                )
                if not od_rows:
                    continue
                s = pd.Series(
                    index=pd.to_datetime([r.date_time for r in od_rows]).tz_convert("GB"),
                    data=[r.day_ahead for r in od_rows],
                )
            else:
                od_rows = list(
                    AgileData.objects.filter(
                        forecast=fc_obj,
                        region=region,
                        date_time__gte=actual_end.tz_convert("UTC"),
                        date_time__lte=end_gb.tz_convert("UTC"),
                    ).order_by("date_time")
                )
                if not od_rows:
                    continue
                idx = pd.to_datetime([r.date_time for r in od_rows]).tz_convert("GB")
                s = pd.Series(index=idx, data=[r.agile_pred for r in od_rows])
                if show_export:
                    s = import_agile_to_export_agile(s, region=region)
            older_label = pd.Timestamp(fc_obj.created_at).tz_convert("GB").strftime("%d %b %H:%M")
            add_price(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    line=dict(color=self._OLDER_COLORS[i % len(self._OLDER_COLORS)], width=1.5, dash="dot"),
                    name=f"Forecast ({older_label})",
                    hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.2f}} {unit}<extra>{older_label}</extra>",
                )
            )

        # Now vline — appears across all subplots
        figure.add_vline(
            x=now_gb.timestamp() * 1000,
            line_width=2,
            line_dash="dot",
            line_color="rgba(255,255,255,0.45)",
            annotation_text="Now",
            annotation_position="top left",
            annotation_font_color="rgba(255,255,255,0.75)",
            annotation_font_size=11,
        )

        # Generation & demand subplot
        if show_gen and latest is not None:
            fp = list(
                ForecastData.objects.filter(
                    forecast=latest,
                    date_time__gte=prior_gb.tz_convert("UTC"),
                    date_time__lte=end_gb.tz_convert("UTC"),
                ).order_by("date_time")
            )
            h_rows = list(
                History.objects.filter(
                    date_time__gte=prior_gb.tz_convert("UTC"),
                    date_time__lte=now_gb.tz_convert("UTC"),
                )
            )
            if fp:
                add_gen(go.Scatter(
                    x=[r.date_time for r in fp],
                    y=[(r.demand + r.solar + r.emb_wind) / 1000 for r in fp],
                    line={"color": "cyan", "width": 2}, name="Forecast demand",
                ))
                add_gen(go.Scatter(
                    x=[r.date_time for r in fp], y=[r.nuclear / 1000 for r in fp],
                    fill="tozeroy", line={"color": "rgba(160,160,160,1)"},
                    fillcolor="rgba(180,180,180,0.8)", name="Nuclear",
                ))
                add_gen(go.Scatter(
                    x=[r.date_time for r in fp],
                    y=[(r.nuclear + r.bm_wind) / 1000 for r in fp],
                    fill="tonexty", line={"color": "rgba(63,127,63)"},
                    fillcolor="rgba(127,255,127,0.8)", name="Metered wind",
                ))
                add_gen(go.Scatter(
                    x=[r.date_time for r in fp],
                    y=[(r.nuclear + r.bm_wind + r.emb_wind) / 1000 for r in fp],
                    fill="tonexty", line={"color": "rgba(50,150,220)"},
                    fillcolor="rgba(100,200,255,0.7)", name="Embedded wind",
                ))
                add_gen(go.Scatter(
                    x=[r.date_time for r in fp],
                    y=[(r.nuclear + r.bm_wind + r.emb_wind + r.solar) / 1000 for r in fp],
                    fill="tonexty", line={"color": "lightgray", "width": 2},
                    fillcolor="rgba(255,255,127,0.8)", name="Solar",
                ))
            if h_rows:
                add_gen(go.Scatter(
                    x=[r.date_time for r in h_rows],
                    y=[(r.demand + r.solar + (r.total_wind - r.bm_wind)) / 1000 for r in h_rows],
                    line={"color": "#aaaa77", "width": 2}, name="Historic demand",
                ))
            figure.update_yaxes(title_text="Power [GW]", row=2, col=1)

        # Layout
        common_layout = dict(
            barmode="overlay",
            margin=dict(r=10, t=15, l=60, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            height=chart_height,
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor="#212529",
            paper_bgcolor="#1a1d21",
            bargap=0.02,
        )
        if show_gen:
            figure.update_layout(**common_layout)
            figure.update_yaxes(
                title_text=price_display["axis_title"],
                zeroline=True,
                zerolinecolor="#888",
                zerolinewidth=2,
                row=1,
                col=1,
            )
        else:
            figure.update_layout(
                **common_layout,
                yaxis=dict(
                    title=price_display["axis_title"],
                    zeroline=True,
                    zerolinecolor="#888",
                    zerolinewidth=2,
                ),
            )

        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86400000], value="%H:%M<br>%a %d %b"),
                dict(dtickrange=[86400000, None], value="%d %b"),
            ]
        )

        # Forecast list for template checkboxes
        forecast_list = [
            {
                "id": f.id,
                "label": pd.Timestamp(f.created_at).tz_convert("GB").strftime("%d %b %H:%M"),
                "selected": f.id in selected_ids
                or (not selected_ids and f.id == (latest.id if latest else None)),
            }
            for f in recent_forecasts
        ]

        context.update(
            {
                "region": region,
                "region_name": regions.get(region, {}).get("name", region),
                "is_raw_day_ahead_region": raw,
                "days": days,
                "show_band": show_band,
                "show_export": show_export,
                "show_gen": show_gen,
                "summary": summary,
                "api_status": api_status,
                "heat_strip": heat_strip,
                "day_options": self._DAY_OPTIONS,
                "forecast_list": forecast_list,
                "selected_ids": selected_ids,
                "graph": figure.to_html(
                    full_html=False,
                    config={"displayModeBar": False, "responsive": True},
                ),
                "classic_url": f"/{region}/",
            }
        )
        return context


class HistoryV2View(V2NavMixin, HistoryView):
    """v2-styled history / accuracy view — always uses region X (national average)."""

    template_name = "history_v2.html"

    def get_kwargs(self):
        # Force region to X regardless of URL kwarg — no region selector in v2 history.
        kwargs = super().get_kwargs() if hasattr(super(), "get_kwargs") else {}
        kwargs["region"] = "X"
        return kwargs

    def setup(self, request, *args, **kwargs):
        # Force region Z so the main price chart always uses raw day-ahead prices (£/MWh).
        # Display context is overridden in get_context_data to show "National Average".
        kwargs["region"] = "Z"
        super().setup(request, *args, **kwargs)

    # CVD-safe palette (distinguishable under protan/deuteranopia): sky-blue, vermilion, bluish-green.
    # Overrides HistoryView._source_colors so the price chart and error chart use matching colours.
    _source_colors = {
        "AgilePredict": "#56B4E9",
        "AgileForecast": "#D55E00",
        "X2R": "#009E73",
    }

    def _build_metrics_chart(self, metrics_table, price_unit, selected_metric="mae"):
        columns = metrics_table.get("columns", [])
        rows = metrics_table.get("rows", [])
        if not columns or not rows:
            return None

        x_offsets = [c["offset"] for c in columns]
        x_labels = [c["label"] for c in columns]
        metric_key = selected_metric.upper()
        _METRIC_LABELS = {"MAE": "Mean Absolute Error", "RMSE": "Root Mean Squared Error"}
        metric_label = _METRIC_LABELS.get(metric_key, metric_key)

        # Parse all series: (model, param) → [(x_offset, float_value), ...]
        series = {}
        for row in rows:
            model = row["model"]
            param = row["parameter"]
            pts = []
            for i, v in enumerate(row["values"]):
                if i >= len(x_offsets):
                    break
                try:
                    pts.append((x_offsets[i], float(str(v).replace("+", ""))))
                except (ValueError, TypeError):
                    pass
            if pts:
                series[(model, param)] = pts

        # Compute a shared Y scale so both subplots use the same magnitude.
        # Error subplot: [0, shared_max]; Bias subplot: [-shared_max, shared_max].
        # This makes bias directly comparable to error at a glance.
        error_vals = [v for (_, p), pts in series.items() if p == metric_key for _, v in pts]
        bias_vals = [v for (_, p), pts in series.items() if p == "Bias" for _, v in pts]
        if not error_vals:
            return None
        shared_max = max(max(error_vals) * 1.12, 0.1)
        if bias_vals:
            shared_max = max(shared_max, max(abs(v) for v in bias_vals) * 1.12)
        shared_max = round(shared_max + 0.05, 1)  # small padding, round for clean ticks

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(metric_label, "Bias"),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
        )

        seen_models = set()
        for (model, param), pts in series.items():
            if param not in (metric_key, "Bias"):
                continue
            color = self._source_color(model)
            xs, ys = zip(*pts)
            show_legend = model not in seen_models
            seen_models.add(model)
            target_row = 1 if param == metric_key else 2
            fig.add_trace(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(color=color, width=2),
                    name=model,
                    legendgroup=model,
                    showlegend=show_legend,
                    hovertemplate=f"%{{y:.2f}} {price_unit}<extra>{model} {param}</extra>",
                ),
                row=target_row,
                col=1,
            )

        x_range = [-0.4, max(x_offsets) + 0.4] if x_offsets else None

        fig.update_layout(
            height=400,
            margin=dict(r=10, t=32, l=70, b=10),
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#1a1d21",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
        )
        fig.update_xaxes(
            tickvals=x_offsets,
            ticktext=x_labels,
            range=x_range,
        )
        fig.update_xaxes(title_text="Lead time", row=2, col=1)
        fig.update_yaxes(
            title_text=price_unit,
            range=[0, shared_max],
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text=price_unit,
            range=[-shared_max, shared_max],
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=2,
            row=2,
            col=1,
        )

        return fig.to_html(
            full_html=False,
            config={"displayModeBar": False, "responsive": True},
        )

    def _chart_title(self, title_str):
        return None, 15  # no title on price chart in v2; reduce top margin

    def _build_agile_metrics_table(self):
        """Build metrics in Agile p/kWh by forward-converting DA predictions for region X."""
        actual_da = getattr(self, "_day_ahead_raw", pd.Series(dtype=float))
        if actual_da.empty:
            return {}
        actual_agile = day_ahead_to_agile(actual_da, region="X").sort_index()
        forecast_rows = getattr(self, "_history_forecast_rows", [])
        external_rows = getattr(self, "_history_external_rows", {})

        columns_by_offset = {}
        metric_sets = {"AgilePredict": {}, **{label: {} for label in external_rows}}

        for offset_days in range(self.max_offset_days + 1):
            # Predictions are from ForecastData.day_ahead (£/MWh); convert to Agile X
            predicted_da = self.build_predicted_series(
                forecast_rows, offset_days * 24, (offset_days + 1) * 24, value_attr="day_ahead"
            )
            if len(predicted_da) >= 2:
                predicted = day_ahead_to_agile(predicted_da, region="X")
            else:
                predicted = pd.Series(dtype=float)

            metrics = self.calculate_error_metrics(actual_agile, predicted)
            if metrics is not None:
                columns_by_offset[offset_days] = {"label": self.format_offset_label(offset_days), "n": metrics["n"]}
                metric_sets["AgilePredict"][offset_days] = metrics

            for label, ext_rows in external_rows.items():
                # External rows were fetched for region "G"; raw_day_ahead=True gives them in DA
                ext_da = self.build_external_predicted_series(
                    ext_rows, offset_days * 24, (offset_days + 1) * 24, raw_day_ahead=True
                )
                if len(ext_da) >= 2:
                    ext_pred = day_ahead_to_agile(ext_da, region="X")
                else:
                    ext_pred = pd.Series(dtype=float)
                ext_metrics = self.calculate_error_metrics(actual_agile, ext_pred)
                if ext_metrics is None:
                    continue
                if offset_days not in columns_by_offset:
                    columns_by_offset[offset_days] = {"label": self.format_offset_label(offset_days), "n": ""}
                metric_sets[label][offset_days] = ext_metrics

        columns = [{"offset": o, **c} for o, c in sorted(columns_by_offset.items())]
        rows = []
        mobile_rows = []
        for source_label, metrics_by_offset in metric_sets.items():
            if not metrics_by_offset:
                continue
            for metric_key, metric_label in [("mae", "MAE"), ("rmse", "RMSE"), ("bias", "Bias")]:
                rows.append({
                    "model": source_label,
                    "parameter": metric_label,
                    "values": [
                        self.format_metric_values(metrics_by_offset.get(col["offset"]))[metric_key]
                        for col in columns
                    ],
                })
            for col in columns:
                vals = self.format_metric_values(metrics_by_offset.get(col["offset"]))
                if not vals["n"]:
                    continue
                mobile_rows.append({
                    "source": source_label, "label": col["label"],
                    "n": vals["n"], "mae": vals["mae"], "rmse": vals["rmse"], "bias": vals["bias"],
                })
        return {"columns": columns, "rows": rows, "mobile_rows": mobile_rows}

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Override region display — data uses Z (DA £/MWh) but UI shows National Average
        context["region"] = "X"
        from config.utils import regions as _regions
        context["region_name"] = _regions.get("X", {}).get("name", "National Average")
        context["is_raw_day_ahead_region"] = False  # show unit toggle, hide raw-DA-specific UI

        selected_metric = self.request.GET.get("metric", "mae").lower()
        if selected_metric not in ("mae", "rmse"):
            selected_metric = "mae"

        # Default unit_mode is "da": main chart and accuracy chart both in £/MWh.
        # "agile" forward-converts to Agile p/kWh for the accuracy chart only.
        unit_mode = self.request.GET.get("unit_mode", "da").lower()
        if unit_mode not in ("agile", "da"):
            unit_mode = "da"

        if unit_mode == "agile":
            chart_metrics_table = self._build_agile_metrics_table()
            chart_price_unit = "p/kWh"
        else:
            chart_metrics_table = context.get("metrics_table", {})
            chart_price_unit = "£/MWh"

        context["history_region_prefix"] = "/v2/history/"
        context["selected_metric"] = selected_metric
        context["unit_mode"] = unit_mode
        context["source_colors"] = self._source_colors
        context["classic_url"] = "/history/X/"
        context["metrics_chart"] = self._build_metrics_chart(
            chart_metrics_table,
            chart_price_unit,
            selected_metric=selected_metric,
        )
        return context


class StatsV2View(V2NavMixin, StatsView):
    template_name = "stats_v2.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["classic_url"] = "/stats"
        return context


class AboutV2View(V2NavMixin, AboutView):
    template_name = "about_v2.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["classic_url"] = "/about"
        return context


class ApiHowToV2View(V2NavMixin, ApiHowToView):
    template_name = "api_how_to_v2.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["classic_url"] = "/api_how_to"
        return context
