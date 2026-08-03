from hmac import compare_digest
from datetime import datetime, time as datetime_time, timedelta, timezone as dt_timezone
import json
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
from django.utils.dateparse import parse_date, parse_datetime
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

# Create your views here.
from django.views.generic import FormView, TemplateView
from plotly.subplots import make_subplots

from config.settings import GLOBAL_SETTINGS
from config.utils import day_ahead_to_agile, import_agile_to_export_agile

from .external_forecasts import fetch_agileforecast, fetch_x2r, region_rows_from_g
from .forms import ForecastForm, RegistrationForm
from .models import (
    AgileData,
    ExternalForecast,
    ForecastData,
    Forecasts,
    History,
    KofiPayment,
    PlotImage,
    PriceHistory,
    UpdateJob,
)

regions = GLOBAL_SETTINGS["REGIONS"]
PRIOR_DAYS = 2
logger = logging.getLogger("prices.web")
USER_GROUP_NAME = "Users"
PRIVILEGED_GROUP_NAME = "Privileged Users"

_FEATURE_LABELS = {
    "bm_wind": "BM wind (MW)",
    "solar": "Solar (MW)",
    "emb_wind": "Embedded wind (MW)",
    "nuclear": "UK nuclear (MW)",
    "fr_nuclear": "FR nuclear (MW)",
    "gas_ttf": "Gas TTF (€/MWh)",
    "demand": "Demand (MW)",
    "temp_2m": "Temperature (°C)",
    "wind_10m": "Wind speed (m/s)",
    "rad": "Radiation (W/m²)",
    "dispatchable_capacity": "Dispatch capacity (MW)",
    "fr_wind": "FR wind speed (m/s)",
    "fr_rad": "FR solar rad. (W/m²)",
    "peak": "Peak hours (16–19)",
    "weekend": "Weekend",
    "days_ago": "Forecast age (days)",
    "dt": "Lead time (days)",
    "time": "Time of day",
    "dow": "Day of week",
}

# Region X (national average) linear transform: agile_p_per_kwh ≈ day_ahead * _AF_M + _AF_A
# Used to scale model-native SHAP values (£/MWh) to consumer-facing Agile p/kWh for display.
_AF_M, _AF_A = GLOBAL_SETTINGS["REGIONS"]["X"]["factors"]


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


def _export_price_badge(p):
    """Map a p/kWh Agile export price to a Bootstrap badge variant — high prices are good."""
    if p is None:
        return "secondary"
    if p < 0:
        return "danger"
    if p < 15:
        return "secondary"
    if p < 25:
        return "warning text-dark"
    return "success"


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
def healthz(request):
    """Minimal liveness check for the fly.io health monitor: touches the DB but does
    no heavy computation, so a genuinely wedged worker/DB still fails this fast."""
    Forecasts.objects.exists()
    return HttpResponse("OK")


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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        rows = []
        for code, cfg in GLOBAL_SETTINGS["REGIONS"].items():
            factors = cfg.get("factors")
            if cfg.get("raw_day_ahead") or not factors:
                continue  # Z = raw day-ahead, no conversion
            export = cfg.get("export_factors")
            rows.append({
                "code": code,
                "name": cfg.get("name", code),
                "mult": factors[0],
                "peak": factors[1],
                "exp_factor": export[0] if export else None,
                "exp_base": export[1] if export else None,
                "exp_peak": export[2] if export else None,
            })
        # Current value of the time-stepped peak "shift" (added during peak hours).
        shifts = GLOBAL_SETTINGS.get("SHIFTS", {})
        now = timezone.now()
        applicable = [v for d, v in sorted(shifts.items()) if pd.Timestamp(d, tz="GB") <= now]
        context["conversion_regions"] = rows
        context["peak_shift"] = applicable[-1] if applicable else 0.0
        return context


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
            include_plotlyjs="cdn",
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
                subplot_titles=("", "Generation and Demand"),
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
                "t": 10,
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
            full_html=False,
            include_plotlyjs="cdn",
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

    _nav_page = ""

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(
            {
                "region_link_prefix": "/v2/",
                "history_link": "/v2/history/",
                "stats_link": "/v2/stats/",
                "api_link": "/v2/api_how_to/",
                "about_link": "/v2/about/",
                # Only surface the ops pages (health, costs) where the monitor log
                # exists (the dev host) — keeps them off public production.
                "health_link": "/v2/health/" if (Path(settings.BASE_DIR) / "logs" / "uptime_monitor.log").exists() else "",
                "costs_link": "/v2/costs/" if (Path(settings.BASE_DIR) / "logs" / "uptime_monitor.log").exists() else "",
                "home_link": "/v2/X/",
                "is_v2": True,
                "classic_url": "/X/",
                "current_page": self._nav_page,
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


def _stored_external_rows(source, region):
    """Latest stored snapshot for a source (persisted at region G by
    refresh_external_forecasts), derived to the requested region. No network I/O.
    Returns {rows, source_created_at, age_hours} or None if nothing is stored.
    """
    latest = (
        ExternalForecast.objects
        .filter(source=source, region="G")
        .order_by("-source_created_at")
        .first()
    )
    if latest is None:
        return None
    g_rows = [
        {
            "date_time": r.date_time,
            "agile_pred": float(r.agile_pred),
            "agile_low": float(r.agile_low) if r.agile_low is not None else None,
            "agile_high": float(r.agile_high) if r.agile_high is not None else None,
        }
        for r in ExternalForecast.objects.filter(
            source=source, region="G", source_created_at=latest.source_created_at
        ).order_by("date_time")
    ]
    return {
        "rows": region_rows_from_g(g_rows, region),
        "source_created_at": latest.source_created_at,
        "age_hours": (timezone.now() - latest.downloaded_at).total_seconds() / 3600,
    }


_EXT_COLORS = {"AgileForecast": "#D55E00", "X2R": "#009E73"}
_EXT_SOURCES = {
    "AgileForecast": ExternalForecast.SOURCE_AGILEFORECAST,
    "X2R": ExternalForecast.SOURCE_X2R,
}


def _fetch_external_forecasts(ext_labels, region, show_export, prior_gb, end_gb):
    """Build AgileForecast/X2R comparison series from STORED data only.

    The data is refreshed on each `manage.py update` run (refresh_external_forecasts),
    so no live external call is made on the request path — a slow upstream used to
    stall a worker for 5-15s (see the WORKER TIMEOUT outages). Returns (traces,
    statuses) where statuses feed the "Data sources" card.
    """
    traces = []
    statuses = []
    for ext_label in ext_labels:
        ext_source_const = _EXT_SOURCES[ext_label]
        data = _stored_external_rows(ext_source_const, region)
        if data is None:
            statuses.append({"name": ext_label, "health": "fail", "detail": "No data available"})
            continue

        ext_rows = [r for r in data["rows"] if r.get("agile_pred") is not None]
        if not ext_rows:
            statuses.append({"name": ext_label, "health": "fail", "detail": "No data"})
            continue

        ext_s = pd.Series(
            index=pd.to_datetime([r["date_time"] for r in ext_rows]),
            data=[float(r["agile_pred"]) for r in ext_rows],
        ).sort_index()
        try:
            ext_s.index = ext_s.index.tz_convert("GB")
        except TypeError:
            ext_s.index = ext_s.index.tz_localize("GB")
        ext_s = ext_s[(ext_s.index >= prior_gb) & (ext_s.index <= end_gb)]
        if show_export:
            ext_s = import_agile_to_export_agile(ext_s, region=region)
        if ext_s.empty:
            statuses.append({"name": ext_label, "health": "warn", "detail": "No data in range"})
            continue

        created_at = pd.Timestamp(data["source_created_at"]).tz_convert("GB")
        health = "ok" if data["age_hours"] <= 36 else "warn"
        statuses.append({
            "name": ext_label,
            "health": health,
            "detail": f"Stored ({created_at.strftime('%d %b %H:%M')})",
        })

        traces.append({
            "name": f"{ext_label} ({created_at.strftime('%d %b %H:%M')})",
            "color": _EXT_COLORS[ext_label],
            "x": [int(ts.timestamp() * 1000) for ts in ext_s.index],
            "y": [round(float(v), 4) for v in ext_s.values],
        })

    return traces, statuses


@require_GET
def ext_forecast_json(request, region):
    """AJAX endpoint for the AgileForecast/X2R comparison overlay (af=1/x2r=1).

    Split out of GraphV2View so the main page render never blocks on these
    external HTTP calls (5-15s timeouts) — the browser fetches this
    separately once the base chart has already rendered.
    """
    region = region.upper()
    if region not in regions:
        region = "X"
    raw = _is_raw_day_ahead_region(region)

    days = min(max(int(request.GET.get("days", 5)), 1), 14)
    show_export = request.GET.get("export", "0") == "1" and not raw
    show_af = _truthy(request.GET.get("af", "")) and not raw
    show_x2r = _truthy(request.GET.get("x2r", "")) and not raw

    now_gb = pd.Timestamp.now(tz="GB")
    prior_gb = now_gb - pd.Timedelta(hours=12)
    end_gb = now_gb + pd.Timedelta(days=days)

    ext_labels = []
    if show_af:
        ext_labels.append("AgileForecast")
    if show_x2r:
        ext_labels.append("X2R")

    traces, statuses = _fetch_external_forecasts(ext_labels, region, show_export, prior_gb, end_gb)
    return JsonResponse({"traces": traces, "statuses": statuses})


@require_GET
def forecast_list_json(request):
    """List the available forecast runs so URL/API callers can discover valid ``fc`` ids.

    A forecast run is region-agnostic (one Forecasts row spans every region), so this
    endpoint is not region-scoped. The returned ids are exactly what the ``/v2/`` GUI —
    and this endpoint's callers — pass back as ``fc=<id>`` query parameters to pin the
    chart to specific runs. Newest first; ``count`` (1-50, default 8) caps the number
    returned.
    """
    try:
        count = int(request.GET.get("count", 8))
    except (TypeError, ValueError):
        count = 8
    count = min(max(count, 1), 50)

    forecasts = [
        {
            "id": f.id,
            "name": f.name,
            "created_at": f.created_at.isoformat(),
            "label": pd.Timestamp(f.created_at).tz_convert("GB").strftime("%d %b %H:%M"),
        }
        for f in Forecasts.objects.order_by("-created_at")[:count]
    ]
    return JsonResponse({"forecasts": forecasts}, json_dumps_params={"indent": 2})


@require_GET
def robots_txt(request):
    """Serve robots.txt. Blocks the AI scrapers that sweep every region/param
    combination and exhaust the two gunicorn workers (GPTBot was the confirmed
    offender behind the 2026-07-20 outages); asks everything else to be gentle.
    Well-behaved crawlers honour this, cutting load before it reaches the app —
    the rate limiter remains the backstop for those that don't.
    """
    blocked = [
        "GPTBot",          # OpenAI training crawler — confirmed offender
        "OAI-SearchBot",   # OpenAI search indexer (bulk)
        "ClaudeBot",       # Anthropic crawler
        "CCBot",           # Common Crawl
        "PerplexityBot",   # Perplexity
        "Bytespider",      # ByteDance
        "Google-Extended",  # Google AI training (does not affect Search)
        "Amazonbot",
        "meta-externalagent",  # Meta AI
    ]
    lines = ["# AI scrapers that sweep every region/filter combination — blocked to"]
    lines.append("# protect the app's 2 workers; the rate limiter backs this up.")
    for bot in blocked:
        lines += [f"User-agent: {bot}", "Disallow: /", ""]
    lines += ["User-agent: *", "Crawl-delay: 10", "Disallow:", ""]
    return HttpResponse("\n".join(lines), content_type="text/plain")


@csrf_exempt
@require_POST
def kofi_webhook(request):
    """Receive Ko-fi payment notifications (configured in the Ko-fi dashboard).

    Ko-fi POSTs a form field `data` containing JSON with a shared
    `verification_token`. Rejected unless KOFI_VERIFICATION_TOKEN is set and
    matches. Idempotent on kofi_transaction_id, so retries are safe.
    """
    expected = getattr(settings, "KOFI_VERIFICATION_TOKEN", "")
    if not expected:
        return HttpResponseForbidden("Webhook not configured")
    try:
        payload = json.loads(request.POST.get("data", ""))
    except (TypeError, ValueError):
        return HttpResponse("Bad payload", status=400)
    if not compare_digest(str(payload.get("verification_token", "")), expected):
        return HttpResponseForbidden("Forbidden")

    txn = payload.get("kofi_transaction_id") or payload.get("message_id")
    if not txn:
        return HttpResponse("Missing transaction id", status=400)
    ts = parse_datetime(str(payload.get("timestamp", ""))) or timezone.now()
    if timezone.is_naive(ts):
        ts = timezone.make_aware(ts, dt_timezone.utc)
    try:
        amount = float(payload.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0
    KofiPayment.objects.get_or_create(
        kofi_transaction_id=txn,
        defaults={
            "message_id": payload.get("message_id") or "",
            "timestamp": ts,
            "payment_type": payload.get("type") or "",
            "from_name": payload.get("from_name") or "",
            "message": payload.get("message") or "",
            "amount": amount,
            "currency": payload.get("currency") or "GBP",
            "is_public": bool(payload.get("is_public", True)),
            "is_subscription_payment": bool(payload.get("is_subscription_payment", False)),
        },
    )
    return HttpResponse("OK")


def _kofi_totals():
    """Per-currency revenue aggregates + monthly breakdown from KofiPayment."""
    from django.db.models.functions import TruncMonth

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = {"count": KofiPayment.objects.count(), "currencies": [], "months": []}
    for cur in KofiPayment.objects.values_list("currency", flat=True).distinct():
        qs = KofiPayment.objects.filter(currency=cur)
        out["currencies"].append({
            "currency": cur,
            "all_time": float(qs.aggregate(s=Sum("amount"))["s"] or 0),
            "this_month": float(qs.filter(timestamp__gte=month_start).aggregate(s=Sum("amount"))["s"] or 0),
            "last_90d": float(qs.filter(timestamp__gte=now - timedelta(days=90)).aggregate(s=Sum("amount"))["s"] or 0),
        })
    monthly = (
        KofiPayment.objects
        .annotate(m=TruncMonth("timestamp"))
        .values("m", "currency")
        .annotate(total=Sum("amount"), n=Count("id"))
        .order_by("-m", "currency")
    )
    out["months"] = [
        {
            "month": r["m"].strftime("%b %Y"),
            "ym": r["m"].strftime("%Y-%m"),
            "currency": r["currency"],
            "total": float(r["total"] or 0),
            "payments": r["n"],
        }
        for r in monthly[:36]
    ]
    return out


@require_GET
def kofi_summary(request):
    """Token-authed JSON revenue summary — lets the dev costs page read
    production's Ko-fi totals (the webhook only fires at the public prod URL)."""
    forbidden = _forbidden_if_update_token_invalid(request)
    if forbidden:
        return forbidden
    return JsonResponse(_kofi_totals())


# Fly compute pricing (USD/month, approximate published rates): shared CPUs bill
# a per-cpu base with 256MB/cpu included; extra RAM ~$5/GB/mo. Estimate only.
_FLY_SHARED_CPU_BASE = 1.94
_FLY_RAM_PER_GB = 5.0


def _fly_machine_cost(guest):
    cpus = guest.get("cpus", 1)
    mem_mb = guest.get("memory_mb", 256)
    if guest.get("cpu_kind") != "shared":
        return None  # performance sizes not estimated
    included = 256 * cpus
    extra_gb = max(0, mem_mb - included) / 1024
    return round(_FLY_SHARED_CPU_BASE * cpus + _FLY_RAM_PER_GB * extra_gb, 2)


def _fly_cost_estimate():
    """Live machine inventory + estimated monthly compute cost via the Machines API."""
    import requests as _requests

    token = getattr(settings, "FLY_API_TOKEN", "")
    if not token:
        return {"ok": False, "error": "FLY_API_TOKEN not set"}
    apps = []
    total = 0.0
    since_ym = None
    for app in getattr(settings, "FLY_COST_APPS", ["prices"]):
        try:
            resp = _requests.get(
                f"https://api.machines.dev/v1/apps/{app}/machines",
                headers={"Authorization": f"Bearer {token}"},
                timeout=6,
            )
            resp.raise_for_status()
            machines = []
            for m in resp.json():
                guest = m.get("config", {}).get("guest", {})
                est = _fly_machine_cost(guest)
                if est is not None:
                    total += est
                created = (m.get("created_at") or "")[:7]  # YYYY-MM
                if created:
                    since_ym = created if since_ym is None else min(since_ym, created)
                machines.append({
                    "name": m.get("name", m.get("id", "?")),
                    "group": m.get("config", {}).get("env", {}).get("FLY_PROCESS_GROUP", ""),
                    "state": m.get("state", "?"),
                    "size": f"{guest.get('cpu_kind','?')}-{guest.get('cpus','?')}x {guest.get('memory_mb','?')}MB",
                    "est": est,
                })
            apps.append({"app": app, "machines": machines})
        except Exception as exc:
            apps.append({"app": app, "error": str(exc)[:120], "machines": []})
    priced = any(m.get("est") is not None for a in apps for m in a["machines"])
    return {
        "ok": True,
        "apps": apps,
        "total": round(total, 2) if priced else None,
        # Month the oldest current machine was created — the live estimate is
        # only valid from here; earlier months need FLY_COST_HISTORY.
        "since_ym": since_ym,
    }


class CostsView(V2NavMixin, TemplateView):
    """Dev-only costs vs revenue dashboard: estimated fly.io compute cost from
    the Machines API, and Ko-fi revenue (local table + production's totals via
    the token-authed summary endpoint). 404s on hosts without the dev marker
    so financial figures are never exposed on public production.
    """

    template_name = "costs_v2.html"
    _nav_page = "costs"

    def dispatch(self, request, *args, **kwargs):
        is_dev_host = (Path(settings.BASE_DIR) / "logs" / "uptime_monitor.log").exists()
        if not is_dev_host and not (request.user.is_authenticated and request.user.is_staff):
            from django.http import HttpResponseNotFound

            return HttpResponseNotFound("Not found")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        import requests as _requests

        context = super().get_context_data(**kwargs)
        context["fly"] = _fly_cost_estimate()
        context["kofi_local"] = _kofi_totals()
        context["kofi_configured"] = bool(getattr(settings, "KOFI_VERIFICATION_TOKEN", ""))

        # Production's totals — the Ko-fi webhook fires at the public prod URL,
        # so prod's DB is where real payments accumulate.
        prod = None
        prod_error = None
        url = getattr(settings, "KOFI_PROD_SUMMARY_URL", "")
        token = getattr(settings, "UPDATE_TOKEN", "")
        if url and token:
            try:
                resp = _requests.get(url, headers={"X-Update-Token": token}, timeout=6)
                resp.raise_for_status()
                prod = resp.json()
            except Exception as exc:
                prod_error = str(exc)[:120]
        else:
            prod_error = "KOFI_PROD_SUMMARY_URL or UPDATE_TOKEN not set"
        kofi = prod if (prod and prod.get("months")) else context["kofi_local"]

        # --- One monthly table: cost vs revenue vs net, all in GBP ---
        # Fly has no billing API, so per-month cost comes from FLY_COST_HISTORY
        # breakpoints ("YYYY-MM:usd", each carried forward until the next) —
        # derived from the machine count/size recorded in fly.toml's history.
        # The live Machines API estimate overrides from the month the current
        # machines were created.
        rate = getattr(settings, "FLY_USD_TO_GBP", 0.79)
        live_total = context["fly"].get("total")
        live_since = context["fly"].get("since_ym")
        breakpoints = []
        for part in (getattr(settings, "FLY_COST_HISTORY", "") or "").split(","):
            if ":" in part:
                b_ym, _, b_usd = part.strip().partition(":")
                try:
                    breakpoints.append((b_ym.strip(), float(b_usd)))
                except ValueError:
                    pass
        breakpoints.sort()

        actuals = dict(breakpoints)

        def _cost_usd_for(ym):
            # 1. An actual invoice amount for that month is the truth.
            if ym in actuals:
                return actuals[ym], False
            # 2. Otherwise carry the most recent invoice forward (the current
            #    month isn't invoiced yet). Preferred over the live estimate,
            #    which is compute-only and understates the real bill.
            latest = None
            for b_ym, b_usd in breakpoints:
                if b_ym <= ym:
                    latest = b_usd
            if latest is not None:
                return latest, True
            # 3. Fall back to the live compute estimate.
            if live_total is not None and live_since and ym >= live_since:
                return live_total, True
            return None, False

        # Payment fees: PayPal takes a % + a flat amount PER TRANSACTION (the
        # flat part dominates on small tips), and Ko-fi takes a partner %.
        pp_pct = getattr(settings, "PAYPAL_FEE_PCT", 2.9) / 100
        pp_fixed = getattr(settings, "PAYPAL_FEE_FIXED", 0.30)
        kofi_pct = getattr(settings, "KOFI_PARTNER_FEE_PCT", 5.0) / 100
        kofi_from = getattr(settings, "KOFI_PARTNER_FEE_FROM", "") or ""

        by_month = {}
        for m in kofi.get("months") or []:
            key = m.get("ym") or datetime.strptime(m["month"], "%b %Y").strftime("%Y-%m")
            entry = by_month.setdefault(key, {"label": m["month"], "gross": 0.0, "payments": 0})
            entry["gross"] += m["total"]
            entry["payments"] += m.get("payments", 0)

        for key, entry in by_month.items():
            fee = entry["gross"] * pp_pct + entry["payments"] * pp_fixed
            if kofi_pct and key >= kofi_from:
                fee += entry["gross"] * kofi_pct
            entry["fees"] = round(fee, 2)
            entry["revenue"] = round(entry["gross"] - fee, 2)

        rows = []
        tot_cost = tot_rev = tot_gross = tot_fees = 0.0
        if by_month or breakpoints:
            start = min(list(by_month) + [b[0] for b in breakpoints])
            cursor = timezone.now().replace(day=1)
            while cursor.strftime("%Y-%m") >= start:
                key = cursor.strftime("%Y-%m")
                entry = by_month.get(key, {"label": cursor.strftime("%b %Y"), "revenue": 0.0, "gross": 0.0, "fees": 0.0})
                revenue = round(entry["revenue"], 2)
                tot_gross += entry.get("gross", 0.0)
                tot_fees += entry.get("fees", 0.0)
                cost_usd, estimated = _cost_usd_for(key)
                cost = round(cost_usd * rate, 2) if cost_usd is not None else None
                tot_rev += revenue
                tot_cost += cost or 0
                rows.append({
                    "month": entry["label"],
                    "cost": cost,
                    "estimated": estimated,
                    "revenue": revenue,
                    "net": round(revenue - cost, 2) if cost is not None else None,
                })
                cursor = (cursor - timedelta(days=1)).replace(day=1)

        context["rows"] = rows
        context["total_cost"] = round(tot_cost, 2)
        context["total_revenue"] = round(tot_rev, 2)
        context["total_net"] = round(tot_rev - tot_cost, 2)
        context["total_gross"] = round(tot_gross, 2)
        context["total_fees"] = round(tot_fees, 2)
        context["usd_gbp_rate"] = rate
        context["fee_desc"] = (
            f"{getattr(settings, 'PAYPAL_FEE_PCT', 2.9)}% + "
            f"£{getattr(settings, 'PAYPAL_FEE_FIXED', 0.30):.2f}/payment PayPal"
            + (f", {getattr(settings, 'KOFI_PARTNER_FEE_PCT', 5.0)}% Ko-fi" if kofi_pct else "")
        )
        return context


def _load_uptime_log():
    """Parse logs/uptime_monitor.log into a sorted DataFrame [ts, code, latency, ok].

    Each line is written by bin/uptime_monitor.sh as 'ISO8601Z  http_code  time_total'.
    Returns an empty frame when the log is absent (e.g. on prod, where that cron
    monitor does not run) so callers can render a graceful 'no data' state.
    """
    path = Path(settings.BASE_DIR) / "logs" / "uptime_monitor.log"
    cols = ["ts", "code", "latency", "ok"]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    records = []
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            latency = 0.0
            if len(parts) > 2:
                try:
                    latency = float(parts[2])
                except ValueError:
                    latency = 0.0
            records.append((parts[0], parts[1], latency))

    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records, columns=["ts", "code", "latency"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["ok"] = df["code"].str.match(r"^[23]").fillna(False).astype(bool)
    return df


def _count_episodes(ok_values):
    """Number of distinct outage episodes (runs of consecutive failed checks)."""
    episodes = 0
    in_episode = False
    for ok in ok_values:
        if not ok and not in_episode:
            in_episode = True
            episodes += 1
        elif ok and in_episode:
            in_episode = False
    return episodes


def _episode_list(df, limit=15):
    """Most-recent outage episodes as display dicts (start, end/ongoing, duration, code)."""
    episodes = []
    in_episode = False
    start = None
    last_code = None
    prev_ts = None
    for ts, ok, code in zip(df["ts"], df["ok"], df["code"]):
        if not ok:
            if not in_episode:
                in_episode = True
                start = ts
            last_code = code
        elif in_episode:
            in_episode = False
            episodes.append({"start": start, "end": prev_ts, "code": last_code})
        prev_ts = ts
    if in_episode:
        episodes.append({"start": start, "end": prev_ts, "code": last_code, "ongoing": True})

    out = []
    for e in reversed(episodes[-limit:]):
        # each check is one minute; a single-check episode is ~1 min
        dur_min = int(round((e["end"] - e["start"]).total_seconds() / 60)) + 1
        out.append({
            "start": e["start"].tz_convert("GB").strftime("%d %b %H:%M"),
            "end": "ongoing" if e.get("ongoing") else e["end"].tz_convert("GB").strftime("%H:%M"),
            "duration": f"{dur_min} min",
            "code": e["code"],
            "ongoing": e.get("ongoing", False),
        })
    return out


def _rl_field(line, key):
    """Pull `key=value` out of a rate_limit.log line (value ends at whitespace)."""
    token = f"{key}="
    idx = line.find(token)
    if idx == -1:
        return ""
    return line[idx + len(token):].split()[0]


def _load_rate_limit_offenders(limit=15):
    """Aggregate recent RateLimitMiddleware events (logs/rate_limit.log) by IP."""
    path = Path(settings.BASE_DIR) / "logs" / "rate_limit.log"
    if not path.exists():
        return []
    try:
        with path.open() as fh:
            lines = fh.readlines()[-5000:]
    except OSError:
        return []

    agg = {}
    for line in lines:
        if "prices.ratelimit" not in line:
            continue
        ip = _rl_field(line, "ip")
        if not ip:
            continue
        rec = agg.setdefault(ip, {"ip": ip, "events": 0, "blocks": 0, "last": "", "path": ""})
        rec["events"] += 1
        if "BLOCK ip=" in line:
            rec["blocks"] += 1
        rec["last"] = " ".join(line.split()[:2])[:16]
        rec["path"] = _rl_field(line, "path")
    return sorted(agg.values(), key=lambda r: (r["blocks"], r["events"]), reverse=True)[:limit]


class ProdHealthView(V2NavMixin, TemplateView):
    """Ops dashboard tracking PRODUCTION (prices.fly.dev) health from the local
    uptime_monitor.log. Intended for the dev server, where bin/uptime_monitor.sh
    logs a per-minute ping; on prod the log is absent and the page says so.
    """

    template_name = "health_v2.html"
    _nav_page = "health"

    _WINDOWS = [("24 hours", 1), ("7 days", 7), ("30 days", 30)]
    _DAY_OPTIONS = [1, 7, 14, 30]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        try:
            days = int(self.request.GET.get("days", 7))
        except (TypeError, ValueError):
            days = 7
        days = min(max(days, 1), 90)
        context["days"] = days
        context["day_options"] = self._DAY_OPTIONS

        from prices import blocklist

        context["ratelimit"] = {
            "enabled": getattr(settings, "RATELIMIT_ENABLED", True),
            "enforce": getattr(settings, "RATELIMIT_ENFORCE", True),
            "per_min": getattr(settings, "RATELIMIT_PER_MIN", 60),
            "block_seconds": getattr(settings, "RATELIMIT_BLOCK_SECONDS", 600),
        }
        context["blocklist"] = {
            "enabled": getattr(settings, "BLOCKLIST_ENABLED", False),
            "enforce": getattr(settings, "BLOCKLIST_ENFORCE", True),
            "count": blocklist.status()["count"],
        }
        context["offenders"] = _load_rate_limit_offenders()
        context["response_cache_client_ttl"] = getattr(settings, "RESPONSE_CACHE_CLIENT_TTL", 300)

        df = _load_uptime_log()
        context["monitor_present"] = not df.empty
        if df.empty:
            return context

        now = pd.Timestamp.now(tz="UTC")
        last_ts = df["ts"].iloc[-1]
        last_row = df.iloc[-1]
        age_min = (now - last_ts).total_seconds() / 60.0

        # Freshness — the monitor runs every 5 minutes, so >11 min stale (two
        # missed checks) means the cron has stopped.
        context["monitor_stale"] = age_min > 11
        context["monitor_last"] = last_ts.tz_convert("GB").strftime("%d %b %H:%M")
        context["monitor_age_min"] = int(age_min)

        current_up = bool(last_row["ok"])
        context["current_up"] = current_up
        context["current_code"] = last_row["code"]
        context["current_latency_ms"] = int(round(last_row["latency"] * 1000))

        # How long the current up/down streak has held
        i = len(df) - 1
        oks = df["ok"].tolist()
        while i > 0 and oks[i - 1] == current_up:
            i -= 1
        context["status_since"] = df["ts"].iloc[i].tz_convert("GB").strftime("%d %b %H:%M")

        # Uptime across fixed windows + all history
        windows = []
        for label, wdays in self._WINDOWS + [("All", None)]:
            sub = df if wdays is None else df[df["ts"] >= now - pd.Timedelta(days=wdays)]
            total = len(sub)
            if total == 0:
                windows.append({"label": label, "total": 0, "uptime": "—", "failed": 0, "episodes": 0})
                continue
            failed = int((~sub["ok"]).sum())
            windows.append({
                "label": label,
                "total": total,
                "failed": failed,
                "uptime": f"{100.0 * (total - failed) / total:.2f}",
                "episodes": _count_episodes(sub["ok"].tolist()),
            })
        context["windows"] = windows

        context["episodes"] = _episode_list(df, limit=15)

        # Chart window + latency stats
        win = df[df["ts"] >= now - pd.Timedelta(days=days)]
        ok_win = win[win["ok"]]
        if not ok_win.empty:
            context["latency_avg_ms"] = int(round(ok_win["latency"].mean() * 1000))
            context["latency_p95_ms"] = int(round(ok_win["latency"].quantile(0.95) * 1000))
        context["graph"] = self._build_chart(win)

        return context

    def _build_chart(self, win):
        if win.empty:
            return ""
        figure = go.Figure()
        ok = win[win["ok"]]
        if not ok.empty:
            figure.add_trace(go.Scatter(
                x=ok["ts"].dt.tz_convert("GB"),
                y=(ok["latency"] * 1000).round(0),
                mode="lines",
                line=dict(color="#4a9eff", width=1),
                name="Response time",
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} ms<extra></extra>",
            ))
        fail = win[~win["ok"]]
        if not fail.empty:
            figure.add_trace(go.Scatter(
                x=fail["ts"].dt.tz_convert("GB"),
                y=[0] * len(fail),
                mode="markers",
                marker=dict(color="#dc3545", size=7, symbol="x"),
                name="Down",
                text=fail["code"],
                hovertemplate="%{x|%d %b %H:%M}<br>DOWN (HTTP %{text})<extra></extra>",
            ))
        figure.update_layout(
            template="plotly_dark",
            height=420,
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            margin=dict(l=55, r=15, t=15, b=35),
            hovermode="x",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            yaxis=dict(title="Response time (ms)", rangemode="tozero"),
        )
        return figure.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"displayModeBar": False, "responsive": True},
        )


class LimitationsView(V2NavMixin, TemplateView):
    """Static page describing service limitations — fair-use rate limits, brief
    response caching, the AI-crawler policy, and best-effort availability.
    Linked from the site footer.
    """

    template_name = "limitations.html"
    _nav_page = ""

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["rate_per_min"] = getattr(settings, "RATELIMIT_PER_MIN", 60)
        context["cache_client_ttl"] = getattr(settings, "RESPONSE_CACHE_CLIENT_TTL", 300)
        return context


class GraphV2View(V2NavMixin, TemplateView):
    """Colour-coded bar chart UI — alternative to the accordion-sidebar GraphFormView."""

    template_name = "graph_v2.html"
    _nav_page = "home"

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
        show_fc_gen = show_gen and self.request.GET.get("fg", "0") == "1"
        show_dc = self.request.GET.get("dc", "0") == "1"
        show_af = _truthy(self.request.GET.get("af", "")) and not raw
        show_x2r = _truthy(self.request.GET.get("x2r", "")) and not raw
        # Draw forecasts for their full duration (back over the confirmed-price
        # region) instead of trimming them at the last confirmed slot — mirrors
        # v1's "Allow Forecast Overlap".
        show_overlap = self.request.GET.get("overlap", "0") == "1"
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

        # Lower bound for forecast series: normally the last confirmed slot (so
        # forecasts don't overdraw confirmed prices); with overlap on, extend back
        # to the chart's left edge so each forecast shows for its full duration.
        fc_lower_gb = prior_gb if show_overlap else actual_end
        fc_lower_utc = fc_lower_gb.tz_convert("UTC")

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

        # Single shared read of the latest forecast's ForecastData rows (widest range any
        # of raw-price/gen&dc/SHAP need), replacing what used to be three separate queries
        # over the same (forecast=latest, date_time range) filter.
        fd_latest_rows = []
        if latest is not None:
            fd_latest_rows = list(
                ForecastData.objects.filter(
                    forecast=latest,
                    date_time__gte=prior_gb.tz_convert("UTC"),
                    date_time__lte=end_gb.tz_convert("UTC"),
                ).order_by("date_time").values(
                    "date_time", "day_ahead", "demand", "solar", "emb_wind",
                    "nuclear", "bm_wind", "dispatchable_capacity", "shap_top_features",
                )
            )

        if latest is not None:
            if raw:
                fd_rows = [r for r in fd_latest_rows if r["date_time"] >= fc_lower_utc]
                if fd_rows:
                    primary_s = pd.Series(
                        index=pd.to_datetime([r["date_time"] for r in fd_rows]).tz_convert("GB"),
                        data=[r["day_ahead"] for r in fd_rows],
                    )
            else:
                ad_rows = list(
                    AgileData.objects.filter(
                        forecast=latest,
                        region=region,
                        date_time__gte=fc_lower_utc,
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
        if not primary_s.empty:
            upcoming_fc = primary_s[primary_s.index >= current_slot].iloc[:48]
        else:
            upcoming_fc = pd.Series(dtype=float)

        if not upcoming_fc.empty:
            if show_export:
                # For export, expensive slots are desirable
                best_single_idx = upcoming_fc.idxmax()
                best_single_p = float(upcoming_fc[best_single_idx])
                summary["best_slot_price"] = fmt(best_single_p)
                summary["best_slot_time"] = best_single_idx.strftime("%H:%M")
                summary["best_slot_badge"] = _export_price_badge(best_single_p)
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
                summary["window_badge"] = (
                    _export_price_badge(float(rolling_avg[best_end])) if show_export
                    else _price_badge(float(rolling_avg[best_end]) if not raw else None)
                )
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

        # --- API / data-source health status (binary: ok / fail, no amber) ---
        # NESO: minimum across the three 14-day sub-sources; da_wind excluded (supplementary)
        # BMRS NDF: day-ahead only, threshold is much lower
        _NESO_THRESHOLD = 200   # rows below this → red for any primary NESO sub-source
        _BMRS_THRESHOLD = 40    # NDF is day-ahead only; 40 rows ≈ one full day

        # Read source-row counts from the most recent UpdateJob (written there to survive
        # across processes, since LocMemCache is per-process).
        recent_update_job = UpdateJob.objects.filter(
            job_type=UpdateJob.JOB_UPDATE
        ).order_by("-requested_at").first()
        job_api_status = (recent_update_job.options or {}).get("api_status", {}) if recent_update_job else {}
        source_rows = job_api_status.get("source_rows", {})
        source_details = job_api_status.get("source_details", {})

        # NESO: use per-sub-source minimum if new keys present, else fall back to old aggregate
        _has_neso_sub = "neso_wind" in source_rows and "neso_solar" in source_rows and "neso_demand" in source_rows
        if _has_neso_sub:
            _neso_subs = {
                "wind": source_rows.get("neso_wind", 0),
                "solar": source_rows.get("neso_solar", 0),
                "demand": source_rows.get("neso_demand", 0),
            }
            neso_effective = min(_neso_subs.values())
        else:
            _neso_subs = {}
            neso_effective = source_rows.get("neso", -1)

        _stored_forecast_rows = job_api_status.get("forecast_rows", -1)
        _bmrs_rows = source_rows.get("bmrs", -1)
        _om_rows = source_rows.get("openmeteo", -1)
        _rte_rows = source_rows.get("rte_nuclear", -1)
        _dc_rows = source_rows.get("neso_dc", -1)
        _om_fr_rows = source_rows.get("openmeteo_fr", -1)

        def _bin_health(rows, threshold):
            if rows < 0:
                return "unknown"
            return "ok" if rows >= threshold else "fail"

        def _neso_sub_health(key):
            """ok / warn (CSV backup used) / fail for a single NESO sub-source."""
            d = source_details.get(key, {})
            rows = d.get("rows", -1)
            if rows < 0:
                return "unknown"
            if rows == 0:
                return "fail"
            return "warn" if d.get("fallback") else "ok"

        # NESO overall: worst of its three primary sub-sources
        if _has_neso_sub:
            _sub_healths = [_neso_sub_health(k) for k in ("neso_wind", "neso_solar", "neso_demand")]
            _rank = {"fail": 0, "unknown": 1, "warn": 2, "ok": 3}
            _neso_health = min(_sub_healths, key=lambda h: _rank.get(h, 1))
        else:
            _neso_health = _bin_health(neso_effective, _NESO_THRESHOLD)

        _bmrs_health = _bin_health(_bmrs_rows, _BMRS_THRESHOLD)
        _om_health = _bin_health(_om_rows, _NESO_THRESHOLD)
        _rte_health = _bin_health(_rte_rows, 24)      # ≥24 rows = at least 12h of 30-min data
        _dc_health = _bin_health(_dc_rows, 14)   # ≥14 rows = at least 7 days broadcast
        _om_fr_health = _bin_health(_om_fr_rows, 336)  # ≥336 rows = at least 7 days of 30-min data

        # Octopus: check freshness of PriceHistory (binary: ok if ≤ 26 h old)
        latest_price = PriceHistory.objects.order_by("-date_time").first()
        _octopus_health = "unknown"
        _octopus_hours_old = None
        if latest_price is not None:
            try:
                price_ts = pd.Timestamp(latest_price.date_time).tz_convert("UTC")
                _octopus_hours_old = (pd.Timestamp.now(tz="UTC") - price_ts).total_seconds() / 3600
                _octopus_health = "ok" if _octopus_hours_old <= 26 else "fail"
            except Exception:
                pass

        # Heuristic for old-format source_rows: if the run failed with low forecast_rows and
        # sub-source keys are absent, NESO is the bottleneck (Open-Meteo is full; BMRS is day-ahead).
        if (
            recent_update_job
            and recent_update_job.status == UpdateJob.STATUS_FAILED
            and not _has_neso_sub
            and 0 <= _stored_forecast_rows < _NESO_THRESHOLD
        ):
            _neso_health = "fail"

        # Build detail text
        def _src_detail(group):
            d = source_details.get(group, {})
            label = d.get("label", group)
            rows = d.get("rows", -1)
            error = d.get("error")
            if d.get("fallback"):
                return f"{label}: CSV backup"
            if error and rows == 0:
                return f"{label}: {error}"
            if rows >= 0:
                return f"{label}: {rows} rows"
            return "no data"

        def _neso_detail():
            if _neso_health == "ok":
                return "OK"
            primary = {k: v for k, v in source_details.items()
                       if k in ("neso_wind", "neso_solar", "neso_demand")}
            if primary:
                # Show the worst sub-source (prefer fallback over fail for clarity)
                _rank = {"fail": 0, "unknown": 1, "warn": 2, "ok": 3}
                worst_key = min(primary, key=lambda k: _rank.get(_neso_sub_health(k), 1))
                return _src_detail(worst_key)
            if _stored_forecast_rows >= 0:
                return f"forecast: {_stored_forecast_rows} rows"
            return "insufficient data"

        _rank = {"fail": 0, "unknown": 1, "warn": 2, "ok": 3}
        _worst_health = lambda a, b: min([a, b], key=lambda h: _rank.get(h, 1))

        # NESO consolidated: forecast data + OPMR
        _neso_combined_health = _worst_health(_neso_health, _dc_health)
        if _neso_combined_health == "ok":
            _neso_combined_detail = "OK"
        elif _neso_health != "ok":
            _neso_combined_detail = _neso_detail()
        else:
            _neso_combined_detail = f"Dispatch cap: {_dc_rows} rows" if _dc_rows > 0 else _src_detail("neso_dc")

        # Open-Meteo consolidated: UK + France
        # Only include FR in the combined health if it has been fetched at least once.
        # A never-fetched source (rows=-1, "unknown") should not degrade the UK result.
        if _om_fr_rows >= 0:
            _om_combined_health = _worst_health(_om_health, _om_fr_health)
        else:
            _om_combined_health = _om_health
        if _om_combined_health == "ok":
            _om_combined_detail = "OK"
        elif _om_health != "ok":
            _om_combined_detail = _src_detail("openmeteo")
        elif _om_fr_rows >= 0:
            _om_combined_detail = f"FR: {_om_fr_rows} rows" if _om_fr_rows > 0 else _src_detail("openmeteo_fr")
        else:
            _om_combined_detail = "OK"

        api_sources = [
            {"name": "NESO", "health": _neso_combined_health, "detail": _neso_combined_detail},
            {"name": "BMRS", "health": _bmrs_health, "detail": "OK" if _bmrs_health == "ok" else _src_detail("bmrs")},
            {"name": "Open-Meteo", "health": _om_combined_health, "detail": _om_combined_detail},
            {
                "name": "Octopus",
                "health": _octopus_health,
                "detail": "OK" if _octopus_health == "ok" else (
                    f"stale: {int(_octopus_hours_old)}h old" if _octopus_hours_old is not None else "no data"
                ),
            },
            {
                "name": "FR nuclear",
                "health": _rte_health,
                "detail": "OK" if _rte_health == "ok" else (
                    f"{_rte_rows} rows" if _rte_rows > 0 else _src_detail("rte_nuclear")
                ),
            },
        ]

        overall_health = "ok" if all(s["health"] == "ok" for s in api_sources) else "fail"

        # Summary health for the "Data sources" badge. This reflects the whole
        # ingest pipeline, not just per-source flags: an "unknown" source (e.g.
        # after a failed run that recorded no status) is NOT "OK"; a failed last
        # run is at least a warning; and a stale/absent latest forecast (data no
        # longer being refreshed) is a fault. Without this, a failed update left
        # every source "unknown" and the badge wrongly read "all OK".
        _rank_lvl = {"ok": 0, "warn": 1, "fail": 2}
        _summary_level = max([0] + [_rank_lvl.get(s["health"], 1) for s in api_sources])
        if recent_update_job and recent_update_job.status == UpdateJob.STATUS_FAILED:
            _summary_level = max(_summary_level, 1)
        _forecast_age_h = None
        if latest is not None:
            _forecast_age_h = (
                pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest.created_at).tz_convert("UTC")
            ).total_seconds() / 3600
            if _forecast_age_h > 12:
                _summary_level = 2
        else:
            _summary_level = 2
        summary_health = {0: "ok", 1: "warn", 2: "fail"}[_summary_level]

        # Forecast horizon from latest ForecastData
        future_rows = 0
        if latest is not None:
            future_rows = ForecastData.objects.filter(
                forecast=latest, date_time__gt=now_gb.tz_convert("UTC"),
            ).count()

        # Most recent job time and status for the card header
        last_run_time = None
        last_run_status = None
        if recent_update_job:
            last_run_time = pd.Timestamp(recent_update_job.requested_at).tz_convert("GB").strftime("%d %b %H:%M")
            last_run_status = recent_update_job.status

        api_status = {
            "sources": api_sources,
            "overall_health": overall_health,
            "summary_health": summary_health,
            "forecast_age_h": int(_forecast_age_h) if _forecast_age_h is not None else None,
            "last_run_time": last_run_time,
            "last_run_status": last_run_status,
            "last_success_time": pd.Timestamp(latest.created_at).tz_convert("GB").strftime("%d %b %H:%M") if latest else None,
        }

        # --- Cheap / best windows (upcoming non-overlapping 2h periods) ---
        cheap_windows = []
        if not raw:
            today_gb = now_gb.normalize()
            tomorrow_gb = today_gb + pd.Timedelta("1D")
            # Best available prices: actual (Octopus-confirmed) where available, else forecast
            if not primary_s.empty:
                future_prices = primary_s[primary_s.index > now_gb].copy()
            else:
                future_prices = pd.Series(dtype=float)

            if not actual.empty:
                future_actual = actual[actual.index > now_gb]
                future_prices.update(future_actual)
            horizon = future_prices.iloc[:144]  # up to 3 days of 30-min slots
            n = len(horizon)
            n_slots = self._CHARGE_SLOTS  # 4 slots = 2h
            best_fn = max if show_export else min
            if n >= n_slots:
                prices_arr = horizon.values.tolist()
                times_arr = horizon.index
                avgs = [
                    sum(prices_arr[i:i + n_slots]) / n_slots
                    for i in range(n - n_slots + 1)
                ]
                used_positions: set = set()
                found = []
                for _ in range(5):
                    best_avg = None
                    best_i = None
                    for i, avg in enumerate(avgs):
                        if any(s in used_positions for s in range(i, i + n_slots)):
                            continue
                        if best_avg is None or (show_export and avg > best_avg) or (not show_export and avg < best_avg):
                            best_avg = avg
                            best_i = i
                    if best_i is None:
                        break
                    start_ts = times_arr[best_i]
                    end_ts = times_arr[best_i + n_slots - 1] + pd.Timedelta("30min")
                    day_offset = (start_ts.normalize() - today_gb).days
                    if day_offset == 0:
                        day_label = "Today"
                    elif day_offset == 1:
                        day_label = "Tomorrow"
                    else:
                        day_label = start_ts.strftime("%a %d %b")
                    found.append({
                        "start": start_ts,
                        "label": f"{day_label} {start_ts.strftime('%H:%M')}–{end_ts.strftime('%H:%M')}",
                        "avg": round(float(best_avg), 1),
                        "badge": _price_badge(float(best_avg) if not show_export else None),
                    })
                    # Block out surrounding positions to prevent overlap
                    for s in range(max(0, best_i - n_slots + 1), min(n - n_slots + 1, best_i + n_slots)):
                        used_positions.add(s)
                    for s in range(best_i, best_i + n_slots):
                        used_positions.add(s)
                cheap_windows = sorted(found, key=lambda w: w["start"])

        # --- Colour strip: prefer confirmed actual, else forecast ---
        _strip_parts = [s for s in [primary_s, actual] if not s.empty]
        if _strip_parts:
            strip_s = pd.concat(_strip_parts).sort_index()
            strip_s = strip_s[~strip_s.index.duplicated(keep="last")]
        else:
            strip_s = pd.Series(dtype=float)
        strip_colors = [color_fn(float(p) if not raw else None) for p in strip_s.values]

        # --- Build chart ---
        _STRIP_ROW = 2
        _n_rows = 2 + int(show_gen) + int(show_dc)
        _GEN_ROW  = 3 if show_gen else None
        _DC_ROW = (4 if show_gen else 3) if show_dc else None

        if _n_rows == 4:
            _row_heights     = [0.45, 0.03, 0.28, 0.24]
            _subplot_titles  = ("", "", "Generation & Demand", "Dispatch Capacity")
            chart_height     = 820
        elif show_gen:
            _row_heights     = [0.55, 0.04, 0.41]
            _subplot_titles  = ("", "", "Generation & Demand")
            chart_height     = 660
        elif show_dc:
            _row_heights     = [0.65, 0.04, 0.31]
            _subplot_titles  = ("", "", "Dispatch Capacity")
            chart_height     = 600
        else:
            _row_heights     = [0.94, 0.06]
            _subplot_titles  = ("", "")
            chart_height     = 440

        figure = make_subplots(
            rows=_n_rows,
            cols=1,
            subplot_titles=_subplot_titles,
            shared_xaxes=True,
            vertical_spacing=0.04 if _n_rows > 2 else 0,
            row_heights=_row_heights,
        )

        def add_price(trace):
            figure.add_trace(trace, row=1, col=1)

        def add_gen(trace):
            if _GEN_ROW:
                figure.add_trace(trace, row=_GEN_ROW, col=1)

        def add_dc(trace):
            if _DC_ROW:
                figure.add_trace(trace, row=_DC_ROW, col=1)

        # Uncertainty band — shaded fill drawn first so lines sit on top
        if show_band and not low_s.empty and not high_s.empty and not primary_s.empty:
            aligned_lo = low_s.reindex(primary_s.index)
            aligned_hi = high_s.reindex(primary_s.index)
            add_price(go.Scatter(
                x=list(primary_s.index),
                y=aligned_lo.values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            add_price(go.Scatter(
                x=list(primary_s.index),
                y=aligned_hi.values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(74,158,255,0.15)",
                name="p10–p90",
                hoverinfo="skip",
            ))

        # Primary forecast line
        if not primary_s.empty:
            latest_label = (
                f"Forecast ({pd.Timestamp(latest.created_at).tz_convert('GB').strftime('%d %b %H:%M')})"
                if latest
                else "Forecast"
            )
            if not low_s.empty and not high_s.empty:
                _lo = low_s.reindex(primary_s.index).ffill().bfill().fillna(0)
                _hi = high_s.reindex(primary_s.index).ffill().bfill().fillna(0)
                fc_cd = [[float(lo), float(hi)] for lo, hi in zip(_lo.values, _hi.values)]
                fc_hover = (
                    f"%{{x|%d %b %H:%M}}<br>"
                    f"<b>%{{y:.2f}} {unit}</b><br>"
                    f"<span style='opacity:.65;font-size:.85em'>"
                    f"p10: %{{customdata[0]:.2f}} · p90: %{{customdata[1]:.2f}} {unit}"
                    f"</span><extra>Forecast</extra>"
                )
            else:
                fc_cd = None
                fc_hover = f"%{{x|%d %b %H:%M}}<br><b>%{{y:.2f}} {unit}</b><extra>Forecast</extra>"
            add_price(go.Scatter(
                x=primary_s.index,
                y=primary_s.values,
                mode="lines",
                line=dict(shape="hv", color="#4a9eff", width=2.0),
                name=latest_label,
                customdata=fc_cd,
                hovertemplate=fc_hover,
            ))

        # Older selected forecasts — solid lines
        older = [f for f in forecasts_to_plot if latest is None or f.id != latest.id]
        for i, fc_obj in enumerate(older[:3]):
            if raw:
                od_rows = list(
                    ForecastData.objects.filter(
                        forecast=fc_obj,
                        date_time__gte=fc_lower_utc,
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
                        date_time__gte=fc_lower_utc,
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
            add_price(go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line=dict(shape="hv", color=self._OLDER_COLORS[i % len(self._OLDER_COLORS)], width=1.5),
                name=f"Forecast ({older_label})",
                hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.2f}} {unit}<extra>{older_label}</extra>",
            ))

        # Confirmed actuals — solid white step line, drawn LAST so it always sits
        # on top of the forecast lines and stays clearly visible (issue #91).
        if not actual.empty:
            add_price(go.Scatter(
                x=actual.index,
                y=actual.values,
                mode="lines",
                line=dict(shape="hv", color="#ffffff", width=3.0),
                name="Confirmed",
                hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>%{{y:.2f}} {unit}</b><extra>Confirmed</extra>",
            ))

        # External forecasts (AgileForecast / X2R): fetched client-side via
        # ext_forecast_json after the main chart renders, not here — these make
        # live HTTP calls with 5-15s timeouts and must never block this view.
        ext_statuses = []

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
        if (show_gen or show_dc) and latest is not None:
            fp = fd_latest_rows
            h_rows = list(
                History.objects.filter(
                    date_time__gte=prior_gb.tz_convert("UTC"),
                    date_time__lte=now_gb.tz_convert("UTC"),
                )
            ) if show_gen else []
            if show_gen:
                if fp:
                    add_gen(go.Scatter(
                        x=[r["date_time"] for r in fp],
                        y=[(r["demand"] + r["solar"] + r["emb_wind"]) / 1000 for r in fp],
                        line={"color": "cyan", "width": 2}, name="Forecast demand",
                    ))
                    add_gen(go.Scatter(
                        x=[r["date_time"] for r in fp], y=[r["nuclear"] / 1000 for r in fp],
                        fill="tozeroy", line={"color": "rgba(160,160,160,1)"},
                        fillcolor="rgba(180,180,180,0.8)", name="Nuclear",
                    ))
                    add_gen(go.Scatter(
                        x=[r["date_time"] for r in fp],
                        y=[(r["nuclear"] + r["bm_wind"]) / 1000 for r in fp],
                        fill="tonexty", line={"color": "rgba(63,127,63)"},
                        fillcolor="rgba(127,255,127,0.8)", name="Metered wind",
                    ))
                    add_gen(go.Scatter(
                        x=[r["date_time"] for r in fp],
                        y=[(r["nuclear"] + r["bm_wind"] + r["emb_wind"]) / 1000 for r in fp],
                        fill="tonexty", line={"color": "rgba(50,150,220)"},
                        fillcolor="rgba(100,200,255,0.7)", name="Embedded wind",
                    ))
                    add_gen(go.Scatter(
                        x=[r["date_time"] for r in fp],
                        y=[(r["nuclear"] + r["bm_wind"] + r["emb_wind"] + r["solar"]) / 1000 for r in fp],
                        fill="tonexty", line={"color": "lightgray", "width": 2},
                        fillcolor="rgba(255,255,127,0.8)", name="Solar",
                    ))
                if h_rows:
                    add_gen(go.Scatter(
                        x=[r.date_time for r in h_rows],
                        y=[(r.demand + r.solar + (r.total_wind - r.bm_wind)) / 1000 for r in h_rows],
                        line={"color": "#aaaa77", "width": 2}, name="Historic demand",
                    ))
                if show_fc_gen and older:
                    for i, fc_obj in enumerate(older[:3]):
                        fc_gen_rows = list(
                            ForecastData.objects.filter(
                                forecast=fc_obj,
                                date_time__gte=prior_gb.tz_convert("UTC"),
                                date_time__lte=end_gb.tz_convert("UTC"),
                            ).order_by("date_time")
                        )
                        if not fc_gen_rows:
                            continue
                        color = self._OLDER_COLORS[i % len(self._OLDER_COLORS)]
                        fc_label = pd.Timestamp(fc_obj.created_at).tz_convert("GB").strftime("%d %b %H:%M")
                        add_gen(go.Scatter(
                            x=[r.date_time for r in fc_gen_rows],
                            y=[(r.nuclear + r.bm_wind + r.emb_wind + r.solar) / 1000 for r in fc_gen_rows],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dot"),
                            name=f"Gen ({fc_label})",
                            hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.2f}} GW<extra>Gen {fc_label}</extra>",
                        ))
                        add_gen(go.Scatter(
                            x=[r.date_time for r in fc_gen_rows],
                            y=[(r.demand + r.solar + r.emb_wind) / 1000 for r in fc_gen_rows],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dot"),
                            name=f"Demand ({fc_label})",
                            hovertemplate=f"%{{x|%d %b %H:%M}}<br>%{{y:.2f}} GW<extra>Demand {fc_label}</extra>",
                        ))
                figure.update_yaxes(title_text="Power [GW]", fixedrange=True, row=_GEN_ROW, col=1)

            if show_dc and fp:
                dc_rows = [(r["date_time"], r["dispatchable_capacity"]) for r in fp if r["dispatchable_capacity"] is not None]
                if dc_rows:
                    xs, ys = zip(*dc_rows)
                    add_dc(go.Scatter(
                        x=list(xs),
                        y=list(ys),
                        mode="lines",
                        fill="tozeroy",
                        line={"color": "#fd7e14", "width": 2, "shape": "hv"},
                        fillcolor="rgba(253,126,20,0.25)",
                        name="Dispatch capacity",
                        hovertemplate="%{x|%d %b}<br><b>%{y:.0f} MW</b><extra>Dispatch capacity</extra>",
                    ))
                figure.update_yaxes(title_text="Dispatch capacity [MW]", fixedrange=True, row=_DC_ROW, col=1)

        # Colour strip at bottom of price chart
        if not strip_s.empty:
            figure.add_trace(
                go.Bar(
                    x=strip_s.index,
                    y=[1] * len(strip_s),
                    marker_color=strip_colors,
                    marker_line_width=0,
                    marker_line_color="rgba(0,0,0,0)",
                    width=int(30 * 60 * 1000),
                    offset=0,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=_STRIP_ROW,
                col=1,
            )
        figure.update_yaxes(visible=False, fixedrange=True, row=_STRIP_ROW, col=1)
        figure.update_xaxes(showgrid=False, row=_STRIP_ROW, col=1)

        # Layout
        common_layout = dict(
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
            bargap=0,
        )
        if show_gen or show_dc:
            figure.update_layout(**common_layout)
            figure.update_yaxes(
                title_text=price_display["axis_title"],
                zeroline=True,
                zerolinecolor="#888",
                zerolinewidth=2,
                rangemode="tozero",
                fixedrange=True,
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
                    rangemode="tozero",
                    fixedrange=True,
                ),
            )

        figure.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86400000], value="%H:%M<br>%a %d %b"),
                dict(dtickrange=[86400000, None], value="%d %b"),
            ]
        )

        # SHAP per-slot explanations: scale model-native £/MWh values using the selected
        # region's own linear factor so contributors are in the same units as the price chart.
        _shap_m, _shap_a = regions.get(region, regions["X"])["factors"]
        shap_data = {}
        if latest is not None:
            shap_rows = [r for r in fd_latest_rows if r["shap_top_features"] is not None]
            for row in shap_rows:
                ts_ms = int(pd.Timestamp(row["date_time"]).timestamp() * 1000)
                shap_data[str(ts_ms)] = {
                    "time": pd.Timestamp(row["date_time"]).tz_convert("GB").strftime("%d %b %H:%M"),
                    "price": round(row["day_ahead"] * _shap_m + _shap_a, 1) if row["day_ahead"] is not None else None,
                    "contributors": [
                        {"label": _FEATURE_LABELS.get(item["feature"], item["feature"]),
                         "value": round(item["value"] * _shap_m, 2)}
                        for item in (row["shap_top_features"] or [])
                    ],
                }

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
                "show_fc_gen": show_fc_gen,
                "show_dc": show_dc,
                "show_af": show_af,
                "show_x2r": show_x2r,
                "show_overlap": show_overlap,
                "summary": summary,
                "api_status": api_status,
                "cheap_windows": cheap_windows,
                "price_data_json": json.dumps(
                    [[int(ts.timestamp() * 1000), round(float(p), 4)]
                     for ts, p in strip_s.items() if pd.notna(p)]
                ) if not raw else "[]",
                "now_ms": int(now_gb.timestamp() * 1000),
                "shap_data_json": json.dumps(shap_data),
                "shap_unit": price_display["unit"],
                "day_options": self._DAY_OPTIONS,
                "forecast_list": forecast_list,
                "selected_ids": selected_ids,
                "ext_statuses": ext_statuses,
                "graph": figure.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",
                    config={"displayModeBar": False, "responsive": True},
                ),
                "classic_url": f"/{region}/",
            }
        )
        return context


class HistoryV2View(V2NavMixin, HistoryView):
    """v2-styled history / accuracy view — always uses region X (national average)."""

    template_name = "history_v2.html"
    _nav_page = "history"

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
            fixedrange=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text=price_unit,
            range=[-shared_max, shared_max],
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=2,
            fixedrange=True,
            row=2,
            col=1,
        )

        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
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
    _nav_page = "stats"

    @staticmethod
    def _extra_cache_key():
        f = (
            Forecasts.objects.filter(mean__isnull=False)
            .order_by("-created_at")
            .values_list("created_at", flat=True)
            .first()
        )
        ts = int(f.timestamp()) if f else 0
        return f"stats_v2_extra_{ts}"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["classic_url"] = "/stats"

        cache_key = self._extra_cache_key()
        v2_extra = cache.get(cache_key)
        if v2_extra is not None:
            context.update(v2_extra)
            return context

        if not PriceHistory.objects.exists():
            v2_extra = {
                "trend_chart": "",
                "diagnostic_charts": [],
                "diagnostic_sample_count": 0,
                "diagnostic_unique_slots": 0,
                "diagnostic_date_from": None,
                "diagnostic_date_to": None,
            }
            cache.set(cache_key, v2_extra, timeout=60 * 60 * 24)
            context.update(v2_extra)
            return context

        started = time.monotonic()
        diag = self._build_diagnostic_charts()
        v2_extra = {
            "trend_chart": self._build_trend_chart(),
            "diagnostic_charts": diag["charts"],
            "diagnostic_sample_count": diag["sample_count"],
            "diagnostic_unique_slots": diag["unique_slots"],
            "diagnostic_date_from": diag["date_from"],
            "diagnostic_date_to": diag["date_to"],
            "feature_experiment": self._build_feature_experiment_section(),
            "shap_explanations": self._build_shap_explanations(),
        }
        cache.set(cache_key, v2_extra, timeout=60 * 60 * 24)
        context.update(v2_extra)
        logger.info("Built stats v2 extra context duration_seconds=%.2f", time.monotonic() - started)
        return context

    @staticmethod
    def _build_trend_chart():
        import numpy as np

        qs = list(Forecasts.objects.filter(mean__isnull=False).order_by("created_at").values("created_at", "mean", "stdev"))
        if not qs:
            return ""

        factor = GLOBAL_SETTINGS["REGIONS"]["X"]["factors"][0]
        created = [pd.Timestamp(r["created_at"]) for r in qs]
        means = np.array([r["mean"] * factor for r in qs])
        stdevs = np.array([(r["stdev"] or 0) * factor for r in qs])
        upper = (means + stdevs).tolist()
        lower = np.maximum(0, means - stdevs).tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=created + created[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255,220,0,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            showlegend=True,
            name="±1 Stdev",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=created,
            y=means.tolist(),
            mode="lines+markers",
            line={"color": "#e0e0e0", "width": 2},
            marker={"size": 6},
            name="Mean RMSE",
            hovertemplate="%{x|%d %b %H:%M}<br>RMSE: %{y:.2f} p/kWh<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            height=280,
            margin={"r": 10, "t": 20, "l": 60, "b": 50},
            yaxis={"title": "RMSE [p/kWh]", "rangemode": "tozero"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def _build_diagnostic_charts():
        import numpy as np
        from scipy.stats import gaussian_kde

        _EMPTY = {"charts": [], "sample_count": 0, "unique_slots": 0, "date_from": None, "date_to": None}

        latest = PriceHistory.objects.order_by("-date_time").values_list("date_time", flat=True).first()
        if not latest:
            return _EMPTY
        end_ts = pd.Timestamp(latest)
        start_ts = end_ts - pd.Timedelta("30D")

        actuals_rows = list(PriceHistory.objects.filter(date_time__gte=start_ts).values("date_time", "agile"))
        if not actuals_rows:
            return _EMPTY
        actuals_df = pd.DataFrame(actuals_rows)
        actuals_df["date_time"] = pd.to_datetime(actuals_df["date_time"], utc=True)

        preds_rows = list(
            AgileData.objects
            .filter(region="X", date_time__gte=start_ts, date_time__lte=end_ts)
            .select_related("forecast")
            .values("date_time", "agile_pred", "forecast__created_at")
        )
        if not preds_rows:
            return _EMPTY

        pred_df = pd.DataFrame(preds_rows)
        pred_df.columns = ["date_time", "agile_pred", "forecast_created"]
        pred_df["date_time"] = pd.to_datetime(pred_df["date_time"], utc=True)
        pred_df["forecast_created"] = pd.to_datetime(pred_df["forecast_created"], utc=True)

        df = pred_df.merge(actuals_df, on="date_time", how="inner")
        if df.empty:
            return {"charts": [], "sample_count": 0, "date_from": None, "date_to": None}

        df["error"] = df["agile"] - df["agile_pred"]
        df["dt"] = (df["date_time"] - df["forecast_created"]).dt.total_seconds() / 86400
        df = df[(df["dt"] >= 0) & (df["dt"] <= 14)].copy()
        if df.empty:
            return {"charts": [], "sample_count": 0, "date_from": None, "date_to": None}

        sample_count = len(df)
        unique_slots = df["date_time"].nunique()
        date_from = df["date_time"].min().strftime("%d %b %Y")
        date_to = df["date_time"].max().strftime("%d %b %Y")

        _LAYOUT = dict(
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            margin={"r": 10, "t": 30, "l": 65, "b": 50},
            height=380,
        )

        charts = []

        # Chart 1: Actual vs Predicted Over Time
        grp = df.groupby("date_time")
        median_pred = grp["agile_pred"].median().sort_index()
        p10 = grp["agile_pred"].quantile(0.1).reindex(median_pred.index)
        p90 = grp["agile_pred"].quantile(0.9).reindex(median_pred.index)
        actual_ts = actuals_df.sort_values("date_time")

        fig1 = go.Figure()
        x_rev = list(median_pred.index[::-1])
        fig1.add_trace(go.Scatter(
            x=list(median_pred.index) + x_rev,
            y=list(p90.values) + list(p10.values[::-1]),
            fill="toself",
            fillcolor="rgba(255,100,100,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            showlegend=True,
            name="P10–P90 range",
            hoverinfo="skip",
        ))
        fig1.add_trace(go.Scatter(
            x=median_pred.index,
            y=median_pred.values,
            mode="lines",
            line={"color": "#e05050", "width": 1.5},
            name="Median prediction",
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} p/kWh<extra>Median pred</extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=actual_ts["date_time"],
            y=actual_ts["agile"].values,
            mode="lines",
            line={"color": "#ffc107", "width": 2.5},
            name="Actual",
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} p/kWh<extra>Actual</extra>",
        ))
        fig1.update_layout(**_LAYOUT, yaxis={"title": "Agile Price [p/kWh]"})
        charts.append({
            "title": "Actual vs Predicted Over Time",
            "description": (
                "Actual Agile prices (amber) versus the spread of all forecasts made in the last 30 days. "
                "Red line is the median forecast; the band shows the P10–P90 range across all forecast runs. "
                "Narrower bands near recent dates reflect shorter lead times."
            ),
            "chart": fig1.to_html(full_html=False, include_plotlyjs=False),
        })

        # Chart 2: Prediction vs Actual Scatter
        sample = df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df
        p_min = float(min(sample["agile"].min(), sample["agile_pred"].min()))
        p_max = float(max(sample["agile"].max(), sample["agile_pred"].max()))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[p_min, p_max], y=[p_min, p_max],
            mode="lines",
            line={"color": "grey", "dash": "dash", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ))
        fig2.add_trace(go.Scatter(
            x=sample["agile"],
            y=sample["agile_pred"],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.35,
                "color": sample["dt"],
                "colorscale": "Viridis",
                "colorbar": {"title": "Days<br>ahead", "thickness": 12, "len": 0.75},
            },
            hovertemplate="Actual: %{x:.1f} p/kWh<br>Predicted: %{y:.1f} p/kWh<br>Lead: %{marker.color:.1f}d<extra></extra>",
            showlegend=False,
        ))
        fig2.update_layout(
            **_LAYOUT,
            xaxis={"title": "Actual Agile Price [p/kWh]"},
            yaxis={"title": "Predicted Agile Price [p/kWh]"},
        )
        charts.append({
            "title": "Prediction vs Actual Scatter",
            "description": (
                "Each dot is one half-hour slot. Points on the diagonal indicate a perfect prediction. "
                "Colour shows forecast lead time — dots from short-lead forecasts should cluster closer to the diagonal."
            ),
            "chart": fig2.to_html(full_html=False, include_plotlyjs=False),
        })

        # Chart 3: Residuals Distribution
        errors = df["error"].dropna().values
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=errors,
            nbinsx=60,
            marker_color="#4a90d9",
            opacity=0.75,
            name="Count",
            histnorm="probability density",
        ))
        try:
            kde_fn = gaussian_kde(errors)
            x_range = np.linspace(float(errors.min()), float(errors.max()), 300)
            fig3.add_trace(go.Scatter(
                x=x_range,
                y=kde_fn(x_range),
                mode="lines",
                line={"color": "#ff7f0e", "width": 2},
                name="KDE",
            ))
        except Exception:
            pass
        fig3.add_shape(
            type="line", x0=0, x1=0, y0=0, y1=1, yref="paper",
            line={"color": "rgba(255,255,255,0.5)", "dash": "dot", "width": 1},
        )
        fig3.update_layout(
            **_LAYOUT,
            xaxis={"title": "Error: Actual − Predicted [p/kWh]"},
            yaxis={"title": "Density"},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        charts.append({
            "title": "Residuals Distribution",
            "description": (
                "Distribution of forecast errors. A peak near zero with low spread indicates accurate, unbiased predictions. "
                "Positive values mean the actual price exceeded the forecast."
            ),
            "chart": fig3.to_html(full_html=False, include_plotlyjs=False),
        })

        # Chart 4: Error by Forecast Horizon
        df["dt_day"] = df["dt"].apply(int)
        dt_days = sorted(df["dt_day"].unique())
        fig4 = go.Figure()
        for d in dt_days:
            fig4.add_trace(go.Box(
                y=df[df["dt_day"] == d]["error"].values,
                name=f"{d}d",
                boxmean="sd",
                marker_color="#4a90d9",
                line_color="#4a90d9",
                showlegend=False,
                hovertemplate=f"Lead: {d}d<br>Error: %{{y:.1f}} p/kWh<extra></extra>",
            ))
        fig4.add_shape(
            type="line", x0=-0.5, x1=len(dt_days) - 0.5, y0=0, y1=0,
            line={"color": "rgba(255,255,255,0.4)", "dash": "dot", "width": 1},
        )
        fig4.update_layout(
            **_LAYOUT,
            xaxis={"title": "Forecast lead time"},
            yaxis={"title": "Error: Actual − Predicted [p/kWh]"},
        )
        charts.append({
            "title": "Forecast Error by Horizon",
            "description": (
                "Forecast error distribution grouped by lead time. Boxes show the interquartile range; "
                "the diamond marks ±1 standard deviation. Wider boxes at longer lead times indicate less certainty further ahead."
            ),
            "chart": fig4.to_html(full_html=False, include_plotlyjs=False),
        })

        # Chart 5: Feature Importance
        fi_chart = StatsV2View._build_feature_importance_chart()
        if fi_chart:
            charts.append(fi_chart)

        # Chart 6: SHAP Feature Importance
        shap_chart = StatsV2View._build_shap_importance_chart()
        if shap_chart:
            charts.append(shap_chart)

        return {
            "charts": charts,
            "sample_count": sample_count,
            "unique_slots": unique_slots,
            "date_from": date_from,
            "date_to": date_to,
        }


    @staticmethod
    def _build_feature_importance_chart():
        """Read feature importances from the most recent UpdateJob and return a chart dict."""
        fi_job = (
            UpdateJob.objects
            .exclude(options__feature_importance=None)
            .order_by("-requested_at")
            .first()
        )
        if fi_job is None:
            return None
        fi = fi_job.options.get("feature_importance", {})
        if not fi:
            return None

        sorted_items = sorted(fi.items(), key=lambda x: x[1])
        labels = [_FEATURE_LABELS.get(k, k) for k, _ in sorted_items]
        values = [v for _, v in sorted_items]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#4a90d9",
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            height=max(300, 30 * len(labels) + 60),
            margin={"r": 10, "t": 30, "l": 170, "b": 50},
            font={"size": 12, "color": "#dee2e6"},
            xaxis={"title": "Relative importance (ensemble average)"},
            yaxis={"title": ""},
        )
        return {
            "title": "Feature Importance",
            "description": (
                "Average normalised feature importance across the three ensemble models (CatBoost, LightGBM, ExtraTrees). "
                "Higher bars indicate features that most strongly drive the predicted price. "
                "Updated each time the model retrains."
            ),
            "chart": fig.to_html(full_html=False, include_plotlyjs=False),
        }

    @staticmethod
    def _build_shap_importance_chart():
        """Read SHAP feature importance (mean |SHAP|, LightGBM) from the most recent UpdateJob."""
        shap_job = (
            UpdateJob.objects
            .exclude(options__shap_importance=None)
            .order_by("-requested_at")
            .first()
        )
        if shap_job is None:
            return None
        shap_imp = shap_job.options.get("shap_importance", {})
        if not shap_imp:
            return None

        sorted_items = sorted(shap_imp.items(), key=lambda x: x[1])
        labels = [_FEATURE_LABELS.get(k, k) for k, _ in sorted_items]
        values = [v * _AF_M for _, v in sorted_items]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#28a745",
            hovertemplate="%{y}<br>Mean |SHAP|: %{x:.2f} p/kWh<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            height=max(300, 35 * len(labels) + 60),
            margin={"r": 10, "t": 30, "l": 210, "b": 50},
            font={"size": 12, "color": "#dee2e6"},
            xaxis={"title": "Mean |SHAP value| (p/kWh Agile, region X)"},
            yaxis={"title": ""},
        )
        return {
            "title": "SHAP Feature Importance",
            "description": (
                "Mean absolute SHAP contribution per feature averaged across all three ensemble models "
                "(CatBoost, LightGBM, ExtraTrees), computed on the holdout set. Values are scaled to "
                "Agile p/kWh (region X). The per-slot breakdown on the forecast page averages "
                "CatBoost + LightGBM only — ExtraTrees is too slow for per-row use with 700 trees."
            ),
            "chart": fig.to_html(full_html=False, include_plotlyjs=False),
        }

    @staticmethod
    def _build_shap_explanations(limit=8):
        """Top SHAP feature contributions for the next few upcoming forecast slots."""
        latest_forecast = Forecasts.objects.order_by("-created_at").first()
        if latest_forecast is None:
            return []

        rows = (
            ForecastData.objects
            .filter(forecast=latest_forecast, date_time__gte=pd.Timestamp.now(tz="UTC"))
            .exclude(shap_top_features__isnull=True)
            .order_by("date_time")[:limit]
        )

        explanations = []
        for row in rows:
            contributors = [
                {
                    "label": _FEATURE_LABELS.get(item["feature"], item["feature"]),
                    "value": round(item["value"] * _AF_M, 2),
                }
                for item in (row.shap_top_features or [])
            ]
            if not contributors:
                continue
            explanations.append({
                "time": row.date_time,
                "price": round(row.day_ahead * _AF_M + _AF_A, 1) if row.day_ahead is not None else None,
                "contributors": contributors,
            })
        return explanations

    @staticmethod
    def _build_feature_experiment_section():
        """Build a feature-experiment results chart from the most recent UpdateJob."""
        exp_job = (
            UpdateJob.objects
            .exclude(options__feature_experiment=None)
            .order_by("-requested_at")
            .first()
        )
        if exp_job is None:
            return None
        exp = exp_job.options.get("feature_experiment", {})
        results = exp.get("results", {})
        winner = exp.get("feature_set", "")
        date_str = exp.get("date", "")
        if not results:
            return None

        sorted_items = sorted(results.items(), key=lambda x: x[1]["score"])
        names = [name for name, _ in sorted_items]
        scores = [data["score"] for _, data in sorted_items]
        wmae_vals = [data["wmae"] for _, data in sorted_items]
        wrmse_vals = [data["wrmse"] for _, data in sorted_items]
        colors = ["#28a745" if name == winner else "#4a90d9" for name in names]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colors,
            customdata=list(zip(wmae_vals, wrmse_vals)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score: %{x:.4f}<br>"
                "wMAE: %{customdata[0]:.4f}<br>"
                "wRMSE: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#212529",
            paper_bgcolor="#343a40",
            height=max(280, 32 * len(names) + 80),
            margin={"r": 10, "t": 20, "l": 200, "b": 50},
            xaxis={"title": "Combined score (lower is better)"},
            yaxis={"title": ""},
        )

        try:
            run_date = pd.Timestamp(date_str).strftime("%d %b %Y %H:%M UTC")
        except Exception:
            run_date = date_str

        return {
            "winner": winner,
            "run_date": run_date,
            "chart": fig.to_html(full_html=False, include_plotlyjs=False),
            "set_count": len(results),
        }


class AboutV2View(V2NavMixin, AboutView):
    template_name = "about_v2.html"
    _nav_page = "about"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["classic_url"] = "/about"
        return context


class ModelDetailV2View(V2NavMixin, TemplateView):
    template_name = "model_detail_v2.html"
    _nav_page = "about"


class ApiHowToV2View(V2NavMixin, ApiHowToView):
    template_name = "api_how_to_v2.html"
    _nav_page = "api"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["classic_url"] = "/api_how_to"
        return context
