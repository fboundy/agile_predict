from django.contrib.admin.views.decorators import staff_member_required
from django.urls import path
from .views import (
    AboutView,
    ApiHowToView,
    ColorView,
    GlossaryView,
    GraphFormView,
    HistoryView,
    MetricsView,
    StatsView,
    run_latest_agile,
    run_update,
    stats_plot,
    update_status,
)

urlpatterns = [
    path("update", run_update, name="run_update"),
    path("update/latest_agile", run_latest_agile, name="run_latest_agile"),
    path("update/status", update_status, name="update_status"),
    path("stats/plot/<path:filename>", stats_plot, name="stats_plot"),
    path("color", ColorView.as_view(), name="color"),
    path("stats", StatsView.as_view(), name="api_how_to"),
    path("metrics", staff_member_required(MetricsView.as_view()), name="metrics"),
    path("api_how_to", ApiHowToView.as_view(), name="api_how_to"),
    path("glossary", GlossaryView.as_view(), name="glossary"),
    path("about", AboutView.as_view(), name="about"),
    path("history", HistoryView.as_view(), name="history"),
    path("history/<str:region>/", HistoryView.as_view(), name="history"),
    path("<str:region>/", GraphFormView.as_view(), name="graph"),
    path("", GraphFormView.as_view(), name="graph"),
]
