from django.urls import path
from .views import GraphFormView, UpdateView, LatestAgileView

urlpatterns = [
    path("update/latest", LatestAgileView.as_view(), name="latest"),
    path("update/update", UpdateView.as_view(), name="update"),
    path("<str:region>/", GraphFormView.as_view(), name="graph"),
    path("", GraphFormView.as_view(), name="graph"),
]
