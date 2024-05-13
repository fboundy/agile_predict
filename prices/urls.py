from django.urls import path
from .views import GraphFormView, ApiHowToView, GlossaryView, AboutView

urlpatterns = [
    path("api_how_to", ApiHowToView.as_view(), name="api_how_to"),
    path("glossary", GlossaryView.as_view(), name="glossary"),
    path("about", AboutView.as_view(), name="about"),
    path("<str:region>/", GraphFormView.as_view(), name="graph"),
    path("", GraphFormView.as_view(), name="graph"),
]
