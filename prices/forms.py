from django import forms
from config.settings import GLOBAL_SETTINGS
from .models import Forecasts

REGION_CHOICES = [(r, GLOBAL_SETTINGS["REGIONS"][r]["name"]) for r in GLOBAL_SETTINGS["REGIONS"]]
forecast_choices = [(f.id, f.name) for f in Forecasts.objects.all().order_by("-created_at")][:14]


class RegionForm(forms.Form):
    region = forms.ChoiceField(choices=REGION_CHOICES)

    forecasts_to_plot = forms.MultipleChoiceField(
        initial=forecast_choices[0], widget=forms.CheckboxSelectMultiple, choices=forecast_choices
    )
