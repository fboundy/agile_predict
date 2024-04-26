from django import forms
from config.settings import GLOBAL_SETTINGS

REGION_CHOICES = [(r, GLOBAL_SETTINGS["REGIONS"][r]["name"]) for r in GLOBAL_SETTINGS["REGIONS"]]


class RegionForm(forms.Form):
    # region = forms.CharField(label="Region", max_length=100)
    region = forms.ChoiceField(choices=REGION_CHOICES)
