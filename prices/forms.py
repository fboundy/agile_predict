from django import forms
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from crispy_forms.helper import FormHelper
from crispy_forms.bootstrap import Accordion, AccordionGroup
from crispy_forms.layout import Submit, Field, Layout
from config.settings import GLOBAL_SETTINGS
from .models import Forecasts


class ForecastForm(forms.Form):
    forecasts_to_plot = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=[],  # will be set in __init__
    )
    days_to_plot = forms.ChoiceField(
        initial=("7", "7"),
        choices=[(f"{i}", f"{i}") for i in range(1, 14)],
        required=False,
    )
    show_generation_and_demand = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Shows the generation and demand forecasts that were used to generate the most recent selected price forecast",
    )
    show_forecast_overlap = forms.BooleanField(
        initial=False,
        required=False,
        help_text="Show forecast prices which have now been superseded by the actual Agile prices",
    )
    show_range_on_most_recent_forecast = forms.BooleanField(
        initial=True,
        required=False,
        help_text="Show the 10% and 90% confidence level spread on the most recent forecast. This reflects the model uncertainty, not the weather uncertainty",
    )
    show_export_pricing = forms.BooleanField(
        label="Show export pricing (Beta)",
        initial=False,
        required=False,
        help_text="Show Agile Outgoing export prices instead of Agile import prices",
    )
    show_live_agileforecast = forms.BooleanField(
        label="Show live AgileForecast",
        initial=False,
        required=False,
    )
    show_live_x2r = forms.BooleanField(
        label="Show live X2R",
        initial=False,
        required=False,
    )

    def __init__(self, *args, **kwargs):
        local_realtime_external_forecasts = kwargs.pop(
            "local_realtime_external_forecasts",
            getattr(settings, "LOCAL_REALTIME_EXTERNAL_FORECASTS", False),
        )
        region = kwargs.pop("region", "X").upper()
        super().__init__(*args, **kwargs)

        self.helper = FormHelper()
        self.helper.form_tag = True
        self.fields["forecasts_to_plot"].label = False
        if GLOBAL_SETTINGS["REGIONS"].get(region, {}).get("raw_day_ahead"):
            self.fields.pop("show_export_pricing", None)

        # 🛠️ Move forecast_choices query here
        forecast_choices = [(f.id, f.name) for f in Forecasts.objects.all().order_by("-created_at")[:14]]
        self.fields["forecasts_to_plot"].choices = forecast_choices
        self.fields["forecasts_to_plot"].initial = forecast_choices[0] if forecast_choices else None

        option_fields = [
            Field("days_to_plot", small=True),
            Field("show_generation_and_demand"),
            Field("show_range_on_most_recent_forecast"),
            Field(
                "show_forecast_overlap",
                title=self.base_fields["show_forecast_overlap"].help_text,
                data_bs_toggle="tooltip",
                data_bs_placement="top",
            ),
        ]
        if "show_export_pricing" in self.fields:
            option_fields.insert(1, Field("show_export_pricing"))
        if local_realtime_external_forecasts:
            option_fields.extend(
                [
                    Field("show_live_agileforecast"),
                    Field("show_live_x2r"),
                ]
            )
        else:
            self.fields.pop("show_live_agileforecast", None)
            self.fields.pop("show_live_x2r", None)

        self.helper.layout = Layout(
            Accordion(
                AccordionGroup(
                    "Options",
                    *option_fields,
                ),
                AccordionGroup(
                    "Forecasts",
                    Field("forecasts_to_plot", small=True, label="Test"),
                ),
            ),
            Submit("submit", "Update Chart", css_class="btn btn-light"),
        )


class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.layout = Layout(
            Field("username"),
            Field("email"),
            Field("password1"),
            Field("password2"),
            Submit("submit", "Register", css_class="btn btn-primary w-100"),
        )
