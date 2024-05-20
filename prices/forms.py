from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.bootstrap import Accordion, AccordionGroup
from crispy_forms.layout import Submit, Field, Layout
from config.settings import GLOBAL_SETTINGS
from .models import Forecasts

# REGION_CHOICES = [(r, GLOBAL_SETTINGS["REGIONS"][r]["name"]) for r in GLOBAL_SETTINGS["REGIONS"]]
forecast_choices = [(f.id, f.name) for f in Forecasts.objects.all().order_by("-created_at")][:14]


class ForecastForm(forms.Form):
    forecasts_to_plot = forms.MultipleChoiceField(
        initial=forecast_choices[0], widget=forms.CheckboxSelectMultiple, choices=forecast_choices
    )
    days_to_plot = forms.ChoiceField(
        initial=("7", "7"), choices=[(f"{i}", f"{i}") for i in range(1, 14)], required=False
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

    def __init__(self, *args, **kwargs):
        super(ForecastForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = True
        # self.helper.form_show_labels = False
        self.fields["forecasts_to_plot"].label = False
        self.helper.layout = Layout(
            Accordion(
                AccordionGroup(
                    "Options",
                    Field("days_to_plot", small=True),
                    Field(
                        "show_generation_and_demand",
                    ),
                    Field("show_range_on_most_recent_forecast"),
                    Field(
                        "show_forecast_overlap",
                        title=self.base_fields["show_forecast_overlap"].help_text,
                        data_bs_toggle="tooltip",
                        data_bs_placement="top",
                    ),
                ),
                AccordionGroup(
                    "Forecasts",
                    Field("forecasts_to_plot", small=True, label="Test"),
                ),
            ),
            Submit("submit", "Update Chart", css_class="btn btn-light"),
        )
        print(f">>>base_fields {self.base_fields}")
