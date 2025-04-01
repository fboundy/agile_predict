from rest_framework import serializers
from prices.models import AgileData, Forecasts
import pandas as pd


# These are the serializers for the bulk view


class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgileData
        fields = ["date_time", "agile_pred", "agile_low", "agile_high", "region"]


class PriceForecastSerializer(serializers.ModelSerializer):
    prices = DataSerializer(many=True, read_only=True)

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "prices"]
        depth = 1


# These are the serializers for the filtered view
class FilteredListSerializer(serializers.ListSerializer):
    def to_representation(self, data):
        data = data.filter(region=self.context["region"])
        max_date = min([d.date_time for d in data.all()]) + pd.Timedelta(days=self.context["days"])
        data = data.filter(date_time__lte=max_date)
        return super(FilteredListSerializer, self).to_representation(data)


class FilteredDataSerializer(serializers.ModelSerializer):
    class Meta:
        list_serializer_class = FilteredListSerializer
        model = AgileData
        fields = ["date_time", "agile_pred", "agile_low", "agile_high"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Access context if needed, e.g., context['some_param']
        high_low = self.context.get("high_low")
        if not high_low:
            self.fields.pop("agile_low")
            self.fields.pop("agile_high")

        # Use the context_param as needed in the serializer


class PriceForecastRegionSerializer(serializers.ModelSerializer):
    # prices = FilteredDataSerializer(many=True, read_only=True)
    prices = serializers.SerializerMethodField()

    def get_prices(self, obj):
        context = self.context
        # Add any specific context you want to pass to FilteredDataSerializer
        return FilteredDataSerializer(obj.prices.all(), many=True, context=context).data

    def to_representation(self, data):
        x = super(PriceForecastRegionSerializer, self).to_representation(data)
        x["name"] = f"Region | {self.context['region']} {x['name']}"
        return x

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "prices"]
        depth = 1
