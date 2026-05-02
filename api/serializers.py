from rest_framework import serializers
from prices.models import AgileData, Forecasts
import pandas as pd
from config.utils import import_agile_to_export_agile


# These are the serializers for the bulk view


class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgileData
        fields = ["date_time", "agile_pred", "agile_low", "agile_high", "region"]

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if self.context.get("export"):
            index = pd.to_datetime([instance.date_time])
            for field in ["agile_pred", "agile_low", "agile_high"]:
                converted = import_agile_to_export_agile(
                    pd.Series(index=index, data=[getattr(instance, field)]),
                    region=instance.region,
                )
                data[field] = round(float(converted.iloc[0]), 4)
        return data


class PriceForecastSerializer(serializers.ModelSerializer):
    prices = DataSerializer(many=True, read_only=True)

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "prices"]
        depth = 1


# These are the serializers for the filtered view
class FilteredListSerializer(serializers.ListSerializer):
    def to_representation(self, data):
        data = data.filter(region=self.context["region"]).order_by("date_time")
        first_price = data.first()
        if first_price is None:
            return []

        max_date = first_price.date_time + pd.Timedelta(days=self.context["days"])
        data = data.filter(date_time__lte=max_date)

        if self.context.get("export"):
            data = list(data)
            index = pd.to_datetime([item.date_time for item in data])
            export_values = {}
            for field in ["agile_pred", "agile_low", "agile_high"]:
                converted = import_agile_to_export_agile(
                    pd.Series(index=index, data=[getattr(item, field) for item in data]),
                    region=self.context["region"],
                )
                for item, value in zip(data, converted):
                    export_values.setdefault(item.pk, {})[field] = round(float(value), 4)
            self.child.context["export_values"] = export_values

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

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if self.context.get("export"):
            export_values = self.context.get("export_values", {}).get(instance.pk)
            for field in ["agile_pred", "agile_low", "agile_high"]:
                if field in data:
                    if export_values is not None:
                        data[field] = export_values[field]
                    else:
                        converted = import_agile_to_export_agile(
                            pd.Series(index=pd.to_datetime([instance.date_time]), data=[getattr(instance, field)]),
                            region=self.context["region"],
                        )
                        data[field] = round(float(converted.iloc[0]), 4)
        return data


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
