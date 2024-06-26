from rest_framework import serializers
from prices.models import PriceHistory, AgileData, Forecasts


# class PriceHistorySerializer(serializers.ModelSerializer):
#     class Meta:
#         model = PriceHistory
#         fields = [
#             "date_time",
#             "agile",
#             "day_ahead",
#         ]
#         depth = 1

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
        return super(FilteredListSerializer, self).to_representation(data)


class FilteredDataSerializer(serializers.ModelSerializer):
    class Meta:
        list_serializer_class = FilteredListSerializer
        model = AgileData
        fields = ["date_time", "agile_pred", "agile_low", "agile_high"]


class PriceForecastRegionSerializer(serializers.ModelSerializer):
    prices = FilteredDataSerializer(many=True, read_only=True)

    def to_representation(self, data):
        x = super(PriceForecastRegionSerializer, self).to_representation(data)
        x["name"] = f"Region | {self.context['region']} {x['name']}"
        return x

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "prices"]
        depth = 1
