from rest_framework import serializers
from prices.models import PriceHistory, ForecastData, Forecasts


class PriceHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PriceHistory
        fields = ["date_time", "agile", "day_ahead"]
        depth = 1


class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastData
        fields = ["date_time", "agile_pred"]


class PriceForecastSerializer(serializers.ModelSerializer):
    data = DataSerializer(many=True, read_only=True)

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "data"]
        depth = 1
