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
        fields = ["date_time", "agile_pred", "region"]


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
        fields = ["date_time", "agile_pred"]


class PriceForecastRegionSerializer(serializers.ModelSerializer):
    prices = FilteredDataSerializer(many=True, read_only=True)

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "prices"]
        depth = 1


# These are the serilizers for the Home Assistant view
class HA_FilteredListSerializer(serializers.ListSerializer):

    def to_representation(self, data):
        data = data.filter(region=self.context["region"])
        list_of_dicts = super(HA_FilteredListSerializer, self).to_representation(data)
        y = {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0].keys()}
        print(y)
        return y


class HA_FilteredDataSerializer(serializers.ModelSerializer):
    class Meta:
        list_serializer_class = HA_FilteredListSerializer
        model = AgileData
        fields = ["date_time", "agile_pred"]


class HA_PriceForecastRegionSerializer(serializers.ModelSerializer):
    prices = HA_FilteredDataSerializer(many=True, read_only=True)

    class Meta:
        model = Forecasts
        fields = ["name", "created_at", "prices"]
        depth = 1
