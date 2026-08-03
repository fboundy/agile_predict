from django.contrib import admin

# Register your models here.
from .models import Forecasts, ForecastData, History, KofiPayment, PriceHistory

admin.site.register(Forecasts)
admin.site.register(ForecastData)
admin.site.register(History)
admin.site.register(PriceHistory)


@admin.register(KofiPayment)
class KofiPaymentAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "from_name", "amount", "currency", "payment_type", "is_subscription_payment")
    list_filter = ("currency", "payment_type")
    search_fields = ("from_name", "kofi_transaction_id")
