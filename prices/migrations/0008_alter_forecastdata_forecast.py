# Generated by Django 4.2.11 on 2024-04-24 18:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0007_alter_history_date_time_alter_pricehistory_date_time"),
    ]

    operations = [
        migrations.AlterField(
            model_name="forecastdata",
            name="forecast",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="data",
                to="prices.forecasts",
            ),
        ),
    ]
