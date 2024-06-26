# Generated by Django 4.2.11 on 2024-04-24 18:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0009_remove_history_demand_source"),
    ]

    operations = [
        migrations.AlterField(
            model_name="forecastdata",
            name="agile",
            field=models.DecimalField(decimal_places=2, max_digits=6),
        ),
        migrations.AlterField(
            model_name="forecastdata",
            name="day_ahead",
            field=models.DecimalField(decimal_places=1, max_digits=6),
        ),
        migrations.AlterField(
            model_name="pricehistory",
            name="agile",
            field=models.DecimalField(decimal_places=2, max_digits=6),
        ),
        migrations.AlterField(
            model_name="pricehistory",
            name="day_ahead",
            field=models.DecimalField(decimal_places=1, max_digits=6),
        ),
    ]
