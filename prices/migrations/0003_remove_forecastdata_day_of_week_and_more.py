# Generated by Django 4.2.11 on 2024-04-24 11:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0002_remove_forecastdata_last_updated_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="forecastdata",
            name="day_of_week",
        ),
        migrations.RemoveField(
            model_name="forecastdata",
            name="day_of_year",
        ),
        migrations.RemoveField(
            model_name="forecastdata",
            name="hour_of_day",
        ),
        migrations.RemoveField(
            model_name="history",
            name="day_of_week",
        ),
        migrations.RemoveField(
            model_name="history",
            name="day_of_year",
        ),
        migrations.RemoveField(
            model_name="history",
            name="hour_of_day",
        ),
        migrations.AddField(
            model_name="forecastdata",
            name="demand",
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="history",
            name="demand",
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]