# Generated by Django 4.2.11 on 2024-04-26 09:32

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0017_forecastdata_day_ahead"),
    ]

    operations = [
        migrations.AddField(
            model_name="agiledata",
            name="date_time",
            field=models.DateTimeField(default=datetime.datetime(2024, 1, 1, 0, 0)),
            preserve_default=False,
        ),
    ]
