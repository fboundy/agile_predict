# Generated by Django 4.2.11 on 2024-05-10 12:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0023_forecastdata_emb_wind"),
    ]

    operations = [
        migrations.AddField(
            model_name="history",
            name="total_wind",
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]