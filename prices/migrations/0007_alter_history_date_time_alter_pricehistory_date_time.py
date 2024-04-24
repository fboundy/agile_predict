# Generated by Django 4.2.11 on 2024-04-24 14:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0006_pricehistory_date_time"),
    ]

    operations = [
        migrations.AlterField(
            model_name="history",
            name="date_time",
            field=models.DateTimeField(unique=True),
        ),
        migrations.AlterField(
            model_name="pricehistory",
            name="date_time",
            field=models.DateTimeField(unique=True),
        ),
    ]