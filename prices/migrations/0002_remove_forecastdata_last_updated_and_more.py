# Generated by Django 4.2.11 on 2024-04-24 11:29

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="forecastdata",
            name="last_updated",
        ),
        migrations.RemoveField(
            model_name="history",
            name="last_updated",
        ),
    ]