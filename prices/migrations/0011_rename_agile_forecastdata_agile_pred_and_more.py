# Generated by Django 4.2.11 on 2024-04-24 18:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0010_alter_forecastdata_agile_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="forecastdata",
            old_name="agile",
            new_name="agile_pred",
        ),
        migrations.RemoveField(
            model_name="forecastdata",
            name="day_ahead",
        ),
        migrations.AddField(
            model_name="forecastdata",
            name="agile_actual",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=6),
            preserve_default=False,
        ),
    ]
