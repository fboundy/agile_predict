from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0036_forecastdata_classified_day_ahead"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastdata",
            name="day_ahead_extra_trees",
            field=models.FloatField(blank=True, null=True),
        ),
    ]
