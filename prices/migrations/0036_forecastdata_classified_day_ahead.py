from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0035_requestmetric_requestclientseen"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastdata",
            name="day_ahead_classified",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="forecastdata",
            name="plunge_probability",
            field=models.FloatField(blank=True, null=True),
        ),
    ]
