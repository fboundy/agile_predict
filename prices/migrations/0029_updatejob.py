from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("prices", "0028_forecasts_mean_forecasts_stdev_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="UpdateJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("options", models.JSONField(blank=True, default=dict)),
                ("error", models.TextField(blank=True)),
                ("requested_at", models.DateTimeField(auto_now_add=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={
                "ordering": ["-requested_at"],
            },
        ),
    ]
