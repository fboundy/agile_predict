from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("prices", "0034_externalforecast"),
    ]

    operations = [
        migrations.CreateModel(
            name="RequestMetric",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField()),
                ("hour", models.PositiveSmallIntegerField()),
                (
                    "surface",
                    models.CharField(
                        choices=[
                            ("web", "Web"),
                            ("api", "API"),
                            ("update", "Update"),
                            ("admin", "Admin"),
                            ("static", "Static"),
                        ],
                        max_length=16,
                    ),
                ),
                ("path", models.CharField(max_length=255)),
                ("method", models.CharField(max_length=8)),
                ("status_code", models.PositiveSmallIntegerField()),
                ("request_count", models.PositiveIntegerField(default=0)),
            ],
            options={
                "ordering": ["-date", "-hour", "surface", "path"],
                "unique_together": {("date", "hour", "surface", "path", "method", "status_code")},
            },
        ),
        migrations.CreateModel(
            name="RequestClientSeen",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField()),
                (
                    "surface",
                    models.CharField(
                        choices=[
                            ("web", "Web"),
                            ("api", "API"),
                            ("update", "Update"),
                            ("admin", "Admin"),
                            ("static", "Static"),
                        ],
                        max_length=16,
                    ),
                ),
                ("client_hash", models.CharField(max_length=64)),
            ],
            options={
                "ordering": ["-date", "surface"],
                "unique_together": {("date", "surface", "client_hash")},
            },
        ),
        migrations.AddIndex(
            model_name="requestmetric",
            index=models.Index(fields=["date", "surface"], name="prices_req_date_3fdbcd_idx"),
        ),
        migrations.AddIndex(
            model_name="requestmetric",
            index=models.Index(fields=["date", "hour"], name="prices_req_date_f84444_idx"),
        ),
        migrations.AddIndex(
            model_name="requestclientseen",
            index=models.Index(fields=["date", "surface"], name="prices_req_date_0d40af_idx"),
        ),
    ]
