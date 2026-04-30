from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("prices", "0031_updatejob_job_type"),
    ]

    operations = [
        migrations.CreateModel(
            name="PlotImage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("filename", models.CharField(max_length=255, unique=True)),
                ("content_type", models.CharField(default="image/png", max_length=64)),
                ("content", models.BinaryField()),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["filename"],
            },
        ),
    ]
