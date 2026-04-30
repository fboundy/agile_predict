from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("prices", "0030_updatejob_log_file"),
    ]

    operations = [
        migrations.AddField(
            model_name="updatejob",
            name="job_type",
            field=models.CharField(
                choices=[
                    ("update", "Full update"),
                    ("latest_agile", "Latest Agile prices"),
                ],
                default="update",
                max_length=32,
            ),
        ),
    ]
