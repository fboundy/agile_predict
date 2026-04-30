from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("prices", "0029_updatejob"),
    ]

    operations = [
        migrations.AddField(
            model_name="updatejob",
            name="log_file",
            field=models.CharField(blank=True, max_length=255),
        ),
    ]
