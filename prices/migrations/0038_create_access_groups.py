from django.db import migrations


def create_access_groups(apps, schema_editor):
    group = apps.get_model("auth", "Group")
    group.objects.get_or_create(name="Users")
    group.objects.get_or_create(name="Privileged Users")


class Migration(migrations.Migration):
    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
        ("prices", "0037_forecastdata_day_ahead_extra_trees"),
    ]

    operations = [
        migrations.RunPython(create_access_groups, migrations.RunPython.noop),
    ]
