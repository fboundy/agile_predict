/home/boundys/.fly/bin/fly ssh console -C "python manage.py update --debug" -a prices 
/home/boundys/.fly/bin/fly ssh console -C "python manage.py collectstatic --noinput --clear" -a prices 
# /home/boundys/.fly/bin/fly app restart prices

