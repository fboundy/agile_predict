# Agile Predict v1.2.0

This model forecasts Octopus Agile electricity prices up to 14 days in advance using a Machine Learning model trained
on data from the Balancing Mechanism Reporting System (<a href="https://bmrs.elexon.co.uk/">BRMS</a>), National Grid 
Electricity Supply Operator (<a href="https://www.nationalgrideso.com/data-portal">NG ESO</a>) and weather data from 
<a href="https://open-meteo.com"> open-meteo.com</a>.<p>

Because the app is hosted on a <a href= "http://fly.io">fly.io</a> hobby account the database is only updated when the page
is accessed and if there has been an hour since the last update. This may sometimes cause slight delays in loading. <p>

---

## Developing for this project

This project is made using Python and Django. Here is some instructions to get you started if you want to develop for the project.

### Create a virtual environment

As with all python projects, it is recommended to create a virtual environment. For example, in this project, create a virtual environment using python's built in virtual environment tool `venv` to create an virtual environment in a folder `.venv`:

```
cd agile_predict
python3 -m venv .venv
```

Then, each time you are developing, activate the virtual environment according to the OS you are using.

Windows:

```
./.venv/Scripts/activate
```

### Installing dependencies

Requirements are listed in `requirements.txt`. You may install these however you like. The usual way is via python pip:

```
pip install -r requirements.txt
```

### Running the project

Run the project via the Django manage.py script. It's as simple as:

```
python manage.py runserver
```

Have fun!
