#%%
import requests
from urllib import parse
import pandas as pd
start = pd.Timestamp("2024-05-10")
#%%

sql_query =  '''SELECT COUNT(*) OVER () AS _count, * FROM "f93d1835-75bc-43e5-84ad-12472b180a98" WHERE "DATETIME" >= '2024-05-10' ORDER BY "_id" ASC LIMIT 100'''
params = {'sql': sql_query}

try:
    resposne = requests.get('https://api.nationalgrideso.com/api/3/action/datastore_search_sql', params = parse.urlencode(params))
    data = resposne.json()["result"]
    print(data) # Printing data
except requests.exceptions.RequestException as e:
    print(e.response.text)

#%%

ds={
    "url": "https://api.nationalgrideso.com/api/3/action/datastore_search_sql",
    "params": parse.urlencode(
        {
            "sql": 'SELECT COUNT(*) OVER () AS _count, * FROM "f93d1835-75bc-43e5-84ad-12472b180a98" WHERE "DATETIME" >= ' +f"'{pd.Timestamp(start).strftime("%Y-%m-%d")}'"  + 'ORDER BY "_id" ASC LIMIT 200'
        }
    ),
    "record_path": ["result", "records"],
    "date_col": "DATETIME",
    "cols": ["SOLAR", "WIND"],
    "rename": ["solar", "total_wind"],
}

r=requests.get(ds['url'], params=ds['params'])
# %%
