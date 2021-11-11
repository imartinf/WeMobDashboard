# WeMobDashboard
A very early stage tool for loading, analyzing and visualizing data from the WeMob Database, Using MySQL and Python DASH (with Folium maps)

## Usage
You can use this repo using Jupyter Notebook or running the run.py script.

1. **Using Jupyter Notebook**. Simply execute connect_to_db.ipynb in your prefered Notebook.

2. **Using Python script**. Run this:  
`pip install -r requeriments.txt`  
`python run.py [--limit LIMIT] conf`    
where `LIMIT` is the number of rows to load from the data_input table and `conf` is your JSON config file
