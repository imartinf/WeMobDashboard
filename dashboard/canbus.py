"""
    This module creates and sets up the dash app that visualizes the database. It is is structured as a notebook for now

    SOURCES
    * Create Dash app: https://towardsdatascience.com/hello-covid-world-python-dash-and-r-shiny-comparison-cc97afef9d82
    * Creating Maps with Dash and Folium: https://medium.com/@shachiakyaagba_41915/integrating-folium-with-dash-5338604e7c56
    * Interesting template: https://medium.com/analytics-vidhya/python-dash-data-visualization-dashboard-template-6a5bff3c2b76
    * Another example: https://medium.com/analytics-vidhya/building-a-dashboard-app-using-plotlys-dash-a-complete-guide-from-beginner-to-pro-61e890bdc423
"""

import folium
import os
import pandas as pd
import json

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


import flask
import plotly.graph_objects as go
import plotly.io as pio

import datetime

from dashboard.app import app


# We start with a sample of the dataframe for test speed
df_s = pd.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/db_1month_9trucks.csv')
# df_s = df_s.sample(frac=0.1)

# Timestamp as datetime
df_s.timestamp = pd.to_datetime(df_s.timestamp)

df_s.loc[:,"color"] = ""
df_s.loc[df_s.speed==0.0, "color"] = "orange"
df_s.loc[(df_s.speed>0.0) & (df_s.speed<=50.0), "color"].color = "blue"
df_s.loc[(df_s.speed>50.0) & (df_s.speed<=90.0), "color"] = "green"
df_s.loc[(df_s.speed>90.0), "color"] = "darkred"

dplates = []
for p in df_s.plate.unique():
    dplates.append(dict(label=p, value=p))


# ### IMPLEMENTATION ON DASH
# 
# Integrating the map on a Dashboard, both in native Python or Jupyter Notebook


def update_map(df):
    m = folium.Map(location=[40.416775, -3.703790], zoom_start=6)
    for _,row in df.iterrows():
        folium.Circle(
            radius=100,
            location=[row.latitude, row.longitude],
            popup=row.plate,
            color=row.color,
            fill=True,
            fillOpacity=1
        ).add_to(m)
    return m

# We start with the heading

heading = html.H2(children='WEMOB DB DASHBOARD', className="bg-primary text-white p-2")

# This is the sidebar that let us choose several options such as: truck, dates

controls = dbc.Form(
    [
        html.P('Choose a vehicule', style={
            'textAlign': 'center'
        }),
        dcc.Dropdown(
            id='dropdown',
            options=dplates,
            className='dbc-dark',
            value=[], # default value
            multi=True
        ),
        html.Br(),
        html.Div([
            dbc.Checklist(id='select-all',
                options=[{'label': ' Select All', 'value': 1}], value=[])
            ], id='checklist-container'),
        html.Br(),
        html.P('Choose a Time Range', style={
            'textAlign': 'center'
        }),
        # The DatePicker let us choose the days we want to be displayed. If we want to display hour, minute and
        # seconds as well we must change this into DateTimePickerRange (more complicated)
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df_s.timestamp.min().date(),
            max_date_allowed=df_s.timestamp.max().date(),
            start_date=df_s.timestamp.min().date(),
            end_date=df_s.timestamp.max().date()
        ),
        html.Div(id='selected_date', style={
            'textAlign': 'center'
        })
    ]
)
sidebar = html.Div(
    [
        html.H2('Parameters'),
        html.Hr(),
        controls
    ]
)

content = html.Div(children=[
    # If using Plotly uncomment this
    # dcc.Graph(
    #     id='example-map',
    #     figure=fig
    # )
    html.Iframe(id='map', srcDoc=open('src/m.html', 'r').read(), width='100%',
    height= '900px')
    ]
)

# App callbacks

# Main callback, updates filtered data when some option is selected
@app.callback(Output('dframe', 'data'), Input('dropdown','value'),  Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'))
def update_data(plates, start_date, end_date):
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    dat = df_s[df_s.plate.isin(list(plates))]
    dat = dat[df_s.timestamp.dt.date>=start]
    dat = dat[df_s.timestamp.dt.date<=end]
    return dat.to_json()

# 'Select all' callbacks for plates dropdown checkmark
# Source: https://community.plotly.com/t/adding-a-select-all-button-to-a-multi-select-dropdown/8849/4

@app.callback(
    Output('dropdown', 'value'),
    [Input('select-all', 'value')],
    [State('dropdown', 'options')])
def test(selected, options):
    if len(selected) > 0:
        return [i['value'] for i in options]
    raise PreventUpdate()


@app.callback(
    Output('checklist-container', 'children'),
    [Input('dropdown', 'value')],
    [State('dropdown', 'options'),
        State('select-all', 'value')])
def tester(selected, options_1, checked):

    if len(selected) < len(options_1) and len(checked) == 0:
        raise PreventUpdate()

    elif len(selected) < len(options_1) and len(checked) == 1:
        return  dcc.Checklist(id='select-all',
                    options=[{'label': ' Select All', 'value': 1}], value=[])

    elif len(selected) == len(options_1) and len(checked) == 1:
        raise PreventUpdate()

    return  dcc.Checklist(id='select-all',
                    options=[{'label': ' Select All', 'value': 1}], value=[1])

# Callback that updates the map, having the map div as output

@app.callback(Output('map','srcDoc'),Input('dframe', 'data'))
def submit(dat):
    if dat is not None:
        file = '/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/m.html'
        if os.path.exists(file): os.remove(file)
        m = update_map(pd.read_json(dat))
        m.save(file)
        return open(file, 'r').read()
    
# Callback that shows the range selected
# @app.callback(Output('selected_date','children'), [Input('range_slider','value')])
# def show_selected_date(value):
#    return 'Selected: {}'.format(datetime.datetime.strptime(value,"%Y%m%d%H%M%S"))


layout = html.Div(
    [
        heading,
        dbc.Row([
            dbc.Col(html.Div(sidebar), width=3),
            dbc.Col(html.Div(content), width={'offset': 1}),
            dcc.Store(id='dframe')
        ])
    ]
)