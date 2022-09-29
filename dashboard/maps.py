import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
from holoviews.operation.datashader import datashade
import plotly.express as px
from plotly.colors import sequential
import numpy as np
import pandas as pd
import hvplot.pandas
from holoviews.operation.datashader import datashade

from dashboard.app import app


df = pd.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/intervals_1month_9trucks.csv')
# df = df[df.plate.isin(df.plate.unique()[:10])]
df["easting"], df["northing"] = hv.Tiles.lon_lat_to_easting_northing(
df["begin_long"], df["begin_lat"]
)
# Mapping status strings to integers for columns
d_colors = dict([(y,x+1) for x,y in enumerate(sorted(set(df["status"])))])

# pd.options.plotting.backend = 'holoviews'
hv.extension('plotly')

mapbox_token = open("/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/conf/.mapbox_token").read()

layout = dbc.Container([
    html.H1("HEATMAP OF STOP INTERVALS IN WEMOB TRUCKS",  className="bg-primary text-white p-2 mb-2 text-center"),
    dbc.Row(id="contents", children=[
        dcc.Graph(id="graph")
    ]),
    dbc.Row([
        html.P("X-Axis:"),
        dcc.RangeSlider(id="x_limits", min=0, max=df["delta"].max(), value=[0,df["delta"].max()], 
                marks={0: '0', df["delta"].max(): str(df["delta"].max())},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
        html.P("Status:"),
        dcc.Dropdown(
                    id='status',
                    options=[dict(label=s, value=s) for s in df["status"].unique()],
                    value=df["status"].unique(), # default value
                    multi=True
                ),
        html.P("Plate:"),
        dcc.Dropdown(
                    id='plate',
                    options=[dict(label=p, value=p) for p in df["plate"].unique()],
                    value=df["plate"].unique(), # default value
                    multi=True
                ),
        html.P("Driver:"),
        dcc.Dropdown(
                    id='driver',
                    options=[dict(label=d, value=d) for d in df["begin_driver"].unique()],
                    value=df["begin_driver"].unique(), # default value
                    multi=True
                )
    ]),
], fluid=True,)

@app.callback(
    Output("contents", "children"),
    # Output("graph", "figure"), 
    [Input("x_limits", "value"),
    Input("status", "value"),
    Input("plate", "value"),
    Input("driver", "value")])
def update_contents(xaxis_limits,status,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.status.isin(list(status))]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]
    dataset = hv.Dataset(dff)
    points = hv.Points(
        dataset, kdims=["easting", "northing"], vdims=["delta"]
    ).opts(hv.opts.Points(
        color=[d_colors[v] for v in dff["status"]],
        size=hv.dim("delta").log10()*5,
        cmap="magma"
    ))
    # points = datashade(points)
    tiles = hv.Tiles().opts(mapboxstyle="light", accesstoken=mapbox_token)
    selection_linker = hv.selection.link_selections.instance()
    overlay = selection_linker(tiles * points)
    overlay.opts(
        title="Mapbox Points with %d points" % len(dataset)
    )
    hist = selection_linker(hv.operation.histogram(dataset, dimension="delta", normed=False))
    #  color="status", marginal="box"
    return to_dash(app, [overlay + hist], reset_button=True, button_class=dbc.Button).children
    # hv.extension('bokeh')
    # overlay = dff.hvplot.points(
    #     x='begin_long', 
    #     y='begin_lat',
    #     xaxis=None,
    #     yaxis=None,
    #     hover_cols=['status', 'delta'], 
    #     c='status', 
    #     title='Stop intervals', 
    #     geo=True,
    #     tiles='CartoLight'
    # )
    # print(overlay)

    # return overlay