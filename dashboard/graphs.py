import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from hdbscan import HDBSCAN

import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
from holoviews.operation.datashader import datashade

from dashboard.app import app

EARTH_RADIUS=63710000

df = pd.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/intervals_07-01-22.csv')
# df = df[df.plate.isin(df.plate.unique()[:10])]
df["easting"], df["northing"] = hv.Tiles.lon_lat_to_easting_northing(
df["begin_long"], df["begin_lat"]
)

# ESTO ES UN POCO CHAPUZA. ME VOY A CARGAR LAS PARADAS "TRUCK OFF" DE MENOS DE 5 MINUTOS
df = df[(df["status"] != "truck off") | (df["status"] == "truck off") & (df["delta"] >= 5)]

mapbox_token = open("/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/conf/.mapbox_token").read()

layout = dbc.Container([
        html.Div([
        html.H1("STOP INTERVALS IN WEMOB TRUCKS",  className="bg-primary text-white p-2 mb-2 text-center"),
        dbc.Tabs([
            dbc.Tab(dcc.Graph(id="graph"), label="Duration"),
            dbc.Tab(dcc.Graph(id="pie"), label="Pie Chart"),
            dbc.Tab(dcc.Graph(id="graph1"), label="Distance Graph"),
            dbc.Tab(dcc.Graph(id="graph2"), label="Distance Hist"),
            dbc.Tab(id="dbscan_tab", children=[
                html.Div(id="dbscan"),
                dcc.Input(id="eps", type="number", placeholder="epsilon (m)", value=50),
                dcc.Input(id="min_samples", type="number", placeholder="min samples", value=10)
            ], label="DBSCAN"),
            dbc.Tab(id="hdbscan_tab", children=[
                html.Div(id="hdbscan"),
                dcc.Input(id="min_cluster_size", type="number", placeholder="epsilon (m)", value=10),
                dcc.Input(id="min_points", type="number", placeholder="min samples", value=100),
                dcc.Input(id="h_eps", type="number", placeholder="epsilon (m)", value=100),
            ], label="HDBSCAN")
        ]),
        dbc.Row([
            dbc.Col([
                html.P("Duration limits:"),
                dcc.RangeSlider(id="x_limits", min=0, max=int(df["delta"].max()), value=[0,df["delta"].max()], 
                    marks={0: '0', df["delta"].max(): str(df["delta"].max())},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=6),
            dbc.Col([
                html.P("Distance limits:"),
                dcc.RangeSlider(id="x_limits_d", min=0, max=5000000, value=[0,5000000], 
                    marks={0: '0', 5000000: str(5000000)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=6)
        ]),
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
    ])
])
@app.callback(
    Output("graph", "figure"), 
    [Input("x_limits", "value"),
    Input("plate", "value"),
    Input("driver", "value")])
def update_duration_graph(xaxis_limits,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]
    fig = px.histogram(dff, x="delta", color="status", marginal="box")
    return fig

@app.callback(
    Output("graph1", "figure"), 
    [Input("x_limits", "value"),
    Input("x_limits_d", "value"),
    Input("plate", "value"),
    Input("driver", "value")])
def update_distance_graph(xaxis_limits,distance_limits,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]

    X = dff[["begin_lat", "begin_long"]].to_numpy()
    # print(X)
    neigh = NearestNeighbors(n_neighbors=2, metric='haversine')
    nbrs = neigh.fit(X)

    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    distances = distances[distances>=distance_limits[0]/EARTH_RADIUS]
    distances = distances[distances<=distance_limits[1]/EARTH_RADIUS]

    fig = px.line(y=distances*EARTH_RADIUS, title='Distance to nearest neighbor')

    fig.update_xaxes(title_text='Sample index')
    fig.update_yaxes(title_text='Distance')

    return fig

@app.callback(
    Output("graph2", "figure"), 
    [Input("x_limits", "value"),
    Input("x_limits_d", "value"),
    Input("plate", "value"),
    Input("driver", "value")])
def update_distance_hist(xaxis_limits,distance_limits,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]

    X = dff[["begin_lat", "begin_long"]].to_numpy()
    # print(X)
    neigh = NearestNeighbors(n_neighbors=2, metric='haversine')
    nbrs = neigh.fit(X)

    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    distances = distances[distances>distance_limits[0]/EARTH_RADIUS]
    distances = distances[distances<distance_limits[1]/EARTH_RADIUS]

    fig = px.histogram(distances*EARTH_RADIUS, title='Distance to nearest neighbor', marginal="box")

    fig.update_xaxes(title_text='Distance')
    fig.update_yaxes(title_text='Count')

    return fig

@app.callback(
    Output("pie", "figure"), 
    [Input("x_limits", "value"),
    Input("plate", "value"),
    Input("driver", "value")])
def update_pie_chart(xaxis_limits,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]

    fig = px.pie(dff, values="delta", names="status", hover_data=['delta'])

    return fig

@app.callback(
    Output("dbscan", "children"),
    [Input("eps", "value"),
    Input("min_samples", "value"),
    Input("x_limits", "value"),
    Input("plate", "value"),
    Input("driver", "value")
    ]
)


def update_dbscan_map(e, ms,xaxis_limits,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]


    dff = dff[dff["status"] != "missing data"]
    X = dff[["begin_lat", "begin_long"]].to_numpy()
    # X = np.radians(X)
    clustering = DBSCAN(eps=e/EARTH_RADIUS, min_samples=ms, metric="haversine")
    labels = clustering.fit_predict(X)

    labels = [int(l) for l in labels]
    nb_labels = len(set(labels)) - 1

    dff["label"]=labels
    dff = dff[dff["label"] != -1]

    dataset = hv.Dataset(dff)
    points = hv.Points(
        dataset, kdims=["easting", "northing"], vdims=["delta", "label"]
    ).opts(hv.opts.Points(
        color=hv.dim("label"),
        cmap="magma"
        # size=hv.dim("delta").log10()*5
    ))
    tiles = hv.Tiles().opts(mapboxstyle="light", accesstoken=mapbox_token)

    overlay = (tiles * points)
    overlay.opts(
        title="DBSCAN CLUSTERING WITH EPS = %d AND MIN_SAMPLES = %d (%d POINTS) | %d CLUSTERS FOUND" % (e, ms, len(dataset), nb_labels)
    ) 

    hist = hv.Histogram(np.histogram(labels, bins=np.arange(0,102,1)))

    table = hv.Table(dff.sort_values(by="label")[["label", "plate", "begin_lat", "begin_long", "start"]])

    return to_dash(app, [overlay + table], reset_button=True, button_class=dbc.Button).children



@app.callback(
    Output("hdbscan", "children"),
    [Input("min_cluster_size", "value"),
    Input("min_points", "value"),
    Input("h_eps", "value"),
    Input("x_limits", "value"),
    Input("plate", "value"),
    Input("driver", "value")
    ]
)

def update_hdbscan_map(mcs, mpts, e, xaxis_limits,plates,drivers):
    dff = df[df['delta'] <= xaxis_limits[1]]
    dff = dff[dff['delta'] >= xaxis_limits[0]]
    dff = dff[dff.plate.isin(list(plates))]
    dff = dff[dff.begin_driver.isin(list(drivers))]

    m_pts = None if mpts==0 else mpts 

    dff = dff[dff["status"] != "missing data"]
    X = dff[["begin_lat", "begin_long"]].to_numpy()
    # X = np.radians(X)
    clustering = HDBSCAN(min_cluster_size=mcs, min_samples=m_pts, cluster_selection_epsilon=e/EARTH_RADIUS, metric="haversine")
    clustering.fit(X)

    labels = clustering.labels_
    nb_labels = labels.max() + 1

    dff["label"]=labels
    dff["sil"] = silhouette_samples(X, labels, metric='haversine')
    dff = dff[dff["label"] != -1]

    centroids = pd.DataFrame(columns=["label", "easting", "northing", "n_points", "silhouette", "ispdi"])

    for label, sample in dff.groupby("label"):
        east = sample.easting.mean()
        north = sample.northing.mean()
        npts = len(sample)
        sil = sample.sil.mean()
        centroids = centroids.append({"label": label, "easting": east, "northing": north, "n_points": npts, "silhouette":sil, "ispdi": False}, ignore_index=True)

    dataset = hv.Dataset(dff)
    points = hv.Points(
        dataset, kdims=["easting", "northing"], vdims=["delta", "label"]
    )

    cents = hv.Points(
        centroids, kdims=["easting", "northing"]
    ).opts(hv.opts.Points(
        color="red"
    ))

    points.opts(
        hv.opts.Points(
        color=hv.dim("label"),
        alpha=0.25,
        cmap="magma")
    )

    tiles = hv.Tiles().opts(mapboxstyle="light", accesstoken=mapbox_token)

    overlay = (tiles * points * cents)
    overlay.opts(
        title="HDBSCAN CLUSTERING WITH MIN_CLUSTER_SIZE = %d AND MIN_SAMPLES = %d (%d POINTS) | %d CLUSTERS FOUND" % (mcs, mpts, len(dataset), nb_labels)
    )

    hist = hv.Histogram(np.histogram(labels, bins=np.arange(0,102,1)))

    table = hv.Table(centroids)

    return to_dash(app, [overlay + table], reset_button=True, button_class=dbc.Button).children