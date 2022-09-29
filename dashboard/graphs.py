from turtle import width
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from hdbscan import HDBSCAN

import folium

from dashboard.app import app

EARTH_RADIUS=6371000

df = pd.read_csv('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/intervals_1month_9trucks.csv')
# df = df[df.plate.isin(df.plate.unique()[:10])]

mapbox_token = open("/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/conf/.mapbox_token").read()

controls = [
    html.P("Duration limits:"),
    dcc.RangeSlider(id="x_limits", min=0, max=int(df["delta"].max()), value=[0,df["delta"].max()], 
            marks={0: '0', df["delta"].max(): str(df["delta"].max())},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ,
    html.P("Distance limits:"),
    dcc.RangeSlider(id="x_limits_d", min=0, max=5000000, value=[0,5000000], 
            marks={0: '0', 5000000: str(5000000)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ,
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
]

content = dbc.Tabs([
            dbc.Tab(dcc.Graph(id="graph"), label="Duration"),
            dbc.Tab(dcc.Graph(id="pie"), label="Pie Chart"),
            dbc.Tab(dcc.Graph(id="graph1"), label="Distance Graph"),
            dbc.Tab(dcc.Graph(id="graph2"), label="Distance Hist"),
            dbc.Tab(id="dbscan_tab", children=[
                html.Div(children=[html.Iframe(id="dbscan", srcDoc=None, width='100%', height='500px')]),
                dcc.Input(id="eps", type="number", placeholder="epsilon (m)", value=50),
                dcc.Input(id="min_samples", type="number", placeholder="min samples", value=10)
            ], label="DBSCAN"),
            dbc.Tab(id="hdbscan_tab", children=[
                html.Iframe(id="hdbscan", srcDoc=None, width='100%', height='500px'),
                dcc.Input(id="min_cluster_size", type="number", placeholder="epsilon (m)", value=10),
                dcc.Input(id="min_points", type="number", placeholder="min samples", value=100),
                dcc.Input(id="h_eps", type="number", placeholder="epsilon (m)", value=100),
            ], label="HDBSCAN")
        ])


layout = dbc.Container([
        html.Div([
        html.H1("STOP INTERVALS IN WEMOB TRUCKS",  className="bg-primary text-white p-2 mb-2 text-center"),
        dbc.Row([
            dbc.Col(html.Div(controls), width=3),
            dbc.Col(html.Div(content), width={'offset': 1}),
        ])
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
    Output("dbscan", "srcDoc"),
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

    if ms==0 : raise dash.exceptions.PreventUpdate


    dff = dff[dff["status"] != "missing data"]
    X = dff[["begin_lat", "begin_long"]].to_numpy()
    X = np.radians(X)
    clustering = DBSCAN(eps=e/EARTH_RADIUS, min_samples=ms, metric="haversine")
    labels = clustering.fit_predict(X)

    labels = [int(l) for l in labels]
    nb_labels = len(set(labels)) - 1

    dff["label"]=labels
    dff = dff[dff["label"] != -1]

    data = dff.copy()

    ## create color column
    lst_elements = sorted(list(data["label"].unique()))
    lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in 
              range(len(lst_elements))]
    data["color"] = data["label"].apply(lambda x: 
                lst_colors[lst_elements.index(x)])

    ## create size column (scaled)
    scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
    data["size"] = scaler.fit_transform(
               data["delta"].values.reshape(-1,1)).reshape(-1)

    ## initialize the map with the starting location
    m = folium.Map(location=[40.416775, -3.703790], zoom_start=6)

    ## add points
    data.apply(lambda row: folium.CircleMarker(
           location=[row["begin_lat"],row["begin_long"]], popup=row["delta"],
           color=row["color"], fill=True,
           radius=row["size"]).add_to(m), axis=1)


    ## add html legend
    legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+"label"+""":</b><br>"""
    for i in lst_elements:
        legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
        fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
        </i>&nbsp;"""+str(i)+"""<br>"""
    legend_html = legend_html+"""</div>"""
    leg2 = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+ str(nb_labels) + " CLUSTERS FOUND" +"""</b></div>"""
    m.get_root().html.add_child(folium.Element(leg2))

    m.save("/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/m_dbscan.html")
    return open('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/m_dbscan.html', 'r').read()



@app.callback(
    Output("hdbscan", "srcDoc"),
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

    if mpts==0: raise dash.exceptions.PreventUpdate

    dff = dff[dff["status"] != "missing data"]
    X = dff[["begin_lat", "begin_long"]].to_numpy()
    # X = np.radians(X)
    clustering = HDBSCAN(min_cluster_size=mcs, min_samples=mpts, metric="haversine")
    # clustering = HDBSCAN(min_cluster_size=mcs, min_samples=mpts, cluster_selection_epsilon=e/EARTH_RADIUS, metric="haversine")
    clustering.fit(X)

    labels = clustering.labels_
    nb_labels = labels.max() + 1

    dff["label"]=labels
    dff["sil"] = silhouette_samples(X, labels, metric='haversine')
    dff = dff[dff["label"] != -1]

    centroids = pd.DataFrame(columns=["label", "easting", "northing", "n_points", "silhouette", "ispdi"])

    data = dff.copy()

    ## create color column
    lst_elements = sorted(list(data["label"].unique()))
    lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in 
                range(len(lst_elements))]
    data["color"] = data["label"].apply(lambda x: 
                lst_colors[lst_elements.index(x)])

    ## create size column (scaled)
    scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
    data["size"] = scaler.fit_transform(
                data["delta"].values.reshape(-1,1)).reshape(-1)

    ## initialize the map with the starting location
    m = folium.Map(location=[40.416775, -3.703790], zoom_start=6)

    ## add points
    data.apply(lambda row: folium.CircleMarker(
            location=[row["begin_lat"],row["begin_long"]], popup=row["plate"],
            color=row["color"], fill=True,
            radius=row["size"]).add_to(m), axis=1)

    m.save("/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/m_hdbscan.html")
    return open('/Users/imartinf/Documents/UPM/MUIT_UPM/BECA/CODE/WeMobDashboard/src/m_hdbscan.html', 'r').read()