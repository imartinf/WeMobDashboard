### Import Packages ###
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
### Import Dash Instance and Pages ###
from dashboard.app import app
from dashboard import graphs, maps, canbus
### Page container ###
page_container = html.Div(
    children=[
        # represents the URL bar, doesn't render anything
        dcc.Location(
            id='url',
            refresh=False,
        ),
        # content will be rendered in this element
        html.Div(id='page-content')
    ]
)
### Index Page Layout ###
index_layout = html.Div(
    children=[
        dcc.Link(
            children='CANBUS DATA',
            href='/canbus',
        ),
        html.Br(),
        dcc.Link(
            children='INTERVALS GRAPH AND STATISTICS',
            href='/graphs',
        ),
        html.Br(),
        dcc.Link(
            children='INTERVALS HEATMAP',
            href='/maps',
        ),
    ]
)
### Set app layout to page container ###
app.layout = page_container
### Assemble all layouts ###
app.validation_layout = html.Div(
    children = [
        page_container,
        index_layout,
        canbus.layout,
        graphs.layout,
        maps.layout,
    ]
)
### Update Page Container ###
@app.callback(
    Output(
        component_id='page-content',
        component_property='children',
        ),
    [Input(
        component_id='url',
        component_property='pathname',
        )]
)
def display_page(pathname):
    if pathname == '/':
        return index_layout
    elif pathname == '/graphs':
        return graphs.layout
    elif pathname == '/maps':
        return maps.layout
    elif pathname == '/canbus':
        return canbus.layout
    else:
        return '404'