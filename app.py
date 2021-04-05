import os
import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from random import randint

import pandas as pd
import plotly.express as px
from astropy.io import ascii
import astropy.units as u
import numpy as np
import plotly.graph_objs as go

import gen_plots

server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets, external_scripts=[
  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML',
])

data = ascii.read('data/TESS_CVZ_brightgiants_goodsample_v1.02.dat',format='csv')
df = pd.DataFrame.from_records(data, columns=data.dtype.names)

mask = (df['numax_dnu_consistent'] == 1) & (df['lum_flag_BHM'] == 1)
df = df[mask]

def get_selection(selection_data):
    ind = []
    #only one curve so we dont care about curveNumber
    for point in selection_data["points"]:
        ind.append(point["pointNumber"])
    return ind


app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1(children="Seismology and Galactic Archaeology in the TESS Southern Continuous Viewing Zone",className = 'eight columns', style={'font-family':"Arial", 'color': "#8B0000", 'fontSize': 32}),
            html.Div([
                html.A([html.Img(src="https://www.asterochronometry.eu/images/Asterochronometry_full.jpg", className = 'four columns',
                     style={'height':'13%',
                            'width':'13%',
                            'float':'right',
                            'position':'relative',
                            'margin-top':10,
                            'margin-left':10,
                            'margin-right':10,
                            'margin-bottom':10
                            },
                )], href="https://www.asterochronometry.eu/",id="astrologo"),
                dbc.Tooltip(" Visit our webside ",target="astrologo",placement="bottom",style={'padding':'5px','color':"white",'background-color':"black"}),
            ]),


        ], className = "row"),
    ]),
    dcc.Markdown('''This dashboard is intended to allow basic exploration of the data set presented in [Mackereth et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.1947M/abstract). Here, we only show a subset of the highest quality data from the full catalogue presented there, selecting stars whose asteroseismic parameters agreed between three independent pipelines, and whose luminosities derived asteroseismically and photometrically agree within $1\sigma$.
                 ''',
                 className="twelve columns"),
    html.Br(),
    dcc.Markdown('''First, we will look at the fundamental properties of the stars themselves, before we dive deeper into their orbits and positions in the Milky Way. The following panels show, from left to right, top to bottom:
                    - The colour-magnitude diagram
                    - Stellar mass vs radius
                    - the positions of the stars on the sky in ecliptic coordinates''',
                 className="twelve columns"),
    html.Br(),
    html.Div(id='my-output'),
    html.Br(),
    html.Div([
    html.Div([
        dcc.Graph(
            id='color-magnitude-scatter',figure={
                                        "layout": {
                                            "paper_bgcolor": "#FFFFFF",
                                            "plot_bgcolor": "#FFFFFF",
                                        }
                                    },
                                    #config={"scrollZoom": True, "displayModeBar": True}
        )
    ], style={'width': '39%', 'display': 'inline-block', 'float':'left'}),

    html.Div([
        dcc.Graph(id='mass-radius-scatter', figure={
                                        "layout": {
                                            "paper_bgcolor": "#FFFFFF",
                                            "plot_bgcolor": "#FFFFFF",
                                        }
                                    },)
    ], style={'display': 'inline-block', 'width': '39%', 'float':'left'}),

    html.Div([
        dcc.Graph(id='polar-scatter', figure={
                                        "layout": {
                                            "paper_bgcolor": "#192444",
                                            "plot_bgcolor": "#192444",
                                        }
                                    },)
    ], style={'display': 'inline-block', 'width': '20%', 'float':'left'}),
    ]),
    html.Div([
    html.Div([html.Label('Power Spectrum Axis Scale:'),
              dcc.RadioItems(id="scale-selector",options=[{'label':'Linear', 'value':'lin'}, {'label':'Logarithmic', 'value':'log'}], value='lin'),
              dcc.Markdown('Stellar properties:')],
              style={'display': 'inline-block', 'width': '15%', 'float':'left', 'margin':'2%'}),
    html.Div([dcc.Graph(id='psd-plot',)],
             style={'display': 'inline-block', 'width': '80%', 'float':'right'}),
    ]),
    html.Br(),
    html.Div([html.H2(children='A closer look at the properties of the star', className = 'eight columns', style={'font-family':"Arial", 'color': "#8B0000", 'fontSize': 32})]),
    html.Div([dcc.Graph(id='orbit-plot',)], style={'display': 'inline-block', 'width':'50%', 'float':'left'}),
    html.Div([dcc.Graph(id='abundance-plot',)], style={'display': 'inline-block', 'width':'50%', 'float':'left'})])



@app.callback(
    Output("color-magnitude-scatter", "figure"),
    Output("mass-radius-scatter", "figure"),
    Output("polar-scatter", "figure"),
    Output("psd-plot", "figure"),
    Output("orbit-plot", "figure"),
    Input("color-magnitude-scatter", "selectedData"),
    Input("mass-radius-scatter", "selectedData"),
    Input("polar-scatter", "selectedData"),
    Input("scale-selector", "value")
)
def callback(cmdselection,mrselection,polarselection,scaleselection):
    ctx = dash.callback_context
    prop_id = ""
    prop_type = ""
    log=False
    points = []
    types = np.array(["color-magnitude-scatter", "mass-radius-scatter", "polar-scatter"])
    selects = [cmdselection,mrselection,polarselection]
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    if prop_id in types:
        which = np.where(types == prop_id)[0][0]
        print(which)
        selected_data = selects[which]
        if selected_data and selected_data['points']:
            thispoints = get_selection(selected_data)
            points.extend(thispoints)
    if len(points) == 0:
        points = [0]
    print(points)
    if scaleselection:
        if scaleselection == 'log':
            log = True
    figs = [gen_plots.generate_cmd(df, points),
            gen_plots.generate_massradius(df, points),
            gen_plots.generate_polar(df, points),
            gen_plots.generate_psd(points,log=log),
            gen_plots.generate_orbit(points)]
    return figs

if __name__ == '__main__':
    app.run_server(debug=True)
