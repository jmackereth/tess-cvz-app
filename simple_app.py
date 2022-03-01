import os
import dash
import flask
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from random import randint

import pandas as pd
import plotly.express as px
from astropy.io import ascii
import astropy.units as u
import numpy as np
import plotly.graph_objs as go

import json

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
            html.H1(children="First look: Asteroseismology in the TESS Southern Continuous Viewing Zone",className = 'eight columns', style={'font-family':"Arial", 'color': "#8B0000", 'fontSize': 38}),
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
                dbc.Tooltip(" Visit our website ",target="astrologo",placement="bottom",style={'padding':'5px','color':"white",'background-color':"black"}),
            ]),


        ], className = "row"),
    ]),
    html.Div([
    dcc.Markdown(r'This dashboard is intended to allow basic exploration of the data set presented in [Mackereth et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.1947M/abstract). Here, we only show a subset of the highest quality data from the full catalogue presented there, selecting stars whose asteroseismic parameters agreed between three independent pipelines, and whose luminosities (intrinsic brightness) derived both asteroseismically and photometrically agree within 1-sigma.',
                 className="twelve columns"),
    html.Br(),
    dcc.Markdown(['''The following panels show, from left to right, top to bottom:''',
                  '''- **The colour-magnitude diagram**: colour is an analog of the temperature of the star, with cooler stars on the right, hotter on the left. Magnitude denotes the intrinsic brightness of the star. The dependence of the stellar brightness on the star's radius is shown by the colour bar.''',
                  '''- **Stellar mass vs radius**: these are measured by modelling the interior structure of the star and comparing it with the seismic data (below)''',
                  '''- **Positions of the stars on the sky in ecliptic coordinates**: the TESS-CVZ is concentrated in one small spot on the sky''',
                  '''- **The power spectrum of oscillations for the selected star**: shows you which frequencies are most important in the stars natural oscillation. Notice how most stars have a lot of low frequency noise, but many show clear higher frequency oscillations (these appear as semi-regularly spaced peaks in the spectrum). The red band demonstrates where the automated pipeline found such higher frequency oscillations. For stars with higher frequency oscillations, you may need to toggle the plot to log-scale using the radio buttons to see the detail.''',
                  '''''',
                  ''' _Try clicking a star in the top panels to see it's power spectrum below. There are some notes below the figures on interesting populations in the data set. You can **zoom and pan** to explore in more detail, and compare stars by selecting multiple points with the **lasso tool**, or **shift + click**._'''
                  ],
                 className="twelve columns"),
    ], style={'padding': '0px 40px 0px 40px'}),
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
              ],
              style={'display': 'inline-block', 'width': '15%', 'float':'left', 'margin':'2%'}),
    html.Div([dcc.Graph(id='psd-plot',)],
             style={'display': 'inline-block', 'width': '80%', 'float':'right'}),
    ]),
    html.Br(),
    html.Div([
    dcc.Markdown(['''Some things you should be able to spot:''',
                  '''- _Brighter stars have lower frequency oscillations with very high amplitudes!_ This is because they are very large (notice that they have the largest Radii), and so the soundwaves propogate more slowly through their atmospheres, but generate larger brightness fluctuations.''',
                  '''- _There is a population of stars with very similar radii, but a large range in mass._ These are 'Red Clump' giants, that are burning Helium in their core. Their radii are nearly constant since the energy output comes is dependent on the Helium core mass, which is nearly constant, with only a slight dependence on the stellar properties. You may be able to notice that the structure of oscillations in these stars is quite different when you look at their power spectra.  ''',
                  '''- _The data are very noisy!_ TESS was designed for detecting planets, not measuring stellar properties, so much of the data appear very noisy (and in some cases, the pipeline is not working well!). However, for stars with many 'sectors' of observations, the data start to look less noise dominated. The brighter stars tend to be less noisy also, since they send us the most photons!'''],
                 className="twelve columns"),
    ], style={'padding': '0px 40px 0px 40px'}),
    dcc.Store(id='selected-points')
])


@app.callback(
    Output("color-magnitude-scatter", "figure"),
    Output("mass-radius-scatter", "figure"),
    Output("polar-scatter", "figure"),
    Output("psd-plot", "figure"),
    Output("selected-points", "data"),
    Input("color-magnitude-scatter", "selectedData"),
    Input("mass-radius-scatter", "selectedData"),
    Input("polar-scatter", "selectedData"),
    Input("scale-selector", "value"),
    Input("selected-points", "data")
)
def callback(cmdselection,mrselection,polarselection,scaleselection,selectedpoints):
    log=False
    ctx = dash.callback_context
    prop_id = ""
    prop_type = ""
    if selectedpoints:
        points = json.loads(selectedpoints)
    else:
        points = []
    types = np.array(["color-magnitude-scatter", "mass-radius-scatter", "polar-scatter"])
    selects = [cmdselection,mrselection,polarselection]
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    if prop_id in types:
        points = []
        which = np.where(types == prop_id)[0][0]
        print(which)
        selected_data = selects[which]
        if selected_data and selected_data['points']:
            thispoints = get_selection(selected_data)
            points.extend(thispoints)
    if len(points) == 0:
        points = [0]
    if scaleselection == 'log':
        log = True
    print(points)
    json_out = json.dumps(points)
    figs = [gen_plots.generate_cmd(df, points),
            gen_plots.generate_massradius(df, points),
            gen_plots.generate_polar(df, points),
            gen_plots.generate_psd(points,log=log),
            json_out]
    return figs

if __name__ == '__main__':
    app.server.run()