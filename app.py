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

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

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

with open('data/standardised_psds_goodsample_v1.02.npy', 'rb') as f:
    psds = np.load(f)
    grid = np.load(f)
    source_ids = np.load(f)

allvxvv = np.dstack([np.array(df['ra'], dtype=np.float64), np.array(df['dec'], dtype=np.float64), np.array(1/df['parallax'], dtype=np.float64), np.array(df['pmra'], dtype=np.float64), np.array(df['pmdec'], dtype=np.float64), np.array(df['radial_velocity'], dtype=np.float64)])[0]
allorbits = Orbit(allvxvv, radec=True, ro=8.175, vo=220.)
#common, indx1, indx2 = np.intersect1d(source_ids,df['source_id'],return_indices=True)

#df = df.iloc[indx2]
#psds = psds[indx1]

def generate_cmd(df, selected_data):
    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=False,
        autosize=True,
        hovermode="closest",
        xaxis=go.layout.XAxis(title=r'$$(J-K)$$', range=[0.5,1.05]),
        yaxis=go.layout.YAxis(title=r'$$M_{K_S}$$', range=[0.5,-4.3])
    )
    if selected_data:
        select_indices = selected_data
    else:
        select_indices = None
    hovertemplate = "<b> %{text}</b><br><br> N_sectors: %{customdata:.0i}<extra></extra>"
    trace = dict(text=list(map(lambda item: "Gaia DR2 "+ str(item), df['source_id'])),
                 customdata=df['N_sectors'],
                 type='scattergl',
                 x=df['jmag']-df['kmag'],
                 y=df['kmag']-(5*np.log10(1000/df['parallax'])-5),
                 mode='markers',
                 hovertemplate=hovertemplate,
                 selectedpoints=select_indices,
                 showlegend=False,
                 marker={"color": "Black", "size":7, "line":{"width":0.}, "opacity":0.1},
                 selected={"marker":{"size":13, "color":'#BB5566', "opacity":1.}},
                 unselected={"marker":{"opacity":0.1}}
                 )
    return {"data": [trace,], "layout":layout}

def generate_massradius(df, selected_data):
    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=False,
        autosize=True,
        hovermode="closest",
        xaxis=go.layout.XAxis(title=r'$$M_{\mathrm{BHM}}\ \mathrm{[M_{\odot}]}$$', range=[0.75,2]),
        yaxis=go.layout.YAxis(title=r'$$R_{\mathrm{BHM}}\ \mathrm{[R_{\odot}]}$$', range=[4,25])
    )
    if selected_data:
        select_indices = selected_data
    else:
        select_indices = None
    hovertemplate = "<b> %{text}</b><br><br> N_sectors: %{customdata:.0i}<extra></extra>"
    trace = dict(text=list(map(lambda item: "Gaia DR2 "+ str(item), df['source_id'])),
                 customdata=df['N_sectors'],
                 type='scattergl',
                 x=df['mass_PARAM_BHM'],
                 y=df['rad_PARAM_BHM'],
                 mode='markers',
                 hovertemplate=hovertemplate,
                 selectedpoints=select_indices,
                 showlegend=False,
                 marker={"color": "Black", "size":7, "line":{"width":0.}, "opacity":0.1},
                 selected={"marker":{"size":13, "color":'#BB5566', "opacity":1.}},
                 unselected={"marker":{"opacity":0.1}}
                 )
    return {"data": [trace,], "layout":layout}

def generate_polar(df, selected_data):
    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=False,
        autosize=True,
        hovermode="closest",
        margin=dict(l=0, r=0, t=0, b=0),
        polar={"radialaxis":{"range":[-90,-65], "title":'$$\mathrm{ecliptic\ latitude}$$'}}
    )
    if selected_data:
        select_indices = selected_data
    else:
        select_indices = None
    hovertemplate = "<b> %{text}</b><br><br> N_sectors: %{customdata:.0i}<extra></extra>"
    trace = dict(text=list(map(lambda item: "Gaia DR2 "+ str(item), df['source_id'])),
                 customdata=df['N_sectors'],
                 type='scatterpolargl',
                 r=df['ecl_lat'],
                 theta=df['ecl_lon'],
                 mode='markers',
                 hovertemplate=hovertemplate,
                 selectedpoints=select_indices,
                 showlegend=False,
                 marker={"color": df['N_sectors'], "size":7, "line":{"width":0.}, "opacity":0.1},
                 selected={"marker":{"size":13, "color":'#BB5566',  "opacity":1.}}
                 )
    return {"data": [trace,], "layout":layout}

def generate_psd(select, log=False):
    if not log:
        xaxis = go.layout.XAxis(title=r'$$\nu\ \mathrm{[\mu Hz]}$$', range=[0.,100.])
        yaxis = go.layout.YAxis(title=r'$$\mathrm{PSD}\ \mathrm{\left[\frac{ppm}{\mu Hz}\right]}$$')
    else:
        xaxis = go.layout.XAxis(title=r'$$\nu\ \mathrm{[\mu Hz]}$$', type='log', range=[0,np.log10(100.)])
        yaxis = go.layout.YAxis(title=r'$$\mathrm{PSD}\ \mathrm{\left[\frac{ppm}{\mu Hz}\right]}$$', type='log')
    layout = go.Layout(
        showlegend=False,
        autosize=True,
        hovermode="closest",
        xaxis=xaxis,
        yaxis=yaxis,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
    )
    traces = []
    for ii,i in enumerate(select):
        if ii == len(select)-1:
            trace = dict(customdata=df['N_sectors'],
                         type='scatter',
                         x=grid,
                         y=psds[i]/1e6,
                         mode='lines',
                         showlegend=False,
                         line={"color":"Black", "width":1.0}
                         )
        else:
            trace = dict(customdata=df['N_sectors'],
                         type='scatter',
                         x=grid,
                         y=psds[i]/1e6,
                         mode='lines',
                         showlegend=False,
                         line={"color":"Black", "width":(ii+1)/(len(select)+1), "opacity":0.5}
                         )
        traces.append(trace)
    if len(select) == 1:
        fig = go.Figure({"data": traces, "layout":layout})
        fig.add_vrect(x0=df['numax_BHM'].iloc[i]-df['numax_err_BHM'].iloc[i], x1=df['numax_BHM'].iloc[i]+df['numax_err_BHM'].iloc[i],
                     line_width=0, fillcolor="red", opacity=0.2)
        fig.update_xaxes(gridcolor='#cfcfcf', zerolinecolor='Black')
        fig.update_yaxes(gridcolor='#cfcfcf', zerolinecolor='Black')
        return fig
    else:
        return {"data": traces, "layout":layout}

def generate_orbit(select):
    orbits = allorbits[select]
    tot_frame = []
    tot_xs = []
    tot_ys = []
    for ii, i in enumerate(select):
        thisorb = orbits[ii]
        if df['age_PARAM_BHM'][i] != -9999.:
            ts = np.linspace(0., -1*df['age_PARAM_BHM'][i]*u.Gyr, int(df['age_PARAM_BHM'][i]*1000))
        else:
            ts = np.linspace(0., -2.*u.Gyr, 2000)
        thisorb.integrate(ts, MWPotential2014)
        xs = thisorb.x(ts)
        ys = thisorb.y(ts)
        nframe = int(len(ts)/10)
        frames = []
        for i in range(nframe-1):
            frames.append(go.Frame(data=[go.Scatter(x=xs[i*10:i*10+10], y=ys[i*10:i*10+10], mode='lines')]))
        tot_frame.append(frames)
        tot_xs.append(xs)
        tot_ys.append(ys)
    layout = go.Layout( xaxis=dict(scaleanchor='y', scaleratio=1, autorange=False),
                        yaxis=dict(range=[np.min(ys), np.max(ys)], autorange=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])])
    if len(tot_frame) == 1:
        fig = go.Figure(data=[go.Scatter(x=[tot_xs[0][0],], y=[tot_ys[0][0],]), go.Scatter(x=allorbits.x(), y=allorbits.y(), mode='markers', marker=dict(size=1, color='#BB5566', opacity=0.5))],
                        layout=layout,
                        frames = tot_frame[0])
    return fig


def get_selection(selection_data):
    ind = []
    #only one curve so we dont care about curveNumber
    for point in selection_data["points"]:
        ind.append(point["pointNumber"])
    return ind

#cmdfig = px.scatter(x=df.jmag-df.kmag, y=df.kmag-(5*np.log10(1000/df.parallax)-5), color=df.N_sectors, custom_data=[df.source_id])
#cmdfig.update_layout(clickmode='event+select')
#cmdfig.update_traces(marker_size=4)

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
    [
    Input("polar-scatter", "selectedData"),
    Input("mass-radius-scatter", "selectedData"),
    ]
)
def update_cmd(polarselect, mrselect):
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    if prop_id == "polar-scatter":
        if polarselect is None:
            selected_data = 0
        else:
            selected_data = get_selection(polarselect)
    elif prop_id == "mass-radius-scatter":
        if mrselect is None:
            selected_data = 0
        else:
            selected_data = get_selection(mrselect)
    else:
        selected_data = 0
    return generate_cmd(df, selected_data)

@app.callback(
    Output("mass-radius-scatter", "figure"),
    [
    Input("polar-scatter", "selectedData"),
    Input("color-magnitude-scatter", "selectedData")
    ]
)
def update_mr(polarselect, cmdselect):
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    if prop_id == "polar-scatter":
        if polarselect is None:
            selected_data = 0
        else:
            selected_data = get_selection(polarselect)
    elif prop_id == "color-magnitude-scatter":
        if cmdselect is None:
            selected_data = 0
        else:
            selected_data = get_selection(cmdselect)
    else:
        selected_data = 0
    return generate_massradius(df, selected_data)

@app.callback(
    Output("polar-scatter", "figure"),
    [
    Input("color-magnitude-scatter", "selectedData"),
    Input("mass-radius-scatter", "selectedData")
    ]
)
def update_polar(cmdselect, mrselect):
    ctx = dash.callback_context

    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    if prop_id == "color-magnitude-scatter":
        if cmdselect is None:
            selected_data = 0
        else:
            selected_data = get_selection(cmdselect)
    elif prop_id == "mass-radius-scatter":
        if mrselect is None:
            selected_data = 0
        else:
            selected_data = get_selection(mrselect)
    else:
        selected_data = 0
    return generate_polar(df, selected_data)

@app.callback(
    Output("psd-plot", "figure"),
    [
    Input("color-magnitude-scatter", "selectedData"),
    Input("mass-radius-scatter", "selectedData"),
    Input("polar-scatter", "selectedData"),
    Input("scale-selector", "value")
    ]
)
def update_psd(cmdselect, mrselect, polarselect, scale):
    # Find which one has been triggered
    ctx = dash.callback_context
    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    log= False
    if prop_id == "color-magnitude-scatter":
        if cmdselect is None:
            select = [0]
        else:
            select = get_selection(cmdselect)
        if scale == 'log':
            log = True

    elif prop_id == "mass-radius-scatter":
        if mrselect is None:
            select = [0]
        else:
            select = get_selection(mrselect)
        if scale == 'log':
            log = True
        last_trigger = 'mr'
    elif prop_id == "polar-scatter":
        if polarselect is None:
            select = [0]
        else:
            select = get_selection(polarselect)
        if scale == 'log':
            log = True
        last_trigger = 'polar'
    elif prop_id == "scale-selector":
        if scale == 'log':
            log = True
        if cmdselect is None:
            select=[0]
        else:
            select = get_selection(cmdselect)
    else:
        select = [0]

    return generate_psd(select, log=log)

@app.callback(
    Output("orbit-plot", "figure"),
    [
    Input("color-magnitude-scatter", "selectedData"),
    Input("mass-radius-scatter", "selectedData"),
    Input("polar-scatter", "selectedData")
    ]
)
def update_orbit(cmdselect, mrselect, polarselect):
    # Find which one has been triggered
    ctx = dash.callback_context
    prop_id = ""
    prop_type = ""
    if ctx.triggered:
        splitted = ctx.triggered[0]["prop_id"].split(".")
        prop_id = splitted[0]
        prop_type = splitted[1]
    log= False
    if prop_id == "color-magnitude-scatter":
        if cmdselect is None:
            select = [0]
        else:
            select = get_selection(cmdselect)
    elif prop_id == "mass-radius-scatter":
        if mrselect is None:
            select = [0]
        else:
            select = get_selection(mrselect)
        last_trigger = 'mr'
    elif prop_id == "polar-scatter":
        if polarselect is None:
            select = [0]
        else:
            select = get_selection(polarselect)
        last_trigger = 'polar'
    else:
        select = [0]

    return generate_orbit(select)

if __name__ == '__main__':
    app.server.run()
