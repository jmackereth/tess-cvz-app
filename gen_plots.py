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

def generate_flexible(df, selected_data):
    layout = go.Layout(
        clickmode="event+select",
        dragmode="lasso",
        showlegend=False,
        autosize=True,
        hovermode="closest",
        xaxis_title='mass',
        yaxis_title='age',
        annotations = list([
            dict(text='x axis:', x=0.0, y=1.15, xref='paper', yref='paper', showarrow=False ),
            dict(text='y axis:', x=0.4, y=1.15, xref='paper', yref='paper', showarrow=False ),
        ]),
        #xaxis=go.layout.XAxis(title=r'$$(J-K)$$', range=[0.5,1.05]),
        #yaxis=go.layout.YAxis(title=r'$$M_{K_S}$$', range=[0.5,-4.3])
        updatemenus=[dict(
            active=1,
            buttons=list([
                dict(label='age', method='update',
                     args=[{'x' : [list(df['age_PARAM_BHM'])]},
                           {'xaxis' : {'title' : 'age'}}]),
                dict(label='mass', method='update',
                     args=[{'x' : [list(df['mass_PARAM_BHM'])]},
                           {'xaxis' : {'title' : 'mass'}}]),
                dict(label='radius', method='update',
                     args=[{'x' : [list(df['rad_PARAM_BHM'])]},
                           {'xaxis' : {'title' : 'radius'}}]),
                dict(label='[Fe/H]', method='update',
                     args=[{'x' : [list(df['feh_SKYMAPPER'])]},
                           {'xaxis' : {'title' : '[Fe/H]'}}]),
                dict(label='T_eff', method='update',
                     args=[{'x' : [list(df['Teff_SKYMAPPER'])]},
                           {'xaxis' : {'title' : 'T_eff'}}])
            ]),
            x=0.10,
            y=1.15,
            xanchor="left",
            yanchor="top",
            showactive=True
        ), dict(
            active=0,
            buttons=list([
                dict(label='age', method='update',
                     args=[{'y' : [list(df['age_PARAM_BHM'])]},
                           {'yaxis' : {'title' : 'age'}}]),
                dict(label='mass', method='update',
                     args=[{'y' : [list(df['mass_PARAM_BHM'])]},
                           {'yaxis' : {'title' : 'mass'}}]),
                dict(label='radius', method='update',
                     args=[{'y' : [list(df['rad_PARAM_BHM'])]},
                           {'yaxis' : {'title' : 'radius'}}]),
                dict(label='[Fe/H]', method='update',
                     args=[{'y' : [list(df['feh_SKYMAPPER'])]},
                           {'yaxis' : {'title' : '[Fe/H]'}}]),
                dict(label='T_eff', method='update',
                     args=[{'y' : [list(df['Teff_SKYMAPPER'])]},
                           {'yaxis' : {'title' : 'T_eff'}}])

            ]),
            x=0.45,
            y=1.15,
            xanchor="left",
            yanchor="top",
            showactive=True
        )]
    )

    if selected_data:
            select_indices = selected_data
    else:
        select_indices = None
    hovertemplate = "<b> %{text}</b><br><br> N_sectors: %{customdata:.0i}<extra></extra>"

    fig = go.Figure(data=[go.Scatter(x=df['mass_PARAM_BHM'], y=df['age_PARAM_BHM'], mode='markers')],
                    layout=layout)

    return fig


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
    fig = go.Figure(data=[go.Scatter(x=[tot_xs[0][0],], y=[tot_ys[0][0],]), go.Scatter(x=allorbits.x(), y=allorbits.y(), mode='markers', marker=dict(size=1, color='#BB5566', opacity=0.5))],
                    layout=layout,
                    frames = tot_frame[0])
    return fig
