import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
import base64
import time

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# Generate curve data
t = np.linspace(0, 2*np.pi, 32)
x = np.sin(t)
y = np.cos(t)+1
xPlate = np.array([-3,3])
yPlate = np.array([0,0])

xm = -2
xM = 2
ym = -3
yM = 10

N = 50



app = Dash(__name__,meta_tags=[{'name':'viewport','content':'width=device-width, initial-scale=1.5, maximum-scale=1.2, minimum-scale=0.5,'}],external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([


        dbc.Row([dbc.Col([html.Div([html.H4('Bouncing Ball'),html.P('Input Params'),
        html.Div(
        children=[dcc.Markdown('Plate Frequency $$[\omega]$$',mathjax=True),
        dcc.Slider(0.1, 10, value=2,id="w0",tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(
        children=[html.P('Plate Displacement'),
        dcc.Slider(0.1, 10, value=0.5,id="A0",tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(
        children=[html.P('Ball Initial Position'),
        dcc.Slider(0, 5, value=1,id="x0",tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(
        children=[html.P('Ball Initial Velocity'),
        dcc.Slider(-1, 1, value=0,id="v0",tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(
        children=[html.P('Coefficient of Restitution'),
        dcc.Input(id='Coeff', type='number', min=0, max=1,value=0.99)]),
        html.Div(
        children=[html.P('Animation Speed'),
        dcc.Slider(1, 100, value=80,id="AnimSpeed",tooltip={"placement": "bottom", "always_visible": True})]),
        html.Div(
        children=[html.P('Simulation Time'),
        dcc.Slider(1, 100, value=5,id="Nframes",tooltip={"placement": "bottom", "always_visible": True})])]), ], width=3,xs=12,lg=3),
        dbc.Col(dcc.Loading(
            id="loading-1",
            type="default",
            children=dcc.Graph(id="loading-output-1")), width=3,xs=12,lg=3),
        dbc.Col(dcc.Loading(
            id="loading-2",
            type="default",
            children=dcc.Graph(id="loading-output-2")), width=6,xs=12,lg=6),    


           
    ])])
# app.layout = html.Div([
#         dbc.Card(
#             dbc.CardBody([

#         dbc.Row([dbc.Col([html.Div([html.H4('Bouncing Ball'),html.P('Input Params'),
#         html.Div(
#         children=[dcc.Markdown('Plate Frequency $$[\omega]$$',mathjax=True),
#         dcc.Slider(0.1, 10, value=1,id="w0",tooltip={"placement": "bottom", "always_visible": True})]),
#         html.Div(
#         children=[html.P('Plate Displacement'),
#         dcc.Slider(0.1, 10, value=0.2,id="A0",tooltip={"placement": "bottom", "always_visible": True})]),
#         html.Div(
#         children=[html.P('Ball Initial Position'),
#         dcc.Slider(0, 5, value=1,id="x0",tooltip={"placement": "bottom", "always_visible": True})]),
#         html.Div(
#         children=[html.P('Ball Initial Velocity'),
#         dcc.Slider(-1, 1, value=0,id="v0",tooltip={"placement": "bottom", "always_visible": True})]),
#         html.Div(
#         children=[html.P('Coefficient of Restitution'),
#         dcc.Input(id='Coeff', type='number', min=0, max=1,value=0.99)]),
#         html.Div(
#         children=[html.P('Animation Speed'),
#         dcc.Slider(1, 100, value=10,id="AnimSpeed",tooltip={"placement": "bottom", "always_visible": True})]),
#         html.Div(
#         children=[html.P('Simulation Time'),
#         dcc.Slider(1, 100, value=10,id="Nframes",tooltip={"placement": "bottom", "always_visible": True})])]), ], width=3),
#         dbc.Col(dcc.Loading(
#             id="loading-1",
#             type="default",
#             children=dcc.Graph(id="loading-output-1")), width=5),
#         dbc.Col(dcc.Loading(
#             id="loading-2",
#             type="default",
#             children=dcc.Graph(id="loading-output-2")), width=3),    


           
#     ])]))])





@app.callback(
    Output("loading-output-1", "figure"),
        Output("loading-output-2", "figure"),

    Input('AnimSpeed', 'value'),
    Input('w0','value'),
    Input('A0','value'),
    Input('x0','value'),
    Input('v0','value'),
    Input('Coeff','value'), Input('Nframes','value'))
   
# def solveDynamics(d):



#         return xVec,yVec


def updateFigure(animSpeed,w0,A0,x0,v0,coeff,Nframes):

    # xVec, yVec = solveDynamics(0)

    t = 0; A = A0;dz0=v0;z0=x0;w=w0;g=9.81;
    t0 = 0; Tol = animSpeed/1000;tf=Nframes;absTol = 1e-10;
    z=10;ii=1;
    yVec=[]; xVec=[];tVec=[];

    while t<tf:
        while z>0: 
            t=t+Tol;
            z=-A*np.sin(w*t)-0.5*g*t**2+(dz0+g*t0+A*w*np.cos(w*t0))*(t-t0)+A*np.sin(w*t0)+0.5*g*t0**2+z0;
            yVec.append(A*np.sin(w*t)); xVec.append(z+yVec[-1]); tVec.append(t);
        a=t-Tol;b=t; 
        fc = 1;
        while abs(fc)>absTol:
            fa=-A*np.sin(w*a)-0.5*g*a**2+(dz0+g*t0+A*w*np.cos(w*t0))*(a-t0)+A*np.sin(w*t0)+0.5*g*t0**2+z0;
            fb=-A*np.sin(w*b)-0.5*g*b**2+(dz0+g*t0+A*w*np.cos(w*t0))*(b-t0)+A*np.sin(w*t0)+0.5*g*t0**2+z0;
            c=(a*fb-b*fa)/(fb-fa);
            fc=-A*np.sin(w*c)-0.5*g*c**2+(dz0+g*t0+A*w*np.cos(w*t0))*(c-t0)+A*np.sin(w*t0)+0.5*g*t0**2+z0;
            if fa*fc<0:
                b=c;
            else:
                a=c;
            
            t=c;
        
        z=1;
        dz0 = -coeff*(-A*w*np.cos(w*t)-g*t+(dz0+g*t0+A*w*np.cos(w*t0)));
        y0 = A*np.sin(w*t);z0=0;t0 = t; yVec[-1]=y0; xVec[-1]=y0;  tVec[-1]= t;

    Nframes = len(xVec)

    fig1 = go.Figure(
    data=[go.Scatter(x=x, y=y+x0,
                     mode="lines",
                     line=dict(width=2, color="blue")),
        go.Scatter(x=xPlate, y=yPlate,
                     mode="lines",
                     line=dict(width=2, color="red"))],
    layout=go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),

        title_text="Animation", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args = [None, {"frame": {"duration": 1/animSpeed*100, 
                                                                        "redraw": False},
                                                              "fromcurrent": True, 
                                                              "transition": {"duration": 0}}])])]),
    frames=[go.Frame(
        data=[go.Scatter(
            x=x,
            y=y+xVec[k],
            mode="lines",
            line=dict(width=2, color="blue")),
            go.Scatter(
            x=xPlate,
            y=yPlate+yVec[k],
            mode="lines",
            line=dict(width=2, color="red"))])#marker=dict(color="red", size=10))])
        for k in range(Nframes)])
    fig1.update_yaxes(scaleanchor="x",scaleratio=1)

    phaseFig = go.Figure(
    data=[go.Scatter(x=tVec, y=xVec,
                     mode="lines",
                     line=dict(width=2, color="blue"),name="Ball"),
        go.Scatter(x=tVec, y=yVec,
                     mode="lines",
                     line=dict(width=2, color="red"),name="Plate")])
    phaseFig.update_layout(xaxis_title="Time [s]",yaxis_title="Displacement")

    fig1.update_layout(height = 600)
    fig1.layout.update(showlegend=False)


# fig.show()

    return fig1,phaseFig



if __name__ == "__main__":
    app.run_server(debug=True)






# # Create figure
# 

