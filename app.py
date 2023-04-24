from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from odeSolver import *

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H4('Interactive color selection with simple Dash example'),
    html.P("Select color:"),
    dcc.Dropdown(
        id="getting-started-x-dropdown",
        options=['Gold', 'MediuTurquoise', 'LightGreen'],
        value='Gold',
        clearable=False,
    ),
    dcc.Graph(id="getting-started-x-graph"),
])


@app.callback(
    Output("getting-started-x-graph", "figure"), 
    Input("getting-started-x-dropdown", "value"))
def display_color(color):
    rr = solve()
    fig = go.Figure(
        data=go.Bar(y=rr, # replace with your own data source
                    marker_color=color))
    return fig



def dynamics(t,y,simVars):
    y=y.reshape(y.size,1)
    dxdt=np.zeros((y.size,1)) #Create vector
    dxdt[0::2]=y[1::2]
    dxdt[1]=(-4*y[0]/simVars["R"]*simVars["g"]*np.cos(y[0])-5*y[5]**2*np.sin(y[0])*np.cos(y[0])-6*y[5]*y[3])/5 #thetaddot
    dxdt[5]=(2.5*y[5]*y[1]*np.sin(y[0])*np.cos(y[0])-1.5*(+y[5]*y[1]*np.sin(y[0]))*np.cos(y[0])+1.5*y[3]*y[1]*np.sin(y[0]))*4/(1-np.cos(y[0])**2)
    dxdt[3]=-dxdt[5]*np.cos(y[0])+y[5]*y[1]*np.sin(y[0])#ksiddot
    dxdt[6]=simVars["R"]*y[1]*np.sin(y[0])*np.sin(y[4])-simVars["R"]*y[3]*np.cos(y[4])-simVars["R"]*y[5]*np.cos(y[0])*np.cos(y[4])
    dxdt[8]=-simVars["R"]*y[1]*np.sin(y[0])*np.cos(y[4])-simVars["R"]*y[3]*np.sin(y[4])-simVars["R"]*y[5]*np.sin(y[0])*np.cos(y[4])
    dxdt[10]=simVars["R"]*y[1]*np.cos(y[0])
    return dxdt.reshape(1,y.size)

def createCoinData(i,odeObj):
    pos=odeObj.resY[i][0::2]
    thetaDum=np.linspace(0,2*np.pi,num=20)
    xxx=odeObj.simVars["R"]*np.cos(thetaDum);yyy=odeObj.simVars["R"]*np.sin(thetaDum);zzz=np.zeros(20)
    theta=pos[0];ksi=pos[1];phi=pos[2]
    #pdb.set_trace()
    rotMat=inv(np.array([[np.cos(ksi)*np.cos(phi)-np.sin(ksi)*np.sin(phi)*np.cos(theta),np.cos(ksi)*np.sin(phi)+np.sin(ksi)*np.cos(theta)*np.cos(phi),np.sin(ksi)*np.sin(theta)],[-np.sin(ksi)*np.cos(phi)-np.cos(ksi)*np.cos(theta)*np.sin(phi), -np.sin(ksi)*np.sin(phi)+np.cos(ksi)*np.cos(theta)*np.cos(phi), np.cos(ksi)*np.sin(theta)],[np.sin(phi)*np.sin(theta), -np.sin(theta)*np.cos(phi), np.cos(theta)]]))
    posVec=np.zeros((3,1))
    rotVec=np.zeros((xxx.size,3))
    for iii in range(xxx.size):
        posVec=[xxx[iii],yyy[iii],zzz[iii]]
        rotVec[iii,:]=np.matmul(rotMat,posVec)
    xData=rotVec[:,0]+pos[3];yData=rotVec[:,1]+pos[4];zData=rotVec[:,2]+pos[5]
    #pdb.set_trace()
    return {"xData":xData,"yData":yData,"zData":zData}
def solve():
    y0=np.zeros((1,12))
    #Take initCond
    file=open("iniCond.txt","r")

    a=file.read().split('\n')
    y0=np.asarray([[float(a[num]) for num in range(1,24,2)]])
    R=float(a[25])
    tEnd=float(a[27])
    file.close()
    solverEq=odeSolver(fun=dynamics,tSpan=[0,tEnd],iniCond=y0,iniStepSize=0.01,errorThresh=1e-6, maxStepSize=1e-1,simVars={"R":R,"g":9.81})
    solverEq.solveSystem()

    return solverEq.resTime

if __name__ == "__main__":

    app.run_server(debug=True)
