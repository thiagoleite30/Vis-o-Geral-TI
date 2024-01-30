import dash
import dash_bootstrap_components as dbc
from flask import Flask, redirect, url_for
from flask_session import Session 

FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

server = Flask(__name__)

app = dash.Dash(__name__, server=server, url_base_pathname='/dash/', external_stylesheets=[
    FONT_AWESOME, dbc.themes.BOOTSTRAP, dbc_css], title="Visão Geral Service Desk")


app.scripts.config.serve_locally = True
server = app.server


@server.route('/')
def dash_app():
    return redirect(url_for('/dash/'))
'''
if __name__ == '__main__':
    #app.run_server(debug=True, port=8050)
    server.run(debug=True, port=8040, host='0.0.0.0')
'''