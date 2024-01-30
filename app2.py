import dash
import dash_bootstrap_components as dbc
from flask import Flask, redirect, url_for, session, request

from authlib.integrations.flask_client import OAuth
from dash import html

from decouple import config


FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

server = Flask(__name__)

app = dash.Dash(__name__, server=server, url_base_pathname='/dash/', external_stylesheets=[
    FONT_AWESOME, dbc.themes.BOOTSTRAP, dbc_css], title="Visão Geral Service Desk")

server.secret_key = config('APP_SECRET_KEY')
oauth = OAuth(server)
app.scripts.config.serve_locally = True
server = app.server

oauth.register(
    'azure',
    client_id=config('CLIENT_ID'),
    client_secret=config('CLIENT_SECRET2'),
    authorize_url='https://login.microsoftonline.com/e01cae07-6251-49c7-8c5e-d6f07cffecc7/oauth2/v2.0/authorize',
    authorize_params=None,
    access_token_url='https://login.microsoftonline.com/e01cae07-6251-49c7-8c5e-d6f07cffecc7/oauth2/v2.0/token',
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri=config('URL_REDIRECT'),
    client_kwargs={'scope': 'User.Read'},
)



@server.route('/')
def dash_app():
    return redirect(url_for('login'))


@server.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    return oauth.azure.authorize_redirect(redirect_uri)


@server.route('/logoff')
def logoff():
    # Limpa a sessão do usuário
    session.clear()

    # Redireciona para a página de login ou página inicial
    return redirect(url_for('login'))


@server.route('/authorize')
def authorize():
    token = oauth.azure.authorize_access_token()
    resp = oauth.azure.get('https://graph.microsoft.com/v1.0/me', token=token)
    profile = resp.json()
    session['user'] = profile
    # return f'Olá, {profile["displayName"]}!'
    return redirect('/dash/')

# Função para verificar se o usuário está autenticado


def user_is_authenticated():
    return 'user' in session

# Proteção da rota do Dash


@server.before_request
def before_request():
    # Verifica se a rota acessada é a do Dash e se o usuário está autenticado
    if '/dash/' in request.path:
        if not user_is_authenticated():
            return redirect('/login')
