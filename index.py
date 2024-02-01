from dash import html, dcc, Input, Output, State, callback, Dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_loading_spinners as dls

# import from folders/theme changer
from app2 import *
from dash_bootstrap_templates import ThemeSwitchAIO
from datetime import datetime, date


from Data_Work.Data_Work import Data_Work
from flask import session


FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"


# ========== Styles ============ #
tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor": "top",
               "y": 0.9,
               "xanchor": "left",
               "x": 0.1,
               "title": {"text": None},
               "font": {"color": "white"},
               "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l": 10, "r": 10, "t": 10, "b": 10}
}

config_graph = {"displayModeBar": False, "showTips": False}

template_theme1 = "flatly"
template_theme2 = "darkly"
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.DARKLY


# ===== Reading n cleaning File ====== #
df = pd.read_csv('Todos_Chamados.csv')

df['DATA_ABERTURA'] = pd.to_datetime(
    df['DATA_ABERTURA'], format='ISO8601')
df['DATA_FECHAMENTO'] = pd.to_datetime(
    df['DATA_FECHAMENTO'], format='ISO8601')
df['DATA_ALVO'] = pd.to_datetime(
    df['DATA_ALVO'], format='ISO8601')

ultima_atualizacao = df.sort_values(
    by='DATA_ABERTURA', ascending=False).head(1)['DATA_ABERTURA'][0]
ultima_atualizacao_str = ultima_atualizacao.strftime(("%d/%m/%Y %H:%M:%S"))

df_cru = df.copy()
opcoes = [{'label': str(int(numero)), 'value': int(
    numero)} for numero in list(df['ANO_ABERTURA'].unique()) if (np.isnan(numero) == False)]
opcoes_tipo_chamado = []
opcoes_tipo_chamado += [{'label': tipo, 'value': tipo}
                        for tipo in list(df['TIPO_CHAMADO'].unique())]

opcoes_grupos_operadores = []
opcoes_grupos_operadores += [{'label': grupo, 'value': grupo}
                             for grupo in list(df['GRUPO_OPERADOR'].unique())]

opcoes_fial_solicitante = []
opcoes_fial_solicitante += [{'label': filial, 'value': filial}
                            for filial in list(df['FILIAL_SOLICITANTE'].unique()) if filial is not None]
opcoes_fial_solicitante = [
    item for item in opcoes_fial_solicitante if item['label'] is not np.nan and item['value'] is not np.nan]


# =========  Layout  =========== #

def create_layout():
    return dbc.Container(children=[
        # Armazenamento de dataset
        # dcc.Store(id='dataset', data=df_store),
        dcc.Location(id='url', refresh=False),
        # Layout
        # Row 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Legend("Service Desk")
                            ], sm=8),
                            dbc.Col([
                                html.I(className="fa fa-database",
                                       style={'aria-hidden': 'true', 'font-size': '300%'})
                            ], sm=4, align="center")
                        ]),

                        dbc.Row([
                            dbc.Col([
                                ThemeSwitchAIO(aio_id="theme", themes=[
                                    url_theme1, url_theme2]),
                                html.Div(id='mensagem_boas_vindas'),
                                html.Div(
                                    html.H6(
                                        f'Última Atuliazação de dados: {ultima_atualizacao_str}'),
                                    style={'margin-top': '20px'}
                                ),
                            ])
                        ], style={'margin-top': '10px'}),
                        dbc.Row([
                            # dbc.Button(
                            #    "Sair", href="http://localhost:8040/logoff",
                                # target="_blank"
                                # )
                                ], className="my-4")

                    ])
                ], style=tab_card)
            ], sm=12, lg=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row(
                            dbc.Col([
                                dls.Ring([
                                    dcc.Graph(id='indicator1', className='dbc',
                                              config=config_graph)
                                ],
                                    show_initially=True,
                                    id="loading-indicator1"
                                ),
                            ])
                        ),
                    ])
                ], style=tab_card)
            ], sm=12, lg=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row(
                            dbc.Col([
                                dls.Ring([
                                    dcc.Graph(id='indicator2', className='dbc',
                                              config=config_graph)
                                ],
                                    show_initially=True,
                                    id="loading-indicator2"
                                ),
                            ])
                        ),

                    ])
                ], style=tab_card)
            ], sm=12, lg=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row(
                            dbc.Col([
                                dls.Ring([
                                    dcc.Graph(id='indicator3', className='dbc',
                                              config=config_graph)
                                ],
                                    show_initially=True,
                                    id="loading-indicator3"
                                ),
                            ])
                        )
                    ])
                ], style=tab_card)
            ], sm=12, lg=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row(
                            dbc.Col([
                                dls.Ring([
                                    dcc.Graph(id='indicator4', className='dbc',
                                              config=config_graph)
                                ],
                                    show_initially=True,
                                    id="loading-indicator4"
                                ),
                            ])
                        )
                    ])
                ], style=tab_card)
            ], sm=12, lg=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            html.Div(
                                children=[
                                    html.H5('Periodo', style={
                                        'margin-top': '15px',
                                        'text-align': 'center'}),
                                    html.Div(

                                        dcc.DatePickerRange(
                                            id='my-date-picker-range1',
                                            minimum_nights=0,
                                            min_date_allowed=date(
                                                datetime.now().year, 1, 1),
                                            # max_date_allowed=date(
                                            #    datetime.now().year, datetime.now().month, datetime.now().day),
                                            max_date_allowed=date.today(),
                                            # initial_visible_month=date(data_atual.year, 1, 1),
                                            display_format='DD/MM/YYYY',
                                            start_date=date(
                                                datetime.now().year, 1, 1),
                                            # start_date=date(data_old.year, data_old.month, data_old.day),
                                            end_date=date(
                                                datetime.now().year, datetime.now().month, datetime.now().day),
                                        ),
                                        style={'width': '100%',
                                               'margin': 'auto',
                                               'text-align': 'center'}
                                    ),
                                    html.Div(
                                        id='output-container-date-picker-range')
                                ],
                            )
                        ]),
                        dbc.Row(
                            children=[

                                html.Div(
                                    html.Legend(
                                        "Selecione ano inicial e final para comparação: ",
                                        style={'font-size': '12px', 'text-align': 'center'}),
                                ),
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.Select(
                                            id='select-year1',
                                            disabled=False,
                                            options=opcoes,
                                            value=datetime.now().year - 1
                                        ),
                                    ]),

                                ]),
                                dbc.Col([
                                    dbc.InputGroup([
                                        dbc.Select(
                                            id='select-year2',
                                            disabled=True,
                                            options=opcoes,
                                            value=2024
                                        ),
                                    ]),
                                    dcc.Interval(id="interval3",
                                                 interval=180000),
                                ]),
                            ],

                        )
                    ])
                ], style=tab_card)
            ], sm=12, lg=2)
        ], className='g-2 my-auto', style={'margin-top': '7px'}),

        # Row 2
        dbc.Row([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Accordion([
                                dbc.AccordionItem([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H5(f'Ano:', style={
                                                        'text-align': 'center'}),
                                                    dbc.InputGroup([
                                                        dbc.Select(
                                                            id='select-year3',
                                                            disabled=False,
                                                            options=opcoes,
                                                            value=datetime.now().year - 1
                                                        ),
                                                    ]),
                                                ]),
                                            ], style=tab_card),], sm=12, lg=2),
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H5(f'Tipo de Chamado:', style={
                                                        'text-align': 'center'}),
                                                    dbc.InputGroup([
                                                        dcc.Dropdown(
                                                            id='selec-call-type1',
                                                            options=opcoes_tipo_chamado,
                                                            multi=True,
                                                            value=[
                                                                valores['value'] for valores in opcoes_tipo_chamado if 'value' in valores],
                                                            style={
                                                                'width': '100%'}
                                                        ),
                                                    ]),
                                                ]),
                                            ], style=tab_card),
                                        ]),
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H5(f'Grupo de Operadores:', style={
                                                        'text-align': 'center'}),
                                                    dbc.InputGroup([
                                                        dcc.Dropdown(
                                                            id='select-operator-group1',
                                                            options=opcoes_grupos_operadores,
                                                            multi=True,
                                                            value=[
                                                                valores['value'] for valores in opcoes_grupos_operadores if 'value' in valores],
                                                            style={
                                                                'width': '100%'}
                                                        ),
                                                    ]),
                                                ]),
                                            ], style=tab_card),
                                        ]),
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H5(f'Filial:', style={
                                                        'text-align': 'center'}),
                                                    dbc.InputGroup([
                                                        dcc.Dropdown(
                                                            id='select-filia1',
                                                            options=opcoes_fial_solicitante,
                                                            multi=True,
                                                            value=[
                                                                valores['value'] for valores in opcoes_fial_solicitante if 'value' in valores],
                                                            style={
                                                                'width': '100%'}
                                                        ),
                                                    ]),
                                                ]),
                                            ], style=tab_card),
                                        ]),
                                    ]),
                                ], title="Opções de Filtragem"),
                            ]),
                        ]),
                    ]),
                ]),
            ], className='g-2 my-auto', style={'margin-top': '7px'}),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row(
                                        dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph1', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph1"
                                            ),
                                        ])
                                    )
                                ])
                            ], style=tab_card)
                        ])
                    ], className='g-2 my-auto', style={'margin-top': '7px'}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row(
                                        dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph6', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph6"
                                            ),
                                        ])
                                    )
                                ])
                            ], style=tab_card)
                        ])
                    ], className='g-2 my-auto', style={'margin-top': '7px'}),
                ], sm=12, lg=6),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row(
                                        dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph2', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph2"
                                            ),
                                        ])
                                    )
                                ])
                            ], style=tab_card)
                        ], sm=12, lg=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph3', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph3"
                                            ),
                                        ])
                                    ])
                                ])
                            ], style=tab_card)
                        ], sm=12, lg=6)
                    ], className='g-2 my-auto', style={'margin-top': '7px'}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph5', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph5"
                                            ),
                                        ])
                                    ])
                                ])
                            ], style=tab_card)
                        ], sm=12, lg=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph4', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph4"
                                            ),
                                        ])
                                    ])
                                ])
                            ], style=tab_card)
                        ], sm=12, lg=6)
                    ], className='g-2 my-auto', style={'margin-top': '7px'}),
                ], sm=12, lg=6),
            ], className='g-2 my-auto', style={'margin-top': '7px'}),
        ], className='g-2 my-auto', style={'margin-top': '7px'}),

        # Row 3

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph7', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph7"
                                            ),
                                            ])
                                ])
                            ])
                        ], style=tab_card)
                    ]),
                ])
            ], sm=12, lg=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row(
                                    dbc.Col([
                                            dls.Ring([
                                                dcc.Graph(id='graph8', className='dbc',
                                                          config=config_graph)
                                            ],
                                                show_initially=True,
                                                id="loading-graph8"
                                            ),
                                            ])
                                )
                            ])
                        ], style=tab_card)
                    ]),
                ]),
            ], sm=12, lg=6),
        ], className='g-2 my-auto', style={'margin-top': '7px'}),

    ], fluid=True, style={'height': '100vh'})


app.layout = create_layout
# ======== Callbacks ========== #
# =========Mensagem de boas vindas ============#


@app.callback(
    Output('mensagem_boas_vindas', 'children'),
    # Supondo que você tem um componente dcc.Location no seu layout
    [Input('url', 'pathname')]
)
def update_boas_vindas(pathname):
    # Acessa a sessão dentro do contexto do callback
    nome_usuario = session.get('user', {}).get('displayName', 'Visitante')
    return f"Seja bem-vindo(a), {nome_usuario}!"


@app.callback(Output('selec-call-type1', 'options'),
              Output('select-operator-group1', 'options'),
              Output('select-filia1', 'options'),
              Input('select-year3', 'value'))
def get_data_filter(ano):
    df_cru = df.query(f'ANO_ABERTURA == {ano}')
    opcoes_tipo_chamado = []
    opcoes_tipo_chamado += [{'label': tipo, 'value': tipo}
                            for tipo in list(df_cru['TIPO_CHAMADO'].unique())]

    opcoes_grupos_operadores = []
    opcoes_grupos_operadores += [{'label': grupo, 'value': grupo}
                                 for grupo in list(df_cru['GRUPO_OPERADOR'].unique())]

    opcoes_fial_solicitante = []
    opcoes_fial_solicitante += [{'label': filial, 'value': filial}
                                for filial in list(df_cru['FILIAL_SOLICITANTE'].unique()) if filial is not None]
    opcoes_fial_solicitante = [
        item for item in opcoes_fial_solicitante if item['label'] is not np.nan and item['value'] is not np.nan]

    return opcoes_tipo_chamado, opcoes_grupos_operadores, opcoes_fial_solicitante


# my-date-picker-range1


@app.callback(
    Output('output-container-date-picker-range', 'children'),
    Input('my-date-picker-range1', 'start_date'),
    Input('my-date-picker-range1', 'end_date'),
    Input('select-year1', 'value'),
    Input('select-year2', 'value'),
)
def update_output(start_date, end_date, ano1, ano2):
    string_prefix = f'Período para comparação entre os anos {ano1} e {ano2}: '
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%d/%m/%Y')
        string_prefix = '\n\n' + string_prefix + start_date_string + ' Até '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%d/%m/%Y')
        string_prefix = string_prefix + end_date_string
    if len(string_prefix) == len('Período selecionado: '):
        return 'Select a date to see it displayed here'
    else:
        return string_prefix


# indicator 1


@app.callback(
    Output('indicator1', 'figure'),
    Input('my-date-picker-range1', 'start_date'),
    Input('my-date-picker-range1', 'end_date'),
    Input('select-year1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def indicator1(start_date, end_date, ano_comparacao, toggle):
    template = template_theme1 if toggle else template_theme2
    ano_comparacao = int(ano_comparacao)

    df11 = df.set_index('DATA_ABERTURA')
    df11.sort_index(inplace=True)

    df11_atual = df11[start_date:end_date]

    data_inicio = pd.to_datetime(start_date)
    data_fim = pd.to_datetime(end_date)

    diff_year = datetime.now().year - ano_comparacao

    data_inicio = data_inicio - pd.DateOffset(years=diff_year)
    data_fim = data_fim - pd.DateOffset(years=diff_year)

    df11_anterior = df11[data_inicio.strftime(
        format='%Y-%m-%d'):data_fim.strftime(format='%Y-%m-%d')]
    # Calcular a variação percentual
    percentual_variacao = (
        (df11_atual.shape[0] - df11_anterior.shape[0]) / df11_anterior.shape[0])

    fig11 = go.Figure()
    fig11.add_trace(go.Indicator(
        mode='number+delta',
        title={"text": f"<span style='font-size:90%'>Total Chamados</span><br><br>"},
        value=df11_atual.shape[0],
        number={'valueformat': '.0f'},
        delta={'relative': True, 'valueformat': '.01%', 'reference': df11_anterior.shape[0],
               'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    # Adicionar anotação com o texto explicativo
    fig11.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.005,  # A posição y pode precisar ser ajustada dependendo do layout do seu gráfico
        text=f"<span style='font-size:70%'>Maior em relação ao ano de {ano_comparacao} com </br></br>{df11_anterior.shape[0]} chamados abertos no periodo.</span>" if percentual_variacao >= 0 else f"<span style='font-size:70%'>Menor em relação ao ano de {ano_comparacao} com </br></br>{df11_anterior.shape[0]} chamados abertos no periodo..</span>",
        showarrow=False,
        font=dict(size=16, color="green" if percentual_variacao >= 0 else "red")
    )
    fig11.update_layout(main_config, height=230, template=template)

    return fig11

# indicator 2


@app.callback(
    Output('indicator2', 'figure'),
    Input('my-date-picker-range1', 'start_date'),
    Input('my-date-picker-range1', 'end_date'),
    Input('select-year1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def indicator2(start_date, end_date, ano_comparacao, toggle):
    template = template_theme1 if toggle else template_theme2
    ano_comparacao = int(ano_comparacao)
    data_work = Data_Work()
    data_atual = datetime.now()
    query = data_work.get_data_range_closed(
        start_date, end_date, ano_comparacao)
    # print(f'Query nno indicator 2 e datas recebidas: {query}')
    df12 = df.query(query)

    # print(f'Chamados abertos em {start_date} e encerrados até {end_date}')

    df12_atual = df12.query(f'ANO_FECHAMENTO == {data_atual.year}').groupby(
        'ANO_FECHAMENTO')['NUMERO_CHAMADO'].size().sum()
    df12_anterior = df12.query(f'ANO_FECHAMENTO == {ano_comparacao}').groupby(
        'ANO_FECHAMENTO')['NUMERO_CHAMADO'].size().sum()
    # Calcular a variação percentual
    percentual_variacao_fechados = (
        (df12_atual - df12_anterior) / df12_anterior)
    fig12 = go.Figure()
    fig12.add_trace(go.Indicator(
        mode='number+delta',
        title={
            "text": f"<span style='font-size:80%'>Total Chamados Fechados</span><br><br>"},
        value=df12_atual,
        number={'valueformat': '.0f'},
        delta={'relative': True, 'valueformat': '.01%', 'reference': df12_anterior,
               'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    # Adicionar anotação com o texto explicativo
    fig12.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.005,  # A posição y pode precisar ser ajustada dependendo do layout do seu gráfico
        text=f"<span style='font-size:70%'>Maior em relação ao ano de {ano_comparacao} com </br></br>{df12_anterior} chamados fechados no periodo.</span>" if percentual_variacao_fechados >= 0 else f"<span style='font-size:70%'>Menor em relação ao ano de {ano_comparacao} com </br></br>{df12_anterior} chamados fechados no periodo.</span>",
        showarrow=False,
        font=dict(
            size=16, color="green" if percentual_variacao_fechados >= 0 else "red")
    )

    fig12.update_layout(main_config, height=230, template=template)
    return fig12

# indicator 3


@app.callback(
    Output('indicator3', 'figure'),
    Input('select-year1', 'value'),
    Input('my-date-picker-range1', 'start_date'),
    Input('my-date-picker-range1', 'end_date'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def indicator3(ano_comparacao, start_date, end_date, toggle):
    template = template_theme1 if toggle else template_theme2
    pd.options.mode.chained_assignment = None
    data_atual = datetime.now()
    ano_comparacao = int(ano_comparacao)
    data_work = Data_Work()
    query = data_work.get_data_range_closed(
        start_date, end_date, ano_comparacao)
    df14 = df.query(query)
    serie_anterior = df14[df14['ANO_FECHAMENTO'] == int(ano_comparacao)].groupby(
        'Dentro do Prazo')['NUMERO_CHAMADO'].count()
    serie_atual = df14[df14['ANO_FECHAMENTO'] == data_atual.year].groupby(
        'Dentro do Prazo')['NUMERO_CHAMADO'].count()
    # print(serie_atual, serie_anterior)
    if not 1 in serie_atual.index:
        serie_atual.at[1] = 0

    if not 1 in serie_anterior.index:
        serie_anterior.at[1] = 0
    # Calcular a variação percentual
    if serie_anterior[1] != 0:
        percentual_variacao_fechados = (
            (serie_atual[1] - serie_anterior[1]) / serie_anterior[1])
    else:
        percentual_variacao_fechados = 0
    fig14 = go.Figure()
    fig14.add_trace(go.Indicator(
        mode='number+delta',
        title={
            "text": f"<span style='font-size:80%'>Total dentro do Prazo</span><br><br>"},
        value=serie_atual[1],
        delta={'relative': True, 'valueformat': '.01%', 'reference': serie_anterior[1], 'increasing': {
            'color': "green"}, 'decreasing': {'color': "red"}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    if percentual_variacao_fechados != 0:
        texto = f"<span style='font-size:70%'>Maior em relação ao ano de {ano_comparacao} com</br></br> {serie_anterior[1]} chamados dentro do prazo.</span>" if percentual_variacao_fechados >= 0 else f"<span style='font-size:70%'>Menor em relação ao ano de {ano_comparacao} com </br></br>{serie_anterior[1]} chamados dentro do prazo.</span>"
    else:
        texto = f"<span style='font-size:70%'>Igual ao ano de {ano_comparacao} com</br></br> {serie_anterior[1]} chamados fechados dentro do prazo.</span>"

    # Adicionar anotação com o texto explicativo
    fig14.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.005,  # A posição y pode precisar ser ajustada dependendo do layout do seu gráfico
        text=texto,
        showarrow=False,
        font=dict(size=16, color="green" if percentual_variacao_fechados >= 0 else "red"))
    fig14.update_layout(main_config, height=230, template=template)
    return fig14

# indicator 4


@app.callback(
    Output('indicator4', 'figure'),
    Input('select-year1', 'value'),
    Input('my-date-picker-range1', 'start_date'),
    Input('my-date-picker-range1', 'end_date'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def indicator4(ano_comparacao, start_date, end_date, toggle):
    template = template_theme1 if toggle else template_theme2
    data_atual = datetime.now()
    ano_comparacao = int(ano_comparacao)
    data_work = Data_Work()
    df15 = df.query(data_work.get_data_range_closed(
        start_date, end_date, ano_comparacao))
    serie_anterior = df15[df15['ANO_FECHAMENTO'] == int(ano_comparacao)].groupby(
        'Dentro do Prazo')['NUMERO_CHAMADO'].count()
    serie_atual = df15[df15['ANO_FECHAMENTO'] == data_atual.year].groupby(
        'Dentro do Prazo')['NUMERO_CHAMADO'].count()

    if not 0 in serie_atual.index:
        serie_atual.at[0] = 0

    if not 0 in serie_anterior.index:
        serie_anterior.at[0] = 0

    # Calcular a variação percentual
    if serie_anterior[0] != 0:
        percentual_variacao_fechados = (
            (serie_atual[0] - serie_anterior[0]) / serie_anterior[0])
    else:
        percentual_variacao_fechados = 0

    fig15 = go.Figure()
    fig15.add_trace(go.Indicator(
        mode='number+delta',
        title={
            "text": f"<span style='font-size:80%'>Total fora do Prazo</span><br><br>"},
        value=serie_atual[0],
        number={'valueformat': '.0f'},
        delta={'relative': True, 'valueformat': '.01%', 'reference': serie_anterior[0],
               'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    if percentual_variacao_fechados != 0:
        texto = f"<span style='font-size:70%'>Maior em relação ao ano de {ano_comparacao} com</br></br> {serie_anterior[0]} chamados fora do prazo.</span>" if percentual_variacao_fechados >= 0 else f"<span style='font-size:70%'>Menor em relação ao ano de {ano_comparacao} com </br></br>{serie_anterior[0]} chamados fora do prazo.</span>"
    else:
        texto = f"<span style='font-size:70%'>Igual ao ano de {ano_comparacao} com</br></br> {serie_anterior[0]} chamados fechados fora do prazo.</span>"

    # Adicionar anotação com o texto explicativo
    fig15.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.005,  # A posição y pode precisar ser ajustada dependendo do layout do seu gráfico
        text=texto,
        showarrow=False,
        font=dict(
            size=16, color="red" if percentual_variacao_fechados >= 0 else "green")
    )
    fig15.update_layout(main_config, height=230, template=template)
    return fig15

# graph 1: Os N Maiores Ofensores


@app.callback(
    Output('graph1', 'figure'),
    Input('selec-call-type1', 'value'),
    Input('select-operator-group1', 'value'),
    Input('select-filia1', 'value'),
    Input('select-year3', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph1(opcoes_tipo_selecionadas, opcoes_operadores_selecionadas, filiais, ano, toggle):
    template = template_theme1 if toggle else template_theme2
    # print(f'Opções selecionadas: {opcoes_tipo_selecionadas}')
    # lista_query = [f'TIPO_CHAMADO == {item["value"]}' for item in opcoes_tipo_selecionadas if item['label']
    #               is not np.nan and item['value'] is not np.nan]
    query = '('
    for index, value in enumerate(opcoes_tipo_selecionadas):
        if index < len(opcoes_tipo_selecionadas) - 1:
            query += f'TIPO_CHAMADO == "{value}" or '
        else:
            query += f'TIPO_CHAMADO == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(opcoes_operadores_selecionadas):
        if index < len(opcoes_operadores_selecionadas) - 1:
            query += f'GRUPO_OPERADOR == "{value}" or '
        else:
            query += f'GRUPO_OPERADOR == "{value}")'

    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(filiais):
        if index < len(filiais) - 1:
            query += f'FILIAL_SOLICITANTE == "{value}" or '
        else:
            query += f'FILIAL_SOLICITANTE == "{value}")'

    query += f' and (ANO_ABERTURA == {ano})'
    # print(f'QUERY RESULTADO: {query}')

    if len(opcoes_tipo_selecionadas) > 0 and len(opcoes_operadores_selecionadas) > 0 and len(filiais) > 0:
        df4 = df.query(query).groupby(
            ['MES_ABERTURA', 'CATEGORIA']).size().reset_index(name="QUANTIDADE_CHAMADOS")
        total_chamados_por_mes = df.query(query).groupby('MES_ABERTURA').size()
        # Calcular o percentual de cada categoria em relação ao total de chamados por mês
        df4['PERCENTUAL'] = df4.apply(lambda row: "{:.2f}%".format(
            (row['QUANTIDADE_CHAMADOS'] / total_chamados_por_mes[row['MES_ABERTURA']]) * 100), axis=1)
        top_categorias = df4.groupby('CATEGORIA')[
            'QUANTIDADE_CHAMADOS'].sum().nlargest(5).index
        df_top_categorias = df4[df4['CATEGORIA'].isin(top_categorias)]
        # Calculando o total de incidentes abertos para traçar uma linha
        df4_group = df.query(query).groupby(
            'MES_ABERTURA')['NUMERO_CHAMADO'].count().reset_index(name='QUANTIDADE_CHAMADOS')

        fig4 = px.line(df_top_categorias, y="QUANTIDADE_CHAMADOS",
                       x="MES_ABERTURA", color="CATEGORIA", hover_data={'PERCENTUAL': True})
        fig4.add_trace(go.Scatter(y=df4_group["QUANTIDADE_CHAMADOS"], x=df4_group["MES_ABERTURA"], mode='markers+lines',
                                  fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='Total de Incidentes Abertos'))

        # Atualizar titulos dos eixos
        fig4.update_layout(xaxis_title="Mês", yaxis_title="Quantidade de Chamados",
                           title=f"Os 5 Maiores Ofensores em {ano}")
        fig4.update_layout(main_config, height=220,
                           template=template)
        fig4.update_layout(margin=dict(l=20, r=10, t=30, b=30))
        fig4.update_layout(legend=dict(
            x=1.22,
            y=0.5,
            xanchor='right',
            yanchor='middle'
        ))
        fig4.update_layout(
            legend=dict(
                font=dict(
                    size=7  # Ajuste para um tamanho menor se necessário
                )
            ))
        return fig4
    else:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=[], y=[]))
        fig4.update_layout(
            title={'text': f'<span>Existem campos de filtragem em branco. </br></br></br>Favor selecionar ao menos uma opção por campo.</span>',
                   'y': 0.5,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle'},
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        fig4.update_layout(main_config, height=220,
                           template=template)
        return fig4


# graph 2: Top 5 Incidentes por ano


@app.callback(
    Output('graph2', 'figure'),
    Input('selec-call-type1', 'value'),
    Input('select-operator-group1', 'value'),
    Input('select-filia1', 'value'),
    Input('select-year3', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph2(opcoes_tipo_selecionadas, opcoes_operadores_selecionadas, filiais, ano, toggle):
    template = template_theme1 if toggle else template_theme2
    # print(f'Opções selecionadas: {opcoes_tipo_selecionadas}')
    # lista_query = [f'TIPO_CHAMADO == {item["value"]}' for item in opcoes_tipo_selecionadas if item['label']
    #               is not np.nan and item['value'] is not np.nan]
    query = '('
    for index, value in enumerate(opcoes_tipo_selecionadas):
        if index < len(opcoes_tipo_selecionadas) - 1:
            query += f'TIPO_CHAMADO == "{value}" or '
        else:
            query += f'TIPO_CHAMADO == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(opcoes_operadores_selecionadas):
        if index < len(opcoes_operadores_selecionadas) - 1:
            query += f'GRUPO_OPERADOR == "{value}" or '
        else:
            query += f'GRUPO_OPERADOR == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(filiais):
        if index < len(filiais) - 1:
            query += f'FILIAL_SOLICITANTE == "{value}" or '
        else:
            query += f'FILIAL_SOLICITANTE == "{value}")'
    query += f' and (ANO_ABERTURA == {ano})'
    # print(f'QUERY RESULTADO: {query}')

    if len(opcoes_tipo_selecionadas) > 0 and len(opcoes_operadores_selecionadas) > 0 and len(filiais) > 0:

        df7 = df.query(query).groupby(
            ['MES_ABERTURA', 'CATEGORIA']).size().reset_index(name="Quantidade de Chamados")
        df7_com_total = df7.groupby('CATEGORIA')[
            'Quantidade de Chamados'].sum().reset_index()
        df_top_categorias = df7_com_total.sort_values(
            by='Quantidade de Chamados', ascending=False).head(5)
        total_outros = df7_com_total.sort_values(by='Quantidade de Chamados', ascending=False).tail(
            len(df7_com_total)-5)['Quantidade de Chamados'].sum()
        nova_linha = pd.DataFrame(
            [{'CATEGORIA': 'Outros', 'Quantidade de Chamados': total_outros}])
        df_top_categorias = pd.concat(
            [df_top_categorias, nova_linha], ignore_index=True)
        fig7 = go.Figure()
        fig7.add_trace(go.Pie(
            labels=df_top_categorias['CATEGORIA'], values=df_top_categorias['Quantidade de Chamados'], hole=.7))
        fig7.update_layout(
            main_config, height=220, title=f'Top 5 Chamados em {ano}.', template=template)
        fig7.update_layout(margin=dict(l=30, r=30, t=30, b=30))
        fig7.update_layout(legend=dict(
            x=1.05,
            y=0.5,
            xanchor='left',
            yanchor='middle',
        ),
        )
        fig7.update_layout(
            legend=dict(
                font=dict(
                    size=7  # Ajuste para um tamanho menor se necessário
                )
            ))
        # fig7.show(config={'legend_clickable': True})
        return fig7
    else:
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=[], y=[]))
        fig7.update_layout(
            title={'text': f'<span>Existem campos de filtragem em branco. </br></br></br>Favor selecionar ao menos uma opção por campo.</span>',
                   'y': 0.5,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle'},
            xaxis={'visible': False},
            yaxis={'visible': False},
        )
        fig7.update_layout(
            main_config, height=220, template=template)
        return fig7
# Graph 3: Chamados Abertos por dia do Mês em 2024


@app.callback(
    Output('graph3', 'figure'),
    Input('selec-call-type1', 'value'),
    Input('select-operator-group1', 'value'),
    Input('select-filia1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph3(opcoes_tipo_selecionadas, opcoes_operadores_selecionadas, filiais, toggle):
    template = template_theme1 if toggle else template_theme2
    data_atual = datetime.now()

    query = '('
    for index, value in enumerate(opcoes_tipo_selecionadas):
        if index < len(opcoes_tipo_selecionadas) - 1:
            query += f'TIPO_CHAMADO == "{value}" or '
        else:
            query += f'TIPO_CHAMADO == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(opcoes_operadores_selecionadas):
        if index < len(opcoes_operadores_selecionadas) - 1:
            query += f'GRUPO_OPERADOR == "{value}" or '
        else:
            query += f'GRUPO_OPERADOR == "{value}")'

    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(filiais):
        if index < len(filiais) - 1:
            query += f'FILIAL_SOLICITANTE == "{value}" or '
        else:
            query += f'FILIAL_SOLICITANTE == "{value}")'

    if len(opcoes_tipo_selecionadas) > 0 and len(opcoes_operadores_selecionadas) > 0 and len(filiais) > 0:
        df5 = df.query(f'{query} and (ANO_ABERTURA == {data_atual.year} and MES_ABERTURA == {data_atual.month})').groupby('DIA_ABERTURA')[
            'NUMERO_CHAMADO'].size().reset_index(name='Quantidade de Chamados')
        fig5 = go.Figure(go.Scatter(
            x=df5['DIA_ABERTURA'], y=df5['Quantidade de Chamados'], mode='markers+lines', fill='tonexty'
        ))

        # Atualizar titulos dos eixos
        fig5.update_layout(xaxis_title="Dia",
                           yaxis_title="Quantidade de Chamados")
        '''
        fig5.add_annotation(text='Chamados Abertos por dia do Mês',
                            xref="paper", yref="paper",
                            font=dict(
                                size=15,
                                color='gray'
                            ),
                            align="center", bgcolor="rgba(0,0,0,0.8)",
                            x=0.05, y=0.55, showarrow=False)
                            '''
        fig5.add_annotation(text=f"Média: {round(df5['Quantidade de Chamados'].mean(), 2)}",
                            xref="paper", yref="paper",
                            font=dict(
                                size=20,
                                color='gray'
                            ),
                            align="center", bgcolor="rgba(0,0,0,0.8)",
                            x=0.05, y=0.35, showarrow=False)
        fig5.update_layout(main_config, height=220,
                           title=f"Chamados Abertos por dia do Mês em {data_atual.year}", template=template)
        fig5.update_layout(margin=dict(l=30, r=30, t=30, b=30))
        return fig5
    else:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=[], y=[]))
        fig5.update_layout(
            title={'text': f'<span>Existem campos de filtragem em branco. </br></br></br>Favor selecionar ao menos uma opção por campo.</span>',
                   'y': 0.5,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle'},
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        fig5.update_layout(main_config, height=220,
                           template=template)
        return fig5

# Graph 4: Média Diária nos meses de 2023


@app.callback(
    Output('graph4', 'figure'),
    Input('select-year3', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph4(ano, toggle):
    template = template_theme1 if toggle else template_theme2
    df6 = df.query(f'ANO_ABERTURA == {ano}').groupby('MES_ABERTURA')[
        'NUMERO_CHAMADO'].size().reset_index(name='Quantidade de Chamados')
    df6_group = df.query(f'ANO_ABERTURA == {ano}').groupby(['MES_ABERTURA', 'DIA_ABERTURA'])[
        'NUMERO_CHAMADO'].size().reset_index(name='Quantidade de Chamados')
    total_por_mes = df6_group.groupby('MES_ABERTURA')[
        'Quantidade de Chamados'].sum().reset_index()
    dias_por_mes = df6_group.groupby('MES_ABERTURA')[
        'DIA_ABERTURA'].size().reset_index()
    media_diaria = total_por_mes['Quantidade de Chamados'] / \
        dias_por_mes['DIA_ABERTURA']
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    fig6 = go.Figure(go.Scatter(x=meses, y=media_diaria,
                     mode='markers+lines', fill='tonexty', name='Média Diária'))
    # Atualizar titulos dos eixos
    fig6.update_layout(xaxis_title="Mês", yaxis_title="Média de Chamados",
                       title=f"Média diária por mês de chamados abertos em {ano}")
    '''
    fig6.add_annotation(text=f'Média Diária ao Mês {ano}',
                        xref="paper", yref="paper",
                        font=dict(
                            size=15,
                            color='gray'
                        ),
                        align="center", bgcolor="rgba(0,0,0,0.8)",
                        x=0.05, y=0.55, showarrow=False)
    '''
    fig6.add_annotation(text=f"Média Anual: {round(df6['Quantidade de Chamados'].mean(), 2)}",
                        xref="paper", yref="paper",
                        font=dict(
                            size=20,
                            color='gray'
                        ),
                        align="center", bgcolor="rgba(0,0,0,0.8)",
                        x=0.05, y=0.35, showarrow=False)
    fig6.update_layout(main_config, height=220,
                       title=f"Média Diária nos meses de {ano}", template=template)
    fig6.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    return fig6

# Graph 5: Top 5 Solicitantes por Chamados


@app.callback(
    Output('graph5', 'figure'),
    Input('select-year3', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph5(ano, toggle):
    template = template_theme1 if toggle else template_theme2
    df8 = df.query(f'ANO_ABERTURA == {ano}').groupby('SOLICITANTE')[
        'NUMERO_CHAMADO'].size().reset_index(name="Quantidade de Chamados")
    top_solicitante = df8.groupby('SOLICITANTE')[
        'Quantidade de Chamados'].sum().nlargest(5).index
    df_top_solicitante = df8[df8['SOLICITANTE'].isin(top_solicitante)]
    df_top_solicitante.sort_values(
        by='Quantidade de Chamados', ascending=True, inplace=True)

    fig8 = go.Figure(go.Bar(
        x=df_top_solicitante['Quantidade de Chamados'],
        y=df_top_solicitante['SOLICITANTE'],
        orientation='h',
        textposition='auto',
        text=df_top_solicitante['Quantidade de Chamados'],
        insidetextfont=dict(family='Times', size=12)))
    # Atualizar o layout para remover as grades
    fig8.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False),
        yaxis_title="Solicitante Name",
        title=f"Top 5 Solicitantes por Chamados em {ano}",

    )
    fig8.update_layout(main_config, height=220, template=template)
    fig8.update_layout(margin=dict(l=30, r=30, t=30, b=30))
    return fig8

# Graph 6: Percentual de chamados resolvidos por abertos por ano


@app.callback(
    Output('graph6', 'figure'),
    Input('select-year3', 'value'),
    Input('selec-call-type1', 'value'),
    Input('select-operator-group1', 'value'),
    Input('select-filia1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph6(ano, opcoes_tipo_selecionadas, opcoes_operadores_selecionadas, filiais, toggle):
    template = template_theme1 if toggle else template_theme2
    # ano = '2022'
    query = '('
    for index, value in enumerate(opcoes_tipo_selecionadas):
        if index < len(opcoes_tipo_selecionadas) - 1:
            query += f'TIPO_CHAMADO == "{value}" or '
        else:
            query += f'TIPO_CHAMADO == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(opcoes_operadores_selecionadas):
        if index < len(opcoes_operadores_selecionadas) - 1:
            query += f'GRUPO_OPERADOR == "{value}" or '
        else:
            query += f'GRUPO_OPERADOR == "{value}")'

    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(filiais):
        if index < len(filiais) - 1:
            query += f'FILIAL_SOLICITANTE == "{value}" or '
        else:
            query += f'FILIAL_SOLICITANTE == "{value}")'

    if len(opcoes_tipo_selecionadas) > 0 and len(opcoes_operadores_selecionadas) > 0 and len(filiais) > 0:
        df9_chamados_fechados = df.query(f'{query} and (ANO_FECHAMENTO == {ano})'
                                         ' and FECHADO == 1').groupby('MES_FECHAMENTO')['NUMERO_CHAMADO'].size()
        df9_chamados_abertos = df.query(
            f'{query} and (ANO_ABERTURA == {ano})').groupby('MES_ABERTURA')['NUMERO_CHAMADO'].size()
        total_chamados_abertos_periodo = df.query(
            f'{query} and (ANO_ABERTURA == {ano})').groupby('MES_ABERTURA')['NUMERO_CHAMADO'].size().sum()
        total_chamados_fechados_periodo = df.query(
            f'{query} and (ANO_FECHAMENTO == {ano}) and FECHADO == 1').groupby('MES_ABERTURA')['NUMERO_CHAMADO'].size().sum()
        percentual_fechados_periodo = "{:.2f}%".format(
            (total_chamados_fechados_periodo / total_chamados_abertos_periodo) * 100)
        print(f'Chamados: {df9_chamados_fechados.head()}')
        fig9 = go.Figure()
        # Adicionar as barras
        fig9.add_trace(go.Bar(
            x=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
            y=df9_chamados_fechados,
            name='Chamados Fechados',
            marker=dict(color='teal'),
            text=df9_chamados_fechados,  # Isso coloca a contagem em cima de cada barra
            # Posiciona o texto fora das barras (no topo)
            textposition='outside',
        ))

        # Adicionar a segunda barra (chamados abertos)
        fig9.add_trace(go.Scatter(
            x=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
            y=df9_chamados_abertos,
            name='Chamados Abertos',
            # marker=dict(color='orange'),
            # text=df9_chamados_abertos,
            # textposition='outside',
            mode='lines+markers',
            line=dict(color='orange'),
        ))

        # Adicionar a anotação de texto para a porcentagem de resolvidos
        fig9.add_annotation(
            x=0,  # Posição aproximada no meio do gráfico
            # Posiciona um pouco acima do valor mais alto
            y=max(df9_chamados_fechados) + 800,
            text=f"{percentual_fechados_periodo} dos chamados abertos foram resolvidos em {ano}.",
            showarrow=False,
            font=dict(size=16, color="white"),
            align="center",
            bgcolor="black",
            opacity=0.4
        )

        # Remover a legenda, se aplicável
        fig9.update_layout(showlegend=False)
        fig9.update_layout(main_config, height=220, template=template)
        fig9.update_layout(margin=dict(l=20, r=10, t=30, b=30))
        return fig9
    else:
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=[], y=[]))
        fig9.update_layout(
            title={'text': f'<span>Existem campos de filtragem em branco. </br></br></br>Favor selecionar ao menos uma opção por campo.</span>',
                   'y': 0.5,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle'},
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        fig9.update_layout(main_config, height=220,
                           template=template)
        return fig9
# Graph 7: Heatmap de horários com maior volume


@app.callback(
    Output('graph7', 'figure'),
    Input('select-year3', 'value'),
    Input('selec-call-type1', 'value'),
    Input('select-operator-group1', 'value'),
    Input('select-filia1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph7(ano, opcoes_tipo_selecionadas, opcoes_operadores_selecionadas, filiais, toggle):
    template = template_theme1 if toggle else template_theme2
    # ano = '2024'
    # df11 = df.query(f'ANO_ABERTURA == {ano}')
    query = '('
    for index, value in enumerate(opcoes_tipo_selecionadas):
        if index < len(opcoes_tipo_selecionadas) - 1:
            query += f'TIPO_CHAMADO == "{value}" or '
        else:
            query += f'TIPO_CHAMADO == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(opcoes_operadores_selecionadas):
        if index < len(opcoes_operadores_selecionadas) - 1:
            query += f'GRUPO_OPERADOR == "{value}" or '
        else:
            query += f'GRUPO_OPERADOR == "{value}")'

    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(filiais):
        if index < len(filiais) - 1:
            query += f'FILIAL_SOLICITANTE == "{value}" or '
        else:
            query += f'FILIAL_SOLICITANTE == "{value}")'

    query += f' and (ANO_ABERTURA == {ano})'

    if len(opcoes_tipo_selecionadas) > 0 and len(opcoes_operadores_selecionadas) > 0 and len(filiais) > 0:
        df11 = df.query(query)
        df11.DATA_ABERTURA = pd.to_datetime(
            df.DATA_ABERTURA, format='%Y-%m-%dT%H:%M:%S.%f%z')
        # Extrair o dia da semana (0 = segunda-feira, 6 = domingo)
        df11['DIA_DA_SEMANA'] = df11['DATA_ABERTURA'].dt.dayofweek
        # Crie um dicionário de mapeamento dos dias da semana
        dias_da_semana_map = {
            0: "Dom",
            1: "Seg",
            2: "Ter",
            3: "Qua",
            4: "Qui",
            5: "Sex",
            6: "Sáb"
        }
        df11['DIA_DA_SEMANA_NOME'] = df11['DIA_DA_SEMANA'].map(
            dias_da_semana_map)
        # Garanta a ordem correta dos dias da semana no DataFrame original
        df11['DIA_DA_SEMANA_NOME'] = pd.Categorical(df11['DIA_DA_SEMANA_NOME'],
                                                    categories=[
                                                        "Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
                                                    ordered=True)

        heatmap_data = df11.groupby(
            ['DIA_DA_SEMANA_NOME', 'HORA_ABERTURA']).size().unstack(fill_value=0)
        # Crie o heatmap com os dados processados
        fig = px.imshow(heatmap_data,
                        labels={"x": "Hora do Dia", "y": "Dia da Semana",
                                "color": "Contagem de Chamados"},
                        color_continuous_scale="Viridis")

        # Personalize o layout do gráfico
        fig.update_layout(
            title=f"Mapa de Calor dos horários com maior volume de chamados em {ano}",
            xaxis_title="Hora do Dia",
            yaxis_title="Dia da Semana",
            xaxis=dict(tickmode='array', tickvals=list(range(24)),
                       ticktext=[str(h) for h in range(24)]),
        )

        fig.update_layout(main_config, height=450, template=template)
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=50))

        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.update_layout(
            title={'text': f'<span>Existem campos de filtragem em branco. </br></br></br>Favor selecionar ao menos uma opção por campo.</span>',
                   'y': 0.5,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle'},
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        fig.update_layout(main_config, height=220,
                          template=template)
        return fig
# Graph 8: Top 5 Locais de atendimento entre as Top5 categorias de incidentes por filial


@app.callback(
    Output('graph8', 'figure'),
    Input('selec-call-type1', 'value'),
    Input('select-operator-group1', 'value'),
    Input('select-filia1', 'value'),
    Input('select-year3', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def graph8(opcoes_tipo_selecionadas, opcoes_operadores_selecionadas, filiais, ano, toggle):
    template = template_theme1 if toggle else template_theme2
    # ano = 2024
    query = '('
    for index, value in enumerate(opcoes_tipo_selecionadas):
        if index < len(opcoes_tipo_selecionadas) - 1:
            query += f'TIPO_CHAMADO == "{value}" or '
        else:
            query += f'TIPO_CHAMADO == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(opcoes_operadores_selecionadas):
        if index < len(opcoes_operadores_selecionadas) - 1:
            query += f'GRUPO_OPERADOR == "{value}" or '
        else:
            query += f'GRUPO_OPERADOR == "{value}")'
    if len(query) > 0 and len(opcoes_tipo_selecionadas) > 0 and len(filiais) > 0:
        query += ' and ('
    for index, value in enumerate(filiais):
        if index < len(filiais) - 1:
            query += f'FILIAL_SOLICITANTE == "{value}" or '
        else:
            query += f'FILIAL_SOLICITANTE == "{value}")'
    query += f' and (ANO_ABERTURA == {ano})'
    # print(f'QUERY RESULTADO: {query}')

    if len(opcoes_tipo_selecionadas) > 0 and len(opcoes_operadores_selecionadas) > 0 and len(filiais) > 0:
        df13 = df.query(query).groupby(
            ['NUMERO_CHAMADO', 'MES_ABERTURA', 'LOCAL_ATENDIMENTO', 'CATEGORIA']).size().reset_index(name='Quantidade de Chamados')
        top_categorias = df13.groupby(
            'CATEGORIA')['Quantidade de Chamados'].sum().nlargest(5).index
        # Filtrando o DataFrame para incluir apenas as top 5 categorias
        df_top5_categorias = df13[df13['CATEGORIA'].isin(top_categorias)]
        # Filtrando o DataFrame Resultante das top 5 categorias para incluir apenas os top 5 locais de atendimento
        top_locais = df_top5_categorias.groupby('LOCAL_ATENDIMENTO')[
            'Quantidade de Chamados'].sum().nlargest(5).index
        df_top5_locais = df_top5_categorias[df_top5_categorias['LOCAL_ATENDIMENTO'].isin(
            top_locais)]
        # Mes que deseja contultar
        mes = 1
        df_mes = df_top5_locais[df_top5_locais['MES_ABERTURA'] == mes]
        fig = px.bar(df_mes, x='CATEGORIA', y='Quantidade de Chamados', color='LOCAL_ATENDIMENTO',
                     title=f'Top 5 Locais de Atendimento Entre as Top 5 Categorias - {ano}')

        fig.update_layout(main_config, height=450, template=template)
        fig.update_layout(margin=dict(l=20, r=10, t=60, b=50))
        fig.update_layout(
            legend=dict(
                x=1.22,
                y=0.5,
                xanchor='right',
                yanchor='middle'
            )
        )
        fig.update_layout(legend=dict(font=dict(size=7)))

        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[]))
        fig.update_layout(
            title={'text': f'<span>Existem campos de filtragem em branco. </br></br></br>Favor selecionar ao menos uma opção por campo.</span>',
                   'y': 0.5,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'middle'},
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        fig.update_layout(main_config, height=220,
                          template=template)
        return fig


# Run server
if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
    # server.run(debug=False, host='0.0.0.0')
