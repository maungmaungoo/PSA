from dotenv import load_dotenv
import os
from wordcloud import WordCloud
import base64
from io import BytesIO
import plotly.graph_objects as go
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
from datetime import date, datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import connectorx as cx

# Database
db = "sqlite://tweets.db"
table_columns = ['ID', 'Date/Time', 'Username', 'Tweet', 'Tweet ID', 'Location', 'Latitude', 'Longitude', 'Hashtag', 'Sentiment']
database_columns = ['id', 'date', 'screen_name', 'tweet', 'tweet_id', 'location', 'latitude', 'longitude', 'hashtags', 'sentiment']
combined_columns = dict(zip(database_columns, table_columns))

# Mapbox
load_dotenv()
MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN')

# PLT
plt.style.use('fivethirtyeight')

now = datetime.now()
end_date_year = now.year + 1
separator = ', '

def select_all_sentiments():
    """
    Query sentiments in the tweets table
    :return: Sentiments from database
    """
    sql = "SELECT sentiment FROM tweets"
    sentiments = cx.read_sql(db, sql)
    return sentiments

def select_date_sentiments():
    """
    Query date and sentiment in the tweets table
    :return: Dates and Sentiments from database
    """
    sql = "SELECT date, sentiment FROM tweets"
    results = cx.read_sql(db, sql)
    return results

def select_sentiment(search_string):
    """
    Query column(s) from `search_string` in the tweets table
    :param search_string: column name(s) to Query 
    :return:
    """
    sql = "SELECT {} FROM tweets".format(search_string)
    results = cx.read_sql(db, sql)
    return results

def get_dates():
    """
    Query date in the tweets table
    :return:
    """
    sql = "SELECT date FROM tweets"
    dates = cx.read_sql(db, sql)
    return dates

def select_location():
    """
    Query date, location and sentiment in the tweets table
    :return: Date, Location, Latitude, Longitude and Sentiment from database
    """
    sql = "SELECT date, location, latitude, longitude, sentiment FROM tweets"
    results = cx.read_sql(db, sql)
    return results

def select_sentiment_by_date(search_string, search_date):
    """
    Query column(s) from `search_string` in the tweets table where date LIKE `search_date`
    :param search_string: column name(s) to Query
    :param search_date: date to search
    :return: Query results from search_string and search_date
    """
    sql = "SELECT {} FROM tweets WHERE date IN ({})".format(search_string, search_date)
    sentiments = cx.read_sql(db, sql)
    return sentiments

# Initialise the app
app = dash.Dash(__name__, update_title=None)

# Define the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Get dates
def get_complete_date(sdate, edate):
    sdate = pd.to_datetime(sdate)
    edate = pd.to_datetime(edate)
    complete_dates = pd.date_range(sdate.date(),edate.date(),freq='d')
    complete_dates = complete_dates.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
    return complete_dates

# Get max and min dates
def get_max_min_date():
    dates = get_dates()
    dates = dates['date'].to_list()
    max_date = max(dates)
    min_date = min(dates)
    complete_dates = get_complete_date(min_date, max_date)
    date_from_db = []
    for date in dates:
        if date not in date_from_db:
            date_from_db.append(date)
    date_from_db = set(date_from_db)
    complete_dates = set(complete_dates)
    disable_dates = date_from_db.union(complete_dates)  - date_from_db.intersection(complete_dates)
    disable_dates = list(disable_dates)
    return [max_date, min_date, disable_dates]

max_min_date = get_max_min_date()
max_date = max_min_date[0]
min_date = max_min_date[1]
disable_dates = max_min_date[2]

nav_bar = children = [
                    html.H2('Dash - SENTIMENT ANALYSIS'),
                    html.Div(
                        children=[
                            html.A(children='Go to index', href='/'),
                            html.Br(),
                            html.A(children='View pie chart', href='/pie'),
                            html.Br(),
                            html.A(children='View pie chart by date', href='/piebydate'),
                            html.Br(),
                            html.A(children='View sentiment by date', href='/sentimentbydate'),
                            html.Br(),
                            html.A(children='View data table', href='/datatable'),
                            html.Br(),
                            html.A(children='Download information', href='/download'),
                            html.Br(),
                            html.A(children='View Map in real time', href='/map'),
                            html.Br(),
                            html.A(children='View Map by date', href='/map_by_date'),
                            html.Br(),
                            html.A(children='View Wordcloud', href='/wordcloud'),
                        ],
                        className="navbar",
                    ),
                    html.Div(
                        children=[
                            html.P("Made with"),
                            html.P("♥", id="love")
                        ],
                        className='low',
                    ),
                ]

pie_chart = children = [
                    html.A(children='Live Pie chart ⬈', href='/pie', className='redirect_links'),
                    dcc.Graph(
                        id='donut-chart',
                        style={"height": "80%", "width": "80%", "margin-top": "8vh", "font-weight": "bold"},
                    ),
                    dcc.Interval(id="donut-trigger", interval=1000, n_intervals=0)
                ]

pie_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',    
                ),  # Define the left element
                html.Div(
                    pie_chart,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(output=Output('donut-chart', 'figure'),
              inputs=[Input('donut-trigger', 'n_intervals')])
def update_graph_live(n):
    # Load data 
    tweets = select_all_sentiments()
    
    counts = tweets['sentiment'].value_counts().to_dict()

    labels = ['POSITIVE','NEGATIVE','NEUTRAL']
    values = [counts["POSITIVE"], counts["NEGATIVE"], counts["NEUTRAL"]]

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', hole=.3, marker={ "colors" : ['green','red','grey']})])
    fig.update_layout(uniformtext_minsize=16, uniformtext_mode='hide', plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)', font={"color": "white"})
    return fig

pie_by_date_chart = children = [
                    html.A(children='Pie chart by date ⬈', href='/piebydate', className='redirect_links'),
                    html.P('Select Date: '),
                    dcc.DatePickerRange(
                        id='datepicker-pie',
                        day_size = 50,
                        reopen_calendar_on_clear=True,
                        clearable = True,
                        min_date_allowed = min_date,
                        max_date_allowed = max_date,
                        start_date = min_date,
                        end_date = max_date,
                        disabled_days=disable_dates,
                        minimum_nights = 0,
                        updatemode = 'singledate',
                        className = 'dcc_compon',
                    ),
                    dcc.Graph(
                        id='donut-chart-by-date',
                        style={"height": "80%", "width": "80%", "margin-top": "1vh", "font-weight": "bold"},
                    ),
                ]

pie_by_date_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    pie_by_date_chart,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(Output('donut-chart-by-date', 'figure'),
              [Input('datepicker-pie', 'start_date'), Input('datepicker-pie', 'end_date')])
def update_graph_by_date(start_date, end_date):
    complete_date = get_complete_date(start_date, end_date)
    search_date =  separator.join(["\'"+value+"\'" for value in complete_date])
    search_string = 'sentiment'
    tweets = select_sentiment_by_date(search_string, search_date)
    counts = tweets['sentiment'].value_counts().to_dict()
    labels = ['POSITIVE','NEGATIVE','NEUTRAL']
    values = [counts["POSITIVE"], counts["NEGATIVE"], counts["NEUTRAL"]]

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', hole=.3, marker={ "colors" : ['green','red','grey']})])
    fig.update_layout(uniformtext_minsize=16, uniformtext_mode='hide', plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)', font={"color": "white"})
    return fig

sentiment_by_date_layout_main = children = [
                    html.A(children='Sentiment by date ⬈', href='/sentimentbydate', className='redirect_links'),
                    html.P('Select Date: '),
                    dcc.DatePickerRange(
                        id='date_range',
                        day_size = 50,
                        reopen_calendar_on_clear=True,
                        clearable = True,
                        min_date_allowed = min_date,
                        max_date_allowed = max_date,
                        start_date = min_date,
                        end_date = max_date,
                        disabled_days=disable_dates,
                        minimum_nights = 0,
                        updatemode = 'singledate',
                        className = 'dcc_compon',
                        style={'backgroundColor': '#31302F', 'margin-top': '1vh'},
                    ),
                    dcc.Graph(
                        id='date-chart',
                        style={"height": "55%", "margin-top": "8vh", "font-weight": "bold"}
                    ),   
                ]

sentiment_by_date_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    sentiment_by_date_layout_main,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(Output('date-chart', 'figure'),
              [Input('date_range', 'start_date')], [Input('date_range', 'end_date')])
def date_chart(start_date, end_date):
    tweets = select_date_sentiments()
    df = tweets.groupby(["date","sentiment"]).size().reset_index().rename(columns={0:'count'})
    df = pd.pivot_table(df, values='count', index=["date"], columns=['sentiment'])
    df.columns = ['positive', 'negative', 'neutral']
    df = df.reset_index()
    fig = px.line(df, x="date", y=df.columns,
                hover_data={"date": "|%B %d, %Y"},
                title='Sentiment data per month', template='plotly_dark')
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    fig.update_layout(uniformtext_minsize=16, uniformtext_mode='hide', plot_bgcolor='rgba(0, 0, 0, 0)',
                            paper_bgcolor='rgba(0, 0, 0, 0)', font={"color": "white"})
    return fig

specific_columns = ['ID', 'Date/Time', 'Username', 'Tweet', 'Hashtag', 'Sentiment']
data_table_layout_main = children = [
                    html.A(children='Data Table ⬈', href='/datatable', className='redirect_links'),
                    html.P('Select Date: '),
                    dcc.DatePickerRange(
                        id='datepicker-table',
                        day_size = 50,
                        reopen_calendar_on_clear=True,
                        clearable = True,
                        min_date_allowed = min_date,
                        max_date_allowed = max_date,
                        start_date = max_date,
                        end_date = max_date,
                        disabled_days=disable_dates,
                        minimum_nights = 0,
                        updatemode = 'singledate',
                        className = 'dcc_compon',
                    ),
                    dash_table.DataTable(
                        id='data-table',
                        columns =[{'name':'{}'.format(col), 'id': '{}'.format(col)} for col in specific_columns],
                        page_size = 20,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                        style_cell={
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white',
                            'textAlign': 'left',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            # 'minWidth': '80px', 'width': '80px', 'maxWidth': '80px',
                        },
                        style_table={
                            'margin-top':'3vh',
                            'height': '65vh',
                            'overflowY': 'auto',
                            'overflowX': 'auto'
                        },
                    ),
                ]

data_table_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    data_table_layout_main,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(Output('data-table', 'data'),
              [Input('datepicker-table', 'start_date')],
              [Input('datepicker-table', 'end_date')])
def data_table_update(start_date, end_date):
    complete_date = get_complete_date(start_date, end_date)
    search_date = separator.join(["\'"+value+"\'" for value in complete_date])
    search_string = 'id, date, screen_name, tweet, hashtags, sentiment'
    tweets = select_sentiment_by_date(search_string, search_date)
    data = []
    for index, row in tweets.iterrows():
        data.append({str(specific_columns[0]):str(row['id']),specific_columns[1]:row['date'],
                    specific_columns[2]:row['screen_name'],specific_columns[3]:row['tweet'],
                    specific_columns[4]:row['hashtags'],specific_columns[5]:row['sentiment']})
    return data

download_layout_main = children = [
                    html.A(children='Download Information ⬈', href='/download', className='redirect_links'),
                    dcc.DatePickerRange(
                        id='datepicker-download',
                        day_size = 50,
                        reopen_calendar_on_clear=True,
                        clearable = True,
                        min_date_allowed = min_date,
                        max_date_allowed = max_date,
                        start_date = min_date,
                        end_date = max_date,
                        disabled_days=disable_dates,
                        minimum_nights = 0,
                        updatemode = 'singledate',
                        className = 'dcc_compon',
                    ),
                    dcc.Checklist(
                        id="all-or-none",
                        options=[{"label": "Select All", "value": "All"}],
                        value=[],
                        labelStyle={"display": "inline-block"},
                    ),
                    html.Br(),
                    dcc.Checklist(
                        id='download-checklist',
                        options=[
                            {'label': value, 'value': key} for key,value in combined_columns.items()
                        ],
                        value=[],
                    ),
                    html.Br(),
                    html.Button('Create Download Link', id='download-button', n_clicks=0, style={'width':'16vw', 'color':'green'}),
                    html.Br(),
                    dcc.Loading(
                        id='loading',
                        type='default',
                        children=html.Div(
                            id="download-area",
                            className="block",
                            children=[]
                        )
                    ),
                ]

download_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    download_layout_main,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(
    Output("download-checklist", "value"),
    [Input("all-or-none", "value")],
    [State("download-checklist", "options")],
)
def select_all_none(all_selected, options):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none

def build_download_button(uri_xlsx, uri_csv):
    """Generates a download button for the resource"""
    buttons = html.Div(
        children=[
            html.Form(
                action=uri_xlsx,
                method="get",
                children=[
                    html.Button(
                        className="button",
                        type="submit",
                        children=[
                            "download excel file"
                        ],
                        style={'width':'16vw', 'color':'green'}
                    )
                ]
            ),
            html.Br(),
            html.Form(
                action=uri_csv,
                method="get",
                children=[
                    html.Button(
                        className="button",
                        type="submit",
                        children=[
                            "download csv file"
                        ],
                        style={'width':'16vw', 'color':'green'}
                    )
                ]
            )
        ]
    )
    return buttons

@app.callback(
    Output("download-area", "children"),
    [Input("download-button", "n_clicks")],
    [Input('datepicker-download', 'start_date')],
    [Input('datepicker-download', 'end_date')],
    [State("download-checklist", "value")]
)
def generate_xlsx_csv_files(n_clicks, start_date, end_date, values):
    if n_clicks > 0:
        if values == []:
            return
        else:
            search_string = ', '.join([value for value in values])
            complete_date = get_complete_date(start_date, end_date)
            search_date = separator.join(["\'"+value+"\'" for value in complete_date])
            tweets = select_sentiment_by_date(search_string, search_date)
            tweets = pd.DataFrame(tweets, columns=[value for key,value in combined_columns.items() if key in values])
            today = now.strftime("%Y%m%dT%H%M%S")
            with pd.ExcelWriter(f'static/{today}.xlsx') as writer:
                tweets.to_excel(writer, sheet_name='Twitter Sentiment Analysis', index=False)
            uri_xlsx = f'static/{today}.xlsx'
            tweets.to_csv(f'static/{today}.csv', index=False)
            uri_csv = f'static/{today}.csv'
            return [build_download_button(uri_xlsx, uri_csv)]

def get_color_code(p_and_n):
    res = []
    for pn in p_and_n:
        if pn[0] == pn[1]:
            res.append(0.5)
        elif pn[0] > pn[1]:
            res.append(1)
        else:
           res.append(0)
    return res

map_layout_main = children = [
                    html.A(children='Geographic Information in real time ⬈', href='/map', className='redirect_links'),
                    dcc.Graph(
                        id='map',
                        config={'displayModeBar': False},
                        style={"height": "80%", "width": "80%", "font-weight": "bold"}
                    ),
                    dcc.Interval(id="map-trigger", interval=5*1000, n_intervals=0),
                ]

map_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    map_layout_main,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(Output('map', 'figure'),
              Input('map-trigger', 'n_intervals'))
def update_map_live(n):
    tweets = select_location()
    df = pd.DataFrame(tweets, columns=["date","location","latitude","longitude","sentiment"])
    df = df.groupby(["date","location","latitude","longitude","sentiment"]).size().reset_index().rename(columns={0:'count'})
    position = df[["location","latitude","longitude"]].drop_duplicates(['location']).set_index(["location"])
    df = pd.pivot_table(df, values='count', index=["date","location"], columns=['sentiment'])
    df.columns = ['positive', 'negative', 'neutral']
    df = df.join(position)
    df = df.fillna(0)
    df['size'] = df['neutral'].apply(lambda x: (np.sqrt(x/100) + 1) if x > 500 else (np.log(x) / 2 + 1)).replace(np.NINF, 0)
    df['size'] = df['size'] * 5
    p_and_n = list(zip(df['positive'].tolist(),df['negative'].tolist()))
    res = get_color_code(p_and_n)
    df['color'] = res
    day = datetime.today().strftime('%Y-%m-%d')
    tmp = df.xs(day)
    # Create the figure and feed it all the prepared columns
    fig = go.Figure(
        go.Scattermapbox(
            lat=tmp['latitude'],
            lon=tmp['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=tmp['size'],
                color=tmp['color'],
                colorscale=[[0,'rgb(255,0,0)'],[1,'rgb(0,255,0)']],
            ),
            customdata=np.stack(
                (pd.Series(tmp.index),
                tmp['positive'],
                tmp['negative'],
                tmp['neutral']),
                axis=-1
            ),
            hovertemplate="""
                <extra></extra>
                <em>%{customdata[0]}</em><br>
                POSITIVE  %{customdata[1]}<br>
                NEGATIVE  %{customdata[2]}<br>
                NEUTRAL  %{customdata[3]}
            """,
        )
    )

    # Specify layout information
    fig.update_layout(
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN, #
            # center=go.layout.mapbox.Center(lat=45, lon=-73),
            # zoom=1
        ),
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
    )
    return fig

def update_map_by_date(_start_date, _end_date):
    global max_date
    global min_date
    global disable_dates
    max_min_date = get_max_min_date()
    max_date = max_min_date[0]
    min_date = max_min_date[1]
    disable_dates = max_min_date[2]

    tweets = select_location()

    df = pd.DataFrame(tweets, columns=["date","location","latitude","longitude","sentiment"])
    df = df.groupby(["date","location","latitude","longitude","sentiment"]).size().reset_index().rename(columns={0:'count'})
    position = df[["location","latitude","longitude"]].drop_duplicates(['location']).set_index(["location"])
    df = pd.pivot_table(df, values='count', index=["date","location"], columns=['sentiment'])
    df.columns = ['positive', 'negative', 'neutral']
    df = df.join(position)
    df = df.fillna(0)
    df['size'] = df['neutral'].apply(lambda x: (np.sqrt(x/100) + 1) if x > 500 else (np.log(x) / 2 + 1)).replace(np.NINF, 0)
    df['size'] = df['size'] * 5
    p_and_n = list(zip(df['positive'].tolist(),df['negative'].tolist()))
    res = get_color_code(p_and_n)
    df['color'] = res
    start_day = _start_date
    end_day = _end_date
    complete_date = get_complete_date(start_day, end_day)
    m = df.index.get_level_values('date').isin(complete_date)
    df = df[m].reset_index(level=0, drop=True)
    tmp = df
    # Create the figure and feed it all the prepared columns
    fig = go.Figure(
        go.Scattermapbox(
            lat=tmp['latitude'],
            lon=tmp['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=tmp['size'],
                color=tmp['color'],
                colorscale=[[0,'rgb(255,0,0)'],[1,'rgb(0,255,0)']],
            ),
            customdata=np.stack(
                (pd.Series(tmp.index),
                tmp['positive'],
                tmp['negative'],
                tmp['neutral']),
                axis=-1
            ),
            hovertemplate="""
                <extra></extra>
                <em>%{customdata[0]}</em><br>
                POSITIVE  %{customdata[1]}<br>
                NEGATIVE  %{customdata[2]}<br>
                NEUTRAL  %{customdata[3]}
            """,
        )
    )

    # Specify layout information
    fig.update_layout(
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN, #
            # center=go.layout.mapbox.Center(lat=45, lon=-73),
            # zoom=1
        ),
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
    )
    return fig

map_by_date_layout_main = children = [
                    html.A(children='Geographic Information by date ⬈', href='/map_by_date', className='redirect_links'),
                    html.P('Select date: '),
                    dcc.DatePickerRange(
                        id='datepicker-map',
                        day_size = 50,
                        reopen_calendar_on_clear=True,
                        clearable = True,
                        min_date_allowed = min_date,
                        max_date_allowed = max_date,
                        start_date = min_date,
                        end_date = max_date,
                        disabled_days=disable_dates,
                        minimum_nights = 0,
                        updatemode = 'singledate',
                        className = 'dcc_compon',
                        style={'backgroundColor': '#31302F', 'margin-top': '1vh'},
                    ),
                    dcc.Graph(
                        figure=update_map_by_date(min_date, max_date),
                        id='map-by-date',
                        config={'displayModeBar': False},
                        style={"height": "80%", "width": "80%", "font-weight": "bold"}
                    ),
                ]

map_by_date_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    map_by_date_layout_main,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

@app.callback(Output('map-by-date', 'figure'),
                [Input('datepicker-map', 'start_date')],
                [Input('datepicker-map', 'end_date')])
def change_map_by_date(start_date, end_date):
    return update_map_by_date(start_date, end_date)

wordcloud_layout_main = children = [
                    html.A(children='Wordcloud of hashtags ⬈', href='/wordcloud', className='redirect_links'),
                    html.P("Wordcloud Image:"),
                    html.Img(
                        id='wordcloud',
                        style={"height": "55%", "width": "80%", "margin-top": "1vh", "margin-left": "6vw", "font-weight": "bold"}
                    ),
                    html.P('Top Hashtags: '),
                    html.Div(
                        id="hashtags_links",
                    ),
                    dcc.Interval(id="wordcloud-trigger", interval=5*1000, n_intervals=0)
                ]

wordcloud_layout = html.Div(
    children=[
        html.Div(className='row',
            children=[
                html.Div(
                    nav_bar,
                    className='three columns div-user-controls',
                ),  # Define the left element
                html.Div(
                    wordcloud_layout_main,
                    className='nine columns div-for-charts bg-grey',
                )  # Define the right element
            ]
        )
    ]
)

def generate_html_a(word):
    buttons = html.Div(
        children=[
            html.A(
                className="hashtag_link",
                children=[
                    html.Button(
                        className="hashtag_button",
                        children=[word],
                        style={'width':'auto', 'color':'green', 'margin':'5px'},
                    ),
                ],
                target='_blank',
                href='https://twitter.com/search?q=%23'+word,
            )
        ],
        style={'float':'left'}
    )
    return buttons

@app.callback(Output('hashtags_links', 'children'),
                Input('wordcloud-trigger', 'n_intervals'))
def generate_hashtag_links(n):
    search_string = 'hashtags'
    tweets = select_sentiment(search_string)
    tweets = tweets[search_string].to_list()
    words = [splited.replace(" ", "") for tweet in tweets if tweet for splited in tweet.split(',')]
    cnt = Counter()
    for word in words:
        cnt[word] += 1
    words = [pair[0] for pair in sorted(dict(cnt).items(), reverse=True, key=lambda item: item[1])]
    return [generate_html_a(word) for word in words[0:40]]

@app.callback(Output('wordcloud', 'src'),
              Input('wordcloud-trigger', 'n_intervals'))
def generate_wordcloud(n):
    search_string = 'hashtags'
    tweets= select_sentiment(search_string)
    tweets = tweets[search_string].to_list()
    words = [tweet for tweet in tweets if tweet]
    words = pd.Series(words).str.cat(sep=' ')
    
    wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='#1E1E1E').generate(words)
    wordcloud_img = wordcloud.to_image()
    with BytesIO() as buffer:
        wordcloud_img.save(buffer, 'png')
        w_img = base64.b64encode(buffer.getvalue()).decode()
    img_uri = "data:image/png;base64," + w_img
    return img_uri

index_page = html.Div(
    children=[
        html.Div(
            children=[
                html.H1("EXAMPLE TEXT"),
                html.P("This defines the alignment along the main axis. It helps distribute extra free space leftover when either all the flex items on a line are inflexible, or are flexible but have reached their maximum size. It also exerts some control over the alignment of items when they overflow the line."),
            ],
            className='top',
        ),
        html.Div(
            pie_chart,
            className='div1',
        ),
        html.Div(
            sentiment_by_date_layout_main,
            className='div2',
        ),
        html.Div(
            pie_by_date_chart,
            className='div3',
        ), 
        html.Div(
            data_table_layout_main,
            className='div4',
        ),
        html.Div(
            map_layout_main,
            className='div5',
        ),
        html.Div(
            map_by_date_layout_main,
            className='div6',
        ),
        html.Div(
            wordcloud_layout_main,
            className='div7',
        ),
        html.Div(
            children=[
                html.P("Made with"),
                html.P("♥", id="love")
            ],
            className='low',
        ),
    ],
    className='parent',
)

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/pie':
        return pie_layout
    elif pathname == '/piebydate':
        return pie_by_date_layout
    elif pathname == '/sentimentbydate':
        return sentiment_by_date_layout
    elif pathname == '/datatable':
        return data_table_layout
    elif pathname == '/download':
        return download_layout
    elif pathname == '/map':
        return map_layout
    elif pathname == '/map_by_date':
        return map_by_date_layout
    elif pathname == '/wordcloud':
        return wordcloud_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1',debug=True)