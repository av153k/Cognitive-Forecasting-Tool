from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename
from flask_wtf import Form
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms.validators import DataRequired
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField,SelectField,TextField,
                     TextAreaField,SubmitField)

import numpy
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import Series
import pandas as pd
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
import warnings
from pmdarima import auto_arima

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime


warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config['SECRET_KEY'] = "3XXbpJ>ADP$wW"

colors = {
    'background': '#202020',
    'text': '#7FDBFF'
}

def create_plot():
    df = pd.read_csv('../../Data/Results.csv')
    data = [
        go.Scatter(
            x=df['timestamp'], # assign x as the dataframe column 'x'
            y=df['value'],
            mode = 'lines',

            marker = {
                'size': 12,
                'color': 'rgb(134,188,37)',
                'symbol': 'pentagon',
                'line': {'width': 2}
                }
        )
    ]

    layout = {
        'title' : 'Forecasted Price',
        'font': dict(size=12, color='#FFFFFF'),
        'hovermode' : 'closest',
        'plot_bgcolor' : colors['background'],
        'paper_bgcolor' : colors['background'],
        'width': 1100,
        'height': 450,
        'xaxis': dict(showgrid=False),
        'yaxis': dict(showgrid=False)}


    fig = go.Figure(data=data, layout=layout)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

class InfoForm(FlaskForm):
    commodity = SelectField(u'Select a Commodity', choices=[('sty', 'Styrene'), ('ben', 'Benzene'),('eth', 'Ethylene')])
    f_period = SelectField(u'Select a Forecast length', choices=[('1M', '1-Month'), ('2M', '2-Month'),('3M', '3-Month'),
    ('4M', '4-Month'), ('5M', '5-Month'),('6M', '6-Month')])

### ARIMAX 1-Month ###
################################################################################################################################################################
def process_data1():

    series = pd.read_excel('../../Data/Styrene-Net Industry Average 2010-2015.xlsx', header=0,
                           index_col=0, parse_dates=True)

    series.index.freq = 'MS'
    data = series.copy()

    actuals = pd.read_excel('../../Data/Styrene-Net Industry Average 2015-2018 Actuals.xlsx',
                            header=0, index_col=0, parse_dates=True)

    actuals.index.freq = 'MS'

    #Test ranges
    data = data['2010-01-01':]

    model = SARIMAX(np.log(data['Styrene']), order=(1,1,1), enforce_invertibility = False,
                    exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    #auto_arima(data['Styrene'], seasonal=True, m=12, enforce_invertibility = False, exog = data[['Oil_Lag']]).summary()

    preds = []

    for i in actuals.index:
        df = actuals.loc[i,:]
        df = pd.DataFrame(df).T
        yhat_log = model.forecast(steps = 1, exog = df[['Oil_Lag', 'Gas_Lag']])
        yhat = numpy.exp(yhat_log)
        preds.append(yhat)
        act = pd.Series(actuals.loc[i,:])
        act = pd.DataFrame(act).T
        data = pd.concat([data, act], axis = 0)
        model = SARIMAX(np.log(data['Styrene']), order=(1,1,1), enforce_invertibility = False,
                        exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    df = pd.DataFrame({'timestamp': [i.index for i in preds], 'value':[round(i[0],2) for i in preds]})
    df['timestamp'] = df.timestamp.apply(lambda x: str(x).split('[')[1].split(']')[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv('../../Data/Results.csv', index = False)

################################################################################################################################################################


### ARIMAX 2-Month ###
################################################################################################################################################################
def process_data2():
    series = pd.read_excel('../../Data/Styrene-Net Industry Average 2010-2015.xlsx', header=0,
                           index_col=0, parse_dates=True)
    series.index.freq = 'MS'

    data = series.copy()

    actuals = pd.read_excel('../../Data/Styrene-Net Industry Average 2015-2018 Actuals.xlsx',
                            header=0, index_col=0, parse_dates=True)

    actuals.index.freq = 'MS'

    #Test ranges
    data = data['2010-01-01':]

    model = SARIMAX(np.log(data['Styrene']), order=(1,1,1), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    #auto_arima(data['Styrene'], seasonal=True, m=12, enforce_invertibility = False, exog = data[['Oil_Lag']]).summary()

    preds = []

    for i in actuals.index:
        df = actuals.loc[i,:]
        df = pd.DataFrame(df).T
        fd = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd.set_index = i+1
        fd = pd.DataFrame(fd).T
        df = pd.concat([df, fd])
        yhat_log = model.forecast(steps = 2, exog = df[['Oil_Lag', 'Gas_Lag']])
        yhat_log = yhat_log[[1]]
        yhat = numpy.exp(yhat_log)
        preds.append(yhat)
        act = pd.Series(actuals.loc[i,:])
        act = pd.DataFrame(act).T
        data = pd.concat([data, act], axis = 0)
        model = SARIMAX(np.log(data['Styrene']), order=(1,1,1), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    df = pd.DataFrame({'timestamp': [i.index for i in preds], 'value':[round(i[0],2) for i in preds]})
    df['timestamp'] = df.timestamp.apply(lambda x: str(x).split('[')[1].split(']')[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv('../../Data/Results.csv', index = False)

################################################################################################################################################################


### ARIMAX 3-Month ###
################################################################################################################################################################
def process_data3():
    series = pd.read_excel('../../Data/Styrene-Net Industry Average 2010-2015.xlsx', header=0,
                           index_col=0, parse_dates=True)
    series.index.freq = 'MS'

    data = series.copy()

    actuals = pd.read_excel('../../Data/Styrene-Net Industry Average 2015-2018 Actuals.xlsx',
                            header=0, index_col=0, parse_dates=True)

    actuals.index.freq = 'MS'

    #Test ranges
    data = data['2010-01-01':]

    model = SARIMAX(np.log(data['Styrene']), order=(1,1,1), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    #auto_arima(data['Styrene'], seasonal=True, m=12, enforce_invertibility = False, exog = data[['Oil_Lag']]).summary()

    preds = []

    for i in actuals.index:
        df = actuals.loc[i,:]
        df = pd.DataFrame(df).T
        fd = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd.set_index = i+1
        fd = pd.DataFrame(fd).T

        fd2 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd2.set_index = i+2
        fd2 = pd.DataFrame(fd2).T

        df = pd.concat([df, fd, fd2])
        yhat_log = model.forecast(steps = 3, exog = df[['Oil_Lag', 'Gas_Lag']])
        yhat_log = yhat_log[[2]]
        yhat = numpy.exp(yhat_log)
        preds.append(yhat)
        act = pd.Series(actuals.loc[i,:])
        act = pd.DataFrame(act).T
        data = pd.concat([data, act], axis = 0)
        model = SARIMAX(np.log(data['Styrene']), order=(1,1,1), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    df = pd.DataFrame({'timestamp': [i.index for i in preds], 'value':[round(i[0],2) for i in preds]})
    df['timestamp'] = df.timestamp.apply(lambda x: str(x).split('[')[1].split(']')[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv('../../Data/Results.csv', index = False)

################################################################################################################################################################


### ARIMAX 4-Month ###
################################################################################################################################################################
def process_data4():
    series = pd.read_excel('../../Data/Styrene-Net Industry Average 2010-2015.xlsx', header=0,
                           index_col=0, parse_dates=True)
    series.index.freq = 'MS'

    data = series.copy()

    actuals = pd.read_excel('../../Data/Styrene-Net Industry Average 2015-2018 Actuals.xlsx',
                            header=0, index_col=0, parse_dates=True)

    actuals.index.freq = 'MS'

    #Test ranges
    data = data['2010-01-01':]

    model = SARIMAX(np.log(data['Styrene']), order=(1,1,2), seasonal_order=(0,0,1,12), enforce_invertibility = False,
                    exog = data[['Oil_Lag', 'Gas_Lag']]).fit()


    #auto_arima(data['Styrene'], seasonal=True, m=12, enforce_invertibility = False,
    #exog = data[['Oil_Lag', 'Gas_Lag']]).summary()


    preds = []

    for i in actuals.index:
        df = actuals.loc[i,:]
        df = pd.DataFrame(df).T
        fd = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd.set_index = i+1
        fd = pd.DataFrame(fd).T

        fd2 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd2.set_index = i+2
        fd2 = pd.DataFrame(fd2).T

        fd3 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd3.set_index = i+3
        fd3 = pd.DataFrame(fd3).T

        df = pd.concat([df, fd, fd2, fd3])
        yhat_log = model.forecast(steps = 4, exog = df[['Oil_Lag', 'Gas_Lag']])
        yhat_log = yhat_log[[3]]
        yhat = numpy.exp(yhat_log)
        preds.append(yhat)
        act = pd.Series(actuals.loc[i,:])
        act = pd.DataFrame(act).T
        data = pd.concat([data, act], axis = 0)

        model = SARIMAX(np.log(data['Styrene']), order=(1,1,2), seasonal_order=(0,0,1,12), enforce_invertibility = False,
                        exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    df = pd.DataFrame({'timestamp': [i.index for i in preds], 'value':[round(i[0],2) for i in preds]})
    df['timestamp'] = df.timestamp.apply(lambda x: str(x).split('[')[1].split(']')[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv('../../Data/Results.csv', index = False)

################################################################################################################################################################


### ARIMAX 5-Month ###
################################################################################################################################################################
def process_data5():
    series = pd.read_excel('../../Data/Styrene-Net Industry Average 2010-2015.xlsx', header=0,
                           index_col=0, parse_dates=True)
    series.index.freq = 'MS'

    data = series.copy()

    actuals = pd.read_excel('../../Data/Styrene-Net Industry Average 2015-2018 Actuals.xlsx',
                            header=0, index_col=0, parse_dates=True)

    actuals.index.freq = 'MS'

    #Test ranges
    data = data['2010-01-01':]

    model = SARIMAX(np.log(data['Styrene']), order=(1,1,2), seasonal_order=(0,0,1,12), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    #auto_arima(data['Styrene'], seasonal=True, m=12, enforce_invertibility = False, exog = data[['Oil_Lag']]).summary()

    preds = []

    for i in actuals.index:
        df = actuals.loc[i,:]
        df = pd.DataFrame(df).T
        fd = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd.set_index = i+1
        fd = pd.DataFrame(fd).T

        fd2 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd2.set_index = i+2
        fd2 = pd.DataFrame(fd2).T

        fd3 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd3.set_index = i+3
        fd3 = pd.DataFrame(fd3).T

        fd4 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd4.set_index = i+4
        fd4 = pd.DataFrame(fd4).T

        df = pd.concat([df, fd, fd2, fd3, fd4])
        yhat_log = model.forecast(steps = 5, exog = df[['Oil_Lag', 'Gas_Lag']])
        yhat_log = yhat_log[[4]]
        yhat = numpy.exp(yhat_log)
        preds.append(yhat)
        act = pd.Series(actuals.loc[i,:])
        act = pd.DataFrame(act).T
        data = pd.concat([data, act], axis = 0)
        model = SARIMAX(np.log(data['Styrene']), order=(1,1,2), seasonal_order=(0,0,1,12), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    df = pd.DataFrame({'timestamp': [i.index for i in preds], 'value':[round(i[0],2) for i in preds]})
    df['timestamp'] = df.timestamp.apply(lambda x: str(x).split('[')[1].split(']')[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv('../../Data/Results.csv', index = False)

################################################################################################################################################################


### ARIMAX 6-Month ###
################################################################################################################################################################
def process_data6():
    series = pd.read_excel('../../Data/Styrene-Net Industry Average 2010-2015.xlsx', header=0,
                           index_col=0, parse_dates=True)
    series.index.freq = 'MS'

    data = series.copy()

    actuals = pd.read_excel('../../Data/Styrene-Net Industry Average 2015-2018 Actuals.xlsx',
                            header=0, index_col=0, parse_dates=True)

    actuals.index.freq = 'MS'

    #Test ranges
    data = data['2010-01-01':]

    model = SARIMAX(np.log(data['Styrene']), order=(1,1,2), seasonal_order=(0,0,1,12), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    #auto_arima(data['Styrene'], seasonal=True, m=12, enforce_invertibility = False, exog = data[['Oil_Lag']]).summary()

    preds = []

    for i in actuals.index:
        df = actuals.loc[i,:]
        df = pd.DataFrame(df).T
        fd = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd.set_index = i+1
        fd = pd.DataFrame(fd).T

        fd2 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd2.set_index = i+2
        fd2 = pd.DataFrame(fd2).T

        fd3 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd3.set_index = i+3
        fd3 = pd.DataFrame(fd3).T

        fd4 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd4.set_index = i+4
        fd4 = pd.DataFrame(fd4).T

        fd5 = pd.DataFrame(data = [df['Oil_Lag'], df['Gas_Lag']])
        fd5.set_index = i+5
        fd5 = pd.DataFrame(fd5).T

        df = pd.concat([df, fd, fd2, fd3, fd4, fd5])
        yhat_log = model.forecast(steps = 6, exog = df[['Oil_Lag', 'Gas_Lag']])
        yhat_log = yhat_log[[5]]
        yhat = numpy.exp(yhat_log)
        preds.append(yhat)
        act = pd.Series(actuals.loc[i,:])
        act = pd.DataFrame(act).T
        data = pd.concat([data, act], axis = 0)
        model = SARIMAX(np.log(data['Styrene']), order=(1,1,2), seasonal_order=(0,0,1,12), enforce_invertibility = False, exog = data[['Oil_Lag', 'Gas_Lag']]).fit()

    df = pd.DataFrame({'timestamp': [i.index for i in preds], 'value':[round(i[0],2) for i in preds]})
    df['timestamp'] = df.timestamp.apply(lambda x: str(x).split('[')[1].split(']')[0])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv('../../Data/Results.csv', index = False)

################################################################################################################################################################


@app.route('/', methods =['POST', 'GET', 'DELETE'])
def get_files():

    form = InfoForm()
    if request.method == 'POST':
        if request.form['btn_identifier'] == 'client_id_btn':
            session['commodity'] = form.commodity.data
            session['f_period'] = form.f_period.data

            if session['f_period'] == '1M':
                process_data1()

            elif session['f_period'] == '2M':
                process_data2()

            elif session['f_period'] == '3M':
                process_data3()

            elif session['f_period'] == '4M':
                process_data4()

            elif session['f_period'] == '5M':
                process_data5()

            elif session['f_period'] == '6M':
                process_data6()

            bar = create_plot()

            input = pd.read_csv('../../Data/Results.csv')
            input.set_index(['timestamp','value'], inplace=True)
            input = input.T

            return render_template('chart1.html', plot=bar, data = input.to_html(index=False, classes = 'table table-striped table-dark'))

        #return redirect("http://www.example.com")

    return render_template('index.html', form = form) #circulation view

####################

if __name__ == '__main__':
    app.run(debug=True)
