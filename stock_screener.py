import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import yfinance as yf
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

app = dash.Dash(__name__)

def fetch_financial_data(ticker):
 data = yf.download(ticker, start='2021-01-01', end='2023-01-01')
 data = data.reset_index()
 data = data[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
 return data

def preprocess_data(data):
 data = data.dropna()
 scaler = MinMaxScaler()
 data[['Open', 'Close', 'High', 'Low', 'Volume']] = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low', 'Volume']])
 return data

def generate_stock_plot(data):
 candlestick = go.Candlestick(
 x=data['Date'],
 open=data['Open'],
 close=data['Close'],
 high=data['High'],
 low=data['Low'],
 name='Candlestick'
 )
 layout = go.Layout(title='Stock Price Analysis', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
 fig = go.Figure(data=[candlestick], layout=layout)
 return fig

def predict_stock_prices(data, ticker):
 X = data.drop(['Date', 'Close'], axis=1)
 y = data['Close']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 model = LinearRegression()
 model.fit(X_train, y_train)

 train_predictions = model.predict(X_train)
 train_rmse = mean_squared_error(y_train, train_predictions, squared=False)

 print(f"Train RMSE: {train_rmse}")
 test_predictions = model.predict(X_test)
 test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

 print(f"Test RMSE: {test_rmse}")
 future_dates = pd.date_range(start='2023-01-02', end='2023-02-01')
 future_data = pd.DataFrame({'Date': future_dates})
 future_data = fetch_financial_data(ticker)
 future_data = preprocess_data(future_data)
 future_predictions = model.predict(future_data.drop(['Date', 'Close'], axis=1))

 return future_dates, future_predictions

app.layout = html.Div(
 children=[
 html.H1("Stock Price Analysis"),
 dcc.Input(id="ticker-input", placeholder="Enter ticker symbol", type="text"),
 html.Button("Submit", id="submit-button", n_clicks=0), dcc.Graph(id="stock-graph")
 ]
)

@app.callback(
 Output("stock-graph", "figure"),
 [Input("submit-button", "n_clicks")],
 [State("ticker-input", "value")]
)

def update_stock_graph(n_clicks, ticker):
 if n_clicks > 0 and ticker:
    data = fetch_financial_data(ticker)
    data = preprocess_data(data)
    fig = generate_stock_plot(data)
    future_dates, future_predictions = predict_stock_prices(data, ticker)
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name="Predicted Close"))
    return fig
 return go.Figure()

if __name__ == "__main__":
 app.run_server(debug=True)
