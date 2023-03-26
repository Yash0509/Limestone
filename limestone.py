import pandas as pd
stock_prices=pd.read_csv("data_challenge_stock_prices.csv")
index_prices=pd.read_csv("data_challenge_index_prices.csv")
stock_returns=stock_prices.pct_change()*10000
index_returns=index_prices.pct_change()*10000
# print(stock_returns)
