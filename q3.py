import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# load data
stock_prices=pd.read_csv("data_challenge_stock_prices.csv")
index_prices=pd.read_csv("data_challenge_index_prices.csv")

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
stock_prices_imputed = imputer.fit_transform(stock_prices)
index_prices_imputed = imputer.fit_transform(index_prices)
# partition stocks into sectors
#  M = 4 
sectors = np.random.randint(0, 4, size=100)

# loop over sectors
for sector in range(4):
    # create subset of stock prices for stocks in this sector
    sector_prices = stock_prices_imputed[:, sectors == sector]
    sector_prices = pd.DataFrame(sector_prices, index=stock_prices_imputed.index, columns=stock_prices_imputed.columns[sectors == sector])
    
    # compute returns for each stock in this sector
    sector_returns = (sector_prices.shift(-1) - sector_prices) / sector_prices
    index_prices_1 = pd.DataFrame(index_prices_imputed, index=index_prices_imputed.index, columns=index_prices_imputed.columns[sectors == sector])
    # loop over indices
    for i in range(15):
        # fit regression model
        X = sector_returns.values
        y = index_prices_1.iloc[:, i].values
        model = LinearRegression().fit(X, y)
        
        # evaluate model performance
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        if r2 >= 0.4:
            print(f'Index {i} is a function of stocks in sector {sector} with R^2={r2:.2f}')


#We couldn't get the exact solution here but this iis the stuff that we tried.