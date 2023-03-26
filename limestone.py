import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

stock_prices=pd.read_csv("data_challenge_stock_prices.csv")
index_prices=pd.read_csv("data_challenge_index_prices.csv")
stock_returns=stock_prices.pct_change()*10000
index_returns=index_prices.pct_change()*10000
#print(index_returns)

fig, axs = plt.subplots(10, 10, figsize=(20, 20))
for i in range(100):
    axs[i//10, i%10].hist(stock_prices.iloc[i], bins=20)
plt.show()

# Use k-means clustering to group the stocks based on their returns
kmeans = KMeans(n_clusters=11, random_state=0).fit(stock_prices.T)
stock_clusters = kmeans.labels_

# Examine the clusters to identify sectors
sectors = {}
for i in range(11):
    sector_stocks = np.where(stock_clusters == i)[0]
    sectors[f"sector_{i}"] = sector_stocks
    print(f"Sector {i}: {sector_stocks}")

# Plot the stock clusters
print(stock_clusters)
plt.scatter(stock_prices.mean(), stock_prices.std(), c=stock_clusters)
plt.xlabel("Mean Return (bps)")
plt.ylabel("Standard Deviation of Return (bps)")
plt.show()