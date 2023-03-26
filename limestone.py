import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

stock_prices=pd.read_csv("../data_challenge_stock_prices.csv")
index_prices=pd.read_csv("../data_challenge_index_prices.csv")
stock_returns=stock_prices.pct_change()*10000
index_returns=index_prices.pct_change()*10000


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
stock_prices_imputed = imputer.fit_transform(stock_returns)
# stock_returns.drop(0)
# index_returns.drop(0)
# print(stock_returns)

# fig, axs = plt.subplots(10, 10, figsize=(20, 20))
# for i in range(100):
#     axs[i//10, i%10].hist(stock_prices.iloc[i], bins=20)
# plt.show()

# # Use k-means clustering to group the stocks based on their returns
# kmeans = KMeans(n_clusters=5, random_state=0).fit(stock_prices.T)
# stock_clusters = kmeans.labels_

# # Examine the clusters to identify sectors
# sectors = {}
# for i in range(5):
#     sector_stocks = np.where(stock_clusters == i)[0]
#     sectors[f"sector_{i}"] = sector_stocks
#     print(f"Sector {i}: {sector_stocks}")

# # Plot the stock clusters
# print(stock_clusters)
# plt.scatter(stock_prices.mean(), stock_prices.std(), c=stock_clusters)
# plt.xlabel("Mean Return (bps)")
# plt.ylabel("Standard Deviation of Return (bps)")
# plt.show()

wcss = []
for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(stock_prices_imputed.T)
        wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("Within-Cluster Sum of Squares")
plt.show()

# # Based on the elbow method, use 4 clusters to identify sectors
kmeans = KMeans(n_clusters=4, random_state=0).fit(stock_prices_imputed.T)
stock_clusters = kmeans.labels_

# # Examine the clusters to identify sectors
sectors = {}
for i in range(4):
    sector_stocks = np.where(stock_clusters == i)[0]
    sectors[f"sector_{i+1}"] = sector_stocks
    print(f"Sector {i+1}: {sector_stocks}")
print(index_prices.cov())