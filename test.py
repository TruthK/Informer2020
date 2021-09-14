from utils.tools import *

import pandas as pd

# NMI_matrix(pd.read_csv("data/traffic/traffic.csv"), 'date')
df = pd.read_csv("data/traffic/traffic_NMI.csv")
data = pd.read_csv("data/traffic/traffic.csv")
cols = data.columns.tolist()[1:]
print(cols)
x = df.as_matrix()
print(x.shape)
# 导入聚类分析工具KMeans
from sklearn.cluster import KMeans

if int(len(cols) / 4) > 3:
    kind_num = int(len(cols) / 6)
else:
    kind_num = 3
# 传入要分类的数目
kms = KMeans(n_clusters=kind_num)
y = kms.fit_predict(x)
for i in range(len(cols)):
    cols[i] = cols[i] + "_" + str(y[i])

cols = sorted(cols, key=lambda x: x.split("_")[1])
print(cols)
for i in range(len(cols)):
    cols[i] = cols[i].split("_")[0]
df_res = pd.concat([data[data.columns[0]], data[cols]], axis=1)
df_res.to_csv('data/traffic/traffic_knn.csv', index=False)