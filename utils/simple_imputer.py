import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df = pd.read_csv("../data/electricity/electricity_miss_10.csv", header=None)

df_orgin = pd.read_csv("../data/electricity/electricity.csv", header=None)

ss = StandardScaler()
orgin_data = df_orgin.iloc[1:, 1:].values

df = df.iloc[1:, 1:]
data = df.values
res = imp.fit_transform(data)

print('------------------mse--------------------------')
print(metrics.mean_squared_error(ss.fit_transform(orgin_data), ss.fit_transform(res)))
print('-------------------mae------------------------------')
print(metrics.mean_absolute_error(ss.fit_transform(orgin_data), ss.fit_transform(res)))

res = pd.DataFrame(res)

date = pd.DataFrame(df_orgin[df_orgin.columns[0]][1:].values)
res = pd.concat([date, res], axis=1)
res.columns = df_orgin.iloc[0].values
res.to_csv("../data/electricity/electricity_10_simple.csv", index=False)
