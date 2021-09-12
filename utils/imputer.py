## 图像显示中文的问题
import matplotlib
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

sns.set(font="Kaiti", style="ticks", font_scale=1.4)

## 导入本小节会使用到的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor



def Multivariate_Interpolation(v):
    imp = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=30, random_state=1997)
    imp.fit(v)
    IteraList = imp.transform(v)
    IteraNp = np.array(IteraList)
    return IteraNp


## 读取用于演示的数据集
df = pd.read_csv("../data/ETT/ETTh1_miss.csv", header=None)
df_orgin = pd.read_csv("../data/ETT/ETTh1.csv", header=None)


ss = StandardScaler()

df = df.iloc[1:, 1:]
data = df.values
res = Multivariate_Interpolation(data)

orgin_data = df_orgin.iloc[1:, 1:].values
pd.DataFrame(res).to_csv("ex_ett.csv", index=False)
print('------------------mse--------------------------')
print(metrics.mean_squared_error(ss.fit_transform(orgin_data), ss.fit_transform(res)))
print('-------------------mae------------------------------')
print(metrics.mean_absolute_error(ss.fit_transform(orgin_data), ss.fit_transform(res)))
