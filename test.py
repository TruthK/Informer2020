from utils.tools import *

import pandas as pd

# NMI_matrix(pd.read_csv("data/traffic/traffic.csv"), 'date')
d = NMI_matrix(pd.read_csv("data/electricity/electricity.csv"), 'date')
print(d[0])
d[0].to_csv("data/electricity/electricity_NMI.csv", index=False)
