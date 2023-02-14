import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, normalize

dataset = pd.read_csv('TestingDatasets/pr_data/concatenatedData.csv')

lmax_col = []
absmax_col = []
lamda_range = dataset.iloc[:, 18:1312].dropna(1)
for i in range(lamda_range.shape[0]):
    # This is currently filling lambda_max with absorbance values
    # if the index == 0 then it is the actual wavelength of max absorbancy
    max_col = max(lamda_range.iloc[i, :].items(), key=lambda pair: pair[1])
    lmax_col.append(max_col[0])
    absmax_col.append(max_col[1])

df = dataset.iloc[:, 6:14]
df = df.assign(lambda_max=lmax_col, max_absorbance=absmax_col).dropna()
print(df.head(3))

# plt.matshow(df.corr())
# plt.title("Correlation Matrix")
# plt.show()

# All values
vals = df.iloc[:, :-1].values

# # Trying values where lambda_max isnt 300
# vals = df.loc[df["lambda_max"] != 300]

cols = list(df.columns)
cols.remove("lambda_max")
col_data = zip(cols, [vals[:, i] for i in range(8)])
for cName, cData in col_data:
    plt.scatter(cData, df["max_absorbance"].values, color='red')
    plt.xlabel(cName)
    plt.ylabel("Max Absorbance")
    plt.title("{} vs Max Absorbance".format(cName))
    plt.show()