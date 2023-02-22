import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, normalize

dataset = pd.read_csv('TestingDatasets/subtracted_and_concated.csv').drop(0)
dataset = dataset.dropna(axis=1)

df = dataset.iloc[:, 7:-1]
df = df.drop(['time of last reagent added', 'last reagent added', 'temp', 'time'], axis=1)
df = df[~df['sodium_borohydride'].astype(str).str.startswith('sodium_borohydride')]

lmax_col = []
absmax_col = []
lamda_range = df.iloc[:, 4:-1]
for i in range(lamda_range.shape[0]):
    # This is currently filling lambda_max with absorbance values
    # if the index == 0 then it is the actual wavelength of max absorbancy
    max_col = max(lamda_range.iloc[i, :].items(), key=lambda pair: pair[1])
    lmax_col.append(max_col[0])
    absmax_col.append(max_col[1])

df = df.iloc[:, 0:4]
df_final = df.assign(lambda_max=lmax_col, max_absorb=absmax_col)

corrmat = df_final.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

names = list(df_final.columns)[0:4]
for i in range(4):
    plt.scatter(sorted(df_final.iloc[:, i].values), sorted(df_final['lambda_max']))
    plt.xlabel(names[i])
    plt.ylabel("Lambda-max")
    plt.title("{} vs Lambda-max".format(names[i]))
    plt.show()


