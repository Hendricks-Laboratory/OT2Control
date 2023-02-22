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

# Regression model testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics

model_data = pd.DataFrame(columns=['Model', 'Mean-Squared Error', 'Mean-Absolute Error',
                                   'Root Mean-Squared Error'])

X = df_final.iloc[:, 0:4].values
y = df_final.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Basic Regression
lreg = LinearRegression()
lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)

model_data = model_data.append({'Model': "Multiple Regression",
                                'Mean-Squared Error': metrics.mean_squared_error(y_test, y_pred),
                                'Mean-Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
                                'Root Mean-Squared Error': np.sqrt(metrics.mean_squared_error(y_test, y_pred))},
                               ignore_index=True)

# Polynomial Regression
poly_feats = PolynomialFeatures(degree=4)
X_poly = poly_feats.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred = poly_reg.predict(X_poly)

model_data = model_data.append({'Model': "Polynomial Regression",
                                'Mean-Squared Error': metrics.mean_squared_error(y, y_pred),
                                'Mean-Absolute Error': metrics.mean_absolute_error(y, y_pred),
                                'Root Mean-Squared Error': np.sqrt(metrics.mean_squared_error(y, y_pred))},
                               ignore_index=True)

# SVM Regression
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train, y_train)
y_pred = svr_reg.predict(X_test)

model_data = model_data.append({'Model': "SVM Regression (RBF Kernel)",
                                'Mean-Squared Error': metrics.mean_squared_error(y_test, y_pred),
                                'Mean-Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
                                'Root Mean-Squared Error': np.sqrt(metrics.mean_squared_error(y_test, y_pred))},
                               ignore_index=True)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=300, max_features='sqrt', max_depth=5, random_state=18)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

model_data = model_data.append({'Model': "Random-Forest Regression (300 estimators)",
                                'Mean-Squared Error': metrics.mean_squared_error(y_test, y_pred),
                                'Mean-Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
                                'Root Mean-Squared Error': np.sqrt(metrics.mean_squared_error(y_test, y_pred))},
                               ignore_index=True)

model_data.head(4)
