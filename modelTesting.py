import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from ml_models import DummyMLModel, PolynomialRegression

df = pd.read_csv('TestingDatasets/brooklyn_listings.csv')
df = df[['price', 'bathrooms', 'sqft']].dropna()
poly_model = PolynomialRegression(None, None, None, 100, 5, ['bathrooms', 'sqft'])

prediction, predOrder = poly_model.training(df, 1)
print(prediction.shape)

# Visualize
df.plot(kind='scatter', x='sqft', y='price', color='red')
plt.scatter(df['sqft'], prediction, color='green')
plt.xlabel("Square-footage")
plt.ylabel("Price (in millions)")
plt.xlim([0, 3000])
plt.savefig("sqft-vs-price.png")

df.plot(kind='scatter', x='bathrooms', y='price', color='red')
plt.scatter(df['bathrooms'], prediction, color='green')
plt.xlabel("Number of bathrooms in property")
plt.ylabel("Price (in millions)")
plt.xlim([0, 200])
plt.savefig("bathroooms-vs-price.png")
plt.show()
