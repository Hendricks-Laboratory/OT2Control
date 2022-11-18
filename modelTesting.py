import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_models import DummyMLModel, PolynomialRegression

df = pd.read_csv('TestingDatasets/brooklyn_listings.csv')
df = df[['price', 'bathrooms', 'sqft']].dropna()
poly_model = PolynomialRegression(None, None, None, 100, 5, ['bathrooms', 'sqft'])

prediction, predOrder = poly_model.training(df, 1)

# Visualize
x_cords = np.linspace(df['bathrooms'].min(), df['bathrooms'].max(), len(df['bathrooms']))
y_cords = np.linspace(df['sqft'].min(), df['sqft'].max(), len(df['sqft']))
z_cords = np.split(prediction, len(prediction)//2)
[X, Y, Z] = np.meshgrid(x_cords, y_cords, z_cords)

# print("x-shape ", x_cords.shape)
# print("y-shape ", y_cords.shape)
# print("z-shape ", prediction.shape)
# print(prediction)

fig, ax = plt.subplots(1, 1)
ax.contourf(X, Y, Z)
ax.set_title('Bathrooms & Square Footage vs Price')
ax.set_xlabel('Bathrooms')
ax.set_ylabel('Square Footage')
plt.show()


