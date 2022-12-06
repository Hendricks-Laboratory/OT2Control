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
x_cords = np.linspace(df['bathrooms'].min(), df['bathrooms'].max(), len(df['bathrooms']))
y_cords = np.linspace(df['sqft'].min(), df['sqft'].max(), len(df['sqft']))
[X, Y] = np.meshgrid(x_cords, y_cords)
print("x-shape ", x_cords.shape)
print("y-shape ", y_cords.shape)
print("z-shape ", prediction.shape)
# print(prediction)



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x_cords.flatten(), y_cords.flatten(), prediction.flatten())
ax.set_title('Bathrooms & Square Footage vs Price')
ax.set_xlabel('Bathrooms')
ax.set_ylabel('Square Footage')
plt.show()


