from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

#loading the dataset
data = fetch_california_housing()
df= pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

#splitting the dataset

x= df.drop('PRICE', axis=1)
y= df['PRICE']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)

#Training the model

model= LinearRegression()
model.fit(x_train,y_train)

#predicting and evaluating the model
y_pred= model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
#Scatter plot of actual vs predicted prices

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred, color='blue', alpha= 0.6)
plt.plot ([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], color='red',lw=2, linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.tight_layout()
plt.show()
