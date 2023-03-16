import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#__________Applying Linear Regresstion using Build-in Model__________#

df_train = pd.read_csv('train.csv')   # Loading Dataset training use your own path as where you are placing the test.csv and trian.csv   
df_test = pd.read_csv('test.csv')     # Loading Dataset testing

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Removing NaN values for the Dataset
# find NaN values in y_train and y_test
train_nan_indices = np.isnan(y_train)
test_nan_indices = np.isnan(y_test)

# remove NaN values from y_train and corresponding rows from x_train
y_train = y_train[~train_nan_indices]
x_train = x_train[~train_nan_indices]

# remove NaN values from y_test and corresponding rows from x_test
y_test = y_test[~test_nan_indices]
x_test = x_test[~test_nan_indices]

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


# Scalling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Applying Linear Reggression Model which we imported from Scikit-Learn
model = LinearRegression().fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled) # Testing Model Accuracy
print(r2_score(y_test, y_pred))
print('Weights:', model.coef_)
print('Bias:', model.intercept_)

# Applying Graphical Analysis for Best fit Line
plt.figure(figsize=(8,8))
plt.scatter(x_test_scaled,y_test,color='red',label='GT')
plt.show()

#__________Now instead of using build-in model we will create it from using NumPy__________#

# Again loading and Removing the NaN values from Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# find NaN values in y_train and y_test
train_nan_indices = np.isnan(y_train)
test_nan_indices = np.isnan(y_test)

# remove NaN values from y_train and corresponding rows from x_train
y_train = y_train[~train_nan_indices]
x_train = x_train[~train_nan_indices]

# remove NaN values from y_test and corresponding rows from x_test
y_test = y_test[~test_nan_indices]
x_test = x_test[~test_nan_indices]

# Training the Model

# Set learning rate and number of iterations
learning_rate = 0.0001
epochs = 1000

# Initialize coefficients and bias
a_0 = 0
a_1 = 0

# Train the model using gradient descent
for i in range(epochs):
    y_pred = a_0 * x_train + a_1
    error = y_pred - y_train
    mse = np.mean((y_train - y_pred)**2) # Compute mean squared error
    a_0 -= learning_rate * np.mean(error * x_train)
    a_1 -= learning_rate * np.mean(error)

# Testing Model Accuracy

y_prediction = a_1 + a_0 * x_test
print('R2 Score:',r2_score(y_test,y_prediction))
print("Coefficients:", a_0)
print("Intercept:", a_1)

# Now applying Graphical Analysis using Matplotlib for Best Fit Line

y_plot = []
for i in range(100):
    y_plot.append(a_1 + a_0 * i)
plt.figure(figsize=(8,8))
plt.scatter(x_test,y_test,color='red',label='GT')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'pred')
plt.legend()
plt.show()
