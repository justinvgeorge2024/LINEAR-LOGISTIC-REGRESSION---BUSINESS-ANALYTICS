import pandas as pd
import numpy as np

data = pd.read_csv('input.csv')


# mat = np.array(df.values,'float')
# print(mat)


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = data.iloc[:, 0].values.reshape(-1, 1)                                                             # values converts it into a numpy array
Y = data.iloc[:, 1].values.reshape(-1, 1)                                                         # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()                                                              # create object for the class

linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
print("coefficient :",linear_regressor.coef_)
print("intercept  :",linear_regressor.intercept_)
print("equation is : y=",linear_regressor.coef_," x + ",linear_regressor.intercept_)
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()