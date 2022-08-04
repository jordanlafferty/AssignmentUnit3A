import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns  # Convention alias for Seaborn

path_to_file = '/Users/jordanlafferty/Desktop/auto_mpg_1983.csv'
df = pd.read_csv(path_to_file)

# print(df.describe().round(2).T)
y = df['mpg']
X = df[['cylinders', 'displacement', 'horsepower', 'weight']]

variables = ['cylinders', 'displacement', 'horsepower', 'weight']

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

for index, var in enumerate(variables):
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='mpg', data=df, ax=ax[int(np.ceil(index / 4))][int(np.mod(index, 2))])

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Predict the MPG')
print('enter the weight: ')
weight_num = input()
print('enter the displacement: ')
displacement_num = input()
print('enter number of cylinders: ')
cylinder_count = input()
print('enter the horsepower: ')
horsepower_num = input()

score = regressor.predict([[int(cylinder_count), int(displacement_num), int(horsepower_num),  int(weight_num)]])
print('The predicted MPG is: ', score[0])

y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

print('test: ', regressor.score(X_test, y_test))
print('model: ', regressor.score(X_train, y_train))
