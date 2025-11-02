import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'Hours_studied' : [1,2,3,4,5,6,7,8,9,10],
    'Score': [12, 25,32,40, 50, 55, 65, 72, 80, 90]
}
df = pd.DataFrame(data)
X = df[['Hours_studied']]
Y = df[['Score']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("predictaion:", y_pred)

mse = mean_squared_error(Y_test, y_pred)
print("Mean squared Error:", mse)

X_id =X['Hours_studied'].values
y_pred_full = model.predict(X).flatten()

plt.scatter(X_id, Y,color='green', label='Actual scores')
plt.plot(X_id, y_pred_full, color='red', label='prediction line')
plt.xlabel('Hours studied')
plt.ylabel('Scores')

plt.legend()
plt.show()
