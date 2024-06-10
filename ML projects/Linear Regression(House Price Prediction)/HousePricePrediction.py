import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm

data = pd.read_csv('data.csv')

X= np.array(data['sqft_living']).reshape(-1,1)
Y= np.array(data['price']).reshape(-1,1)

X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size=0.30)

model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

plt.scatter(X_test,Y_test,color='blue',label="Data")

plt.plot(X_test,Y_pred,color="black",label="predicted")
plt.xlabel('Sqft')
plt.ylabel('Price')
plt.legend()
plt.show()

print("R2 Score=",round(sm.r2_score(Y_test,Y_pred),2))