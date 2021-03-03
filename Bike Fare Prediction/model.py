import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle



data = pd.read_csv("bikes.csv")


x = data[['temperature', 'humidity', 'windspeed']]
sc = StandardScaler()
x = sc.fit_transform(x)
y = data['count']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

lr = LinearRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)

print("R2 Score",r2_score(y_test,pred))


with open('bike_model.pkl', 'wb') as file:
    pickle.dump(lr, file)




