import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('final_data.csv')

#Making X matrix and Y prediction variable
Xmat = data.drop(columns=['avg_rating', 'Unnamed: 0', 'game_order', 'is_check', 'is_check_mate']).to_numpy(dtype=np.float64)
Y = data['avg_rating'].to_numpy(dtype=np.float64)

#Splitting into train, test (80% train, 20% test)
Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.2, random_state=4)
n, d = Xmat_train.shape

model = RandomForestRegressor(n_estimators = 100, random_state = 4)
model.fit(Xmat_train, Y_train)

Yhat = model.predict(Xmat_test)

rmse = mean_squared_error(Y_test, Yhat, squared = False)

print(f'Root Mean Squared Error = {rmse}')