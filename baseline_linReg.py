import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('final_data.csv')

    columns_to_keep = ['result', 'eco', 'move_no', 'avg_rating']
    data = data[columns_to_keep]
    data.to_csv('baseline_data.csv')
    X = data.drop(columns=['avg_rating']).to_numpy(dtype=np.float64)
    y = data['avg_rating'].to_numpy(dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred, squared = False)
    r2 = r2_score(y_test, y_pred)

    print(f'Root Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

if __name__ == "__main__":
    main()