import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def accuracy(Y, Yhat):
    """
    ***Taken from HW 4***

    Function for computing accuracy 
    
    Y is a vector of the true ratings and Yhat is a vector of estimated ratings (0-3000)

    If the prediction is within 50 of the true rating, it counts as accurate
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):

        if ((y >= yhat-102) and (y <= yhat + 102)) : acc += 1 

    return acc/len(Y) * 100


def chessData():
    data = pd.read_csv('final_data.csv')

    #Making X matrix and Y prediction variable
    Xmat = data.drop(columns=['avg_rating', 'Unnamed: 0', 'game_order', 'is_check', 'is_check_mate']).to_numpy(dtype=np.float64)
    Y = data['avg_rating'].to_numpy(dtype=np.float64)

    #Splitting into train, validation, test (80% train, 10% validation, 10% test)
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.1, random_state=4)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.125, random_state=4)
    n, d = Xmat_train.shape

    Xmat_train, Y_train = np.array(Xmat_train), np.array(Y_train)
    Xmat_val, Y_val = np.array(Xmat_val), np.array(Y_val)
    Xmat_test, Y_test = np.array(Xmat_test), np.array(Y_test)


    #Grid search to optimize hyperparameters
    #Trying powers of 2 for hidden layer sizes for better efficiency
    candidate_grid = {
    'hidden_layer_sizes': [(256), (64, 64), (32, 32, 32), (128, 64, 32), (128, 64), (256, 32, 16)],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.0001, 0.00001],
    }
    model = MLPRegressor(hidden_layer_sizes= [128, 64, 32], learning_rate = 'adaptive', alpha = 0.0001, learning_rate_init = 0.01, max_iter = 25, verbose = True, solver = 'sgd', activation = 'tanh')
    
    model.fit(Xmat_train, Y_train)

    #UNCOMMENT THESE ONLY TO SEE GRID SEARCH
    # grid_search = GridSearchCV(model, candidate_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
    #Fitting the model with grid search (note: alpha is lambda value for L2 Regularization)
    # grid_search.fit(Xmat_train, Y_train)
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    
    return model, Xmat_train, Y_train, Xmat_val, Y_val, Xmat_test, Y_test

def main():
    #Getting X and Y data
    model, Xmat_train, Y_train, Xmat_val, Y_val, Xmat_test, Y_test = chessData()
    #Training and validation accuracies
    
    train_acc = accuracy(Y_train, model.predict(Xmat_train))
    val_acc = accuracy(Y_val, model.predict(Xmat_val))
    rmse = mean_squared_error(Y_val, model.predict(Xmat_val), squared = False)
    

    print(f"Training accuracy: {train_acc:.0f}%, Validation accuracy: {val_acc:.0f}% \n")
    
    print(f'RMSE: {rmse} \n')
    # print(best_params)
    

if __name__ == "__main__":
    main()