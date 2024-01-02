# Using-Neural-Networks-to-estimate-online-Chess-Ratings
This is a project I completed for my ML class but have been working on post-submission as well. I use data from Lichess.org's open database to estimate the rating of online chess games.
This analysis focuses on "classical" type games (10 minutes +) and is limited to games of length 75 moves or less


Baseline:
I use previous research accuracy metrics, a basic linear regression, and an unoptimized random forest as my baselines

Neural Networks:
1. Sklearn model, I used grid search to find optimal organization of layers.
2. Pytorch model, Used dropout for further regularization.

Technical Details:
Stochastic Gradient Descent (SGD) + Mini-batch gradient descent with Backpropagation
Mean Squared Error Loss function (MSE)
tanH activation function

Encodings:
Chess moves are encoded in a tuple format as follows: (checkmate, check, promotion, capture, queenside castle, kingside castle, king, queen, rook, bishop, knight, pawn)
Example-- the move move Bxc6 (Bishop captures the piece on square c6) would be represented as:
(0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0).
