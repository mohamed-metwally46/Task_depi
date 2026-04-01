import numpy as np 

# Gradinet Decent Implementation

def gradient_decent_1d(learning_rate = 0.1 , iter = 50 , start =10):
    x = start 
    history = [x]
    for _ in range(iter):
        gradient = 2 * x   
        x = x-(learning_rate * gradient)
        history.append(x)
    return x,history

def gradient_decent_2D (learning_rate = 0.1 , iter = 50 , start =(10,10)):
    x,y = start
    history = [(x,y)]
    for _ in range(iter):
        grad_x = 2*x
        grad_y = 2*y
        x = x - (learning_rate * grad_x)
        y = y - (learning_rate * grad_y)
        history.append((x,y))
    return (x,y),history

def linear_regression_gd(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    theta = np.zeros(2)         # [bias, weight]
    X_b = np.c_[np.ones(m), X] # add bias column
    cost_history = []

    for _ in range(iterations):
        predictions = X_b @ theta
        errors = predictions - y
        gradients = (1 / m) * X_b.T @ errors
        theta -= learning_rate * gradients
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)

    return theta, cost_history