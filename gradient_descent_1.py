# Source: https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21


# Base model

import numpy as np

def gradient_descent(start, gradient, learn_rate, max_iter, tol=0.01):
    steps = [start]  # history tracking
    x = start

    for _ in range(max_iter):
        diff = learn_rate * gradient(x)
        if np.abs(diff) < tol:
            break
        x = x - diff
        steps.append(x)  # history tracing

    return steps, x
