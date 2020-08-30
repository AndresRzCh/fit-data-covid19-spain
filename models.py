import numpy as np


def logistic_model(x, a, b, c, d):

    """"Uses Logistic Model (https://en.wikipedia.org/wiki/Logistic_regression) to fit the curve of infected
        individuals to a Logistic Curve f(x, a, b, c) = c / (1 + exp(-(x-b)/a))

        Parameters:
        - x : (float) Time to evaluate the infected curve
        - a, b, c : (float) Logistic Curve paramters"""

    return c / (1 + np.exp(-(x - b) / a)) + d


def exponential_model(x, a, b, c, d):

    """"Uses Logistic Model (https://en.wikipedia.org/wiki/Logistic_regression) to fit the curve of infected
            individuals to a Logistic Curve f(x, a, b, c) = a exp(b(x-c))

            Parameters:
            - x : (float) Time to evaluate the infected curve
            - a, b, c : (float) Logistic Curve paramters"""

    return a * np.exp(b * (x-c)) + d
