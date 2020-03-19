import numpy as np
from scipy import interpolate, integrate


def sir_model(x, initial_infected, beta, gamma, population, initial_recovered=0, tmax=365, n=1000):

    """"Uses SIR Model (https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model)
        to return the value of the Infected Curve at time x, given the SIR Model parameters.

        Parameters:
        - x : (float) Time to evaluate the infected curve
        - initial_infected : (float) SIR Model I0 parameter. Initial infected subjects
        - beta : (float) Transmission rate
        - gamma : (float) Typical time between contacts
        - population : (float) Susceptible population
        - initial_recovered : (float) Initial recovered individuals (Default 0)
        - tmax : (float) Time for integrating the differential equations, in days (Default 365)
        - n : (int) Number of time points for integrating the differential equations (Default 1000)"""

    initial_susceptible = population - initial_infected - initial_recovered  # Everyone who is susceptible to infection
    t = np.linspace(0, tmax, n)  # Time vector for integrating

    def derivatives(y, _):

        """SIR Model Differential Equations

            Parameters:
            - y : (np.ndarray) Array containing [Susceptible, Infected, Recovered] points
            - _ : (None) Empty parameter for consistency with scipy.integrate.odeint method"""

        s, i, _ = y
        derivative_a = -beta * s * i / population  # dS/dt
        derivative_b = beta * s * i / population - gamma * i  # dI/dt
        derivative_c = gamma * i  # dR / dt
        return derivative_a, derivative_b, derivative_c

    y0 = initial_susceptible, initial_infected, initial_recovered  # Initial conditions vector
    sol = integrate.odeint(derivatives, y0, t)  # Integrate the SIR equations over the time grid, total_time
    infected = sol[:, 1]  # Infected individuals for each day
    interp = interpolate.interp1d(t, infected, fill_value='extrapolate')  # Creates an interpolator with the vectors
    return interp(x)


def logistic_model(x, a, b, c):

    """"Uses Logistic Model (https://en.wikipedia.org/wiki/Logistic_regression) to fit the curve of infected
        individuals to a Logistic Curve f(x, a, b, c) = c / (1 + exp(-(x-b)/a))

        Parameters:
        - x : (float) Time to evaluate the infected curve
        - a, b, c : (float) Logistic Curve paramters"""

    return c / (1 + np.exp(-(x - b) / a))


def exponential_model(x, a, b, c):

    """"Uses Logistic Model (https://en.wikipedia.org/wiki/Logistic_regression) to fit the curve of infected
            individuals to a Logistic Curve f(x, a, b, c) = a exp(b(x-c))

            Parameters:
            - x : (float) Time to evaluate the infected curve
            - a, b, c : (float) Logistic Curve paramters"""

    return a * np.exp(b * (x-c))