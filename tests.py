import covid19
import numpy as np
from models import sir_model, logistic_model, exponential_model
import pandas as pd

# Read Data from CSSEGISandData's GitHub Repository
data = covid19.get_data('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
today_data = [110238, 10003, 26743]

# Append the last minute data manually to the -csv data
confirmed, deaths, active = covid19.clean_data(data, 'Spain', threshold=[50, 10, 50], lastminute=today_data)

# # Compare between different countries
countries_to_compare = ['Spain', 'Italy', 'France', 'Germany', 'US']
covid19.compare(data, countries_to_compare, 'figures\\compare.png', threshold=[1000, 250, 1000])

# SIR Model
sol = covid19.fit(active,  sir_model, [0, 0, 0, 1E5], [np.inf, np.inf, np.inf, 1E8], [1, 3, 1/50, 5E7],
                  title='SIR Model', tmax=50)

# Logistic Model
covid19.fit(confirmed,  logistic_model, [0, 0, 0], [np.inf, np.inf, np.inf], [1, 1, 1],
            title='Logistic Model Confirmed', tmax=50)
covid19.fit(deaths,  logistic_model, [0, 0, 0], [np.inf, np.inf, np.inf], [1, 1, 1],
            title='Logistic Model Deaths', tmax=50, scale=1)