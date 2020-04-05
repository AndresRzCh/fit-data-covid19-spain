import covid19
import numpy as np
from models import sir_model, logistic_model, exponential_model
import pandas as pd

# Read Data from CSSEGISandData's GitHub Repository
data = covid19.get_data('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
today_data = [172541, 18056, 67504]

# Append the last minute data manually to the -csv data
confirmed, deaths, active = covid19.clean_data(data, 'Spain', threshold=[2500, 5000, 2500], lastminute=None)
print(deaths.reset_index())

# # Compare between different countries
countries_to_compare = ['Spain', 'Italy', 'France', 'Germany', 'US']
covid19.compare(data, countries_to_compare, 'figures\\compare.png', threshold=[1000, 250, 1000])

# SIR Model
sol = covid19.fit(active,  sir_model, [0, 0, 1, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf],
                  [1, 1.5, 5E5, 1E8, 0], title='SIR Model', tmax=200, days_range=30)

# Logistic Model
covid19.fit(confirmed,  logistic_model, [0, 0, 0], [np.inf, np.inf, np.inf], [1, 1, 1],
            title='Logistic Model Confirmed', tmax=100)
covid19.fit(deaths,  logistic_model, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf], [1, 1, 1E5],
            title='Logistic Model Deaths', tmax=100, scale=1)
