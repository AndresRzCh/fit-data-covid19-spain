import covid19
import numpy as np
from models import sir_model, logistic_model

# Get data From ISCIII website

data = covid19.get_data('https://covid19.isciii.es/resources/serie_historica_acumulados.csv', drop_last=6,
                        date_label='FECHA', country_label='CCAA', confirmed_label='CASOS',
                        deaths_label='Fallecidos', recovered_label='Recuperados', dayfirst=True,
                        sumcolumns={'CASOS':['CASOS', 'PCR+', 'TestAc+']})

# Clean all the data with some Threshold to remove the first days of wrong data
confirmed, deaths, active = covid19.clean_data(data, None, threshold=[1000, 1, 1000])
print('Last Update: ', confirmed.index.max())

covid19.daily_increases(data, None)

# Compare all regions
covid19.compare(data, data['Country'].drop_duplicates(), 'figures\\compare.png')

# Fit the SIR Model
covid19.fit(active,  sir_model, [0, 0, 1, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf],
            [1, 5, 15, 1E6, 0], title='SIR Model', tmax=70, days_range=20)

# Fit the Logistic Model for confirmed cases
covid19.fit(confirmed,  logistic_model, [0, 0, 0], [np.inf, np.inf, np.inf], [1, 1, 1],
            title='Logistic Model Confirmed', tmax=70, days_range=20)

# Fit the Logistic Model for deaths
covid19.fit(deaths,  logistic_model, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf], [1, 1, 1],
            title='Logistic Model Deaths', tmax=70, scale=1, days_range=20)