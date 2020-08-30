import covid19
import pandas as pd
import numpy as np
from models import sir_model, logistic_model


def full_test():
    github = 'https://raw.githubusercontent.com/montera34/escovid19data/master/data/output/covid19-provincias' \
             '-spain_consolidated.csv '

    data = covid19.get_data(github, date_label='date', ccaa_label='ccaa', drop_first=0, drop_last=0, dayfirst=True,
                            ccaa=None, from_date='07/01/2020', to_date='08/23/2020',
                            name_maps={'casos': 'Casos', 'hospitalized': 'Ingresados', 'intensive_care': 'UCI',
                                       'deceased': 'Muertos'},
                            sumcolumns={'casos': ['num_casos', 'num_casos_prueba_pcr', 'num_casos_prueba_test_ac',
                                                  'num_casos_prueba_otras', 'num_casos_prueba_desconocida']})

    covid19.fit(data['UCI'], logistic_model, [0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf], [100, 100, 1, 1],
                title='Logistic UCI', tmax=100, days_range=14)

    covid19.fit(data['Muertos'], logistic_model, [0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf], [100, 100, 1, 1],
                title='Logistic Deaths', tmax=100, days_range=14)

    covid19.fit(data['Casos'], logistic_model, [0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf], [100, 100, 1, 1],
                title='Logistic Cases', tmax=100, days_range=14)

    sample = pd.read_csv('sample.csv')
    sample['Date'] = pd.to_datetime(sample['Date'])
    sample = sample.set_index('Date')

    covid19.fit(sample['IA'], logistic_model, [0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf], [100, 100, 1, 1],
                title='Logistic IA', tmax=100, days_range=14, scale=1)

    covid19.fit(sample['U14D'], logistic_model, [0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf], [100, 100, 1, 1],
                title='Logistic U14D', tmax=100, days_range=14)


full_test()
print(covid19.predict('IA'))
print(covid19.predict('U14D'))