import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pandas.plotting import register_matplotlib_converters
from fbprophet import Prophet
import datetime
import os, sys

register_matplotlib_converters()


def get_data(csv_route, date_label='Date', country_label='Country', confirmed_label='Confirmed',
             deaths_label='Deaths', recovered_label='Recovered', drop_first=0, drop_last=0, dayfirst=False,
             sumcolumns=None):

    """ Get the data from a .csv file. This .csv file must contain a time series with
        data for each date in one column, and must contain a 'Country' column.
        Please, refer to https://github.com/datasets/covid-19 for
        an example of the time series format. Returns a pandas.DataFrame object.

        Parameters:
        - csv_route : (str) Location to the .csv file (can be a local folder or an URL)
        - date_label : (str) Name of the date column in the provided CSV file (Default 'Date')
        - country_label : (str) Name of the country (or state) column in the provided CSV file (Default 'Country')
        - confirmed_label : (str) Name of the confirmed cases column in the provided CSV file (Default 'Confirmed')
        - deaths_label : (str) Name of the deaths column in the provided CSV file (Default 'Deaths')
        - recovered_label : (str) Name of the recovered column in the provided CSV file (Default ('Recovered')
        - drop_first : (int) Number of first lines to drop (Default 0)
        - drop_last : (int) Number of last lines to drop (Default 0)
        - dayfirst : (bool) True if date has the day before month (Default False)
        - sumcolumns : (dict) If not None, for each key it takes the columns from a list and adds it (Default None)
        """

    df = pd.read_csv(csv_route, skiprows=drop_first, skipfooter=drop_last, engine='python', encoding='utf-8')  # Read the CSV
    if sumcolumns:
        for col in sumcolumns:
            df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), sumcolumns[col]].max(axis=1)
    df = df.rename({date_label: 'Date', country_label: 'Country', confirmed_label: 'Confirmed',
                    deaths_label: 'Deaths', recovered_label: 'Recovered'}, axis=1) # Rename using labels provided
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=dayfirst)  # Converts date strings to timestamp
    return df[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']]


def clean_data(data, country=None, threshold=None, from_date=None):

    """ Select one (or none) country to study, and returns the same DataFrame read from
        the .csv file, filtered by country, and applying a minimum threshold in order to
        ignore insignificant values. Returns a pandas.Series object.

        Parameters:
        - data : (pandas.DataFrame) Output of get_data() method
        - country: (str) Country to study. Use None for studying global data. (Default None)
        - threshold: (int) Minimum number of cases to consider (Default 10)"""

    if threshold is None:
        threshold = [0, 0, 0]

    if country:
        data = data[data['Country'] == country]  # Filter by country when it's given

    if from_date:
        data = data[data['Date'] > pd.to_datetime(from_date, format='%m/%d/%Y')]

    data = data.groupby('Date').sum()  # Group by date and sum cases
    data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']  # Active Cases

    confirmed = data.loc[data['Confirmed'] >= threshold[0], 'Confirmed'].drop_duplicates()
    deaths = data.loc[data['Deaths'] >= threshold[1], 'Deaths'].drop_duplicates()
    active = data.loc[data['Active'] >= threshold[2], 'Active'].drop_duplicates()

    return confirmed, deaths, active


def format_axis(ax, values, days_range, tmax=None):

    """ Given an matplotlib.pyplot.axis object, it returns the same object formatting the date values on the x-axis

        Parameters:
        - ax (matplotlib.pyplot.axis) The axis to format
        - values (pandas.DataFrame or pandas.Series) The data which appears on the plot
        - days_range (int) Days between x-label ticks
        - tmax (int) Maximum number of days to show, default chooses the lenght of values (Default None)"""

    initial_date = values.index.values[0]  # Select the initial date

    if tmax:
        days_ticks = range(tmax)  # Expand or truncate to tmax if necesary
    else:
        days_ticks = range(len(values))  # Automatically chooses the lenght of values

    # Format the labels to day / month strings
    days_labels = [str(pd.to_datetime(initial_date + np.timedelta64(i, 'D')).day).zfill(2) + '/' +
                   str(pd.to_datetime(initial_date + np.timedelta64(i, 'D')).month).zfill(2) for i in days_ticks]

    ax.set_xticks(days_ticks[::days_range])  # Define the matplotlib xticks
    ax.set_xticklabels(days_labels[::days_range])  # Define the matplotlib xlabels
    return ax


def compare(data, countries, filename='figures\\compare.png', threshold=None, scale=1000):

    """ Given a list of countries, compares their number of cases setting them to the
        same origin. Returns a matplotlib.figure.Figure object.

        Parameters:
        - data : (pandas.DataFrame) Output of get_data() method
        - countries : (list) List of strings containing the countries to compare
        - filename : (str) Location and filename to save the plot (Default 'figures\\compare.png')
        - threshold : (int) Minimum number of cases to consider for setting the origin (Default 100)
        - scale :  (int) Scale for the y-axis (Default 1000)"""

    if threshold is None:
        threshold = [100, 10, 100]

    values = [clean_data(data, i, threshold) for i in countries]  # Get all values for each country

    fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)  # Creates the Figure

    for i in range(len(values)):
        ax1.plot(values[i][0].values / scale, label=countries[i], marker='x')  # Confirmed Cases
        ax2.plot(values[i][1].values, label=countries[i], marker='x')  # Deaths

    ax1.legend().get_frame().set_alpha(0.5)  # Style the legend
    ax2.legend().get_frame().set_alpha(0.5)

    ax1.grid(b=True, which='major', c='k', lw=0.25, ls='-')  # Style the grid
    ax2.grid(b=True, which='major', c='k', lw=0.25, ls='-')

    ax1.set_xlabel('Days')  # Set the x-axis label
    ax2.set_xlabel('Days')

    ax1.set_ylabel('Confirmed (' + str(scale) + ')')  # Set the y-axis label
    ax2.set_ylabel('Deaths')

    ax1.yaxis.set_tick_params(length=0)  # Style the y-axis labels
    ax2.yaxis.set_tick_params(length=0)

    ax1.xaxis.set_tick_params(length=0)  # Style the x-axis labels
    ax2.xaxis.set_tick_params(length=0)

    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(False)  # Style the figure box
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(filename)  # Save the figure into the route
    return values


def daily_increases(data, country, filename='figures\\daily.png', threshold=None, scale=None):

    """ Given a list of countries, plots the daily confirmed/deaths with respect to the previous day.

        Parameters:
        - data : (pandas.DataFrame) Output of get_data() method
        - countries : (list) List of strings containing the countries to compare
        - filename : (str) Location and filename to save the plot (Default 'figures\\compare.png')
        - scale :  (list) Scale for the y-axis for confirmed and deaths (Default None)"""

    if scale is None:
        scale = [1000, 1]

    confirmed, deaths, _ = clean_data(data, country, threshold)  # Clean the data
    confirmed = confirmed - confirmed.shift(1) # Calculate the increases
    deaths = deaths  -deaths.shift(1)

    fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)  # Creates the Figure

    ax1.bar(range(len(confirmed)), confirmed.values / scale[0])  # Confirmed Cases
    ax2.bar(range(len(deaths)), deaths.values / scale[1])  # Deaths

    print()

    ax1.grid(b=True, which='major', c='k', lw=0.25, ls='-')  # Style the grid
    ax2.grid(b=True, which='major', c='k', lw=0.25, ls='-')

    ax1.set_xlabel('Date')  # Set the x-axis label
    ax2.set_xlabel('Date')

    ax1.set_ylabel('Confirmed (' + str(scale[0]) + ')')  # Set the y-axis label
    ax2.set_ylabel('Deaths (' + str(scale[1]) + ')')

    ax1.yaxis.set_tick_params(length=0)  # Style the y-axis labels
    ax2.yaxis.set_tick_params(length=0)

    ax1.xaxis.set_tick_params(length=0)  # Style the x-axis labels
    ax2.xaxis.set_tick_params(length=0)

    ax1 = format_axis(ax1, confirmed, 7)
    ax2 = format_axis(ax2, confirmed, 7)

    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(False)  # Style the figure box
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(filename)  # Save the figure into the route
    return confirmed, deaths


def fit(values, model, lbounds, gbounds, guess=None, route='figures\\', title='Model',
        tmin=0, tmax=50, scale=1000, days_range=7):

    """Fit the real data to a model. Lower and greater bounds must be provided
       in order to fit a problem with many free parameters. An initial guess
       should be provided in order to improve the accuracy of the model.

       Parameters:
       - values : (pandas.Series) Output from clean_data() method
       - model : (models.model) A function from models.py library
       - lbounds : (list) List of floats for lower bounds of each free parameter of the model
       - gounds : (list) List of floats for greater bounds of each free parameter of the model
       - guess : (list) List of floats for an initial guess between bounds (default None)
       - route : (str) Location to save the plot (Default 'figures\\')
       - title : (str) Title for the figure. It is used for the filename of the plot (default 'Model')
       - tmin : (int) First day to show in the figure using the model (default 0)
       - tmax : (int) Number of days to show in the figure using the model (default 150)
       - scale :  (int) Scale for the y-axis (Default 1000)
       - days_range : (int) Space between days when plotting results"""

    route = route + title.lower().replace(' ', '_') + '.png'

    days = np.arange(0, len(values))  # Calculates a time array, one day for each point in data
    sol = optimize.curve_fit(model, days, values, p0=guess, bounds=(lbounds, gbounds))[0]  # Fit the curve

    days = np.arange(0, len(values))  # Defines the days array for the real data plot
    t = np.arange(tmin, tmax)  # Defines the time array for the model plot

    fig, ax = plt.subplots(figsize=(6, 4), ncols=1)  # Creates the Figure

    ax.plot(days, values / scale, 'k', alpha=0.5, lw=2, marker='x', label='Real Data')  # Plot the real data
    ax.plot(t, model(t, *sol) / scale, 'r', alpha=0.5, lw=3, label='Model')  # Plot the model data

    ax.legend().get_frame().set_alpha(0.5)  # Style the legend
    ax.grid(b=True, which='major', c='k', lw=0.25, ls='-')  # Style the grid

    ax.set_xlabel('Date')  # Set the x-axis label
    ax.set_ylabel('Number (' + str(scale) + ')')  # Set the y-axis label

    ax.yaxis.set_tick_params(length=0)  # Style the y-axis labels
    ax.xaxis.set_tick_params(length=0)  # Style the x-axis labels

    ax.set_xlim([tmin, tmax])

    ax = format_axis(ax, values, days_range, tmax)

    ax.set_title(title)  # Set the title

    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)  # Style the figure box
    plt.tight_layout()
    fig.savefig(route)  # Save the figure into the route
    return sol


class suppress_stdout_stderr(object):

    """A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through)."""

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def predict(y, filename='sample.csv', periods=25):

    """Predict the shape of a curve given a .csv file with a column named "date" for the time series and another
       column "y", which is the feature to predict.

           Parameters:
           - filename : (str) Name of the .csv file containing the time series
           - y : (str) Feature to predict
           - periods : (int) periods parameter for Prophet.make_future_dataframe() function"""

    file = pd.read_csv(filename)
    file['Date'] = pd.to_datetime(file['Date'])
    df = pd.DataFrame()
    df['ds'] = file['Date']
    df['y'] = file[y]
    file = file[['Date', y]].rename({y: 'Real'}, axis=1)

    model = Prophet(yearly_seasonality=False, n_changepoints=24, daily_seasonality=True)
    with suppress_stdout_stderr():
        model.fit(df)
    future = model.make_future_dataframe(periods=25)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    fig.savefig('figures\\prophet_' + y.lower() + '.png')

    today = pd.to_datetime('today').date()
    today = np.datetime64(datetime.datetime(today.year, today.month, today.day, 0, 0, 0, 0))
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    sol = pd.concat([forecast.loc[forecast['ds'] == today + np.timedelta64(i, 'D'), cols] for i in [-1, 0, 1]])
    renaming = {'ds': 'Date', 'yhat': 'Feature', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}
    return sol.rename(renaming, axis=1).merge(file, on='Date', how='left')
