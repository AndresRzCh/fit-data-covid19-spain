import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pandas.plotting import register_matplotlib_converters
from fbprophet import Prophet
import datetime
import os

register_matplotlib_converters()


def get_data(csv_route, date_label='Date', ccaa_label='CCAA', name_maps=None, drop_first=0, drop_last=0,
             dayfirst=False, sumcolumns=None, ccaa=None, from_date=None, to_date=None):

    """ Get the data from a .csv file. Returns a pandas.DataFrame object.

        Parameters:
        - csv_route : (str) Location to the .csv file (can be a local folder or an URL)
        - date_label : (str) Name of the date column in the provided CSV file (Default 'Date')
        - ccaa_label : (str) Name of the state column in the provided CSV file (Default 'CCAA')
        - name_maps : (dict) Dictionary to rename the columns (Default None)
        - drop_first : (int) Number of first lines to drop (Default 0)
        - drop_last : (int) Number of last lines to drop (Default 0)
        - dayfirst : (bool) True if date has the day before month (Default False)
        - sumcolumns : (dict) If not None, for each key it takes the columns from a list and adds it (Default None)
        - ccaa : (str) Keel only one state (Default None)
        - from_date : (str) Date in %m/%d/%Y format to start
        - to_date : (str) Date in %m/%d/%Y format to end
        """

    df = pd.read_csv(csv_route, skiprows=drop_first, skipfooter=drop_last, engine='python', encoding='utf-8')  # CSV
    df = df.rename({date_label: 'Date', ccaa_label: 'CCAA'}, axis=1)

    if sumcolumns:
        for col in sumcolumns:
            df[col] = df[sumcolumns[col]].apply(lambda x: np.sum(x), axis=1)

    df[df.columns.values[0]] = pd.to_datetime(df[df.columns.values[0]], dayfirst=dayfirst)  # Converts to timestamp

    if ccaa:
        df = df[df['CCAA'] == df]  # Filter by state when it's given

    if from_date:
        df = df[df['Date'] >= pd.to_datetime(from_date, format='%m/%d/%Y')]

    if to_date:
        df = df[df['Date'] <= pd.to_datetime(to_date, format='%m/%d/%Y')]

    if name_maps:
        df = df.rename(name_maps, axis=1)  # Rename using labels provided
        return df[['Date'] + list(name_maps.values())].groupby('Date').sum()
    return df.groupby('Date').sum()


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


def predict(y, filename='sample.csv'):

    """Predict the shape of a curve given a .csv file with a column named "date" for the time series and another
       column "y", which is the feature to predict.

           Parameters:
           - filename : (str) Name of the .csv file containing the time series
           - y : (str) Feature to predict"""

    file = pd.read_csv(filename)
    file['Date'] = pd.to_datetime(file['Date'])
    df = pd.DataFrame()
    df['ds'] = file['Date']
    df['y'] = file[y]
    file = file[['Date', y]].rename({y: 'Real'}, axis=1)

    model = Prophet(yearly_seasonality=False, n_changepoints=50, daily_seasonality=True, interval_width=0.999)
    with suppress_stdout_stderr():
        model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    fig.savefig('figures\\prophet_' + y.lower() + '.png')

    today = pd.to_datetime('today').date()
    today = np.datetime64(datetime.datetime(today.year, today.month, today.day, 0, 0, 0, 0))
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    sol = pd.concat([forecast.loc[forecast['ds'] == today + np.timedelta64(i, 'D'), cols] for i in [-1, 0, 1]])
    renaming = {'ds': 'Date', 'yhat': 'Feature', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}
    return sol.rename(renaming, axis=1).merge(file, on='Date', how='left')
