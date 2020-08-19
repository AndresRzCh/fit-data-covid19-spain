# Fit Data COVID-19

Fit updated data for COVID-19 (https://github.com/datasets/covid-19) to different models, 
or compare cases between countries.

In this particular case, for Spain, the SIR Model gives the following figure:

![](figures/sir_model.png  "SIR Model")

Fitting the logistic model to new dates we get the following:

![](figures/logistic_model_confirmed.png "Logistic Model" )

Daily new cases and deaths for Spain:

![](figures/daily.png "Daily Cases")

Different confirmed cases for all the counties in Spain:

![](figures/compare.png "Compare Countries")

Simple prediction using Facebook Prophet's Time Series:
 
![](figures/prophet_ia.png "IA")

## Usage example

Edit `tests.py` to use the desired parameters, and then run `python tests.py`. 
All plots will be created in `~\figures`.
