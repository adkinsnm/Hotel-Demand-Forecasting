# **Hotel Demand Forecasting — ISA 444 Final Project**

*Forecasting daily demand for 17 hotel properties using classical, machine learning, deep learning, and foundation models.*

This project implements a complete end-to-end forecasting workflow using many of the models and techniques we learned throughout the year in ISA444.
The goal is to forecast 28 days of hotel demand using multiple forecasting families and compare their performance.

## **Project Overview**

This repository contains the full workflow used to:

* Load and preprocess hotel demand data

* Perform 5-fold rolling time-series cross-validation

* Fit the forecasting and evaluate

* Select the best model based on accuracy metrics

* Generate final 28-day forecasts

* Visualize actual vs. forecasted demand for each property

* Save evaluation results and final predictions to CSV files

## **Dataset**

Source: sample_hotels.parquet (not included due to agreement with partnering company)

Structure:

* `unique_id` — hotel property ID

* `ds` — timestamp (daily)

* `y` — observed demand


**Data Preparation**

*Renamed columns into standard forecasting format:*

* Property → `unique_id`  
* Date     → `ds`  
* Demand   → `y`

*Convert*

* ds → datetime

* categorical features → category dtype

* y → float

*Sorted by hotel and date*

* Ensures proper temporal ordering.

*Removed problematic hotel*

* `Hotel_77` had long a long period of zero demand.
It was excluded because it was breaking TimeGPT and would degrade the model accuracy.

*Train/test split*

* `Train`: before 2023-06-01

* `Test`: on/after 2023-06-01

**Cross-Validation Strategy**

All models used 5-fold non-overlapping time-series cross-validation:

* `Horizon`: 28 days

* `Step size`: 28 days

* `Windows`: 5

* Evaluated separately per hotel (`unique_id`)

## Models Implemented
### 1. Baseline Statistical Models (StatsForecast)

* `Naive`

This model is great for creating a baseline in forecasting. The forecast for all future periods is equal to the last observed value. Any other model should outperform the Naive model to be considered useful. It's downsides are that it doens't take any seasonality or trends into account.

* `Seasonal Naive`

This is a very similar model to the `Naive` Model, but the difference is that it takes values from the last season (weekly) rather than just the last observed value. The fact that it captures seasonality and the patterns caused by it would make it slightly more accurate than the classic `Naive`, but it is still considered to be a baseline model for other models to beat.

* `AutoETS`

This is an exponential smoothing model that automatically selects the components it needs. ETS models are great for series with trends & seasonality and are very popular in forecasting in general. Overall, it's considered to be a relatively accurate model.
* `AutoARIMA`

Automatically fits an ARIMA(p,d,q) model using statistical patterns in autocorrelation to model the future. This is one of the more popular models that was used in this project and also ended up being one of the most accurate. It handles trends and autocorrelation very well, making it a common use for those looking to forecast future values. 

### 2. Machine Learning Model (MLForecast)

Model Implemented: `LightGBM Regressor`

* Lag features: 1, 7, 14, 28

* Rolling statistical transforms

* Automatic feature engineering

* Cross-validated using MLForecast’s built-in CV

* These models operate on a transformed cross-sectional version of the time series.

### 3. Foundation Model — `TimeGPT` (Nixtla)

Model Implemented: `TimeGPT`

* TimeGPT was the **strongest** performing model.

* Automatically handles seasonality, trends, holidays, covariates

* Learns from massive general pretraining across global time-series

* Requires no manual feature engineering

4. Deep Learning Models — (NeuralForecast)

* `AutoNBEATS`

A deep learning model based on fully connected neural networks that uses backward and forward stacks to model trend, seasonality, and overall generic patterns. One of the reasons it would be used is because it learns patterns and trends directly from the data without any manipulation needing to take place. Along with this, it's best for medium to long-term horizons, which is what we were dealing with in this scenario.

* `AutoNHITS`

This was actually a successor to the `NBEATS` model and is meant to be an improvement on it. It's widely considered to be one of the more accurate models at forecasting, and has placed very high in competitions in that field. The improvements were to its ability forecast from longer horizons than `NBEATS`.


### Evaluation showed `TimeGPT` consistently achieved the lowest MAE and RMSE for the majority of hotels.

**Model Comparison & Selection**
Metrics:

* `ME`

* `RMSE`

* `MAPE` (used in original evaluation, but realized it may be misleading)

*Counted “wins”*

For each hotel and each metric, the model with the lowest error was recorded.
TimeGPT won most hotels across both `RMSE` and `MAE`.

Final Selected Model: `TimeGPT`

**Final 28-Day Forecast**


`timegpt_forecast = nixtla_client.forecast(
    df=train_clean,
    h=28,
    freq="D",
    time_col="ds",
    target_col="y"
)`


Forecasts were then merged with actual test values to compute final RMSE/MAE on the holdout period.

**Visualization**

For each of the 17 hotels, the following were plotted:

![Hotel Demand](C:/User/Nate Adkins/Pictures/Screenshots/hotel_forecasting_image.png)

* Historical demand

* Forecasted demand

* Separate plots for each model (Naive, ETS, ARIMA, ML, TimeGPT)



**Saved Outputs**

The following files were exported as CSV:

final_eval_table.csv

final_forecasting_table.csv

final_testing_output.csv

**What I Learned:**

* How to design a full forecasting pipeline

From data prep → cross-validation → model comparison → final evaluation.

* The importance of time-series cross-validation

Unlike normal CV, it respects temporal order and avoids leakage.

* How statistical, ML, DL, and AI foundation models differ

Naive/SeasonalNaive: yes

AutoETS/AutoARIMA: fast, classical baselines

LightGBM (MLForecast): needs feature engineering + lag structure

TimeGPT: massively pretrained → strong generalization

**Why TimeGPT performed best**

This model was by far the most accurate, putting the others to shame on how many times it had the best evaluation metrics. 
The reason it performed so much better was its ability to adapt to each hotel and series. 

**Challenges Solved During the Process**

* Handling broken/missing timestamps

* Dropping hotels that didn't have the proper data

* Merging actuals and forecasts for evaluation

* Understanding neuralforecast APIs

* Making changes to data to fit different models

**Final Summary**

This project demonstrates a full forecasting workflow using multiple modeling paradigms.
After evaluating all models under identical conditions, TimeGP due to its superior error performance and visual alignment with actual demand.
