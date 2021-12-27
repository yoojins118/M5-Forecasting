# M5-Forecasting

This example serves to highlight the effectiveness of modeltime and we have
explored several models in order to produce the optimal forecast for M5 dataset (https://www.kaggle.com/c/m5-forecasting-accuracy). 

Objective: Estimmate the unit sales of Walmart retail goods. 

Firstly, we ran modelling for the three item categories (FOODS, HOBBIES, HOUSEHOLDS) 
separately due to low computational power. With enough computational power, we would have
been able to run the original "train" without making the workflow more complex. 

Secondly, we have used a total of 8 different models. The models are mostly boosting 
techniques because from the M5 competition, Boosting appears to give the most accurate
predictions. The models we have used are XGBoost with 4 different hypertuning parameters), 
ARIMA and AUTO ARIMA with XGBoost,Temporal Hierachical Forecasting 
(THIEF) and Prophet with XGBoost. For each of time series, out of these 8 models, 
the model with the lowest Root Mean Square Error (RSME) will be chosen to do the 
28 day forecasting. 
