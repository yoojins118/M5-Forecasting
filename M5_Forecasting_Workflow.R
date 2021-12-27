#--------------------------------------------------------------------------------
# M5 Forecasting 
#--------------------------------------------------------------------------------

# This example serves to highlight the effectiveness of modeltime and we have
# explored several models in order to produce the optimal forecast for M5 dataset. 

# Firstly, we ran modelling for the three item categories (FOODS, HOBBIES, HOUSEHOLDS) 
# separately due to low computational power. With enough computational power, we would have
# been able to run the original "train" without making the workflow more complex. 

# Secondly, we have used a total of 8 different models. The models are mostly boosting 
# techniques because from the M5 competition, Boosting appears to give the most accurate
# predictions. The models we have used are XGBoost with 4 different hypertuning 
# parameters), ARIMA and AUTO ARIMA with XGBoost,Temporal Hierachical Forecasting 
# (THIEF) and Prophet with XGBoost. For each of time series, out of these 8 models, 
# the model with the lowest Root Mean Square Error (RSME) will be chosen to do the 
# 28 day forecasting. 

#--------------------------------------------------------------------------------
#### Loading the libraries  ----
#--------------------------------------------------------------------------------
library(modeltime)
library(tidymodels)
library(tidyverse)
library(timetk)
library(dplyr)
library(lubridate)
library(plotly)
library(xgboost)

#--------------------------------------------------------------------------------
#### Loading the files  ----
#--------------------------------------------------------------------------------
rm(list=ls())

path_root="."
sales <- read.csv(path_root, "sales_train_validation.csv", stringsAsFactors = F)
calendar <- read.csv(path_root, "calendar.csv", stringsAsFactors = F) ; calendar$date <- as.Date(calendar$date)
prices <- read.csv(path_root, "sell_prices.csv", stringsAsFactors = F)

#--------------------------------------------------------------------------------
#### 1. DATA PROCESSING  ----
#--------------------------------------------------------------------------------
### 1.1 Calendar features ----
# Modify the calendar dataset to make the first date as d_1, d_2 and so on
calendar$d <- paste("d_",as.character(1:nrow(calendar)),sep="")

# The dataset has "NA" values for the event_name and event_types. 
# We will fill this as "Nothing" so that it will be easier to create dummy variables
calendar[is.na(calendar)] = "Nothing"

# Melt the sales and left join with calendar. The joined calendar will be the calendar features
train <- sales %>% 
  pivot_longer(
    cols = starts_with("d_"),
    names_to = "day",
    values_to = "value"
  ) %>% 
  left_join(calendar, by = c('day' = 'd'))

# Take the columns to use
train <- train[,c('item_id','dept_id','cat_id','store_id','state_id',
                  'date','value','day','event_name_1','event_type_1',
                  'event_name_2','event_type_2','snap_CA','snap_TX','snap_WI')]

# Separation of the dataset into the 3 categories
train_hobbies <- train %>% filter(cat_id == 'HOBBIES') 
train_foods <- train %>% filter(cat_id == 'FOODS') 
train_household <- train %>% filter(cat_id == 'HOUSEHOLD')

# Creating a duplicate dataset
train_hobbies1 <- train_hobbies
train_foods1 <- train_foods
train_household1 <- train_household

#--------------------------------------------------------------------------------
### 1.2 Product release date ---
## Some products might not be released yet and were released later in the year. It does not make sense
## for the data to have irrelevant zeroes. As such, we would like to remove the irrelevant zeroes for
## some of the item_ids. We will try with hobbies category first.

# To determine when the product was released based on wm_yr_wk
product_release_date <- prices %>% group_by(store_id,item_id)%>% summarize(release= min(wm_yr_wk))

# Left join with the 3 categories dataframe
train_hobbies1<- left_join(train_hobbies1, product_release_date, by = c('store_id','item_id'))  
train_foods1<- left_join(train_foods1, product_release_date, by = c('store_id','item_id'))
train_household1<- left_join(train_household1, product_release_date, by = c('store_id','item_id'))

# Subsetting the calendar columns (my select function has a problem? WHY???)
calendar1 <- calendar[,c('wm_yr_wk','d')]

# Joining with 'd' 
train_hobbies1 <- left_join(train_hobbies1, calendar1, by = c('day' = 'd'))  
train_foods1 <- left_join(train_foods1, calendar1, by = c('day' = 'd'))
train_household1 <- left_join(train_household1, calendar1, by = c('day' = 'd'))

## Remove some rows with 0 sales until it was released. This process also frees up some memory
train_hobbies2 <- train_hobbies1 %>% filter(train_hobbies1$wm_yr_wk >= train_hobbies1$release) 
train_foods2 <- train_foods1 %>% filter(train_foods1$wm_yr_wk >= train_foods1$release) 
train_household2 <- train_household1 %>% filter(train_household1$wm_yr_wk >= train_household1$release) 

#--------------------------------------------------------------------------------
### 1.3 Price features ---
# Basic price aggregations containing max, mean and standard deviation
price_max <- prices %>% group_by(store_id,item_id)%>%summarize(price_max = max(sell_price))
price_sd <- prices %>% group_by(store_id,item_id)%>%summarize(price_sd = sd(sell_price))
price_mean <- prices %>% group_by(store_id,item_id)%>%summarize(price_mean = mean(sell_price))
price_count_prices <- prices %>% group_by(store_id,item_id)%>%summarize(count_prices = n_distinct(sell_price)) #number of times price changed

# Left join aggregations with the "prices" dataset. 
prices <- prices %>%
  left_join(price_max, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id"))
prices <- prices %>%
  left_join(price_sd, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id"))
prices <- prices %>%
  left_join(price_mean, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id"))
prices <- prices %>%
  left_join(price_count_prices, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id"))
rm(price_max)
rm(price_sd)
rm(price_mean)
rm(price_count_prices)

# Normalisation of the price
prices$price_norm <- prices$sell_price/prices$price_max

# Merging the selling_prices data with train data 
train_hobbies2 <- train_hobbies2 %>%
  left_join(prices, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id",
                   "wm_yr_wk" = "wm_yr_wk"))

train_foods2 <- train_foods2 %>%
  left_join(prices, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id",
                   "wm_yr_wk" = "wm_yr_wk"))

train_household2 <- train_household2 %>%
  left_join(prices, 
            by = c("store_id" = "store_id",
                   "item_id" = "item_id",
                   "wm_yr_wk" = "wm_yr_wk"))


#--------------------------------------------------------------------------------
### 1.4 Remove columns that are not needed
train_hobbies_final <- train_hobbies2[,c('item_id','store_id','date','value','event_name_1',
                                         'event_type_1', 'event_name_2','event_type_2',
                                         'snap_CA','snap_TX','snap_WI','sell_price', 'price_max',
                                         'price_sd','price_mean', 'count_prices','price_norm')]

train_foods_final <- train_foods2[,c('item_id','store_id','date','value','event_name_1',
                                      'event_type_1', 'event_name_2','event_type_2',
                                      'snap_CA','snap_TX','snap_WI','sell_price', 'price_max',
                                      'price_sd','price_mean', 'count_prices','price_norm')]

train_household_final <- train_household2[,c('item_id','store_id','date','value','event_name_1',
                                              'event_type_1', 'event_name_2','event_type_2',
                                              'snap_CA','snap_TX','snap_WI','sell_price', 'price_max',
                                              'price_sd','price_mean', 'count_prices','price_norm')]

#--------------------------------------------------------------------------------
### 1.5 Changing to factor types ---
## For the creation of our dummy variable creation in the Recipe part.
train_foods_final$snap_CA <- as.factor(train_foods_final$snap_CA)
train_foods_final$snap_TX <- as.factor(train_foods_final$snap_TX)
train_foods_final$snap_WI <- as.factor(train_foods_final$snap_WI)
train_foods_final <- train_foods_final %>% mutate_if(is.character,factor)

train_hobbies_final$snap_CA <- as.factor(train_hobbies_final$snap_CA)
train_hobbies_final$snap_TX <- as.factor(train_hobbies_final$snap_TX)
train_hobbies_final$snap_WI <- as.factor(train_hobbies_final$snap_WI)
train_hobbies_final <- train_hobbies_final %>% mutate_if(is.character,factor)

train_household_final$snap_CA <- as.factor(train_household_final$snap_CA)
train_household_final$snap_TX <- as.factor(train_household_final$snap_TX)
train_household_final$snap_WI <- as.factor(train_household_final$snap_WI)
train_household_final <- train_household_final %>% mutate_if(is.character,factor)

#--------------------------------------------------------------------------------
### Free up more memory
rm(prices)
rm(calendar1)
rm(calendar)
rm(sales)
rm(product_release_date)
rm(train)
rm(train_hobbies2)
rm(train_foods2)
rm(train_household2)

#--------------------------------------------------------------------------------
#### 2. MODELLING APPROACH---- 
#--------------------------------------------------------------------------------
### 2.1 Sampling by store_id---
## From each of the different categories, select a store_id that will be used to conduct
## our modelling. 
sample_tbl<- train_hobbies_final %>%  #(*Choose category; choose from: train_hobbies_final, train_hobbies_final or train_foods_final) 
  filter(store_id == "CA_1") %>% #(*Choose store_id; choose from: CA_01, CA_02, CA_03, CA_04, TX_01, TX_02, TX_03, WI_01, WI_02, WI_03)

## Visualise the selected item and store as a time series
sample_tbl%>%
  group_by(item_id) %>%
  plot_time_series(date, value, .facet_ncol = 3, .smooth = FALSE) 

#--------------------------------------------------------------------------------
### 2.2 Create NESTED TIME SERIES + Feature engineering (Rolling lag features) ---- 
nested_data_tbl <- sample_tbl %>%
  group_by(item_id) %>%
  extend_timeseries(
    .id_var = item_id,
    .date_var = date,
    .length_future = 28  #Add a bunch of 28 NAs in the time series (To predict 28 days)
  ) %>% 
  
  # Add in time series features (Rolling lag features)
  tk_augment_lags(value, .lags = 28) %>%
  tk_augment_slidify(
    value_lag28,
    .f = ~mean(.,na.rm= TRUE),
    .period = c(7,14,28,28*2),
    .align = "center",
    .partial = TRUE
  ) %>%
  
  nest_timeseries(
    .id_var = item_id, 
    .length_future = 28
  ) %>%
  split_nested_timeseries(
    .length_test = 365 # Train-Test Spilt: 1 year worth of testing because of  year seasonality
  )

nested_data_tbl %>% tail()

#--------------------------------------------------------------------------------
### 2.3 XGBoost Recipe ---
rec_xgb <- recipe(value ~ ., extract_nested_train_split(nested_data_tbl)) %>%   
  step_timeseries_signature(date) %>% 
  #Takes a date and breaks it up into year, month, quarter etc
  step_lag(value, lag = 7:28) %>% 
  #Impute lag days (lag 7 to lag 28)
  step_rm(date) %>%
  #remove date column
  step_zv(all_predictors()) %>%
  #step_zero variance, removes all features that do not have any variance
  step_dummy(all_nominal_predictors(), one_hot = TRUE) # rec_xgl %>% prep() shows all the operations

bake(prep(rec_xgb), extract_nested_train_split(nested_data_tbl)) %>% glimpse()

#--------------------------------------------------------------------------------
### 2.4 Hyperparameter tuning for XGBoost Models---
wflw_xgb_1 <- workflow() %>%
  add_model(boost_tree("regression", learn_rate = 0.001) %>% 
              set_engine("xgboost")) %>%
  add_recipe(rec_xgb)

wflw_xgb_2 <- workflow() %>%
  add_model(boost_tree("regression", learn_rate = 0.010) %>% 
              set_engine("xgboost")) %>%
  add_recipe(rec_xgb)

wflw_xgb_3 <- workflow() %>%
  add_model(boost_tree("regression", learn_rate = 0.100) %>% 
              set_engine("xgboost")) %>%
  add_recipe(rec_xgb)

wflw_xgb_4 <- workflow() %>%
  add_model(boost_tree("regression", learn_rate = 0.35) %>% 
              set_engine("xgboost")) %>%
  add_recipe(rec_xgb)

#--------------------------------------------------------------------------------
### 2.5 Temporal Hierachical Forecasting (THIEF) ----
wflw_thief <- workflow() %>%
  add_model(temporal_hierarchy() %>% set_engine("thief")) %>%
  add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl))%>%
               step_zv(all_predictors()))

#--------------------------------------------------------------------------------
### 2.6 "Boosted" ARIMAs ---
wflw_ARIMA <- workflow() %>% 
  add_model(arima_boost() %>% set_engine("arima_xgboost")) %>%
  add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl))%>%
               step_zv(all_predictors()))

wflw_AUTO_ARIMA <- workflow() %>% 
  add_model(arima_boost() %>% set_engine("auto_arima_xgboost")) %>%
  add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl))%>%
               step_zv(all_predictors()))

#--------------------------------------------------------------------------------
### 2.7 Boosted PROPHET ---
wflw_PROPHET <- workflow() %>% 
  add_model(prophet_boost() %>% set_engine("prophet_xgboost")) %>%
  add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl))%>%
               step_zv(all_predictors()))

#--------------------------------------------------------------------------------
### 2.8 Save models as RDS 
#Model_list
models_m5_forecasting<-list(
  wflw_xgb_1,
  wflw_xgb_2,
  wflw_xgb_3,
  wflw_xgb_4,
  wflw_thief,
  wflw_ARIMA,
  wflw_AUTO_ARIMA,
  wflw_PROPHET
)

#Save as RDS
saveRDS(models_m5_forecasting, file = "M5_Forecasting_Models.rds")
#--------------------------------------------------------------------------------
#### 3. Modelling 
#--------------------------------------------------------------------------------
### 3.1 TRY 1 TIME SERIES ----
try_sample_tbl <- nested_data_tbl %>%
  filter(as.numeric(item_id) %in% 1) %>%
  modeltime_nested_fit(
    model_list = list(
      wflw_xgb_1,
      wflw_xgb_2,
      wflw_xgb_3,
      wflw_xgb_4,
      wflw_thief,
      wflw_ARIMA,
      wflw_AUTO_ARIMA,
      wflw_PROPHET
    ),
    control = control_nested_fit(
      verbose   = TRUE,
      allow_par = FALSE
    )
  )

try_sample_tbl 

#Check Errors ----
try_sample_tbl %>% extract_nested_error_report()

#--------------------------------------------------------------------------------
### 3.2 SCALE ----
# Option 1 - Local CPUs
parallel_start(8)

# Option 2 - Local Spark Session
#options(sparklyr.console.log = TRUE)
#sc <- spark_connect(master = "local")
#parallel_start(sc, .method = "spark")

nested_modeltime_tbl <- nested_data_tbl %>%
  # slice_tail(n = 6) %>%
  modeltime_nested_fit(
    model_list = list(
      wflw_xgb_1,
      wflw_xgb_2,
      wflw_xgb_3,
      wflw_xgb_4,
      wflw_thief,
      wflw_ARIMA,
      wflw_AUTO_ARIMA,
      wflw_PROPHET
    ),
    control = control_nested_fit(
      verbose   = TRUE,
      allow_par = TRUE
    )
  )

nested_modeltime_tbl

#Review Any Errors (Validation) ----
nested_modeltime_tbl %>% extract_nested_error_report()

#Review Test Accuracy (Validation) ----
nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  table_modeltime_accuracy()

#Visualize Test Forecast (Validation) ----
nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  filter(item_id == "HOBBIES_1_070") %>% #Anyhow choose one 
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)


nested_modeltime_subset_tbl <- nested_modeltime_tbl

#--------------------------------------------------------------------------------
#### 4. MODEL SELECTION
#--------------------------------------------------------------------------------
### 4.1 SELECT BEST ----
nested_best_tbl <- nested_modeltime_subset_tbl %>% 
  modeltime_nested_select_best(metric = "rmse")

#Visualize Best Models ----
nested_best_tbl %>%
  extract_nested_test_forecast() %>%
  filter(as.numeric(item_id) %in% 1:12) %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)

#--------------------------------------------------------------------------------
### 4.2 REFIT (Forecasting of the 28 days) ----
nested_best_refit_tbl <- nested_best_tbl %>%
  modeltime_nested_refit(
    control = control_refit(
      verbose   = TRUE,
      allow_par = TRUE
    )
  )

#Review Any Errors ----
nested_best_refit_tbl %>% extract_nested_error_report()

#Visualize Future Forecast ---- 
nested_best_refit_tbl %>%
  extract_nested_future_forecast() %>%
  filter(as.numeric(item_id) %in% 1:12) %>%
  group_by(item_id) %>%
  plot_modeltime_forecast(.facet_ncol = 3)

#--------------------------------------------------------------------------------
####5. Extracting Predictions into CSV file 
#--------------------------------------------------------------------------------
final_forecasted_volumes_tbl <- nested_best_refit_tbl %>% 
  extract_nested_future_forecast(.include_actual=FALSE) %>%  #taking 28 days of forecast only 
  select (-.model_id, -.key, -.model_desc, -.conf_lo, -.conf_hi)  %>%  #remove columns not needed
  spread (key = ".index", value = ".value") 

final_forecasted_volumes_tbl$store_id<-"CA_01" #(*put the store_id here*)

final_forecasted_volumes_tbl <-final_forecasted_volumes_tbl %>% unite( "id", c(item_id, store_id), remove=TRUE)

colnames(final_forecasted_volumes_tbl) <- c("id", "F1", "F2", "F3", "F4", "F5", "F6", 
                                            "F7", "F8", "F9", "F10", "F11", "F12", 
                                            "F13", "F14", "F15", "F16", "F17", "F18", 
                                            "F19", "F20", "F21", "F22", "F23", "F24", 
                                            "F25", "F26", "F27", "F28")

write.csv(final_forecasted_volumes_tbl, file="HOBBIES_CA_01.csv") #(*name the csv here*)

#--------------------------------------------------------------------------------
#### APPENDIX
#--------------------------------------------------------------------------------
###Feature Importance - XGBoost
fit_xgboost <- nested_modeltime_tbl %>%
  filter(as.numeric(item_id) %in% 3)  %>%
  extract_nested_modeltime_table() %>%
  pluck(".model", 2)

fit_xgboost$fit$fit$fit %>% class()

xgboost::xgb.importance(model = fit_xgboost$fit$fit$fit) %>%
  as_tibble() %>%
  mutate(Feature = as_factor(Feature) %>% fct_reorder(Gain)) %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_point()
