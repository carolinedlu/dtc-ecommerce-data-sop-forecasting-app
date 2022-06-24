# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:10:39 2021

@author: 40102891
"""

from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit,KFold
from sklearn.model_selection import train_test_split
import pickle

#######################################  Global Variables  ####################

start_train_date = '2018-02-18'
end_train_date = '2020-04-26'
start_test_date = '2020-05-03'
end_test_date = '2020-07-26'
start_pred_date = '2020-08-02'
end_pred_date = '2020-11-08'

###############################  Data  ########################################

# df = pd.read_csv('sample-df-2.csv')

###############################  Class Definition  ############################

class Model:
    def __init__(self, tso, tst):
        self.tso = tso # time-series object
        self.tst = tst # transformation object
        
        # Auxilliary variables
            # Each Model
        self.predictions_list = []
        self.model_list = []
            # Summary Table
        self.summary_table = []
            # Best Model
        self.best_mape = None
        self.best_smape = None
        self.best_predictions = None
        self.best_model_name = None
        self.best_model = None
        self.cv_split = None
        self.model_kind = None
        self.model_mape = None
        self.model_smape = None
        self.model_predictions = None

    
    def naive(self, plot = False):
        
        # Fitting to Train + Test
        y_pred_sb = pd.Series(data = [self.tso.y_test[-1]]*len(self.tso.X_pred),
                                   index = self.tso.X_pred.index)
        
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)

        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Naive')
            plt.legend()
            plt.show()
        
        
    
        # Update list
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Naive')

        return y_pred

    def seasonal_naive(self,seasonal_period = 52, plot = False):
        
        # Fitting to Train + Test
        end = -seasonal_period + len(self.tso.X_pred)
        if end == 0:
            end = None
        
        y_pred_sb = pd.Series(data = pd.concat([self.tso.y_train, self.tso.y_test]).values[-seasonal_period:end],
                              index = self.tso.X_pred.index)

        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Seasonal Naive')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Seasonal Naive')

        return y_pred
        
        
    def elastic_net(self, plot = False):
        model = self.best_model
        y_pred_sb = self.generate_future_preds("elastic_net",model)
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)

        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Elastic Net')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Elastic-Net')

        return y_pred
    
    def linear_regression(self, plot = False):
        linreg_model = self.best_model
        y_pred_sb = self.generate_future_preds("linreg",linreg_model)
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)

        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb

        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Linear Regression')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Linear Regression')

        return y_pred 
    
    def random_forest(self, plot = False):
        rf_model = self.best_model
        y_pred_sb = self.generate_future_preds("rf",rf_model)
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)

        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Random Forest')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Random Forest')
        
        return y_pred    

    def xgboost(self, plot = False):
        xgb_model = self.best_model
        y_pred_sb = self.generate_future_preds("xgb",xgb_model)
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)

        ##Rescaling
        #mean_val = self.tst.SS_df['response'][0]
        #variance_val = self.tst.SS_df['response'][1]
        #y_pred_sb = y_pred_sb.apply(lambda x: x * variance_val + mean_val)
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for XGBoost')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('XGBoost')

        return y_pred
    
    def light_gbm(self, plot = False):
        lgbm_tuned = self.best_model
        y_pred_sb = self.generate_future_preds("lgb",lgbm_tuned)
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)

        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Light GBM')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Light GBM')

        return y_pred    

    def moving_average(self, seasonal_period = 52,
                       window_size = 2, 
                       growth_factor = False,
                       plot = False):
         
        # Fitting to Train + Test
        y_pred_sb = np.zeros(shape=len(self.tso.X_pred))
        if growth_factor:
            for i in range(len(y_pred_sb)):
                y_pred_sb[i] = np.mean(
                                pd.concat([self.tso.y_train, self.tso.y_test]).values[-seasonal_period + i - window_size - 1: -seasonal_period + i - 1])
        else:
            num = np.zeros(shape = len(self.tso.X_pred))
            den = np.zeros(shape = len(self.tso.X_pred))
            for i in range(len(y_pred_sb)):
                num[i] = np.mean(pd.concat([self.tso.y_train, self.tso.y_test]).values[-seasonal_period + i - window_size - 1: -seasonal_period + i - 1])
                den[i] = np.mean(pd.concat([self.tso.y_train, self.tso.y_test]).values[-2*seasonal_period + i - window_size - 1: -2*seasonal_period + i - 1])
                y_pred_sb[i] = (num[i]/den[i]) * pd.concat([self.tso.y_train, self.tso.y_test]).values[-seasonal_period + i -window_size]

        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index=self.tso.X_pred.index)        
        
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Moving Average')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Moving Average')

        return  y_pred         
                
    def simple_exp_smoothing(self, plot = False):
        ses_model = self.best_model
        y_pred_sb = ses_model.forecast(len(self.tso.X_pred))
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)
        
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Simple Exponential Smoothing')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Simple Exponential Smoothing')

        return y_pred
        
    def holt(self, plot = False):
        holt_model = self.best_model
        y_pred_sb = holt_model.forecast(len(self.tso.X_pred))
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)
        
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Holt Exponential Smoothing')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Holt Exponential Smoothing')

        return y_pred
    
    def holt_winters(self, plot = False):
        holt_model = self.best_model
        y_pred_sb = holt_model.forecast(len(self.tso.X_pred))
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)

        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for Holt-Winters Exponential Smoothing')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('Holt-Winters Exponential Smoothing')

        return y_pred    
    
    def sarimax(self, plot = False):
        model = self.best_model
        y_pred_sb = self.generate_future_preds("sarimax",model)
        y_pred_sb = pd.Series(data = y_pred_sb, 
                              index = self.tso.X_pred.index)
        
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order>0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
            # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)
        
        if plot:
            # Visualize
            plt.figure(figsize = (14,6))
            plt.plot(self.tst.y, label = 'Historical')
            #plt.plot(y_test, label = 'Actuals')
            plt.plot(y_pred, label = 'Predictions')
            plt.title('Forecast vs Actuals and Predictions for SARIMAX')
            plt.legend()
            plt.show()
        
        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('SARIMAX')

        return y_pred

    def sarima(self, plot = False):
        
        model = self.best_model
        y_pred_sb = model.forecast(len(self.tso.X_pred),
                                   exog=None)
        y_pred_sb = pd.Series(data=y_pred_sb,
                              index=self.tso.X_pred.index)
        
        #Rescaling
        if self.tst.ss_y != None:
            y_pred_sb = self.tst.ss_y.inverse_transform(np.array(y_pred_sb).reshape(-1,1))
            y_pred_sb=pd.Series(data=y_pred_sb.reshape(1,- 1)[0],index=self.tso.X_pred.index)
        # Destationarizing
        diff_order = self.tst.y_diff_order
        y_stationary_history = self.tst.y_stationary_history
        while diff_order > 0:
            y_pred_sb = y_stationary_history['diff_{}'.format(diff_order - 1)].iloc[
                            len(self.tso.y_train) + len(self.tso.y_test) - 1] + \
                        np.cumsum(y_pred_sb)
            diff_order = diff_order - 1
        y_pred_b = y_pred_sb
        # Inverse-Boxcox
        # Predictions
        y_pred = self.tst.inverse_boxcox(y_pred_b)

        if plot:
            # Visualize
            plt.figure(figsize=(14, 6))
            plt.plot(self.tst.y, label='Historical')
            #plt.plot(y_test, label='Actuals')
            plt.plot(y_pred, label='Predictions')
            plt.title('Forecast vs Actuals and Predictions for SARIMA')
            plt.legend()
            plt.show()

        
        self.predictions_list.append(y_pred.values.tolist())
        self.model_list.append('SARIMA')

        return y_pred

    def get_inference_results(self,best_model_name, item_name):
        
        self.best_model = self.get_best_model(best_model_name,item_name)
        if best_model_name == 'Naive':
            self.naive()
        elif best_model_name == 'Linear Regression':
            self.linear_regression()
        elif best_model_name == 'Elastic-Net':
            self.elastic_net()
        elif best_model_name == 'Random Forest':
            self.random_forest()
        elif best_model_name == 'XGBoost':
            self.xgboost()
        elif best_model_name == 'Light GBM':
            self.light_gbm()
        elif best_model_name == 'Seasonal Naive':
            self.seasonal_naive()
        elif best_model_name == 'Moving Average':
            self.moving_average()
        elif best_model_name == 'Moving Average Growth Factor':
            self.moving_average(growth_factor=True)
        elif best_model_name == 'Simple Exponential Smoothing':
            self.simple_exp_smoothing()
        elif best_model_name == 'Holt Exponential Smoothing':
            self.holt()
        elif best_model_name == 'Holt-Winters Exponential Smoothing':
            self.holt_winters()
        elif best_model_name == 'SARIMA':
            self.sarima()
        else:
            self.sarimax()
        self.results_table = pd.DataFrame(data = {'model_name':self.model_list,
                                                  'predictions':self.predictions_list},
                                          index = [i+1 for i in range(len(self.model_list))])
        return self.results_table
    
    def generate_future_preds(self,model_name,model):
        y_all = pd.concat([self.tso.y_train[3:], self.tso.y_test])
        for i in range(0,len(self.tso.X_pred.index)-1):
            self.tso.X_pred.iloc[[i]]=self.tso.X_pred.iloc[[i]].fillna(0)
            if model_name == 'sarimax':
                y_pred_next = model.forecast(len(self.tso.X_pred.iloc[[i]]),exog = self.tso.X_pred.iloc[[i]])
            else:
                y_pred_next = model.predict(self.tso.X_pred.iloc[[i]])
            y_all=pd.concat([y_all,pd.Series(y_pred_next,index = self.tso.X_pred.iloc[[i]].index)])
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1']=y_all.iloc[-1]
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted2']=y_all.iloc[-2]
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted3']=y_all.iloc[-3]
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted4']=y_all.iloc[-4]
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted5']=y_all.iloc[-5]
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted6']=y_all.iloc[-6]
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1min']=y_all.iloc[-12:].min()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1max']=y_all.iloc[-12:].max()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1sum']=y_all.iloc[-12:].sum()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1mean']=y_all.iloc[-12:].mean()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1std']=y_all.iloc[-12:].std()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1min_4']=y_all.iloc[-4:].min()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1max_4']=y_all.iloc[-4:].max()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1sum_4']=y_all.iloc[-4:].sum()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1mean_4']=y_all.iloc[-4:].mean()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1std_4']=y_all.iloc[-4:].std()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1min_8']=y_all.iloc[-8:].min()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1max_8']=y_all.iloc[-8:].max()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1sum_8']=y_all.iloc[-8:].sum()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1mean_8']=y_all.iloc[-8:].mean()
            self.tso.X_pred.loc[self.tso.X_pred.index[i+1],'response|shifted1std_8']=y_all.iloc[-8:].std()
        
        y_pred_next_all = y_all.iloc[-len(self.tso.X_pred.index):]
        return y_pred_next_all

    def load_summary_table(self, item_name):
        filename = str(r'./best_models/')+str(item_name)+str(r'/')+str(item_name)+str(r'metrics_.csv')
        summary_table = pd.read_csv(filename)
        return summary_table
        
    def get_best_model(self,best_model_name,item_name):
        if best_model_name == 'Linear Regression':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'Elastic-Net':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'Random Forest':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'XGBoost':              
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'Light GBM':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'Simple Exponential Smoothing':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'Holt Exponential Smoothing':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'Holt-Winters Exponential Smoothing':    
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'SARIMA':
            filename = str(r'./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        elif best_model_name == 'SARIMAX':
            filename = str('./best_models/')+str(item_name)+'/'+str(item_name)+"_"+str(best_model_name)+'_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        else:
            print("Model results doesnot vary much with input")
            loaded_model = None
        return loaded_model


    def find_best_model(self, summary_table, criteria = 'smape',):
        min_metric = summary_table[criteria].min()
        best_model_stats = summary_table[summary_table[criteria] == min_metric]
        self.best_mape = best_model_stats.mape.values[0]
        self.best_model_name = best_model_stats.model_name.values[0]
        self.best_predictions = pd.Series(data = best_model_stats.predictions.values[0],
                                          index = self.tso.X_pred.index)
        self.best_smape = best_model_stats.smape.values[0]
        return self.best_model_name
    
    
#####################################  Testing  ###############################

if __name__ == '__main__':
    df = pd.read_csv('sample-df-4.csv')
    
    start_train_date = '2018-02-18'
    end_train_date = '2020-04-26'
    start_test_date = '2020-05-03'
    end_test_date = '2020-07-19'
    start_pred_date = '2020-07-26'
    end_pred_date = '2020-11-08'
    
    ts = timeSeriesObject.TimeSeriesObject(df)
    ts.set_date_column(date_column = 'order_date')    
    ts.drop_col('sales_units')
    ts.xy_split(response_variable = 'sales_value')
    ts_analyse = analyse.Analyse(ts)
    ts_analyse.visual_summary(ts.y)
    ts_transform = transform.Transform(ts)
    ts_transform.do_boxcox()
    ts_y_stationary = ts_transform.stationarize_y()
    ts_analyse.visual_summary(ts_y_stationary.dropna())
    ts_X_stationary = ts_transform.stationarize_X()
    ts.train_test_split(start_train_date = start_train_date,
                        end_train_date = end_train_date,
                        start_test_date = start_test_date,
                        end_test_date = end_test_date,
                        start_pred_date = start_pred_date,
                        end_pred_date = end_pred_date)   
    models = Model(tso = ts, tst = ts_transform)
    summary_table = models.construct_summary_table()
    best_model = models.find_best_model()