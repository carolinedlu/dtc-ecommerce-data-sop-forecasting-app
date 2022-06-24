# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 06:42:15 2021

@author: 40102891
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import seaborn as sns
# import timeSeriesObject, analyse
from scipy.stats import boxcox
from math import exp, log
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler

#############################  Data  ##########################################

# df = pd.read_csv('sample-df-2.csv')

#################################### Class Definitions ########################

class Transform:
    def __init__(self, tso):
        # Initializing Attributes
        self.tso = tso
        # Auxilliary Attributes
            # X-y 
        self.X = tso.X
        self.y = tso.y
            # y boxcox transformed
        self.yb = tso.y
        self.y_lambda = None
            # Stationary X-y
        self.Xs = tso.X
        self.ys = tso.y

        self.n_lag = None
        self.lags = {}
        self.y_stationary_history = self.tso.y
        self.y_diff_order = None
        self._selected_features = tso._selected_features
        self.ss_x = None
        self.ss_y = None
        
    def do_boxcox(self, lmbda = None, modify = False):
        if modify:
            if lmbda == None:
                yb, self.y_lambda = boxcox(x = self.tso.y.dropna())
            else:
                yb = boxcox(x = self.tso.y.dropna(),
                            lmbda = lmbda)
                self.y_lambda = lmbda
            self.yb = pd.Series(data = list(yb) + [np.nan]*(len(self.tso.y) - len(yb)),
                                index = self.tso.y.index)
            return self.yb
        else:
            self.y_lambda = 1
            self.yb = self.tso.y
            return self.yb
    
    def inverse_boxcox(self, series):
        if self.y_lambda == 0:
            return pd.Series(data = exp(series), index = series.index)
        else:
            return pd.Series(
                data = np.exp(np.log(np.abs(self.y_lambda * np.abs(series.values) + 1))/self.y_lambda),
                index = series.index)

    
    def unstationarize_v2(self, series, initial_vals):
        z = series.values
        for initial_val in initial_vals:
            z = initial_val + np.cumsum(z)
            
        pass
    
    def stationarize_y(self, modify = False):
        if modify == False:
            self.ys = self.yb.rename('diff_0')
            self.y_stationary_history = self.yb.rename('diff_0')
            self.y_diff_order = 0
            self.tso.ys = self.ys
            return self.ys
        else:
            series = self.yb
            y = pd.DataFrame(data = series.values,
                         index = series.index, 
                         columns = ['diff_0'])
            z = y.diff_0
            diff_order = 0
            while smt.adfuller(z.dropna())[1]>0.05:
                diff_order = diff_order + 1
                y['diff_{}'.format(diff_order)] = y['diff_{}'.format(diff_order - 1)].diff(1)
                z = y['diff_{}'.format(diff_order)]
            self.y_stationary_history = y
            self.y_diff_order = diff_order
            self.ys = y['diff_{}'.format(diff_order)]
            self.tso.ys = self.ys
            return self.ys

    def stationarize_X(self, modify = False):
        X = self.tso.X
        if modify == True:
            for col in X.columns.tolist():
                diff_order = 0
                while smt.adfuller(X[col].dropna())[1]>0.05:
                    diff_order = diff_order + 1
                    X[col] = X[col].diff(1)
                X = X.rename(columns = {col:col + '_diff_{}'.format(diff_order)})
        self.Xs = X
        self.tso.Xs = self.Xs
        return self.Xs

    def scaling_X(self, modify = False, scaling_method='standard'):
        X = self.tso.Xs
        if modify == True:
            if scaling_method == 'robust':
                scaler = RobustScaler()
                cols_names = self._selected_features
                X[cols_names] = scaler.fit_transform(X[cols_names])
                self.ss_x = scaler
                self.tso.ss_x = self.ss_x
            elif scaling_method == 'minmax':
                scalar = MinMaxScaler()
                cols_names = self._selected_features
                X[cols_names] = scaler.fit_transform(X[cols_names])
                self.ss_x = scaler
                self.tso.ss_x = self.ss_x
            else:
                scaler = StandardScaler()
                cols_names = self._selected_features
                X[cols_names] = scaler.fit_transform(X[cols_names])
                self.ss_x = scaler
                self.tso.ss_x = self.ss_x
        self.tso.Xs = X
        #print(self.Xs)
        return self.Xs,self.ss_x
        
    def scaling_y(self, modify = False, scaling_method='standard'):
        y = self.tso.ys
        if modify == True:
            if scaling_method == 'robust':
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(np.array(y).reshape(-1, 1))
                self.ss_y = scaler_y
                self.tso.ss_y = self.ss_y 
                self.tso.ys = pd.Series(data=y.reshape(1,- 1)[0],index=self.tso.ys.index)
            elif scaling_method == 'minmax':
                scaler_y = MinMaxScaler()
                y = scaler_y.fit_transform(np.array(y).reshape(-1, 1))
                self.ss_y = scaler_y
                self.tso.ss_y = self.ss_y 
                self.tso.ys = pd.Series(data=y.reshape(1,- 1)[0],index=self.tso.ys.index)
            else:
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(np.array(y).reshape(-1, 1))
                self.ss_y = scaler_y
                self.tso.ss_y = self.ss_y 
                self.tso.ys = pd.Series(data=y.reshape(1,- 1)[0],index=self.tso.ys.index)
        return self.ys,self.ss_y

##########################################  Testing  ##########################

if __name__ == '__main__':
    df = pd.read_csv('sample-df-2.csv')
    ts = timeSeriesObject.TimeSeriesObject(df)
    ts.set_date_column(date_column = 'order_date')    
    ts.drop_col('sales_units')
    ts.xy_split(response_variable = 'sales_value')
    ts_analyse = analyse.Analyse(ts)
    ts_analyse.visual_summary(ts.y)
    ts_transform = Transform(ts)
    ts_boxcox = ts_transform.do_boxcox()
    ts_analyse.visual_summary(ts_boxcox)
    ts_inv_boxcox = ts_transform.inverse_boxcox(ts_boxcox)
    ts_y_stationary = ts_transform.stationarize_y()
    ts_analyse.visual_summary(ts_y_stationary.dropna())
    ts_X_stationary = ts_transform.stationarize_X()
    ts_X_scaled = ts_transform.scaling_X(modify = scale_modify_x,scaling_method=scaling_choice)
    ts_y_scaled = ts_transform.scaling_y(modify = scale_modify_y,scaling_method=scaling_choice)
    # st = ts_transform.stationarize(ts_transform.y)