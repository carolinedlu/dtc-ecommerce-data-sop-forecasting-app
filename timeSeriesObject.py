# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:27:36 2021

@author: 40102891
"""

import pandas as pd
import numpy as np
from scipy import fft
from scipy import signal as sig
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from cmath import phase

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
warnings.simplefilter('ignore')
#######################################  Global Variables  ####################
start_train_date = '18-02-2018'
end_train_date = '26-04-2020'
start_test_date = '03-05-2020'
end_test_date = '26-07-2020'
start_pred_date = '02-08-2020'
end_pred_date = '08-11-2020'

response_variable = 'sales_units'
#######################################  Data  ################################

# df = pd.read_csv('sample-df-2.csv')

#################################### Class Definitions ########################

class TimeSeriesObject:
    
    def __init__(self, df):
        # Instantiating variables
        self.df = df
        
        # Auxilliary variables
            # target 
        self.response_variable = None
            # modified df
        self.dfm = df
            # date column
        self.date_column = None
            # X-y splits
        self.X = None
        self.y = None
            # Stationary X-y
        self.Xs = self.X
        self.ys = self.y
            # feature selection
        self.Xsf = None
        self._selected_features = []
            
            # train-test-pred splits
        self.X_train = None
        self.X_test = None
        self.X_pred = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def set_date_column(self, date_column):
        self.df = self.df.set_index(date_column)
        self.df.index = pd.to_datetime(self.df.index, format = '%d-%m-%Y')
        self.dfm = self.df
        return self
    
    def drop_col(self, drop_col):
        self.dfm = self.dfm.drop(drop_col, axis = 1)
        return self
    
    def response_lag(self, date_column,response_variable):
        lag_list = [1, 2, 3, 4, 5, 6]
        #Create last 6 weeks lags
        for lag in lag_list:
            ft_name = ('response|shifted%s' % lag)
            self.dfm[ft_name] = self.dfm[response_variable].shift(lag)
            # Fill the empty shifted features with 0
            self.dfm[ft_name].fillna(0, inplace=True)
        return self

    def response_rolling(self, date_column, response_variable):
        #print('Rolling window based features')
        # Rolling window based features (window = 3 months)
        # Min value
        f_min = lambda x: x.rolling(window=12, min_periods=1).min()
        # Max value
        f_max = lambda x: x.rolling(window=12, min_periods=1).max()
        # Mean value
        f_mean = lambda x: x.rolling(window=12, min_periods=1).mean()
        # Standard deviation
        f_std = lambda x: x.rolling(window=12, min_periods=1).std()
        # Sum
        f_sum = lambda x: x.rolling(window=12, min_periods=1).sum()

        # Min value
        f_min_4 = lambda x: x.rolling(window=4, min_periods=1).min()
        # Max value
        f_max_4 = lambda x: x.rolling(window=4, min_periods=1).max()
        # Mean value
        f_mean_4 = lambda x: x.rolling(window=4, min_periods=1).mean()
        # Standard deviation
        f_std_4 = lambda x: x.rolling(window=4, min_periods=1).std()
        # Sum
        f_sum_4 = lambda x: x.rolling(window=4, min_periods=1).sum()

        # Min value
        f_min_8 = lambda x: x.rolling(window=8, min_periods=1).min()
        # Max value
        f_max_8 = lambda x: x.rolling(window=8, min_periods=1).max()
        # Mean value
        f_mean_8 = lambda x: x.rolling(window=8, min_periods=1).mean()
        # Standard deviation
        f_std_8 = lambda x: x.rolling(window=8, min_periods=1).std()
        # Sum
        f_sum_8 = lambda x: x.rolling(window=8, min_periods=1).sum()

        function_list = [f_min, f_max, f_mean, f_std, f_sum,f_min_4, f_max_4, f_mean_4, f_std_4, f_sum_4,f_min_8, f_max_8, f_mean_8, f_std_8, f_sum_8]
        function_name = ['min', 'max', 'mean', 'std', 'sum','min_4', 'max_4', 'mean_4', 'std_4', 'sum_4','min_8', 'max_8', 'mean_8', 'std_8', 'sum_8']

        for i in range(len(function_list)):
            self.dfm[('response|shifted1%s' % function_name[i])] = self.dfm['response|shifted1'].transform(function_list[i])
            #print('function', i)
            self.dfm[('response|shifted1%s' % function_name[i])].fillna(0, inplace=True)
        #print(self.dfm)
        self._selected_features = self.dfm.columns
        self._selected_features = [e for e in self._selected_features if e not in (date_column, response_variable)]
        return self

    def fft_features(self, date_column, response_variable,modify=False):
        if modify:
            data_copy = self.dfm.copy()
            data_copy=data_copy.sort_values(by=date_column).reset_index()
            #data_copy=data_copy.drop('index',axis=1)
            data_copy['datetime_index']=list(range(1,len(data_copy)+1))
            #print(data_copy['datetime_index'])
            time = np.array(data_copy['datetime_index'])
            target_lag = np.array(data_copy['response|shifted1'].subtract(data_copy['response|shifted1mean_8']))
            fft_output = fft.fft(target_lag)
            power = np.abs(fft_output)
            freq = fft.fftfreq(len(target_lag))
            mask = freq >= 0
            freq = freq[mask]
            power = power[mask]
            peaks = sig.find_peaks(power[freq >=0], prominence=data_copy['response|shifted1'].mean()*2)[0]
            peak_freq =  freq[peaks]
            peak_power = power[peaks]
            output = pd.DataFrame()
            output['index'] = peaks
            output['freq (1/week)'] = peak_freq
            output['amplitude'] = peak_power
            output['period (month)'] = 1 / peak_freq / 4
            output['fft'] = fft_output[peaks]
            output = output.sort_values('amplitude', ascending=False)
            #print(output)
            filtered_fft_output = np.array([f if i in list(output['index']) else 0 for i, f in enumerate(fft_output)])
            filtered_target_lag = fft.ifft(filtered_fft_output)
            fourier_terms = pd.DataFrame()
            fourier_terms['fft'] = output['fft']
            fourier_terms['freq (1 /week)'] = output['freq (1/week)']
            fourier_terms['amplitude'] = fourier_terms.fft.apply(lambda z: abs(z)) 
            fourier_terms['phase'] = fourier_terms.fft.apply(lambda z: phase(z))
            fourier_terms.sort_values(by=['amplitude'], ascending=[0])
            #print(fourier_terms)
            # Create some helpful labels (FT_1..FT_N)
            fourier_terms['label'] = list(map(lambda n : 'FT_{}'.format(n), range(1, len(fourier_terms) + 1)))
            # Turn our dataframe into a dictionary for easy lookup
            fourier_terms = fourier_terms.set_index('label')
            fourier_terms_dict = fourier_terms.to_dict('index')
            for count,key in enumerate(fourier_terms_dict.keys()):
                if count<4:
                    a = fourier_terms_dict[key]['amplitude']
                    w = 2 * math.pi * (fourier_terms_dict[key]['freq (1 /week)'])
                    p = fourier_terms_dict[key]['phase']
                    data_copy[key] = data_copy['datetime_index'].apply(lambda t: a * math.cos(w*t + p))
                #print(self.dfm[key])

            data_copy['FT_All'] = 0
            for count,column in enumerate(list(fourier_terms.index)):
                if count<4:
                    data_copy['FT_All'] = data_copy['FT_All'] + data_copy[column]
            self.dfm=data_copy
            self.dfm = self.dfm.set_index(date_column)
        #print(self.dfm[['FT_All']  + list(fourier_terms.index)])
        #print(self.dfm.columns)
        self._selected_features = self.dfm.columns
        self._selected_features = [e for e in self._selected_features if e not in (date_column, response_variable)]
        return self

    def date_features(self, date_column):
        self.dfm['quarter_of_year'] = self.dfm.index.quarter
        self.dfm['month_of_year'] = self.dfm.index.month
        self.dfm['week_of_quarter']= pd.to_numeric((self.dfm.index.isocalendar().week - 1 ) % 13 + 1).astype("int64")
        self.dfm['week_in_month'] = pd.to_numeric(self.dfm.index.day/7)
        self.dfm['week_in_month'] = self.dfm['week_in_month'].apply(lambda x: math.ceil(x))
        #print(self.dfm.dtypes)
        #self.dfm.to_csv("weekly_data.csv",index=False)
        return self

    def add_col(self, add_col):
        col_list = self.dfm.columns.tolist() + [add_col] 
        self.dfm = self.df[col_list]
        return self
    
    def xy_split(self, response_variable):
        self.response_variable = response_variable
        self.y = self.dfm[response_variable]
        self.X = self.dfm.drop(response_variable, axis = 1)
        self.Xm = self.X
        self.Xs = self.X
        self.ys = self.y
        return self
        
    def train_test_split(self, 
                         start_train_date,
                         end_train_date,
                         start_test_date,
                         end_test_date,
                         start_pred_date,
                         end_pred_date):
        self.X_train = self.Xs.loc[start_train_date:end_train_date]
        self.y_train = self.ys.loc[start_train_date:end_train_date]
        self.X_test = self.Xs.loc[start_test_date:end_test_date]
        self.y_test = self.ys.loc[start_test_date:end_test_date]
        self.X_pred = self.Xs.loc[start_pred_date:end_pred_date]
        self.y_pred = np.nan * len(self.X_pred)
        #print(len(self.X_train),len(self.y_train),len(self.X_test),len(self.y_test),len(self.X_pred))
        return self
    
##########################################  Testing  ###################################################################

if __name__ == '__main__':
    df = pd.read_csv('sample-df-2.csv')
    start_train_date = '2018-02-18'
    end_train_date = '2020-04-26'
    start_test_date = '2020-05-03'
    end_test_date = '2020-07-26'
    start_pred_date = '2020-08-02'
    end_pred_date = '2020-11-08'
    ts = TimeSeriesObject(df)
    ts.set_date_column(date_column = 'order_date')
    ts.response_lag(date_column = 'order_date',response_variable = 'sales_value')
    ts.response_rolling(date_column = 'order_date',response_variable = 'sales_value')
    ts.fft_features(date_column = 'order_date',response_variable = 'sales_value',modify=False)
    ts.date_features(date_column = 'order_date')
    ts.drop_col('sales_units')
    ts.add_col('sales_units')
    ts.drop_col('sales_units')
    ts.drop_col('Halloween')
    ts.xy_split(response_variable = 'sales_value')
    ts.train_test_split(start_train_date = start_train_date,
                        end_train_date = end_train_date,
                        start_test_date = start_test_date,
                        end_test_date = end_test_date,
                        start_pred_date = start_pred_date,
                        end_pred_date = end_pred_date)
    