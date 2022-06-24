# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:38:54 2020

@author: 40102891
"""
import numpy as np
import warnings

#######################################  Data  ################################
class Metrics:
    
    def __init__(self, y_true, y_predicted = None):

        # Instantiation Attributes
        self.y_true = np.ravel(np.array(y_true))
        if (y_predicted is not None):
            self.y_predicted = np.ravel(np.array(y_predicted))

        # Metrics
        self.cov = np.round(self.coef_variation(), 2)
        self.adi = np.round(self.avg_demand_interval(), 2)
        if (y_predicted is not None):
            self.mae = np.round(self.mean_absolute_error(), 2)
            self.mape = np.round(self.mean_absolute_percentage_error(), 2)
            self.smape = np.round(self.sym_mean_absolute_percentage_error(), 2)
        else:
            warnings.warn("No 'y_predicted' passed")


    ### Class Methods

    # Univariate Metrics

    def coef_variation(self):
        """
        Returns the coefficient of variation of the response time series
        :return: Coefficient of Variation
        """
        return np.std(self.y_true)/np.mean(self.y_true)

    def avg_demand_interval(self):
        """
        Returns the Average Demand Interval for the response time series
        :return: Average Demand Interval
        """
        return len(self.y_true)/sum(self.y_true > 0)

    # Bivariate Metrics

    def mean_absolute_error(self):
        """
        Returns the Mean Absolute Error for the predicted time series 
        with respect to the actual time series
        :return: Mean Absolute Error
        """
        if len(self.y_predicted) == 0:
            return None
        else:
            return np.mean(np.abs(self.y_true - self.y_predicted))

    def mean_absolute_percentage_error(self):
        """
        Returns the Mean Absolute Percentage Error for the predicted 
        time series with respect to the actual time series
        :return: Mean Absolute Percentage Error
        """
        if len(self.y_predicted) == 0:
            return None
        else:
            return 100 * np.mean(np.abs(self.y_true - self.y_predicted)/ self.y_true)

    def sym_mean_absolute_percentage_error(self):
        """
        Returns the Summetric Mean Absolute Percentage Error for the 
        predicted time series with respect to the actual time series
        :return: Symmetric Mean Absolute Percentage Error
        """
        if len(self.y_predicted) == 0:
            return None
        else:
            return 200 * np.mean(np.abs(self.y_true - self.y_predicted)/ (self.y_true + self.y_predicted))

##############################  Testing  ###############################################################################

if __name__ == '__main':
    print('COV: {}'.format(Metrics(y_true = [1,2,0,3]).cov))
    print('ADI: {}'.format(Metrics(y_true = [1,2,0,3]).adi))

    print('MAE: {}'.format(Metrics(y_true = [1,2,1,3], y_predicted= [1,1,1,2]).mae))
    print('MAPE: {}'.format(Metrics(y_true = [1,2,1,3], y_predicted= [1,1,1,2]).mape))
    print('SMAPE: {}'.format(Metrics(y_true = [1,2,1,3], y_predicted= [1,1,1,2]).smape))