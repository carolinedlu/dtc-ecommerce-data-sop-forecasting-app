# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:54:09 2021

@author: 40102891
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import seaborn as sns
# import timeSeriesObject

#############################  Data  ##########################################

# df = pd.read_csv('sample-df-2.csv')

#################################### Class Definitions ########################

class Analyse:
    def __init__(self, tso):
        # Initializing Attributes
        self.tso = tso
        # Auxilliary Attributes
            # X-y (to be replaced by self.tso.X, self.tso.y)
        self.X = tso.X
        self.y = tso.y
            # Intermittency Measure
        self.adi = None
        self.cov = None
        
    def visual_summary(self, series):
        plt.figure(figsize = (10,10))
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan = 2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        
        series.plot(ax = ts_ax, )
        smt.graphics.plot_acf(series, lags=int(len(series)/3), ax=acf_ax)
        smt.graphics.plot_pacf(series, lags=int(len(series)/3), ax=pacf_ax)
        [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
        sns.despine()
        plt.tight_layout()
        return ts_ax, acf_ax, pacf_ax
    
    def _split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def intermittency(self, n_chunks = 6):
        self.cov = np.std(self.y)/np.mean(self.y)
        self.adi = len(self.y)/sum(self.y > 0)
        
        plt.figure(figsize = (6,6))
        i = 0
        for y_chunk in list(self._split(self.y.values, n_chunks)):
            plt.plot((np.std(y_chunk)/np.mean(y_chunk))**2,
                     len(y_chunk)/sum(y_chunk > 0), 
                     marker = 'o', 
                     alpha = ((i+1)/n_chunks),
                     color = 'blue')
            i+=1
        plt.title('COV vs ADI plot for Demand Classification')
        plt.xlabel('COV^2')
        plt.ylabel('ADI')
        plt.xlim([0,1])
        plt.ylim([0,5])
        plt.plot([0.49, 0.49], [0,5], color = 'green')
        plt.plot([0.0, 1.0], [1.32, 1.32], color = 'green')
        plt.annotate('Smooth', xy = (0.2,0.5))
        plt.annotate('Erratic', xy = (0.8, 0.5))
        plt.annotate('Intermittent', xy = (0.2,3))
        plt.annotate('Lumpy', xy = (0.8, 3))
        return plt
    
##########################################  Testing  ##########################

if __name__ == '__main__':
    df = pd.read_csv('sample-df-2.csv')
    ts = timeSeriesObject.TimeSeriesObject(df)
    ts.set_date_column(date_column = 'order_date')    
    ts.drop_col('sales_units')
    ts.xy_split(response_variable = 'sales_value')
    ts_analyse = Analyse(ts)
    ts_analyse.visual_summary(ts.y)
    ts_analyse.intermittency(n_chunks = 7)
        
        
        
    