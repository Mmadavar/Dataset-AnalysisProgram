#!/usr/bin/python3
#
#
#   Date:31 Aug 2020
#   Revised:10 September 2020
#   Version:0.2
#

#   Import Libraries
import os
import csv
import math
import numpy as np
import pandas as pd
from scipy.stats import stats
from scipy.stats.mstats import gmean
from scipy.stats import t
from numpy.linalg import lstsq
from numpy.testing import (assert_allclose, assert_equal, assert_, assert_raises)
from scipy.sparse import rand
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear

import statsmodels.api as sm
import statsmodels.multivariate as smm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.dates as mDates
from matplotlib import rc


#global debug variable
debug = 0

def main():
    # MAIN() main driver function
    # 
    #
    #
    #
    #
    
    #   File Name
    # Use a dataset that you find and put it in the appropriate directory
    # to make it easier put it in your Downloads folder.
    file_name = '../../Downloads/Salaries.csv'
    
    #   Read in the data to a data frame
    data_frame = pd.read_csv(file_name, sep=',')
    if debug:
        print(data_frame)
    
    #   Count the number of observations and columns
    num_observations = data_frame.shape[0]
    num_columns = data_frame.shape[1]
    
    #   Create a new data frame for the descriptive statistics
    #descriptive_stats = pd.DataFrame(columns = ['Count', 'Mean', 'Var', 'Std_Dev', 'Range', 'IQR', 'Median', 'Mode', 'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'UCL','LCL'])
    
    stat_frame = calcEDA(data_frame, [0, 1, 2, 3], 0.05, True)   # List indicates to ignore column 0, 1, 2, and 3
    
    #print(stat_frame)
    

    
def calcEDA(df, ignore_columns = [], alpha = 0.05, print_table = False):
    
    import numpy as np
    from scipy import stats
    
    #   Columns to drop from the calculations
    if(ignore_columns):
        #   axis = 1 to indicate columns (not rows)
        df.drop(df.columns[ignore_columns],axis=1,inplace=True)
    
    stats_df = pd.DataFrame(columns = ['Count', 'Mean', 'Var', 'Std_Dev', 'Range', 'IQR', 'Median', 'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'LCL','UCL', 'LF', 'UF'])
    
    stats_df['Count'] = df.count()
    stats_df['Mean'] = df.mean()
    stats_df['Var'] = df.var()
    stats_df['Std_Dev'] = df.std()
    stats_df['Var'] = df.var()
    stats_df['Median'] = df.median()
    stats_df['Q0'] = df.min()
    stats_df['Q1'] = df.quantile(0.25)
    stats_df['Q2'] = df.median()
    stats_df['Q3'] = df.quantile(0.75)
    stats_df['Q4'] = df.max()
    #stats_df['FNS'] = [df.min(), df.quantile(0.25), df.median(), df.quantile(0.75), df.max()]
    
    #   Calculating the lower and upper confidence limits
    #lower_t = stats.t.ppf(alpha / 2, df.count - 1)
    upper_t = stats.t.ppf(1 - (alpha / 2), stats_df['Count']-1)
    std_err = np.sqrt(stats_df['Var']/stats_df['Count'])
    margin_error = upper_t * std_err
    stats_df['LCL'] = stats_df['Mean'] - margin_error 
    stats_df['UCL'] = stats_df['Mean'] + margin_error 
    
    #   Range
    stats_df['Range'] = stats_df['Q4'] - stats_df['Q0']
    
    #   Interquartile range and lower and upper fences
    stats_df['IQR'] = stats_df['Q3'] - stats_df['Q1']
    stats_df['LF'] = stats_df['Q1'] - (3.0/2.0)*stats_df['IQR']
    stats_df['UF'] = stats_df['Q3'] + (3.0/2.0)*stats_df['IQR']
   
    #   If print_table is set to True create a table of the results
    if(print_table):
        #   Get the number of variables
        count = stats_df.shape[0]
        
        #   Print Table Headers
        print('')
        print('\t\t\t\t\t\t\tExploratory Data Analysis\n')
        print('%15s %22s %15s %15s %15s %15s' % ('Var' , 'Count' , 'Mean' , 'Std Dev', 'Range', 'IQR'))
        
        for k in range(0, count):
            print('\t\t%-18s %12d %15.3f %15.3f %15.3f %15.3f' % (stats_df.index[k], stats_df['Count'].iloc[k], stats_df['Mean'].iloc[k], stats_df['Std_Dev'].iloc[k], stats_df['Range'].iloc[k], stats_df['IQR'].iloc[k]))
            
        
        print('')
        print('%15s %22s %15s %15s %15s %15s' % ('Var' , 'Min' , '25%' , 'Median', '75%', 'Max'))
        for k in range(0, count):
            print('\t\t%-18s %12.3f %15.3f %15.3f %15.3f %15.3f'%(stats_df.index[k], stats_df['Q0'].iloc[k], stats_df['Q1'].iloc[k], stats_df['Median'].iloc[k], stats_df['Q3'].iloc[k],stats_df['Q4'].iloc[k]))
            
        print('')
        print('%15s %19.1f%% %15s %23s %17s' % ('Var' , 100*(1-alpha) , 'Confidence Interval', 'Lower Fence', 'Upper Fence'))
        for k in range(0, count):
            print('\t\t%-18s ( %10.3f - %13.3f   ) %19.3f %15.3f'%(stats_df.index[k], stats_df['LCL'].iloc[k], stats_df['UCL'].iloc[k], stats_df['LF'].iloc[k], stats_df['UF'].iloc[k]))
            
   
   
    return stats_df

  
#   Call and run the driver  
main()