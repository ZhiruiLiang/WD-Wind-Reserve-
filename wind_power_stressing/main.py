'''
[ARPA-E Perform Code]
Function: Set parameters; output forecast and stressed wind power

Author: Zhirui Liang, zliang31@jhu.edu
Date: 2022-10-05
'''
import numpy as np
import pandas as pd
import csv
from Forecast_Error_Generation import ErrorGeneration

month_day = [31,28,31,30,31,30,31,31,30,31,30,31]

# changeable parameters
n_farms = 38 # number of wind farms in NYISO system (33 onshore + 5 offshore)
n_scenarios = 100 # number of stressed scenarios to be generated at each hour
n_hours = 2 # time horizon (how many hours to be considered)
start_year = 2013 # can only be set to 2013 based on available data
start_month = 8 # can be changed between 1 and 12
start_day = 27 # can be changed between 1 and the total number of days in that month
start_hour = 17 # can be changed between 0 and 23

start_point = (np.sum(month_day[0:start_month-1])+start_day-1)*24 + start_hour

header_generator = ['year','month','day','hour','38','42','44','45','46','48','52','95','97','100','102','103',
                    '310','311','312','313','331','375','387','401','418','449','451','640','643','669','936',
                    '1006','1414','1416','1483','1669','1810','1814','1815','1816','1817','1818']

# call function "ErrorGeneration" to generate stressed wind power
stressed_wind_power = np.zeros((n_scenarios*n_hours,n_farms+4))
forecast_wind_power = np.zeros((n_hours,n_farms+4))

for f in range(n_farms):
    for h in range(n_hours):
        power_forecast_final, power_generated_final =ErrorGeneration(f,h,start_point,n_scenarios)
        index_min = h*n_scenarios
        index_max = (h+1)*n_scenarios
        stressed_wind_power[index_min:index_max,0] = int(start_year)
        stressed_wind_power[index_min:index_max,1] = int(start_month)
        stressed_wind_power[index_min:index_max,2] = int(start_day)
        stressed_wind_power[index_min:index_max,3] = int(start_hour+h)
        stressed_wind_power[index_min:index_max,f+4] = power_generated_final

        forecast_wind_power[h,0] = int(start_year)
        forecast_wind_power[h,1] = int(start_month)
        forecast_wind_power[h,2] = int(start_day)
        forecast_wind_power[h,3] = int(start_hour)
        forecast_wind_power[h,f+4] = power_forecast_final

# output results to csv files
with open('output//stressed_wind_power.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(stressed_wind_power)

with open('output//forecast_wind_power.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(forecast_wind_power)