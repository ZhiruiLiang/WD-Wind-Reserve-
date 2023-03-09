import numpy as np
import pandas as pd
import csv

from ForecastErrorGeneration import ErrorGeneration
risk = 1 
n_days = 28
n_hours = n_days*24
n_farms0 = 38
n_farms = 25
n_areas = 12
month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
start_year = 2013
start_month = 2
start_day = 1
start_point = (np.sum(month_day[0:start_month-1])+start_day-1)*24

header_generator = ['year','month','day','hour','38','42','44','45','46','48','52','95','97','100','102','103',
                    '310','311','312','313','331','375','387','401','418','449','451','640','643','669','936',
                    '1006','1414','1416','1483','1669','1810','1814','1815','1816','1817','1818']
header_area = ['year','month','day','hour','0','1','2','3','4','5','6','7','8','9','10','11']
header_system = ['year','month','day','hour','reserve']

upward_reserve_L1 = np.zeros((n_hours,n_farms0+4))
downward_reserve_L1 = np.zeros((n_hours,n_farms0+4))
upward_reserve_L2 = np.zeros((n_hours,n_farms0+4))
downward_reserve_L2 = np.zeros((n_hours,n_farms0+4))
upward_reserve_L3 = np.zeros((n_hours,n_farms0+4))
downward_reserve_L3 = np.zeros((n_hours,n_farms0+4))
upward_reserve_L4 = np.zeros((n_hours,n_farms0+4))
downward_reserve_L4 = np.zeros((n_hours,n_farms0+4))
upward_reserve_L5 = np.zeros((n_hours,n_farms0+4))
downward_reserve_L5 = np.zeros((n_hours,n_farms0+4))

for f in range(n_farms):
    for d in range(n_days):
        n_point = 24
        index_min = d*n_point
        index_max = (d+1)*n_point
        farm_list = [0,1,2,3,4,5,8,10,11,14,16,19,21,22,23,24,26,28,29,31,33,34,35,36,37]

        upward_L1, downward_L1, upward_L2, downward_L2, upward_L3, downward_L3, upward_L4, downward_L4, upward_L5, downward_L5 =ErrorGeneration(f,d,start_point,risk)
        upward_reserve_L1[index_min:index_max,farm_list[f]+4] = upward_L1
        downward_reserve_L1[index_min:index_max,farm_list[f]+4] = downward_L1
        upward_reserve_L2[index_min:index_max,farm_list[f]+4] = upward_L2
        downward_reserve_L2[index_min:index_max,farm_list[f]+4] = downward_L2
        upward_reserve_L3[index_min:index_max,farm_list[f]+4] = upward_L3
        downward_reserve_L3[index_min:index_max,farm_list[f]+4] = downward_L3
        upward_reserve_L4[index_min:index_max,farm_list[f]+4] = upward_L4
        downward_reserve_L4[index_min:index_max,farm_list[f]+4] = downward_L4
        upward_reserve_L5[index_min:index_max,farm_list[f]+4] = upward_L5
        downward_reserve_L5[index_min:index_max,farm_list[f]+4] = downward_L5

        upward_reserve_L1[index_min:index_max,0] = int(start_year)
        downward_reserve_L1[index_min:index_max,0] = int(start_year)
        upward_reserve_L2[index_min:index_max,0] = int(start_year)
        downward_reserve_L2[index_min:index_max,0] = int(start_year)
        upward_reserve_L3[index_min:index_max,0] = int(start_year)
        downward_reserve_L3[index_min:index_max,0] = int(start_year)
        upward_reserve_L4[index_min:index_max,0] = int(start_year)
        downward_reserve_L4[index_min:index_max,0] = int(start_year)
        upward_reserve_L5[index_min:index_max,0] = int(start_year)
        downward_reserve_L5[index_min:index_max,0] = int(start_year)

        upward_reserve_L1[index_min:index_max,1] = int(start_month)
        downward_reserve_L1[index_min:index_max,1] = int(start_month)
        upward_reserve_L2[index_min:index_max,1] = int(start_month)
        downward_reserve_L2[index_min:index_max,1] = int(start_month)
        upward_reserve_L3[index_min:index_max,1] = int(start_month)
        downward_reserve_L3[index_min:index_max,1] = int(start_month)
        upward_reserve_L4[index_min:index_max,1] = int(start_month)
        downward_reserve_L4[index_min:index_max,1] = int(start_month)
        upward_reserve_L5[index_min:index_max,1] = int(start_month)
        downward_reserve_L5[index_min:index_max,1] = int(start_month)

        upward_reserve_L1[index_min:index_max,2] = int(start_day+d)
        downward_reserve_L1[index_min:index_max,2] = int(start_day+d)
        upward_reserve_L2[index_min:index_max,2] = int(start_day+d)
        downward_reserve_L2[index_min:index_max,2] = int(start_day+d)
        upward_reserve_L3[index_min:index_max,2] = int(start_day+d)
        downward_reserve_L3[index_min:index_max,2] = int(start_day+d)
        upward_reserve_L4[index_min:index_max,2] = int(start_day+d)
        downward_reserve_L4[index_min:index_max,2] = int(start_day+d)
        upward_reserve_L5[index_min:index_max,2] = int(start_day+d)
        downward_reserve_L5[index_min:index_max,2] = int(start_day+d)

        upward_reserve_L1[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        downward_reserve_L1[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        upward_reserve_L2[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        downward_reserve_L2[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        upward_reserve_L3[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        downward_reserve_L3[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        upward_reserve_L4[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        downward_reserve_L4[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        upward_reserve_L5[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        downward_reserve_L5[index_min:index_max,3] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

with open('upward_nodal_reserve_L1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(upward_reserve_L1)
with open('downward_nodal_reserve_L1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(downward_reserve_L1)
with open('upward_nodal_reserve_L2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(upward_reserve_L2)
with open('downward_nodal_reserve_L2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(downward_reserve_L2)
with open('upward_nodal_reserve_L3.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(upward_reserve_L3)
with open('downward_nodal_reserve_L3.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(downward_reserve_L3)
with open('upward_nodal_reserve_L4.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(upward_reserve_L4)
with open('downward_nodal_reserve_L4.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(downward_reserve_L4)
with open('upward_nodal_reserve_L5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(upward_reserve_L5)
with open('downward_nodal_reserve_L5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_generator)
    writer.writerows(downward_reserve_L5)


Area = [2, 5, 8, 8, 2 ,8, 4, 9, 4, 4, 9, 9, 5, 5, 5, 5, 8, 5, 5, 5, 5, 9, 9, 9, 9, 2, 9, 4, 4, 4, 7, 5, 0, 1, 7, 1, 7, 7]
downward_20_area = np.zeros((n_hours,n_areas+4))
upward_20_area = np.zeros((n_hours,n_areas+4))
downward_40_area = np.zeros((n_hours,n_areas+4))
upward_40_area = np.zeros((n_hours,n_areas+4))
downward_60_area = np.zeros((n_hours,n_areas+4))
upward_60_area = np.zeros((n_hours,n_areas+4))
downward_80_area = np.zeros((n_hours,n_areas+4))
upward_80_area = np.zeros((n_hours,n_areas+4))
downward_100_area = np.zeros((n_hours,n_areas+4))
upward_100_area = np.zeros((n_hours,n_areas+4))

for i in range(n_areas):
    for j in range(n_farms0):
        if Area[j] == i:
            upward_20_area[:,i+4] = upward_20_area[:,i+4] + upward_reserve_L1[:,j+4]
            downward_20_area[:,i+4] = downward_20_area[:,i+4] + downward_reserve_L1[:,j+4]
            upward_40_area[:,i+4] = upward_40_area[:,i+4] + upward_reserve_L2[:,j+4]
            downward_40_area[:,i+4] = downward_40_area[:,i+4] + downward_reserve_L2[:,j+4]
            upward_60_area[:,i+4] = upward_60_area[:,i+4] + upward_reserve_L3[:,j+4]
            downward_60_area[:,i+4] = downward_60_area[:,i+4] + downward_reserve_L3[:,j+4]
            upward_80_area[:,i+4] = upward_80_area[:,i+4] + upward_reserve_L4[:,j+4]
            downward_80_area[:,i+4] = downward_80_area[:,i+4] + downward_reserve_L4[:,j+4]
            upward_100_area[:,i+4] = upward_100_area[:,i+4] + upward_reserve_L5[:,j+4]
            downward_100_area[:,i+4] = downward_100_area[:,i+4] + downward_reserve_L5[:,j+4]
downward_20_area[:,0:4] = upward_reserve_L1[:,0:4]
upward_20_area[:,0:4] = upward_reserve_L1[:,0:4]
downward_40_area[:,0:4] = upward_reserve_L1[:,0:4]
upward_40_area[:,0:4] = upward_reserve_L1[:,0:4]
downward_60_area[:,0:4] = upward_reserve_L1[:,0:4]
upward_60_area[:,0:4] = upward_reserve_L1[:,0:4]
downward_80_area[:,0:4] = upward_reserve_L1[:,0:4]
upward_80_area[:,0:4] = upward_reserve_L1[:,0:4]
downward_100_area[:,0:4] = upward_reserve_L1[:,0:4]
upward_100_area[:,0:4] = upward_reserve_L1[:,0:4]

with open('upward_area_reserve_L1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(upward_20_area)
with open('downward_area_reserve_L1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(downward_20_area)
with open('upward_area_reserve_L2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(upward_40_area)
with open('downward_area_reserve_L2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(downward_40_area)
with open('upward_area_reserve_L3.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(upward_60_area)
with open('downward_area_reserve_L3.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(downward_60_area)
with open('upward_area_reserve_L4.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(upward_80_area)
with open('downward_area_reserve_L4.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(downward_80_area)
with open('upward_area_reserve_L5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(upward_100_area)
with open('downward_area_reserve_L5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_area)
    writer.writerows(downward_100_area)

downward_20_system = np.zeros((n_hours,1+4))
upward_20_system = np.zeros((n_hours,1+4))
downward_40_system = np.zeros((n_hours,1+4))
upward_40_system = np.zeros((n_hours,1+4))
downward_60_system = np.zeros((n_hours,1+4))
upward_60_system = np.zeros((n_hours,1+4))
downward_80_system = np.zeros((n_hours,1+4))
upward_80_system = np.zeros((n_hours,1+4))
downward_100_system = np.zeros((n_hours,1+4))
upward_100_system = np.zeros((n_hours,1+4))

downward_20_system[:,0:4] = upward_reserve_L1[:,0:4]
upward_20_system[:,0:4] = upward_reserve_L1[:,0:4]
downward_40_system[:,0:4] = upward_reserve_L1[:,0:4]
upward_40_system[:,0:4] = upward_reserve_L1[:,0:4]
downward_60_system[:,0:4] = upward_reserve_L1[:,0:4]
upward_60_system[:,0:4] = upward_reserve_L1[:,0:4]
downward_80_system[:,0:4] = upward_reserve_L1[:,0:4]
upward_80_system[:,0:4] = upward_reserve_L1[:,0:4]
downward_100_system[:,0:4] = upward_reserve_L1[:,0:4]
upward_100_system[:,0:4] = upward_reserve_L1[:,0:4]

downward_20_system[:,4] = np.sum(downward_20_area[:,4:17], axis=1)
upward_20_system[:,4] = np.sum(upward_20_area[:,4:17], axis=1)
downward_40_system[:,4] = np.sum(downward_40_area[:,4:17], axis=1)
upward_40_system[:,4] = np.sum(upward_40_area[:,4:17], axis=1)
downward_60_system[:,4] = np.sum(downward_60_area[:,4:17], axis=1)
upward_60_system[:,4] = np.sum(upward_60_area[:,4:17], axis=1)
downward_80_system[:,4] = np.sum(downward_80_area[:,4:17], axis=1)
upward_80_system[:,4] = np.sum(upward_80_area[:,4:17], axis=1)
downward_100_system[:,4] =np.sum(downward_100_area[:,4:17], axis=1)
upward_100_system[:,4] = np.sum(upward_100_area[:,4:17], axis=1)

with open('upward_system_reserve_L1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(upward_20_system)
with open('downward_system_reserve_L1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(downward_20_system)
with open('upward_system_reserve_L2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(upward_40_system)
with open('downward_system_reserve_L2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(downward_40_system)
with open('upward_system_reserve_L3.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(upward_60_system)
with open('downward_system_reserve_L3.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(downward_60_system)
with open('upward_system_reserve_L4.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(upward_80_system)
with open('downward_system_reserve_L4.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(downward_80_system)
with open('upward_system_reserve_L5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(upward_100_system)
with open('downward_system_reserve_L5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header_system)
    writer.writerows(downward_100_system)