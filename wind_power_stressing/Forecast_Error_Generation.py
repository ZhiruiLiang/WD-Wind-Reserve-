'''
[ARPA-E Perform Code]
Function: Generate forecast errors of weather features; calculate stressed wind power

Author: Zhirui Liang, zliang31@jhu.edu
Date: 2022-10-05
'''
import numpy as np
import pandas as pd
import math 
from distfit import distfit
import warnings
warnings.filterwarnings('ignore')

from Wind_Power_Calculation import PowerCalculation
from PCA_Calculation import PCAcalculation

def ErrorGeneration(f,h,start_point,n_scenarios):  
    turbine_rated = 10*1e6
    farm_loss = 0.15
    farm_information = pd.read_csv('data//38farm_information.csv')
    Pmax = np.array(farm_information['pmax'])
    farm_rated = Pmax[f]*1e6
    
    # load forecast weather data
    df_farm = pd.read_csv('data//data_from_NREL_toolkit//farm_{}_2013.csv'.format(f))
    index_now = start_point+h
    p = np.array(df_farm['surface air pressure (Pa)'])[index_now]
    phi = np.array(df_farm['relative humidity at 2m (%)'])[index_now]
    t = np.array(df_farm['air temperature at 10m (C)'])[index_now]
    D1 = np.array(df_farm['wind direction at 10m (deg)'])[index_now]
    D2 = np.array(df_farm['wind direction at 40m (deg)'])[index_now]
    D3 = np.array(df_farm['wind direction at 60m (deg)'])[index_now]
    D4 = np.array(df_farm['wind direction at 80m (deg)'])[index_now]
    D5 = np.array(df_farm['wind direction at 100m (deg)'])[index_now]
    D6 = np.array(df_farm['wind direction at 120m (deg)'])[index_now]
    D7 = np.array(df_farm['wind direction at 140m (deg)'])[index_now]
    D8 = np.array(df_farm['wind direction at 160m (deg)'])[index_now]
    D9 = np.array(df_farm['wind direction at 200m (deg)'])[index_now]
    V1 = np.array(df_farm['wind speed at 10m (m/s)'])[index_now]
    V2 = np.array(df_farm['wind speed at 40m (m/s)'])[index_now]
    V3 = np.array(df_farm['wind speed at 60m (m/s)'])[index_now]
    V4 = np.array(df_farm['wind speed at 80m (m/s)'])[index_now]
    V5 = np.array(df_farm['wind speed at 100m (m/s)'])[index_now]
    V6 = np.array(df_farm['wind speed at 120m (m/s)'])[index_now]
    V7 = np.array(df_farm['wind speed at 140m (m/s)'])[index_now]
    V8 = np.array(df_farm['wind speed at 160m (m/s)'])[index_now]
    V9 = np.array(df_farm['wind speed at 200m (m/s)'])[index_now]

    # load historical distribution of wind speed forecast error (transition matrix and conditional distributions)
    df_matrix = pd.read_csv('data//transition_matrix_and_functions//transition_matrix_farm{}.csv'.format(f), header=None)
    transition_matrix = np.array(df_matrix)
    dist_1 = distfit()
    dist_21 = distfit()
    dist_22 = distfit()
    dist_23 = distfit()
    dist_24 = distfit()
    dist_25 = distfit()
    dist_1.load('data//transition_matrix_and_functions//dist_1_2_farm{}.pkl'.format(f))
    dist_21.load('data//transition_matrix_and_functions//dist_21_2_farm{}.pkl'.format(f))
    dist_22.load('data//transition_matrix_and_functions//dist_22_2_farm{}.pkl'.format(f))
    dist_23.load('data//transition_matrix_and_functions//dist_23_2_farm{}.pkl'.format(f))
    dist_24.load('data//transition_matrix_and_functions//dist_24_2_farm{}.pkl'.format(f))
    dist_25.load('data//transition_matrix_and_functions//dist_25_2_farm{}.pkl'.format(f))
    
    # generate wind speed forecast errors based on historical distribution
    speed_with_error = np.zeros((n_scenarios))
    speed_error = np.zeros((n_scenarios))
    if V6<4:
        n00 = int(transition_matrix[0,0]*n_scenarios)
        n01 = int(transition_matrix[0,1]*n_scenarios)
        n02 = int(transition_matrix[0,2]*n_scenarios)
        n03 = int(transition_matrix[0,3]*n_scenarios)
        diff0 = n00 + n01 + n02 + n03 - n_scenarios
        n00 = n00 - diff0
        speed_with_error[0:n00] = 1
        speed_error[0:n00] = 0
        speed_with_error[n00:n00+n01] = V6+dist_1.generate(n=n01)
        speed_error[n00:n00+n01] = -dist_1.generate(n=n01)
        speed_with_error[n00+n01:n00+n01+n02] = 20
        speed_error[n00+n01:n00+n01+n02] = 0
        speed_with_error[n00+n01+n02:n00+n01+n02+n03] = 30
        speed_error[n00+n01+n02:n00+n01+n02+n03] = 0
    elif V6<6:
        n10 = int(transition_matrix[1,0]*n_scenarios)
        n11 = int(transition_matrix[1,1]*n_scenarios)
        n12 = int(transition_matrix[1,2]*n_scenarios)
        n13 = int(transition_matrix[1,3]*n_scenarios)
        diff1 = n10 + n11 + n12 + n13 - n_scenarios
        n11 = n11 - diff1
        speed_with_error[0:n10] = 1
        speed_error[0:n10] = 1
        speed_with_error[n10:n10+n11] = V6+dist_21.generate(n=n11)
        speed_error[n10:n10+n11] = -dist_21.generate(n=n11)
        speed_with_error[n10+n11:n10+n11+n12] = 20
        speed_error[n10+n11:n10+n11+n12] = 0
        speed_with_error[n10+n11+n12:n10+n11+n12+n13] = 30
        speed_error[n10+n11+n12:n10+n11+n12+n13] = 0
    elif V6<8:
        n20 = int(transition_matrix[2,0]*n_scenarios)
        n21 = int(transition_matrix[2,1]*n_scenarios)
        n22 = int(transition_matrix[2,2]*n_scenarios)
        n23 = int(transition_matrix[2,3]*n_scenarios)
        diff2 = n20 + n21 + n22 + n23 - n_scenarios
        n21 = n21 - diff2
        speed_with_error[0:n20] = 1
        speed_error[0:n20] = 0
        speed_with_error[n20:n20+n21] = V6+dist_22.generate(n=n21)
        speed_error[n20:n20+n21] = -dist_22.generate(n=n21)
        speed_with_error[n20+n21:n20+n21+n22] = 20
        speed_error[n20+n21:n20+n21+n22] = 0
        speed_with_error[n20+n21+n22:n20+n21+n22+n23] = 30
        speed_error[n20+n21+n22:n20+n21+n22+n23] = 0
    elif V6<10:
        n30 = int(transition_matrix[3,0]*n_scenarios)
        n31 = int(transition_matrix[3,1]*n_scenarios)
        n32 = int(transition_matrix[3,2]*n_scenarios)
        n33 = int(transition_matrix[3,3]*n_scenarios)
        diff3 = n30 + n31 + n32 + n33 - n_scenarios
        n31 = n31 - diff3
        speed_with_error[0:n30] = 1
        speed_error[0:n30] = 0
        speed_with_error[n30:n30+n31] = V6+dist_23.generate(n=n31)
        speed_error[n30:n30+n31] = -dist_23.generate(n=n31)
        speed_with_error[n30+n31:n30+n31+n32] = 20
        speed_error[n30+n31:n30+n31+n32] = 0
        speed_with_error[n30+n31+n32:n30+n31+n32+n33] = 30
        speed_error[n30+n31+n32:n30+n31+n32+n33] = 0
    elif V6<12:
        n40 = int(transition_matrix[4,0]*n_scenarios)
        n41 = int(transition_matrix[4,1]*n_scenarios)
        n42 = int(transition_matrix[4,2]*n_scenarios)
        n43 = int(transition_matrix[4,3]*n_scenarios)
        diff4 = n40 + n41 + n42 + n43 - n_scenarios
        n41 = n41 - diff4
        speed_with_error[0:n40] = 1
        speed_error[0:n40] = 0
        speed_with_error[n40:n40+n41] = V6+dist_24.generate(n=n41)
        speed_error[n40:n40+n41] = -dist_24.generate(n=n41)
        speed_with_error[n40+n41:n40+n41+n42] = 20
        speed_error[n40+n41:n40+n41+n42] = 0
        speed_with_error[n40+n41+n42:n40+n41+n42+n43] = 30
        speed_error[n40+n41+n42:n40+n41+n42+n43] = 0
    elif V6<14:
        n50 = int(transition_matrix[5,0]*n_scenarios)
        n51 = int(transition_matrix[5,1]*n_scenarios)
        n52 = int(transition_matrix[5,2]*n_scenarios)
        n53 = int(transition_matrix[5,3]*n_scenarios)
        diff5 = n50 + n51 + n52 + n53 - n_scenarios
        n51 = n51 - diff5
        speed_with_error[0:n50] = 1
        speed_error[0:n50] = 0
        speed_with_error[n50:n50+n51] = V6+dist_25.generate(n=n51)
        speed_error[n50:n50+n51] = -dist_25.generate(n=n51)
        speed_with_error[n50+n51:n50+n51+n52] = 20
        speed_error[n50+n51:n50+n51+n52] = 0
        speed_with_error[n50+n51+n52:n50+n51+n52+n53] = 30
        speed_error[n50+n51+n52:n50+n51+n52+n53] = 0
    elif V6<25:
        n60 = int(transition_matrix[6,0]*n_scenarios)
        n61 = int(transition_matrix[6,1]*n_scenarios)
        n62 = int(transition_matrix[6,2]*n_scenarios)
        n63 = int(transition_matrix[6,3]*n_scenarios)
        diff6 = n60 + n61 + n62 + n63 - n_scenarios
        n62 = n62 - diff6
        speed_with_error[0:n60] = 1
        speed_error[0:n60] = 15
        speed_with_error[n60:n60+n61] = 10
        speed_error[n60:n60+n61] = 5
        speed_with_error[n60+n61:n60+n61+n62] = 20
        speed_error[n60+n61:n60+n61+n62] = 0
        speed_with_error[n60+n61+n62:n60+n61+n62+n63] = 30
        speed_error[n60+n61+n62:n60+n61+n62+n63] = -5
    else:
        n70 = int(transition_matrix[7,0]*n_scenarios)
        n71 = int(transition_matrix[7,1]*n_scenarios)
        n72 = int(transition_matrix[7,2]*n_scenarios)
        n73 = int(transition_matrix[7,3]*n_scenarios)
        diff7 = n70 + n71 + n72 + n73 - n_scenarios
        n73 = n73 - diff7
        speed_with_error[0:n70] = 1
        speed_error[0:n70] = 30
        speed_with_error[n70:n70+n71] = 10
        speed_error[n70:n70+n71] = 20
        speed_with_error[n70+n71:n70+n71+n72] = 20
        speed_error[n70+n71:n70+n71+n72] = 10
        speed_with_error[n70+n71+n72:n70+n71+n72+n73] = 30
        speed_error[n70+n71+n72:n70+n71+n72+n73] = 0
    speed_with_error = np.where(speed_with_error > 0, speed_with_error, 0) 
    V6_generated = speed_with_error.T
    speed_error = speed_error.T
    
    # call function "PCAcalculation" to find the linear relationship between different weather features
    PCA_results = PCAcalculation(start_point,df_farm)

    # generate forecast errors of other weather features based on PCA results
    p_generated = p-speed_error* PCA_results[0]
    phi_generated = phi-speed_error*PCA_results[1]
    t_generated = t-speed_error*PCA_results[2]
    D1_generated = D1-np.arcsin(speed_error*PCA_results[3])*180/math.pi
    D2_generated = D2-np.arcsin(speed_error*PCA_results[4])*180/math.pi
    D3_generated = D3-np.arcsin(speed_error*PCA_results[5])*180/math.pi
    D4_generated = D4-np.arcsin(speed_error*PCA_results[6])*180/math.pi
    D5_generated = D5-np.arcsin(speed_error*PCA_results[7])*180/math.pi
    D6_generated = D6-np.arcsin(speed_error*PCA_results[8])*180/math.pi
    D7_generated = D7-np.arcsin(speed_error*PCA_results[9])*180/math.pi
    D8_generated = D8-np.arcsin(speed_error*PCA_results[10])*180/math.pi
    D9_generated = D9-np.arcsin(speed_error*PCA_results[11])*180/math.pi
    V1_generated = V1-speed_error*PCA_results[12]
    V2_generated = V2-speed_error*PCA_results[13]
    V3_generated = V3-speed_error*PCA_results[14]
    V4_generated = V4-speed_error*PCA_results[15]
    V5_generated = V5-speed_error*PCA_results[16]
    V6_generated = V6-speed_error*PCA_results[17]
    V7_generated = V7-speed_error*PCA_results[18]
    V8_generated = V8-speed_error*PCA_results[19]
    V9_generated = V9-speed_error*PCA_results[20]

    V1_generated = np.where(V1_generated > 0, V1_generated, 0) 
    V2_generated = np.where(V2_generated > 0, V2_generated, 0) 
    V3_generated = np.where(V3_generated > 0, V3_generated, 0) 
    V4_generated = np.where(V4_generated > 0, V4_generated, 0) 
    V5_generated = np.where(V5_generated > 0, V5_generated, 0) 
    V6_generated = np.where(V6_generated > 0, V6_generated, 0) 
    V7_generated = np.where(V7_generated > 0, V7_generated, 0) 
    V8_generated = np.where(V8_generated > 0, V8_generated, 0) 
    V9_generated = np.where(V9_generated > 0, V9_generated, 0) 
    D1_generated = np.where(D1_generated < 360, D1_generated, D1_generated-360) 
    D1_generated = np.where(D1_generated > 0, D1_generated, D1_generated+360)
    D2_generated = np.where(D2_generated < 360, D2_generated, D2_generated-360) 
    D2_generated = np.where(D2_generated > 0, D2_generated, D2_generated+360)
    D3_generated = np.where(D3_generated < 360, D3_generated, D3_generated-360) 
    D3_generated = np.where(D3_generated > 0, D3_generated, D3_generated+360)
    D4_generated = np.where(D4_generated < 360, D4_generated, D4_generated-360) 
    D4_generated = np.where(D4_generated > 0, D4_generated, D4_generated+360)
    D5_generated = np.where(D5_generated < 360, D5_generated, D5_generated-360) 
    D5_generated = np.where(D5_generated > 0, D5_generated, D5_generated+360)
    D6_generated = np.where(D6_generated < 360, D6_generated, D6_generated-360) 
    D6_generated = np.where(D6_generated > 0, D6_generated, D6_generated+360)
    D7_generated = np.where(D7_generated < 360, D7_generated, D7_generated-360) 
    D7_generated = np.where(D7_generated > 0, D7_generated, D7_generated+360)
    D8_generated = np.where(D8_generated < 360, D8_generated, D8_generated-360) 
    D8_generated = np.where(D8_generated > 0, D8_generated, D8_generated+360)
    D9_generated = np.where(D1_generated < 360, D9_generated, D9_generated-360) 
    D9_generated = np.where(D1_generated > 0, D9_generated, D9_generated+360)
    phi_generated = np.where(phi_generated > 0, phi_generated, 0)
    phi_generated = np.where(phi_generated < 100, phi_generated, 100)

    # call function "PowerCalculation" to calculate stressed wind power based on stressed weather features
    power_forecast = PowerCalculation(p,phi,t,D1,D2,D3,D4,D5,D6,D7,D8,D9,V1,V2,V3,V4,V5,V6,V7,V8,V9)
    power_forecast_final = power_forecast*farm_rated/turbine_rated*(1-farm_loss)/1000
    power_generated = np.zeros((n_scenarios))
    for i in range(n_scenarios):
        power_generated[i] = PowerCalculation(p_generated[i],phi_generated[i],t_generated[i],
                                D1_generated[i],D2_generated[i],D3_generated[i],D4_generated[i],
                                D5_generated[i],D6_generated[i],D7_generated[i],D8_generated[i],
                                D9_generated[i],V1_generated[i],V2_generated[i],V3_generated[i],
                                V4_generated[i],V5_generated[i],V6_generated[i],V7_generated[i],
                                V8_generated[i],V9_generated[i])
    power_generated_final = np.zeros((n_scenarios))
    for i in range(n_scenarios):
        power_generated_final[i] = power_generated[i]*farm_rated/turbine_rated*(1-farm_loss)/1000

    return power_forecast_final, power_generated_final