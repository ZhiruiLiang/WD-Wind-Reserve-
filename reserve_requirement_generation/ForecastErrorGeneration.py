import numpy as np
import pandas as pd
import math 
from scipy.interpolate import interp1d
from distfit import distfit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import sklearn.cluster as cluster
import scipy.io as scio
import csv
from WindPowerFunctions import PowerCalculation
from WindPowerFunctions import PowerCurveGenerator
from WindPowerFunctions import CurveInterpolation


def ErrorGeneration(f,d,start_point,risk):  
    turbine_rated = 10*1e6
    farm_loss = 0.15
    sample = 100
    n_point = 24
    year = 2013
    farm_list = [0,1,2,3,4,5,8,10,11,14,16,19,21,22,23,24,26,28,29,31,33,34,35,36,37]

    farm_information = pd.read_csv('data//38farm_information.csv')
    Pmax = np.array(farm_information['pmax'])
    farm_rated = Pmax[farm_list[f]]*1e6

    df_farm = pd.read_csv('data//data_from_NREL_toolkit//farm_{}_{}.csv'.format(farm_list[f],year))
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
    
    index_min = start_point+d*n_point
    index_max = start_point+(d+1)*n_point

    p = np.array(df_farm['surface air pressure (Pa)'])[index_min:index_max]
    phi = np.array(df_farm['relative humidity at 2m (%)'])[index_min:index_max]
    t = np.array(df_farm['air temperature at 10m (C)'])[index_min:index_max]
    D1 = np.array(df_farm['wind direction at 10m (deg)'])[index_min:index_max]
    D2 = np.array(df_farm['wind direction at 40m (deg)'])[index_min:index_max]
    D3 = np.array(df_farm['wind direction at 60m (deg)'])[index_min:index_max]
    D4 = np.array(df_farm['wind direction at 80m (deg)'])[index_min:index_max]
    D5 = np.array(df_farm['wind direction at 100m (deg)'])[index_min:index_max]
    D6 = np.array(df_farm['wind direction at 120m (deg)'])[index_min:index_max]
    D7 = np.array(df_farm['wind direction at 140m (deg)'])[index_min:index_max]
    D8 = np.array(df_farm['wind direction at 160m (deg)'])[index_min:index_max]
    D9 = np.array(df_farm['wind direction at 200m (deg)'])[index_min:index_max]
    V1 = np.array(df_farm['wind speed at 10m (m/s)'])[index_min:index_max]
    V2 = np.array(df_farm['wind speed at 40m (m/s)'])[index_min:index_max]
    V3 = np.array(df_farm['wind speed at 60m (m/s)'])[index_min:index_max]
    V4 = np.array(df_farm['wind speed at 80m (m/s)'])[index_min:index_max]
    V5 = np.array(df_farm['wind speed at 100m (m/s)'])[index_min:index_max]
    V6 = np.array(df_farm['wind speed at 120m (m/s)'])[index_min:index_max]
    V7 = np.array(df_farm['wind speed at 140m (m/s)'])[index_min:index_max]
    V8 = np.array(df_farm['wind speed at 160m (m/s)'])[index_min:index_max]
    V9 = np.array(df_farm['wind speed at 200m (m/s)'])[index_min:index_max]

    speed_with_error = np.zeros((sample,n_point))
    speed_error = np.zeros((sample,n_point))
    for h in range(24):
        if V6[h]<4:
            n00 = int(transition_matrix[0,0]*sample)
            n01 = int(transition_matrix[0,1]*sample)
            n02 = int(transition_matrix[0,2]*sample)
            n03 = int(transition_matrix[0,3]*sample)
            diff0 = n00 + n01 + n02 + n03 - sample
            n00 = n00 - diff0
            speed_with_error[0:n00,h] = 1
            speed_error[0:n00,h] = 0
            speed_with_error[n00:n00+n01,h] = V6[h]+dist_1.generate(n=n01)
            speed_error[n00:n00+n01,h] = -dist_1.generate(n=n01)
            speed_with_error[n00+n01:n00+n01+n02,h] = 20
            speed_error[n00+n01:n00+n01+n02,h] = 0
            speed_with_error[n00+n01+n02:n00+n01+n02+n03,h] = 30
            speed_error[n00+n01+n02:n00+n01+n02+n03,h] = 0
        elif V6[h]<6:
            n10 = int(transition_matrix[1,0]*sample)
            n11 = int(transition_matrix[1,1]*sample)
            n12 = int(transition_matrix[1,2]*sample)
            n13 = int(transition_matrix[1,3]*sample)
            diff1 = n10 + n11 + n12 + n13 - sample
            n11 = n11 - diff1
            speed_with_error[0:n10,h] = 1
            speed_error[0:n10,h] = 1
            speed_with_error[n10:n10+n11,h] = V6[h]+dist_21.generate(n=n11)
            speed_error[n10:n10+n11,h] = -dist_21.generate(n=n11)
            speed_with_error[n10+n11:n10+n11+n12,h] = 20
            speed_error[n10+n11:n10+n11+n12,h] = 0
            speed_with_error[n10+n11+n12:n10+n11+n12+n13,h] = 30
            speed_error[n10+n11+n12:n10+n11+n12+n13,h] = 0
        elif V6[h]<8:
            n20 = int(transition_matrix[2,0]*sample)
            n21 = int(transition_matrix[2,1]*sample)
            n22 = int(transition_matrix[2,2]*sample)
            n23 = int(transition_matrix[2,3]*sample)
            diff2 = n20 + n21 + n22 + n23 - sample
            n21 = n21 - diff2
            speed_with_error[0:n20,h] = 1
            speed_error[0:n20,h] = 0
            speed_with_error[n20:n20+n21,h] = V6[h]+dist_22.generate(n=n21)
            speed_error[n20:n20+n21,h] = -dist_22.generate(n=n21)
            speed_with_error[n20+n21:n20+n21+n22,h] = 20
            speed_error[n20+n21:n20+n21+n22,h] = 0
            speed_with_error[n20+n21+n22:n20+n21+n22+n23,h] = 30
            speed_error[n20+n21+n22:n20+n21+n22+n23,h] = 0
        elif V6[h]<10:
            n30 = int(transition_matrix[3,0]*sample)
            n31 = int(transition_matrix[3,1]*sample)
            n32 = int(transition_matrix[3,2]*sample)
            n33 = int(transition_matrix[3,3]*sample)
            diff3 = n30 + n31 + n32 + n33 - sample
            n31 = n31 - diff3
            speed_with_error[0:n30,h] = 1
            speed_error[0:n30,h] = 0
            speed_with_error[n30:n30+n31,h] = V6[h]+dist_23.generate(n=n31)
            speed_error[n30:n30+n31,h] = -dist_23.generate(n=n31)
            speed_with_error[n30+n31:n30+n31+n32,h] = 20
            speed_error[n30+n31:n30+n31+n32,h] = 0
            speed_with_error[n30+n31+n32:n30+n31+n32+n33,h] = 30
            speed_error[n30+n31+n32:n30+n31+n32+n33,h] = 0
        elif V6[h]<12:
            n40 = int(transition_matrix[4,0]*sample)
            n41 = int(transition_matrix[4,1]*sample)
            n42 = int(transition_matrix[4,2]*sample)
            n43 = int(transition_matrix[4,3]*sample)
            diff4 = n40 + n41 + n42 + n43 - sample
            n41 = n41 - diff4
            speed_with_error[0:n40,h] = 1
            speed_error[0:n40,h] = 0
            speed_with_error[n40:n40+n41,h] = V6[h]+dist_24.generate(n=n41)
            speed_error[n40:n40+n41,h] = -dist_24.generate(n=n41)
            speed_with_error[n40+n41:n40+n41+n42,h] = 20
            speed_error[n40+n41:n40+n41+n42,h] = 0
            speed_with_error[n40+n41+n42:n40+n41+n42+n43,h] = 30
            speed_error[n40+n41+n42:n40+n41+n42+n43,h] = 0
        elif V6[h]<14:
            n50 = int(transition_matrix[5,0]*sample)
            n51 = int(transition_matrix[5,1]*sample)
            n52 = int(transition_matrix[5,2]*sample)
            n53 = int(transition_matrix[5,3]*sample)
            diff5 = n50 + n51 + n52 + n53 - sample
            n51 = n51 - diff5
            speed_with_error[0:n50,h] = 1
            speed_error[0:n50,h] = 0
            speed_with_error[n50:n50+n51,h] = V6[h]+dist_25.generate(n=n51)
            speed_error[n50:n50+n51,h] = -dist_25.generate(n=n51)
            speed_with_error[n50+n51:n50+n51+n52,h] = 20
            speed_error[n50+n51:n50+n51+n52,h] = 0
            speed_with_error[n50+n51+n52:n50+n51+n52+n53,h] = 30
            speed_error[n50+n51+n52:n50+n51+n52+n53,h] = 0
        elif V6[h]<25:
            n60 = int(transition_matrix[6,0]*sample)
            n61 = int(transition_matrix[6,1]*sample)
            n62 = int(transition_matrix[6,2]*sample)
            n63 = int(transition_matrix[6,3]*sample)
            diff6 = n60 + n61 + n62 + n63 - sample
            n62 = n62 - diff6
            speed_with_error[0:n60,h] = 1
            speed_error[0:n60,h] = 15
            speed_with_error[n60:n60+n61,h] = 10
            speed_error[n60:n60+n61,h] = 5
            speed_with_error[n60+n61:n60+n61+n62,h] = 20
            speed_error[n60+n61:n60+n61+n62,h] = 0
            speed_with_error[n60+n61+n62:n60+n61+n62+n63,h] = 30
            speed_error[n60+n61+n62:n60+n61+n62+n63,h] = -5
        else:
            n70 = int(transition_matrix[7,0]*sample)
            n71 = int(transition_matrix[7,1]*sample)
            n72 = int(transition_matrix[7,2]*sample)
            n73 = int(transition_matrix[7,3]*sample)
            diff7 = n70 + n71 + n72 + n73 - sample
            n73 = n73 - diff7
            speed_with_error[0:n70,h] = 1
            speed_error[0:n70,h] = 30
            speed_with_error[n70:n70+n71,h] = 10
            speed_error[n70:n70+n71,h] = 20
            speed_with_error[n70+n71:n70+n71+n72,h] = 20
            speed_error[n70+n71:n70+n71+n72,h] = 10
            speed_with_error[n70+n71+n72:n70+n71+n72+n73,h] = 30
            speed_error[n70+n71+n72:n70+n71+n72+n73,h] = 0
    speed_with_error = np.where(speed_with_error > 0, speed_with_error, 0) 
    V6_generated = speed_with_error.T
    speed_error = speed_error.T
    
    weather_used = np.hstack((p.reshape(-1,1),phi.reshape(-1,1),t.reshape(-1,1),D1.reshape(-1,1),D2.reshape(-1,1),
                            D3.reshape(-1,1),D4.reshape(-1,1),D5.reshape(-1,1),D6.reshape(-1,1),D7.reshape(-1,1),
                            D8.reshape(-1,1),D9.reshape(-1,1),V1.reshape(-1,1),V2.reshape(-1,1),V3.reshape(-1,1),
                            V4.reshape(-1,1),V5.reshape(-1,1),V6.reshape(-1,1),V7.reshape(-1,1),V8.reshape(-1,1),V9.reshape(-1,1)))

    for i in range(24):
        for j in range(9):
            weather_used[i,3+j] = math.sin(weather_used[i,3+j]/180*math.pi)

    zscoredData = stats.zscore(weather_used)
    pca = PCA().fit(zscoredData)
    eigVals = pca.explained_variance_
    loadings = pca.components_*-1

    original_std=[]
    for i in range(21):
        original_std.append(weather_used[:,i].std())
        
    new_vector= eigVals[0]*loadings[0,:]+eigVals[1]*loadings[1,:]+eigVals[2]*loadings[2,:]

    new_vector_reverse = []
    for i in range(21):
        new_vector_reverse.append(new_vector[i]*original_std[i])
        
    new_vector_scaled = new_vector_reverse/new_vector_reverse[17]

    new_vector_scaled =[[-5.70797309],
                        [ 1.48273962],
                        [-1.00049961],
                        [-0.08503499],
                        [-0.09118537],
                        [-0.09302854],
                        [-0.0919131 ],
                        [-0.09125607],
                        [-0.09056525],
                        [-0.08977991],
                        [-0.08905928],
                        [-0.08803062],
                        [ 0.34943271],
                        [ 0.6077643 ],
                        [ 0.72760302],
                        [ 0.82973781],
                        [ 0.9208548 ],
                        [ 1.        ],
                        [ 1.07376508],
                        [ 1.13630065],
                        [ 1.23868886]]

    p_generated = p.reshape(-1,1)-speed_error* new_vector_scaled[0]
    phi_generated = phi.reshape(-1,1)-speed_error*new_vector_scaled[1]
    t_generated = t.reshape(-1,1)-speed_error*new_vector_scaled[2]
    D1_generated = D1.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[3])*180/math.pi
    D2_generated = D2.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[4])*180/math.pi
    D3_generated = D3.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[5])*180/math.pi
    D4_generated = D4.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[6])*180/math.pi
    D5_generated = D5.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[7])*180/math.pi
    D6_generated = D6.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[8])*180/math.pi
    D7_generated = D7.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[9])*180/math.pi
    D8_generated = D8.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[10])*180/math.pi
    D9_generated = D9.reshape(-1,1)-np.arcsin(speed_error*new_vector_scaled[11])*180/math.pi
    V1_generated = V1.reshape(-1,1)-speed_error*new_vector_scaled[12]
    V2_generated = V2.reshape(-1,1)-speed_error*new_vector_scaled[13]
    V3_generated = V3.reshape(-1,1)-speed_error*new_vector_scaled[14]
    V4_generated = V4.reshape(-1,1)-speed_error*new_vector_scaled[15]
    V5_generated = V5.reshape(-1,1)-speed_error*new_vector_scaled[16]
    V6_generated = V6.reshape(-1,1)-speed_error*new_vector_scaled[17]
    V7_generated = V7.reshape(-1,1)-speed_error*new_vector_scaled[18]
    V8_generated = V8.reshape(-1,1)-speed_error*new_vector_scaled[19]
    V9_generated = V9.reshape(-1,1)-speed_error*new_vector_scaled[20]

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

    power_forecast = []
    for h in range(n_point):
        power_forecast.append(PowerCalculation(p[h],phi[h],t[h],D1[h],D2[h],D3[h],D4[h],D5[h],D6[h],D7[h],
                                            D8[h],D9[h],V1[h],V2[h],V3[h],V4[h],V5[h],V6[h],V7[h],V8[h],V9[h]))
    power_forecast_farm = np.zeros((24,1))
    for h in range(n_point):
        power_forecast_farm[h] = power_forecast[h]*farm_rated/turbine_rated*(1-farm_loss)/1000

    power_generated = np.zeros((24,sample))
    for i in range(sample):
        for h in range(24):
            power_generated[h,i] = PowerCalculation(p_generated[h,i],phi_generated[h,i],t_generated[h,i],
                                    D1_generated[h,i],D2_generated[h,i],D3_generated[h,i],D4_generated[h,i],
                                    D5_generated[h,i],D6_generated[h,i],D7_generated[h,i],D8_generated[h,i],
                                    D9_generated[h,i],V1_generated[h,i],V2_generated[h,i],V3_generated[h,i],
                                    V4_generated[h,i],V5_generated[h,i],V6_generated[h,i],V7_generated[h,i],
                                    V8_generated[h,i],V9_generated[h,i])
    power_generated_farm = np.zeros((24,sample))
    for i in range(sample):
        for h in range(24):
            power_generated_farm[h,i] = power_generated[h,i]*farm_rated/turbine_rated*(1-farm_loss)/1000
    power_max = np.zeros((n_point,1))
    power_min = np.zeros((n_point,1))
    for h in range(n_point):
        power_max[h,:] = np.max(power_generated_farm [h,:])
        power_min[h,:] = np.min(power_generated_farm [h,:])
   

    if risk == 0:
        error_generated = power_forecast_farm - power_generated_farm  
        error_min_20=[]
        error_max_20=[]
        error_min_40=[]
        error_max_40=[]
        error_min_60=[]
        error_max_60=[]
        error_min_80=[]
        error_max_80=[]
        for h in range(n_point):
            error_sort = np.argsort(abs(error_generated[h,:]))
            error_20 = error_sort[0:int(sample*0.2)]
            error_40 = error_sort[0:int(sample*0.4)]
            error_60 = error_sort[0:int(sample*0.6)]
            error_80 = error_sort[0:int(sample*0.8)]
            error_min_20.append(np.min(error_generated[h,error_20]))
            error_max_20.append(np.max(error_generated[h,error_20]))
            error_min_40.append(np.min(error_generated[h,error_40]))
            error_max_40.append(np.max(error_generated[h,error_40]))
            error_min_60.append(np.min(error_generated[h,error_60]))
            error_max_60.append(np.max(error_generated[h,error_60]))
            error_min_80.append(np.min(error_generated[h,error_80]))
            error_max_80.append(np.max(error_generated[h,error_80]))

        upward_20 = np.array(error_max_20)
        downward_20 = -np.array(error_min_20)
        upward_40 = np.array(error_max_40)
        downward_40 = -np.array(error_min_40)
        upward_60 = np.array(error_max_60)
        downward_60 = -np.array(error_min_60)
        upward_80 = np.array(error_max_80)
        downward_80 = -np.array(error_min_80)
        upward_100 = (power_forecast_farm-power_min).reshape(-1)
        downward_100 = -(power_forecast_farm-power_max).reshape(-1)

        upward_L1 = np.where(upward_20> 0, upward_20, 0)
        downward_L1 = np.where(downward_20> 0, downward_20, 0)
        upward_L2 = np.where(upward_40> 0, upward_40, 0)
        downward_L2 = np.where(downward_40> 0, downward_40, 0)
        upward_L3 = np.where(upward_60> 0, upward_60, 0)
        downward_L3 = np.where(downward_60> 0, downward_60, 0)
        upward_L4 = np.where(upward_80> 0, upward_80, 0)
        downward_L4 = np.where(downward_80> 0, downward_80, 0)
        upward_L5 = np.where(upward_100> 0, upward_100, 0)
        downward_L5 = np.where(downward_100> 0, downward_100, 0)

    elif risk == 1:
        difference = np.zeros((n_point,sample))
        probability = np.zeros((n_point,sample))
        risk = np.zeros((n_point,sample))
        error_sort = np.zeros((n_point,sample))
        for h in range(n_point):
            power_sort = np.sort(power_generated_farm[h,:])
            error_sort[h,:] = power_sort - power_forecast_farm[h]
            for i in range(sample):
                if power_sort[i] > power_forecast_farm[h]:
                    difference[h,i] = power_sort[sample-1] - power_sort[i]
                    probability[h,i] = np.sum(power_sort>power_sort[i])/sample
                elif power_sort[i] < power_forecast_farm[h]:
                    difference[h,i] = power_sort[0] - power_sort[i]
                    probability[h,i] = np.sum(power_sort<power_sort[i])/sample
                elif power_sort[i] == power_forecast_farm[h]:
                    difference[h,i] = 0
                    probability[h,i] = 0
                risk[h,i] = difference[h,i]*probability[h,i]

        r1= farm_rated/1000000*0.1
        r2= farm_rated/1000000*0.2
        r3= farm_rated/1000000*0.3
        r4= farm_rated/1000000*0.4
        r5= farm_rated/1000000*0.5
        risk_min_100=[]
        risk_max_100=[]
        risk_min_200=[]
        risk_max_200=[]
        risk_min_300=[]
        risk_max_300=[]
        risk_min_400=[]
        risk_max_400=[]
        risk_min_500=[]
        risk_max_500=[]
        for h in range(n_point):
            error_min_100=[0]
            error_max_100=[0]
            error_min_200=[0]
            error_max_200=[0]
            error_min_300=[0]
            error_max_300=[0]
            error_min_400=[0]
            error_max_400=[0]
            error_min_500=[0]
            error_max_500=[0]
            for i in range(sample):
                if risk[h,i]>=0 and risk[h,i]<=r1:
                    error_max_100.append(error_sort[h,i])
                elif risk[h,i]>=0 and risk[h,i]<=r2:
                    error_max_100.append(error_sort[h,i])
                    error_max_200.append(error_sort[h,i])
                elif risk[h,i]>=0 and risk[h,i]<=r3:
                    error_max_100.append(error_sort[h,i])
                    error_max_200.append(error_sort[h,i])
                    error_max_300.append(error_sort[h,i])
                elif risk[h,i]>=0 and risk[h,i]<=r4:
                    error_max_100.append(error_sort[h,i])
                    error_max_200.append(error_sort[h,i])
                    error_max_300.append(error_sort[h,i])
                    error_max_400.append(error_sort[h,i])
                elif risk[h,i]>=0 and risk[h,i]<=r5:
                    error_max_100.append(error_sort[h,i])
                    error_max_200.append(error_sort[h,i])
                    error_max_300.append(error_sort[h,i])
                    error_max_400.append(error_sort[h,i])
                    error_max_500.append(error_sort[h,i])
                elif risk[h,i]<0 and risk[h,i]>=-r1:
                    error_min_100.append(error_sort[h,i])
                elif risk[h,i]<0 and risk[h,i]>=-r2:
                    error_min_100.append(error_sort[h,i])
                    error_min_200.append(error_sort[h,i])
                elif risk[h,i]<0 and risk[h,i]>=-r3:
                    error_min_100.append(error_sort[h,i])
                    error_min_200.append(error_sort[h,i])
                    error_min_300.append(error_sort[h,i])
                elif risk[h,i]<0 and risk[h,i]>=-r4:
                    error_min_100.append(error_sort[h,i])
                    error_min_200.append(error_sort[h,i])
                    error_min_300.append(error_sort[h,i])
                    error_min_400.append(error_sort[h,i])
                elif risk[h,i]<0 and risk[h,i]>=-r5:
                    error_min_100.append(error_sort[h,i])
                    error_min_200.append(error_sort[h,i])
                    error_min_300.append(error_sort[h,i])
                    error_min_400.append(error_sort[h,i])
                    error_min_500.append(error_sort[h,i])
            risk_min_100.append(-np.min(error_min_100))
            risk_max_100.append(np.max(error_max_100))
            risk_min_200.append(-np.min(error_min_200))
            risk_max_200.append(np.max(error_max_200))
            risk_min_300.append(-np.min(error_min_300))
            risk_max_300.append(np.max(error_max_300))
            risk_min_400.append(-np.min(error_min_400))
            risk_max_400.append(np.max(error_max_400))
            risk_min_500.append(-np.min(error_min_500))
            risk_max_500.append(np.max(error_max_500))
              
        downward_L5 = risk_max_100
        upward_L5 = risk_min_100
        downward_L4 = risk_max_200
        upward_L4 = risk_min_200
        downward_L3 = risk_max_300
        upward_L3 = risk_min_300
        downward_L2 = risk_max_400
        upward_L2 = risk_min_400
        downward_L1 = risk_max_500
        upward_L1 = risk_min_500

    return upward_L1, downward_L1, upward_L2, downward_L2, upward_L3, downward_L3, upward_L4, downward_L4, upward_L5, downward_L5