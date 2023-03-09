'''
[ARPA-E Perform Code]
Function: Use PCA to find the linear relationship between different weather features

Author: Zhirui Liang, zliang31@jhu.edu
Date: 2022-10-05
'''
import numpy as np
import pandas as pd
import math 
from sklearn.decomposition import PCA
from scipy import stats

def PCAcalculation(start_point,df_farm):
    n_point = 24
    index_min = start_point-n_point
    index_max = start_point

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

    weather_used = np.hstack((p.reshape(-1,1),phi.reshape(-1,1),t.reshape(-1,1),D1.reshape(-1,1),D2.reshape(-1,1),
                            D3.reshape(-1,1),D4.reshape(-1,1),D5.reshape(-1,1),D6.reshape(-1,1),D7.reshape(-1,1),
                            D8.reshape(-1,1),D9.reshape(-1,1),V1.reshape(-1,1),V2.reshape(-1,1),V3.reshape(-1,1),
                            V4.reshape(-1,1),V5.reshape(-1,1),V6.reshape(-1,1),V7.reshape(-1,1),V8.reshape(-1,1),V9.reshape(-1,1)))
    for i in range(n_point):
        for j in range(9):
            weather_used[i,3+j] = math.sin(weather_used[i,3+j]/180*math.pi)

    zscoredData = stats.zscore(weather_used)
    pca = PCA().fit(zscoredData)
    eigVals = pca.explained_variance_
    loadings = pca.components_*-1

    original_std=[]
    for i in range(21):
        original_std.append(weather_used[:,i].std())
        
    PCA_results= eigVals[0]*loadings[0,:]+eigVals[1]*loadings[1,:]+eigVals[2]*loadings[2,:]

    return PCA_results