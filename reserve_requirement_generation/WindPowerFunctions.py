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
#%% Power Calculation
def PowerCalculation(p,phi,t,D1,D2,D3,D4,D5,D6,D7,D8,D9,V1,V2,V3,V4,V5,V6,V7,V8,V9):    
    P_nominal = 10*1e6
    cut_in = 4
    cut_out = 25
    speed_rated = 12
    speed_min = 0
    speed_max = 40
    speed_step = 0.1
    h = 120
    d = 205

    df_10MW = pd.read_csv('data//Power_curve_NREL_Reference_10MW.csv')
    Speed = df_10MW['Wind Speed [m/s]'].tolist()
    Cp = df_10MW['Cp [-]'].tolist()
    f = interp1d(Speed, Cp)
    speed_start = 1
    speed_end = 25
    xnew = np.linspace(speed_start, speed_end, num=int((speed_end-speed_start)/speed_step+1), endpoint=True)
    Cp_inter = np.zeros(int(speed_max/speed_step)+1)
    Cp_inter[int(speed_start/speed_step):int(speed_end/speed_step+1)] = f(xnew)

    M_d = 0.0289652
    M_v = 0.018016
    R = 8.31446
    H_n = 10.4*1000
    T = t + 273.15
    TT = 7.5*t/(t+237.3)
    p_sat = 100*6.1078 * 10**TT
    p_v = phi/100 * p_sat
    p_d = p - p_v
    rho_sealevel = (p_d * M_d + p_v * M_v) / R / T 
    rho_120m = rho_sealevel * math.exp(-h/H_n)

    A_rotor = math.pi*(d/2)**2

    a1 = 0.060736668
    a2 = 0.104731106
    a3 = 0.126323631
    a4 = 0.137618881
    a5 = 0.141179426
    a6 = 0.137618881
    a7 = 0.126323631
    a8 = 0.104731106
    a9 = 0.060736668

    phi1 = D1 - D5
    phi2 = D2 - D5
    phi3 = D3 - D5
    phi4 = D4 - D5
    phi5 = D5 - D5
    phi6 = D6 - D5
    phi7 = D7 - D5
    phi8 = D8 - D5
    phi9 = D9 - D5

    U1 = a1* (V1 * math.cos(phi1/180*math.pi))**3
    U2 = a2* (V2 * math.cos(phi2/180*math.pi))**3
    U3 = a3* (V3 * math.cos(phi3/180*math.pi))**3
    U4 = a4* (V4 * math.cos(phi4/180*math.pi))**3
    U5 = a5* (V5 * math.cos(phi5/180*math.pi))**3
    U6 = a6* (V6 * math.cos(phi6/180*math.pi))**3
    U7 = a7* (V7 * math.cos(phi7/180*math.pi))**3
    U8 = a8* (V8 * math.cos(phi8/180*math.pi))**3
    U9 = a9* (V9 * math.cos(phi9/180*math.pi))**3

    U1 = np.maximum(U1,0)
    U2 = np.maximum(U2,0)
    U3 = np.maximum(U3,0)
    U4 = np.maximum(U4,0)
    U5 = np.maximum(U5,0)
    U6 = np.maximum(U6,0)
    U7 = np.maximum(U7,0)
    U8 = np.maximum(U8,0)
    U9 = np.maximum(U9,0)

    V_eq = (U1+U2+U3+U4+U5+U6+U7+U8+U9)**(1/3)
    if math.isnan(V_eq) ==1:
        V_eq = V5
    rho = rho_120m
    df_curve = PowerCurveGenerator(P_nominal, rho, A_rotor, Cp_inter, cut_in, cut_out, speed_rated, speed_min, speed_max, speed_step)
    power_inter = CurveInterpolation(df_curve)
    if V_eq <= speed_max:
        power_calculated = power_inter[int(round(V_eq,2)*100-1)]
    else:
        power_calculated = 0

    return power_calculated

#%% Power Curve Generator
def PowerCurveGenerator(P_nominal, rho, A_rotor, Cp_inter, cut_in, cut_out, speed_rated, speed_min, speed_max, speed_step):

    df_curve = pd.DataFrame(index = np.arange(speed_min, speed_max + speed_step, speed_step), columns = ['wind_speed'])
    df_curve['wind_speed'] = df_curve.index
    df_curve['Cp'] = Cp_inter
     
    # Region II
    df_curve['Power'] = 1/2 * rho * A_rotor * df_curve.Cp * df_curve.wind_speed**3 
     
    # Region I
    df_curve['Power'][df_curve['wind_speed'] < cut_in] = 0
    
    # Region III
    df_curve['Power'][df_curve['Power'] >  P_nominal ] = P_nominal

    # Region IV
    df_curve['Power'][df_curve['wind_speed'] > cut_out] = 0
    df_curve['Power'] = df_curve['Power']/1e3
     
    return df_curve

#%% Curve Interpolation
def CurveInterpolation(df_curve):
    power = df_curve['Power'].tolist()
    wind_speed = df_curve['wind_speed'].tolist()
    ff = interp1d(wind_speed, power)
    start = 0
    end = 40
    step = 0.01
    xnew = np.linspace(start, end, num=int((end-start)/step), endpoint=True)
    power_inter = np.zeros(int(end/step))
    power_inter[int(start/step):int(end/step)] = ff(xnew)
    return power_inter