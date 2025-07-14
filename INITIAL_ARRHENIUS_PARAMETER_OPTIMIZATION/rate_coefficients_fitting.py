import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import csv

R_kcal = 1.98720425864083e-3; # [kcal/K mol]
# Ea [kcal/mol], T [K]

def sort_csv_by_temperature(filename):
    df = pd.read_csv(filename)
    df_sorted = df.sort_values(by='temperature')  # sort by temperature ascending
    df_sorted.to_csv(filename, index=False)       # overwrite the file

def rate_constant(T, b, Ea):
    return 100.0e6 * (T**b) * np.exp(-Ea / (R_kcal*T))

#data = pd.read_csv('C:/CRNN/Validation/LLNL_feed_cons_1ate.1sv')
sort_csv_by_temperature('C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INITIAL_ARRHENIUS_PARAMETER_OPTIMIZATION/INITIAL_FEEED_CONSUMPTION_RATE/NUIG_feed_cons_rate_1b_v2.csv')
data = pd.read_csv('C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INITIAL_ARRHENIUS_PARAMETER_OPTIMIZATION/INITIAL_FEEED_CONSUMPTION_RATE/NUIG_feed_cons_rate_1b_v2.csv')

T_data = data['temperature']
ini_data = data['initial_feed_concentration']
cons_data = data['feed_cons_rate']
reaction_rate_data_er = cons_data / (ini_data**1.0)
reaction_rate_data = reaction_rate_data_er / 1.0
print(reaction_rate_data)

ini_guess = [0, 20]
param_bounds = ([-5, 0], [5, 100])

# Perform the fit
params, covariance = curve_fit(rate_constant, T_data, reaction_rate_data, p0=ini_guess, maxfev=10000, bounds=param_bounds)
params

# Extract the fitted parameters
b_fit, Ea_fit = params
ref_sparsity = 6

print(f"Fitted Parameters:\nb = {b_fit}\nEa = {Ea_fit}")
fig = plt.figure(figsize=(8, 8))
plt.plot(T_data[::ref_sparsity], reaction_rate_data[::ref_sparsity], marker='D', markersize = 12, linestyle = '-', label='Data', color = 'black', linewidth=2)
plt.plot(T_data, rate_constant(T_data, *params), 'r-',label='Fit: b=%5.3f, Ea=%5.3f' % tuple(params), linewidth=2)
plt.xlabel('Temperature [K]', fontsize=22)
plt.ylabel('Rate constant [1/s]', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tick_params(axis='x', direction='in',length=4, width=1.2, top=True)
plt.tick_params(axis='y', direction='in',length=4, width=1.2, right=True)
plt.legend(loc='upper left', fontsize=22)
fig.savefig(f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INITIAL_ARRHENIUS_PARAMETER_OPTIMIZATION/figure_n.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()