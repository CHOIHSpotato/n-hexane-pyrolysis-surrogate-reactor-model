import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# PFR simulation of n-Hexane thermal decomposition

# 1. Mechanism file input
#reaction_mechanism = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/LLNL.yaml' 
#reaction_mechanism = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/JetSurf.yaml'
reaction_mechanism = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/NUIGMech1.1.yaml'


ini_T= np.arange(850,1150,2)
ini_P = np.array([1.0])

array_T=np.repeat(ini_T,len(ini_P))
array_P=np.tile(ini_P,len(ini_T))

composition_0 = 'NC6H14:1.0'

length = 1.6  # *approximate* PFR length [m]
u_0 = 1600.0  # inflow velocity [m/s]
area = 0.0019635  # cross-sectional area [m**2]
n_steps = 10

filename = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INITIAL_ARRHENIUS_PARAMETER_OPTIMIZATION/INITIAL_FEEED_CONSUMPTION_RATE/NUIG_feed_cons_rate_1b_v2.csv'

with open(filename, 'w', newline='') as csvfile:

    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write the header row (optional)
    csvwriter.writerow(['temperature','ini_feed_concentration','feed_cons_rate'])

    for k in range(1,(len(ini_T)*len(ini_P))+1):
        Temperature = array_T[k-1]
        Pressure = array_P[k-1]*100000
        print('Temperature',Temperature,'Pressure',Pressure)

        # import the gas model and set the initial conditions
        gas = ct.Solution(reaction_mechanism)
        gas.TPX = Temperature, Pressure, composition_0
        mass_flow_rate = u_0 * gas.density * area

        # create a new reactor
        r = ct.IdealGasConstPressureReactor(contents=gas, energy='off')
        # create a reactor network for performing time integration
        sim = ct.ReactorNet([r])

        # approximate a time step to achieve a similar resolution as in the next method
        t_total = length / u_0
        dt = t_total / n_steps
        # define time, space, and other information vectors
        t = (np.arange(n_steps+1)) * dt
        z = np.zeros_like(t)
        u = np.zeros_like(t)

        states = ct.SolutionArray(r.thermo)

        # Record the initial state at t = 0
        states.append(r.thermo.state)

        u[0] = mass_flow_rate / area / r.thermo.density
        z[0] = 0

        for n, t_i in enumerate(t[1:], start=1):
            # perform time integration
            sim.advance(t_i)

            # compute velocity and transform into space
            u[n] = mass_flow_rate / area / r.thermo.density
            z[n] = z[n - 1] + u[n] * dt
            states.append(r.thermo.state)

        time_solution = t
        pressure_solution = states.P
        temperature_solution = states.T

        # Calculating feed_conv_rate
        feed_data = states('NC6H14').concentrations
        feed_cons_rate = abs(float(feed_data[1])-float(feed_data[0]))/(float(time_solution[1])) # [mol/L/sec]
        print(feed_cons_rate)

        csvwriter.writerow([float(temperature_solution[0]), float(feed_data[0]), float(feed_cons_rate)])    