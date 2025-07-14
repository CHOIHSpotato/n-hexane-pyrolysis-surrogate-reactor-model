import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import datetime
import multiprocessing
import psutil  # For better CPU core detection
import os

# Restrict threads in Cantera, NumPy, OpenMP, MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# PFR simulation of n-Hwxane thermal decomposition

# 1. Mechanism file input
reaction_mechanism_1 = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/LLNL.yaml' 
reaction_mechanism_2 = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/JetSurf.yaml'
reaction_mechanism_3 = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/NUIGMech1.1.yaml'

# Load the CSV file

# 4-D case
data = np.loadtxt("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_GENERATION/sampling_case_4D.csv", delimiter=",", skiprows=0)

# Get the number of rows (length of the column)
column_length = data.shape[0]
print("Length of the column:", column_length)

## Assign the first column to vector a and second column to vector b for 2D case ##
ini_T = data[:, 0]
ini_P = data[:, 1]
reactor_L = data[:, 2]
reactor_u0 = data[:, 3]

composition_0 = 'NC6H14:1.0, H2O:0.7'
diameter = 0.05              # Diameter of the radiant reactor [m]
area = np.pi * (diameter / 2) ** 2  # Cross-sectional area [m²]
n_steps = 800

# Method: Lagrangian Particle Simulation
def run_simulation(k):
    time.sleep(1)
    """Function to run one PFR simulation"""
    Temperature = ini_T[k-1]
    Pressure = ini_P[k-1]*100000
    length = reactor_L[k-1]
    u_0 = reactor_u0[k-1]
    Temperature_round = round(Temperature, 5)
    Pressure_round = round(Pressure, 5)
    length_round = round(length, 5)
    u_0_round = round(u_0, 5)
    print('Temperature',Temperature_round,'Pressure',Pressure_round)

    # import the gas_1 model and set the initial conditions
    gas_1 = ct.Solution(reaction_mechanism_1)
    gas_1.TPY = Temperature_round, Pressure_round, composition_0
    mass_flow_rate_1 = u_0_round * gas_1.density_mass * area
    # import the gas_2 model and set the initial conditions
    gas_2 = ct.Solution(reaction_mechanism_2)
    gas_2.TPY = Temperature_round, Pressure_round, composition_0
    mass_flow_rate_2 = u_0_round * gas_2.density_mass * area
    # import the gas_3 model and set the initial conditions
    gas_3 = ct.Solution(reaction_mechanism_3)
    gas_3.TPY = Temperature_round, Pressure_round, composition_0
    mass_flow_rate_3 = u_0_round * gas_3.density_mass * area

    # Check the mass flow rate
    print("mass flow rate")
    print(mass_flow_rate_1)
    print(mass_flow_rate_2)
    print(mass_flow_rate_3)

    # Define the length and volume of each reactor segment
    dz = length_round / n_steps
    r_vol = area * dz

    # create a new reactor
    # Isothermal: energy='off' / Adiabatic: energy='on'
    r_1 = ct.IdealGasReactor(contents=gas_1, energy='on')
    r_2 = ct.IdealGasReactor(contents=gas_2, energy='on')
    r_3 = ct.IdealGasReactor(contents=gas_3, energy='on')

    # Volume of each reactor segment
    r_1.volume = r_vol
    r_2.volume = r_vol
    r_3.volume = r_vol

    # create a reservoir to represent the reactor immediately upstream. Note
    # that the gas object is set already to the state of the upstream reactor
    upstream_1 = ct.Reservoir(gas_1, name='upstream_1')
    upstream_2 = ct.Reservoir(gas_2, name='upstream_2')
    upstream_3 = ct.Reservoir(gas_3, name='upstream_3')

    # create a reservoir for the reactor to exhaust into. The composition of
    # this reservoir is irrelevant.
    downstream_1 = ct.Reservoir(gas_1, name='downstream_1')
    downstream_2 = ct.Reservoir(gas_2, name='downstream_2')
    downstream_3 = ct.Reservoir(gas_3, name='downstream_3')

    # The mass flow rate into the reactor will be fixed by using a
    # MassFlowController object.
    m_1 = ct.MassFlowController(upstream_1, r_1, mdot=mass_flow_rate_1)
    m_2 = ct.MassFlowController(upstream_2, r_2, mdot=mass_flow_rate_2)
    m_3 = ct.MassFlowController(upstream_3, r_3, mdot=mass_flow_rate_3)

    # We need an outlet to the downstream reservoir. This will determine the
    # pressure in the reactor. The value of K will only affect the transient
    # pressure difference.
    v_1 = ct.PressureController(r_1, downstream_1, primary=m_1, K=1e-5)
    v_2 = ct.PressureController(r_2, downstream_2, primary=m_2, K=1e-5)
    v_3 = ct.PressureController(r_3, downstream_3, primary=m_3, K=1e-5)

    # create a reactor network for performing time integration
    sim_1 = ct.ReactorNet([r_1])
    sim_2 = ct.ReactorNet([r_2])
    sim_3 = ct.ReactorNet([r_3])

    # define time, space, and other information vectors
    #z = (np.arange(n_steps)+1) * dz
    z = np.linspace(0, length, n_steps + 1)  # Including z=0
    t_r1 = np.zeros_like(z)  # residence time in each reactor
    u1 = np.zeros_like(z)
    t1 = np.zeros_like(z)
    den1 = np.zeros_like(z)
    t_r2 = np.zeros_like(z)  # residence time in each reactor
    u2 = np.zeros_like(z)
    t2 = np.zeros_like(z)
    den2 = np.zeros_like(z)
    t_r3 = np.zeros_like(z)  # residence time in each reactor
    u3 = np.zeros_like(z)
    t3 = np.zeros_like(z)
    den3 = np.zeros_like(z)

    states_1 = ct.SolutionArray(r_1.thermo)
    states_2 = ct.SolutionArray(r_2.thermo)
    states_3 = ct.SolutionArray(r_3.thermo)

    # Record the initial state at t = 0
    states_1.append(r_1.thermo.state)
    states_2.append(r_2.thermo.state)
    states_3.append(r_3.thermo.state)
    u1[0] = mass_flow_rate_1 / area / r_1.thermo.density_mass
    u2[0] = mass_flow_rate_2 / area / r_2.thermo.density_mass
    u3[0] = mass_flow_rate_3 / area / r_3.thermo.density_mass
    den1[0] = r_1.thermo.density_mass
    den2[0] = r_2.thermo.density_mass   
    den3[0] = r_3.thermo.density_mass      

    for n in range(1, n_steps+1):
        # Set the state of the reservoir to match that of the previous reactor
        gas_1.TDY = r_1.thermo.TDY
        upstream_1.syncState()
        gas_2.TDY = r_2.thermo.TDY
        upstream_2.syncState()
        gas_3.TDY = r_3.thermo.TDY
        upstream_3.syncState()

        # integrate the reactor forward in time until steady state is reached
        sim_1.reinitialize()
        sim_1.advance_to_steady_state()
        sim_2.reinitialize()
        sim_2.advance_to_steady_state()
        sim_3.reinitialize()
        sim_3.advance_to_steady_state()

        # compute velocity and transform into time
        den1[n] = r_1.thermo.density_mass
        u1[n] = mass_flow_rate_1 / area / r_1.thermo.density
        t_r1[n] = r_1.mass / mass_flow_rate_1  # residence time in this reactor
        t1[n] = np.sum(t_r1)
        den2[n] = r_2.thermo.density_mass
        u2[n] = mass_flow_rate_2 / area / r_2.thermo.density
        t_r2[n] = r_2.mass / mass_flow_rate_2  # residence time in this reactor
        t2[n] = np.sum(t_r2)
        den3[n] = r_3.thermo.density_mass
        u3[n] = mass_flow_rate_3 / area / r_3.thermo.density
        t_r3[n] = r_3.mass / mass_flow_rate_3  # residence time in this reactor
        t3[n] = np.sum(t_r3)

        # write output data
        states_1.append(r_1.thermo.state)
        states_2.append(r_2.thermo.state)
        states_3.append(r_3.thermo.state)
    
    time_solution_1 = t1
    time_solution_2 = t2
    time_solution_3 = t3
    pressure_solution_1 = states_1.P
    temperature_solution_1 = states_1.T
    pressure_solution_2 = states_2.P
    temperature_solution_2 = states_2.T
    pressure_solution_3 = states_3.P
    temperature_solution_3 = states_3.T

    # Calculating feed_conv_rate
    feed_data_1 = states_1('NC6H14').concentrations
    feed_data_2 = states_2('NC6H14').concentrations
    feed_data_3 = states_3('NC6H14').concentrations

    feed_conv_rate_1 = (float(feed_data_1[0])-float(feed_data_1[n_steps]))*100/float(feed_data_1[0])
    feed_conv_rate_2 = (float(feed_data_2[0])-float(feed_data_2[n_steps]))*100/float(feed_data_2[0])
    feed_conv_rate_3 = (float(feed_data_3[0])-float(feed_data_3[n_steps]))*100/float(feed_data_3[0])

    print("LLNL_feed_conv_rate")
    print(feed_conv_rate_1)
    print("JetSurF_feed_conv_rate")
    print(feed_conv_rate_2)
    print("NUIG_feed_conv_rate")
    print(feed_conv_rate_3)


    # Processing concentration of reactant and product 
    i_var_1 = [gas_1.species_index(s) for s in ['H2','CH4','C2H4','C2H6','C3H6','C4H8-1','NC6H14','C4H10','C5H10-1']] # LLNL
    i_var_2 = [gas_2.species_index(s) for s in ['H2','CH4','C2H4','C2H6','C3H6','C4H81','NC6H14','C4H10','C5H10']] # JetSurf
    i_var_3 = [gas_3.species_index(s) for s in ['H2','CH4','C2H4','C2H6','C3H6','C4H8-1','NC6H14','C4H10','C5H10-1']] # NUIG
    C_11 = states_1.concentrations[:, i_var_1].T
    C_1=C_11[0:9,:]
    C_22 = states_2.concentrations[:, i_var_2].T
    C_2=C_22[0:9,:]
    C_33 = states_3.concentrations[:, i_var_3].T
    C_3=C_33[0:9,:]
    print(C_1.shape)
    print(C_2.shape)
    print(C_3.shape)

    nodedata_time_1 = np.vstack((time_solution_1, temperature_solution_1, pressure_solution_1, C_1, u1, z, den1)).T
    nodedata_time_2 = np.vstack((time_solution_2, temperature_solution_2, pressure_solution_2, C_2, u2, z, den2)).T
    nodedata_time_3 = np.vstack((time_solution_3, temperature_solution_3, pressure_solution_3, C_3, u3, z, den3)).T

    np.savetxt('C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_on/LLNL_Eon_'+str(k)+'.txt', nodedata_time_1)
    np.savetxt('C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_on/JetSurf_Eon_'+str(k)+'.txt', nodedata_time_2)
    np.savetxt('C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_on/NUIG_Eon_'+str(k)+'.txt', nodedata_time_3)

# **Parallel Execution**
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Avoids memory issues on Linux/macOS

    # Dynamically determine number of workers
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in GB
    num_workers = max(1, psutil.cpu_count(logical=False) - 4)  # Leave some free cores
    num_workers = min(num_workers, max(1, int(available_memory / 2)))  # Ensure enough RAM per worker

    print(f"Using {num_workers} CPU cores.")

    start = time.time()

    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        #pool.map(run_simulation, range(1, 2 + 1), chunksize=4)  # For debugging the code with small set
        pool.map(run_simulation, range(1, column_length + 1), chunksize=4)  # Batch tasks

    # Check execution time
    sec = time.time() - start
    times = str(datetime.timedelta(seconds=sec)).split(".")[0]
    print(f"{times} sec")
    print("Completed tasks.")