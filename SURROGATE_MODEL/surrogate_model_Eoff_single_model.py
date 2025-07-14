import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import cantera as ct
import time

###############################################################################
# Two models are used in this code now
####### MODEL1: LLNL ###########

# Base detailed kinetic model for surrogate model training 
# If you want to analyze other models, change the name of pkl, pth, npz file for the models
################################################################################

# Running device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(False)

##################################################################################################
# 0) Calculate initial reactant concentrations in ideal reactor with Cantera software
# Cantera is only used for getting the species molar mass and thermophysical data
# Any hydrocarbon pyrolysis/combustion detailed mechanisms gives the identical data
# Defined mechanisms in this section does not mean this surrogate model following that mechanisms
##################################################################################################

# Cantera loading
reaction_mechanism = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/DETAILED_KINETIC_MODEL/LLNL.yaml'
gas_initial = ct.Solution(reaction_mechanism)
idx_NC6H14 = gas_initial.species_index("NC6H14")
idx_H2O = gas_initial.species_index("H2O")
mw_NC6H14 = gas_initial.molecular_weights[idx_NC6H14]  # [g/mol]
mw_H2O = gas_initial.molecular_weights[idx_H2O]
steam_dilution_ratio = 0.7 # [kg of NC6H14 / kg of H2O]
R_J = 8.314462618 # [J/molK]

print(f"Molecular weight of NC6H14: {mw_NC6H14}")
print(f"Molecular weight of H2O: {mw_H2O}")

def calculate_spec_conc_0_list(
    T_ini_list, P_ini_list, n_exp, ns, device, steam_dilution_ratio, R_J):
    def ini_reactant_concentration(T_ini, P_ini):
        return (P_ini / (R_J * T_ini)) * (1 / (steam_dilution_ratio * (mw_NC6H14 / mw_H2O) + 1))
    spec_conc_0_list_calculated = torch.zeros((n_exp, ns), dtype=torch.float32, device=device)
    for i in range(n_exp):
        T_ini = T_ini_list[i]
        P_ini = P_ini_list[i]
        NC6H14_ini_cons = ini_reactant_concentration(T_ini, P_ini)
        spec_conc_0_list_calculated[i, (ns - 3)] = NC6H14_ini_cons
    return spec_conc_0_list_calculated

################################################################################
# 1) Class for shared utilities and loading datasets
################################################################################

def clamp(tensor, min_val, max_val):
    return torch.clamp(tensor, min_val, max_val)

class RawDataDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, ns, device):
        self.file_paths = file_paths
        self.ns = ns
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        rawdata = np.loadtxt(filepath).T
        rawdata = torch.tensor(rawdata, dtype=torch.float32, device=self.device)
        
        # rawdata: [0]=time, [1]=temperature, [2]=pressure, [3:3+ns]=species concentrations
        ylabel = rawdata[3:3 + self.ns, :] * 1.0e3  # [kmol/m3] -> [mol/m3]
        Tlist = rawdata[1, :]
        Plist = rawdata[2, :]
        spec_conc_0 = ylabel[:, 0]
        tsteps = rawdata[0, :]
        return Tlist, Plist, ylabel, spec_conc_0, tsteps

def load_dataset(file_template, n_exp, ns, device):
    """
    file_template example: "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_off/LLNL_Eoff_{}.txt"
    """
    file_paths = [file_template.format(i+1) for i in range(n_exp)]
    dataset = RawDataDataset(file_paths, ns, device)

    tsteps = torch.stack([dataset[i][4] for i in range(n_exp)], dim=0)
    ylabel = torch.stack([dataset[i][2] for i in range(n_exp)], dim=0)
    Tlist  = torch.stack([dataset[i][0] for i in range(n_exp)], dim=0)
    Plist  = torch.stack([dataset[i][1] for i in range(n_exp)], dim=0)
    spec_conc_0_list= ylabel[:, :, 0]
    yscale = clamp(ylabel.amax(dim=2) - ylabel.amin(dim=2), 1e-6, torch.inf)

    return tsteps, ylabel, Tlist, Plist, spec_conc_0_list, yscale

################################################################################
# 2) Interpolation function of adaptive time silver & CRNN definition
################################################################################

def linear_interpolation(tsteps, values):
    def interpolate(t):
        indices = torch.searchsorted(tsteps, t, right=True).clamp(1, len(tsteps) - 1)
        x0 = tsteps[indices - 1]
        x1 = tsteps[indices]
        y0 = values[indices - 1]
        y1 = values[indices]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (t - x0)
    return interpolate

class CRNN(nn.Module):
    def __init__(self, tsteps):
        super(CRNN, self).__init__()
        self.tsteps = tsteps

    def wrapper(self, itpT, itpP, w_in, w_b, w_out, yscale,
                lb=torch.tensor(1.0e-6, device=device),
                ub=torch.tensor(6.0e1, device=device),
                intermediate_min=torch.tensor(-30.0, device=device),
                intermediate_max=torch.tensor(30.0, device=device),
                R_kcal=torch.tensor(1.9872036e-3, dtype=torch.float32, device=device),
                ll_du=torch.tensor(-1e5, device=device),
                ul_du=torch.tensor(1e5, device=device)
    ):
        class CRNNFunc(nn.Module):
            def __init__(self):
                super(CRNNFunc, self).__init__()

            def forward(self, t, u):
                T = itpT(t)
                P = itpP(t)
                Y = torch.clamp(u, lb, ub)
                logX = torch.log(Y)

                w_v = torch.cat([
                    logX,
                    torch.tensor([-1 / (R_kcal * T), torch.log(T)], device=device)
                ])

                w_in_x = torch.matmul(w_in.T, w_v)
                intermediate = w_in_x + w_b
                intermediate = torch.clamp(intermediate, min=intermediate_min, max=intermediate_max)

                # du
                du = torch.matmul(w_out, torch.exp(intermediate))
                du_clamped = torch.clamp(du, ll_du, ul_du)
                return du_clamped

        return CRNNFunc()

###########################################################################################
# 3) Class for solving ode came from CRNNFunc() and calculating the loss(mean square error)
###########################################################################################

class Trainer:
    def __init__(self, tsteps, crnn, ylabel, yscale, spec_conc_0_list, Tlist, Plist, i_obs,
                 lb=1e-6, ub=6e1):
        self.tsteps = tsteps
        self.crnn = crnn
        self.ylabel = ylabel
        self.yscale = yscale
        self.spec_conc_0_list = spec_conc_0_list
        self.Tlist = Tlist
        self.Plist = Plist
        self.i_obs = i_obs
        self.lb = lb
        self.ub = ub

    def predict_n_ode(self, w_in, w_b, w_out, i_exp):
        u0 = self.spec_conc_0_list[i_exp]
        Tlist_exp = self.Tlist[i_exp, :]
        Plist_exp = self.Plist[i_exp, :]
        timelist_exp = self.tsteps[i_exp, :]

        itpT = linear_interpolation(timelist_exp, Tlist_exp)
        itpP = linear_interpolation(timelist_exp, Plist_exp)

        crnn_func = self.crnn.wrapper(itpT, itpP, w_in, w_b, w_out, self.yscale)
        sol = odeint(crnn_func, u0, timelist_exp, method='dopri5', atol=1e-6, rtol=1e-6)
        return torch.clamp(sol.T, self.lb, self.ub)

################################################################################
# 4) Time MLP -> predict tsteps from 4-dimensional variables
################################################################################

class MultiLayerPerceptron_time(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron_time, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        out = self.fc4(x)
        return out

def enforce_strict(row, eps=1e-5):
    """
    Force tsteps become strictly increasing
    """
    for i in range(1, row.size(0)):
        if row[i] <= row[i-1]:
            row[i] = row[i-1] + eps
    return row

################################################################################
# 5) Load the parameters for CRNN
################################################################################

def load_npz_parameters(file_path):
    data = np.load(file_path, allow_pickle=True)
    parameters = data['parameters']
    final_param = parameters[-1]
    w_in  = torch.tensor(final_param['w_in'],  dtype=torch.float32, device=device)
    w_b   = torch.tensor(final_param['w_b'],   dtype=torch.float32, device=device)
    w_out = torch.tensor(final_param['w_out'], dtype=torch.float32, device=device)
    return w_in, w_b, w_out

################################################################################
# 6) Main function
################################################################################

def main():
    # -----------------
    # Set constants
    # -----------------
    start_time = time.time()

    geometry_df = pd.read_csv("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/sampling_case_4D.csv", header=None)
    n_exp = geometry_df.shape[0] # The number of cases
    print("the number of cases")
    print(n_exp)
    ns = 9 # The number of species
    i_obs = torch.arange(0, ns - 2, device=device)  # Except the resovoir species, idx (0..6)

    # ====== 7.1) MODEL1 data & time prediction model & CRNN parameter ======
    # 1) Original simulation data (from cantera software) ('Eoff' file)
    file_template_MODEL1 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_off/LLNL_Eoff_{}.txt"
    tsteps_MODEL1, ylabel_MODEL1, Tlist_MODEL1, Plist_MODEL1, spec_conc_0_list_MODEL1, yscale_MODEL1 = load_dataset(file_template_MODEL1, n_exp, ns, device)

    # OPTION B. Select it when you use INDIPENDENT_DATASET_CONTAINER
    # In OPTION B, make your own initial conditions list csv file where you want to get species concentration time series data
    # csv file must be in same shape of 'sampling_case_4D.csv' file
    # You can choose 4-dimensional conditions which located in a range of [870 K - 1150 K], [100 kPa - 300 kPa], [0.5 m - 1.0 m], and [2.5 m/s - 5.0 m/s]

    T_ini_list_MODEL1 = torch.tensor(geometry_df.iloc[:, 0].values, dtype=torch.float32, device=device)
    P_ini_list_MODEL1 = torch.tensor(geometry_df.iloc[:, 1].values*1.0e+5, dtype=torch.float32, device=device)
    L_ini_list_MODEL1 = torch.tensor(geometry_df.iloc[:, 2].values, dtype=torch.float32, device=device)
    u0_ini_list_MODEL1 = torch.tensor(geometry_df.iloc[:, 3].values, dtype=torch.float32, device=device)

    # Overlap initial species concentations with calculated one from initial temperature, pressure, reactant composition
    spec_conc_0_list_MODEL1 = calculate_spec_conc_0_list(
    T_ini_list=T_ini_list_MODEL1.cpu().numpy(),
    P_ini_list=P_ini_list_MODEL1.cpu().numpy(),
    n_exp=n_exp,
    ns=ns,
    device=device,
    steam_dilution_ratio=steam_dilution_ratio,
    R_J=R_J)

    # 3) Time prediction MLP load (MODEL1)
    #    min/max from 'pkl' fike, model parameters from 'pth' file
    load_dir_time_MODEL1 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/TIME_PRED_MODEL_PARAMETER_CONTAINER"
    with open(os.path.join(load_dir_time_MODEL1, "min_max_values_mlp_LLNL_4D_time_off.pkl"), 'rb') as f:
        output_scale_time_MODEL1 = pickle.load(f)
    min_val_time_MODEL1 = output_scale_time_MODEL1['min']
    max_val_time_MODEL1 = output_scale_time_MODEL1['max']

    input_scale_time_MODEL1 = np.array([[870, 1.0e5, 0.5, 2.5],
                                   [1150, 3.0e5, 1.0, 5.0]])
    # Structure of time prediction model neural network
    input_node_time = 4
    output_node_time = 800  # ntotal-1 (In this case, 801-1 = 800)
    neurons = 512
    model_time_MODEL1 = MultiLayerPerceptron_time(input_node_time, neurons, output_node_time).to(device)
    model_time_MODEL1.load_state_dict(torch.load(
        os.path.join(load_dir_time_MODEL1, "mlp_weights_LLNL_4D_time_off.pth"),
        map_location=device
    ))
    model_time_MODEL1.eval()

    # 4) MODEL1 tsteps regularization
    input_params_time_MODEL1 = torch.zeros((n_exp, 4), dtype=torch.float32, device=device)
    input_params_time_MODEL1[:,0] = (T_ini_list_MODEL1 - input_scale_time_MODEL1[0,0])/(input_scale_time_MODEL1[1,0] - input_scale_time_MODEL1[0,0])
    input_params_time_MODEL1[:,1] = (P_ini_list_MODEL1 - input_scale_time_MODEL1[0,1])/(input_scale_time_MODEL1[1,1] - input_scale_time_MODEL1[0,1])
    input_params_time_MODEL1[:,2] = (L_ini_list_MODEL1  - input_scale_time_MODEL1[0,2])/(input_scale_time_MODEL1[1,2] - input_scale_time_MODEL1[0,2])
    input_params_time_MODEL1[:,3] = (u0_ini_list_MODEL1 - input_scale_time_MODEL1[0,3])/(input_scale_time_MODEL1[1,3] - input_scale_time_MODEL1[0,3])

    with torch.no_grad():
        pred_time_profile_scaled_MODEL1 = model_time_MODEL1(input_params_time_MODEL1)
    # (n_exp, 800)
    pred_time_profile_MODEL1 = pred_time_profile_scaled_MODEL1 * (max_val_time_MODEL1 - min_val_time_MODEL1) + min_val_time_MODEL1
    # tsteps: concat [first timestep, predictions after the first step]
    time_ini_list_MODEL1 = tsteps_MODEL1[:,0].unsqueeze(1)
    tsteps_calculated_MODEL1 = torch.cat((time_ini_list_MODEL1, pred_time_profile_MODEL1), dim=1)

    # Enforce increasing
    diffs_MODEL1 = tsteps_calculated_MODEL1[:,1:] - tsteps_calculated_MODEL1[:,:-1]
    is_strict_MODEL1 = (diffs_MODEL1>0).all(dim=1)
    bad_rows_MODEL1 = (~is_strict_MODEL1).nonzero(as_tuple=True)[0]
    for i in bad_rows_MODEL1:
        tsteps_calculated_MODEL1[i] = enforce_strict(tsteps_calculated_MODEL1[i])

    # Replace original tsteps
    tsteps_MODEL1 = tsteps_calculated_MODEL1

    # 5) MODEL1 CRNN parameters load
    file_npz_MODEL1 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_LLNL_Eoff_wide_v2.npz"
    w_in_MODEL1, w_b_MODEL1, w_out_MODEL1 = load_npz_parameters(file_npz_MODEL1)

    # 6) MODEL1 Trainer
    i_obs_MODEL1 = torch.arange(0, ns-2, device=device)
    crnn_MODEL1 = CRNN(tsteps_MODEL1)
    trainer_MODEL1 = Trainer(tsteps_MODEL1, crnn_MODEL1, ylabel_MODEL1, yscale_MODEL1, spec_conc_0_list_MODEL1, Tlist_MODEL1, Plist_MODEL1, i_obs_MODEL1)

    all_idx = np.arange(n_exp)

    # -----------------------------------------------------------------------------------------------------
    # 7.3) MODEL1: Save prediction results on the all test or independent(for validating the model) dataset 
    # -----------------------------------------------------------------------------------------------------
    save_dir = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_FIGURE/LLNL_Eoff"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start prediction for {len(all_idx)} test cases ...")

    for idx, i_exp in enumerate(all_idx, 1):
        T_ini = Tlist_MODEL1[i_exp, 0].item()
        P_ini = Plist_MODEL1[i_exp, 0].item()
        L_ini = L_ini_list_MODEL1[i_exp].item()
        u0_ini = u0_ini_list_MODEL1[i_exp].item()

        pred_species = trainer_MODEL1.predict_n_ode(w_in_MODEL1, w_b_MODEL1, w_out_MODEL1, i_exp)[i_obs_MODEL1, :]
        pred_species[:-1,0] = 0.0

        time_array = tsteps_MODEL1[i_exp, :]

        T_profile = Tlist_MODEL1[i_exp, :]
        P_profile = Plist_MODEL1[i_exp, :]
        L_profile = torch.full_like(time_array, L_ini)
        u0_profile = torch.full_like(time_array, u0_ini)

        time_np = time_array.cpu().numpy()
        T_np = T_profile.cpu().numpy()
        P_np = P_profile.cpu().numpy()
        L_np = L_profile.cpu().numpy()
        u0_np = u0_profile.cpu().numpy()
        species_np = pred_species.cpu().numpy()  # shape: (n_species, n_time)

        output_array = np.vstack([
            time_np,
            T_np,
            P_np,
            L_np,
            u0_np,
            *species_np
        ]).T  # shape: (N_time, 5 + n_species)

        #save_path = os.path.join(save_dir, f"pred_LLNLoff_{idx}.txt")
        #np.savetxt(save_path, output_array, fmt="%.6e")
        #print(f"[{idx}/{len(all_idx)}] Saved: {save_path}")

    print("All MODEL1 predictions saved as .txt files!")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.3f} seconds")

    # -----------------------------------------------------------------------------------------------------
    # 7.5) Analyze an accuracy of the model with RMSE, Relative error, NRMSE, and FCD
    # -----------------------------------------------------------------------------------------------------
    result_rows = []

    print("\nCalculating detailed RMSE & Relative Error for each case ...")

    epsilon_rel = 1.0e-5

    for idx, i_exp in enumerate(all_idx, 1):
        T_ini = Tlist_MODEL1[i_exp, 0].item()
        P_ini = Plist_MODEL1[i_exp, 0].item()
        L_ini = L_ini_list_MODEL1[i_exp].item()
        u0_ini = u0_ini_list_MODEL1[i_exp].item()

        # Predicted value
        Y_pred = trainer_MODEL1.predict_n_ode(w_in_MODEL1, w_b_MODEL1, w_out_MODEL1, i_exp)[i_obs_MODEL1, :]
        Y_pred[:-1,0] = 0.0

        # True or refernce value
        Y_true = trainer_MODEL1.ylabel[i_exp, i_obs_MODEL1, :]

        Y_pred_np = Y_pred.cpu().numpy()
        Y_true_np = Y_true.cpu().numpy()

        ns_used = Y_pred_np.shape[0]  # the number of chemical species
        n_time  = Y_pred_np.shape[1]  # the number of time steps

        species_name=["H2","CH4","C2H4","C2H6","C3H6","C4H8-1","NC6H14"]

        for sp_idx in range(ns_used):
            true = Y_true_np[sp_idx, :]
            pred = Y_pred_np[sp_idx, :]

            # Exclude initial molar concentration (which are zero excepting n-hexane)
            true = true[1:]
            pred = pred[1:]

            # Final
            true_final = true[-1]
            pred_final = pred[-1]

            # RMSE (Final)
            rmse_final = np.sqrt((pred_final - true_final)**2)

            # NRMSE (Final)
            nrmse_final = rmse_final / (np.max(true) - np.min(true) + epsilon_rel)

            # Relative Error (%) (Final)
            rel_final = np.abs(pred_final - true_final) / (true_final + epsilon_rel) * 100

            # RMSE (residence time average)
            rmse_time = np.sqrt(np.mean((pred - true)**2))

            # NRMSE (residence time average)
            nrmse_time = rmse_time / (np.max(true) - np.min(true) + epsilon_rel)

            # Relative Error (%) (residence time average)
            rel_time = np.mean(np.abs(pred - true) / (true + epsilon_rel)) * 100

            # Gaussian approximation: mean & std
            mu_true = np.mean(true)
            mu_pred = np.mean(pred)
            sigma_true = np.std(true)
            sigma_pred = np.std(pred)

            # Frechet Distance
            fcd = np.sqrt( (mu_true - mu_pred)**2 + (sigma_true - sigma_pred)**2)

            # Max Norm
            max_diff = np.max(np.abs(pred - true))
            max_norm = max_diff / (np.max(np.abs(true)) + epsilon_rel)

            result_rows.append([
                idx,  # case number
                species_name[sp_idx],  # species index (or species name list)
                T_ini, P_ini, L_ini, u0_ini,  # initial conditions
                rmse_final, nrmse_final, rel_final, 
                rmse_time, nrmse_time, rel_time,
                fcd, max_norm
            ])

        print(f"[{idx}/{len(all_idx)}] done")

    # Convert it to DataFrame
    result_df = pd.DataFrame(
        result_rows,
        columns=[
            "Case_ID", "Species_ID",
            "T_ini [K]", "P_ini [Pa]", "L_ini [m]", "u0_ini [m/s]",
            "RMSE_final", "NRMSE_final", "RelError_final(%)", 
            "RMSE_time_avg", "NRMSE_time_avg", "RelError_time_avg(%)", 
            "FCD", "Max_Norm"
        ]
    )

    # Save path
    csv_save_path = os.path.join(save_dir, "final_species_rmse_relerror_widecond_0417.csv")
    result_df.to_csv(csv_save_path, index=False, float_format="%.6e")
    print(f"\n All species-wise RMSE & Relative Error saved: {csv_save_path}")

if __name__ == "__main__":
    main()
