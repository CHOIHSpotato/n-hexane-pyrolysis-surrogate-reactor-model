import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import cantera as ct

###############################################################################
# Two models are used in this code now
####### MODEL1: LLNL, MODEL2: NUIG ###########

# Base detailed kinetic model for surrogate model training 
# If you want to analyze other models, change the name of pkl, pth, npz file for the models
################################################################################

# Running device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(False)

##################################################################################################
# 0) Calculate initial reactant concentrations in ideal reactor with Cantera software
# Cantera is only used for getting the species molar mass and thermophysical data
# Any hydrocarbon pyrolysis/combustion detailed mechanisms gives identical data
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
    file_template example: "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/LLNL_Eoff_{}.txt"
    file_template example: "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_off/LLNL_Eoff_{}.txt"
    file_template example: "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_on/LLNL_Eon_{}.txt"
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
# 6) Draw prediction of two models at three different conditions in one plot
################################################################################

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_sol_3_two_models(
    i_exp_list,
    solutions_MODEL1, references_MODEL1,
    solutions_MODEL2, references_MODEL2,
    L_ini_list, u0_ini_list,
    Tlist_MODEL1, Plist_MODEL1, Tlist_MODEL2, Plist_MODEL2,
    tsteps_MODEL1, tsteps_MODEL2,
    species_name=["H2","CH4","C2H4","C2H6","C3H6","C4H8-1","NC6H14"],
    # ref_sparsity: setting how many points show up on the plot for ensuring visibility
    ref_sparsity=80,
    cond_colors = ["red", "blue", "green"]
):
    """
    i_exp_list: [i_exp1, i_exp2, i_exp3] (same index)
    solutions_MODEL1, references_MODEL1: (length=3) shape=(ns, Ntime) Tensor list(LLNL)
    solutions_MODEL2, references_MODEL2: (length=3) (NUIG)
    Tlist_MODEL1, Plist_MODEL1, Tlist_MODEL2, Plist_MODEL2, tsteps_MODEL1, tsteps_MODEL2: all (n_exp, nt)
    """

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    # - LLNL: line style='solid', marker='o'
    # - NUIG: line style='dashdot', marker='s'
    linestyle_MODEL1 = 'solid'
    linestyle_MODEL2 = 'dashed'
    marker_MODEL1 = '^'
    marker_MODEL2 = 's'

    # cond_colors[0..2] = Condition 1..3
    info_lines = []

    # saved (T, P, L, u0) information pop on 8th axis
    cond_info_texts = []

    for idx, i_exp in enumerate(i_exp_list):
        c = cond_colors[idx]

        temp_val = Tlist_MODEL1[i_exp, 0].item()  
        pres_val = Plist_MODEL1[i_exp, 0].item() / 1e3  

        # Real L, u0
        length_iexp = L_ini_list[i_exp].item()
        u0_iexp = u0_ini_list[i_exp].item()

        # time axis
        time_cal_MODEL1 = tsteps_MODEL1[i_exp].cpu().numpy()
        time_cal_MODEL2 = tsteps_MODEL2[i_exp].cpu().numpy()

        y_pred_MODEL1 = solutions_MODEL1[idx].cpu().numpy()
        y_ref_MODEL1  = references_MODEL1[idx].cpu().numpy()
        y_pred_MODEL2 = solutions_MODEL2[idx].cpu().numpy()
        y_ref_MODEL2  = references_MODEL2[idx].cpu().numpy()

        # At info_lines, Cond i: T, P, L, u0
        cond_info_texts.append(
            f"Cond {idx+1}: T={temp_val:.2f} K, P={pres_val:.2f} kPa\nL={length_iexp:.2f} m, u0={u0_iexp:.2f} m/s"
        )

        for i_sp in range(len(species_name)):
            ax = axes[i_sp]

            # === MODEL1 ===
            ax.plot(time_cal_MODEL1, y_pred_MODEL1[i_sp],
                    color=c, linestyle=linestyle_MODEL1, linewidth=3)
            # scatter ref
            ax.scatter(time_cal_MODEL1[::ref_sparsity], y_ref_MODEL1[i_sp, ::ref_sparsity],
                       color=c, marker=marker_MODEL1, facecolors='none', s=140)

            # === MODEL2 ===
            ax.plot(time_cal_MODEL2, y_pred_MODEL2[i_sp],
                    color=c, linestyle=linestyle_MODEL2, linewidth=3)
            # scatter ref
            ax.scatter(time_cal_MODEL2[::ref_sparsity], y_ref_MODEL2[i_sp, ::ref_sparsity],
                       color=c, marker=marker_MODEL2, facecolors='none', s=140)

            ax.set_title(species_name[i_sp], fontsize=22)
            ax.set_xlabel("Time [s]", fontsize=21)
            ax.set_ylabel("Concentration [mol/m$^3$]", fontsize=18)
            ax.tick_params(axis='x', direction='in', length=4, width=1.2, top=True, colors='black')
            ax.tick_params(axis='y', direction='in', length=4, width=1.2, right=True, colors='black')
            ax.tick_params(axis='both', labelsize=16, colors='black')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.2)

    # --- Subplot[7]: Legend ---
    axes[7].axis('off')

    # 1) Condition color handles (3)
    cond1_line = mlines.Line2D([], [], color=cond_colors[0], linewidth=3, label='Cond1')
    cond2_line = mlines.Line2D([], [], color=cond_colors[1], linewidth=3, label='Cond2')
    cond3_line = mlines.Line2D([], [], color=cond_colors[2], linewidth=3, label='Cond3')

    # 2) Model style handles
    MODEL1_style = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                               markersize=20, markerfacecolor='none', markeredgecolor='black', label='LLNL')
    MODEL2_style = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                                  markersize=20, markerfacecolor='none', markeredgecolor='black', label='NUIG')

    # cond legend + model style legend
    legend_handles = [cond1_line, cond2_line, cond3_line, MODEL1_style, MODEL2_style]

    axes[7].legend(handles=legend_handles, loc='center', fontsize=26)

    # --- Subplot[8]: Cond info (T,P,L,u0) ---
    axes[8].axis('off')
    info_text = "\n\n".join(cond_info_texts)
    axes[8].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=22)

    fig.tight_layout()
    return fig

def test_3_conditions_two_models(
    trainer_MODEL1, w_in_MODEL1, w_b_MODEL1, w_out_MODEL1,
    trainer_MODEL2, w_in_MODEL2, w_b_MODEL2, w_out_MODEL2,
    test_idx,
    L_ini_list,
    u0_ini_list
):
    # 1) Choose three condition index (low temp., middel temp., and high temp.)
    sorted_test_idx = sorted(test_idx, key=lambda i_exp: trainer_MODEL1.Tlist[i_exp, 0].item())
    n_test = len(sorted_test_idx)
    i_exp1 = sorted_test_idx[n_test // 4]
    i_exp2 = sorted_test_idx[n_test // 2]
    i_exp3 = sorted_test_idx[-2]
    i_exp_list = [i_exp1, i_exp2, i_exp3]

    solutions_MODEL1 = []
    references_MODEL1 = []
    for i_exp in i_exp_list:
        sol_pred = trainer_MODEL1.predict_n_ode(w_in_MODEL1, w_b_MODEL1, w_out_MODEL1, i_exp)
        sol_ref  = trainer_MODEL1.ylabel[i_exp, :, :]
        solutions_MODEL1.append(sol_pred)
        references_MODEL1.append(sol_ref)

    solutions_MODEL2 = []
    references_MODEL2 = []
    for i_exp in i_exp_list:
        sol_pred = trainer_MODEL2.predict_n_ode(w_in_MODEL2, w_b_MODEL2, w_out_MODEL2, i_exp)
        sol_ref  = trainer_MODEL2.ylabel[i_exp, :, :]
        solutions_MODEL2.append(sol_pred)
        references_MODEL2.append(sol_ref)

    # 4) Make the plot
    fig = plot_sol_3_two_models(
        i_exp_list,
        solutions_MODEL1, references_MODEL1,
        solutions_MODEL2, references_MODEL2,
        # L & u0
        L_ini_list, 
        u0_ini_list,
        trainer_MODEL1.Tlist, trainer_MODEL1.Plist,
        trainer_MODEL2.Tlist, trainer_MODEL2.Plist,
        trainer_MODEL1.tsteps, trainer_MODEL2.tsteps
    )
    fig.savefig("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_FIGURE/LLNL_NUIG_off_comparison_revised.png")
    plt.close(fig)

################################################################################
# 7) Main function
################################################################################

def main():
    # -----------------
    # Set constants
    # -----------------
    n_exp = 810 # The number of cases (4D_valid)
    ns = 9 # The number of species
    i_obs = torch.arange(0, ns - 2, device=device)  # Except the resovoir species, idx (0..6)

    # ====== 7.1) MODEL1 data & time prediction model & CRNN parameter ======
    # OPTION A. Select it when you use VALIDATION_DATASET_CONTAINER
    # OPTION A is stand for validating constructed surrogate model
    # 1) Original simulation data (from cantera software) ('Eoff' file)
    file_template_MODEL1 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/LLNL_Eoff_{}.txt"
    tsteps_MODEL1, ylabel_MODEL1, Tlist_MODEL1, Plist_MODEL1, spec_conc_0_list_MODEL1, yscale_MODEL1 = load_dataset(file_template_MODEL1, n_exp, ns, device)

    # 2) reactor geometry (reactor length/inflow rate) 
    loaded_geometry = np.loadtxt("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/reactor_geometry.txt")
    tensor_geometry = torch.tensor(loaded_geometry, dtype=torch.float32, device=device)
    T_ini_list_MODEL1 = Tlist_MODEL1[:,0]
    P_ini_list_MODEL1 = Plist_MODEL1[:,0]
    L_ini_list_MODEL1 = tensor_geometry[:,0]
    u0_ini_list_MODEL1= tensor_geometry[:,1]

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
    file_npz_MODEL1 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_LLNL_Eoff.npz"
    w_in_MODEL1, w_b_MODEL1, w_out_MODEL1 = load_npz_parameters(file_npz_MODEL1)

    # 6) MODEL1 Trainer
    i_obs_MODEL1 = torch.arange(0, ns-2, device=device)
    crnn_MODEL1 = CRNN(tsteps_MODEL1)
    trainer_MODEL1 = Trainer(tsteps_MODEL1, crnn_MODEL1, ylabel_MODEL1, yscale_MODEL1, spec_conc_0_list_MODEL1, Tlist_MODEL1, Plist_MODEL1, i_obs_MODEL1)

    # ====== 7.2) MODEL2 data & time prediction model & CRNN parameter ======
    # OPTION A. Select it when you use VALIDATION_DATASET_CONTAINER
    # OPTION A is stand for validating constructed surrogate model
    # 1) Original simulation data (from cantera software) ('Eoff' file))
    file_template_MODEL2 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/NUIG_Eoff_{}.txt"
    tsteps_MODEL2, ylabel_MODEL2, Tlist_MODEL2, Plist_MODEL2, spec_conc_0_list_MODEL2, yscale_MODEL2 = load_dataset(file_template_MODEL2, n_exp, ns, device)

    # 2) reactor geometry (reactor length/inflow rate) (example: "D:/CRNN/4D_valid/reactor_geometry.txt")
    T_ini_list_MODEL2 = Tlist_MODEL2[:,0]
    P_ini_list_MODEL2 = Plist_MODEL2[:,0]
    L_ini_list_MODEL2 = tensor_geometry[:,0]
    u0_ini_list_MODEL2= tensor_geometry[:,1]
    spec_conc_0_list_MODEL2 = calculate_spec_conc_0_list(
    T_ini_list=T_ini_list_MODEL2.cpu().numpy(),
    P_ini_list=P_ini_list_MODEL2.cpu().numpy(),
    n_exp=n_exp,
    ns=ns,
    device=device,
    steam_dilution_ratio=steam_dilution_ratio,
    R_J=R_J)
    
    # 3) Time prediction MLP load (MODEL2)
    #    min/max from 'pkl' fike, model parameters from 'pth' file
    load_dir_time_MODEL2 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/TIME_PRED_MODEL_PARAMETER_CONTAINER"
    with open(os.path.join(load_dir_time_MODEL2, "min_max_values_mlp_NUIG_4D_time_off.pkl"), 'rb') as f2:
        output_scale_time_MODEL2 = pickle.load(f2)
    min_val_time_MODEL2 = output_scale_time_MODEL2['min']
    max_val_time_MODEL2 = output_scale_time_MODEL2['max']

    input_scale_time_MODEL2 = np.array([[870, 1.0e5, 0.5, 2.5],
                                   [1150, 3.0e5, 1.0, 5.0]]) 

    model_time_MODEL2 = MultiLayerPerceptron_time(input_node_time, neurons, output_node_time).to(device)
    model_time_MODEL2.load_state_dict(torch.load(
        os.path.join(load_dir_time_MODEL2, "mlp_weights_NUIG_4D_time_off.pth"),
        map_location=device
    ))
    model_time_MODEL2.eval()

    # 4) MODEL2 tsteps regularization
    input_params_time_MODEL2 = torch.zeros((n_exp, 4), dtype=torch.float32, device=device)
    input_params_time_MODEL2[:,0] = (T_ini_list_MODEL2 - input_scale_time_MODEL2[0,0])/(input_scale_time_MODEL2[1,0] - input_scale_time_MODEL2[0,0])
    input_params_time_MODEL2[:,1] = (P_ini_list_MODEL2 - input_scale_time_MODEL2[0,1])/(input_scale_time_MODEL2[1,1] - input_scale_time_MODEL2[0,1])
    input_params_time_MODEL2[:,2] = (L_ini_list_MODEL2  - input_scale_time_MODEL2[0,2])/(input_scale_time_MODEL2[1,2] - input_scale_time_MODEL2[0,2])
    input_params_time_MODEL2[:,3] = (u0_ini_list_MODEL2 - input_scale_time_MODEL2[0,3])/(input_scale_time_MODEL2[1,3] - input_scale_time_MODEL2[0,3])

    with torch.no_grad():
        pred_time_profile_scaled_MODEL2 = model_time_MODEL2(input_params_time_MODEL2)
    pred_time_profile_MODEL2 = pred_time_profile_scaled_MODEL2 * (max_val_time_MODEL2 - min_val_time_MODEL2) + min_val_time_MODEL2
    time_ini_list_MODEL2 = tsteps_MODEL2[:,0].unsqueeze(1)
    tsteps_calculated_MODEL2 = torch.cat((time_ini_list_MODEL2, pred_time_profile_MODEL2), dim=1)

    # Enforce increasing
    diffs_MODEL2 = tsteps_calculated_MODEL2[:,1:] - tsteps_calculated_MODEL2[:,:-1]
    is_strict_MODEL2 = (diffs_MODEL2>0).all(dim=1)
    bad_rows_MODEL2 = (~is_strict_MODEL2).nonzero(as_tuple=True)[0]
    for i in bad_rows_MODEL2:
        tsteps_calculated_MODEL2[i] = enforce_strict(tsteps_calculated_MODEL2[i])

    # Replace original tsteps
    tsteps_MODEL2 = tsteps_calculated_MODEL2

    # 5) MODEL2 CRNN parameters
    file_npz_MODEL2 = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_NUIG_Eoff.npz"
    w_in_MODEL2, w_b_MODEL2, w_out_MODEL2 = load_npz_parameters(file_npz_MODEL2)

    # 6) NUIG Trainer
    i_obs_MODEL2 = torch.arange(0, ns-2, device=device)
    crnn_MODEL2 = CRNN(tsteps_MODEL2)
    trainer_MODEL2 = Trainer(tsteps_MODEL2, crnn_MODEL2, ylabel_MODEL2, yscale_MODEL2, spec_conc_0_list_MODEL2, Tlist_MODEL2, Plist_MODEL2, i_obs_MODEL2)

    # ====== 7.3) Train/Valid/Test Split & 3conditions ======
    all_idx = np.arange(n_exp)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    # Plot for comparing three conditions
    test_3_conditions_two_models(
        trainer_MODEL1, w_in_MODEL1, w_b_MODEL1, w_out_MODEL1,
        trainer_MODEL2, w_in_MODEL2, w_b_MODEL2, w_out_MODEL2,
        test_idx,
        L_ini_list_MODEL1,
        u0_ini_list_MODEL1 
    )

    print("Done! 3-condition comparison plot saved.")

if __name__ == "__main__":
    main()
