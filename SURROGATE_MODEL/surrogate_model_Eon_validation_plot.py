import os
import numpy as np
import cantera as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter


###############################################################################
# Two models are used in this code now
####### MODEL1: JetSurf, MODEL2: LLNL ###########

# Base detailed kinetic model for surrogate model training 
# If you want to analyze other models, change the name of pkl, pth, npz file for the models
################################################################################

# Running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
mw_NC6H14 = gas_initial.molecular_weights[idx_NC6H14]
mw_H2O = gas_initial.molecular_weights[idx_H2O]

print(f"MW NC6H14: {mw_NC6H14:.4f} g/mol, MW H2O: {mw_H2O:.4f} g/mol")

def build_spec_conc_0_list(
    T_ini_list, P_ini_list, n_exp, ns, device, steam_dilution_ratio, R_J,
    mw_NC6H14, mw_H2O):
    spec_conc_0_list = torch.zeros((n_exp, ns), dtype=torch.float32, device=device)
    for i in range(n_exp):
        T_ini = T_ini_list[i]
        P_ini = P_ini_list[i]
        NC6H14_conc = (P_ini / (R_J * T_ini)) * (1 / (steam_dilution_ratio * (mw_NC6H14 / mw_H2O) + 1))
        spec_conc_0_list[i, ns - 3] = NC6H14_conc
    return spec_conc_0_list

################################################################################
# 1) Class for shared utilities and loading datasets
################################################################################

n_exp, ns, nr, ntotal = 810, 9, 9, 801
lb = 1e-6
ub = 6e1
intermediate_min = -3.0e+1
intermediate_max = 3.0e+1
R_kcal = 1.9872036e-3
ll_du, ul_du = -1e5, 1e5
steam_dilution_ratio = 0.7 # [kg of NC6H14 / kg of H2O]
R_J = 8.314462618 # [J/molK]

def clamp(tensor, min_val, max_val):
    return torch.clamp(tensor, min_val, max_val)

def enforce_strict(arr, eps=1e-5):
    """Calibrate the time array to be strict increasing"""
    for i in range(1, len(arr)):
        if arr[i] <= arr[i-1]:
            arr[i] = arr[i-1] + eps
    return arr

def linear_interpolation(tsteps, values):
    """Interpolation when tsteps do not fit in Adaptive ODEs solver"""
    def interpolate(t):
        indices = torch.searchsorted(tsteps, t, right=True).clamp(1, len(tsteps) - 1)
        x0 = tsteps[indices - 1]
        x1 = tsteps[indices]
        y0 = values[indices - 1]
        y1 = values[indices]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (t - x0)
    return interpolate

################################################################################
# 2) Two auxillary MLP (Temperature, Time prediction model)
################################################################################

neurons = 512

class MLP_Temp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Temp, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.relu1= nn.ReLU()
        self.fc2 = nn.Linear(neurons, neurons)
        self.relu2= nn.ReLU()
        self.fc3 = nn.Linear(neurons, neurons)
        self.relu3= nn.ReLU()
        self.fc4 = nn.Linear(neurons, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        out = self.fc4(x)
        return out
    
class MLP_Time(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Time, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.relu1= nn.ReLU()
        self.fc2 = nn.Linear(neurons, neurons)
        self.relu2= nn.ReLU()
        self.fc3 = nn.Linear(neurons, neurons)
        self.relu3= nn.ReLU()
        self.fc4 = nn.Linear(neurons, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        out = self.fc4(x)
        return out

################################################################################
# 3) CRNN
################################################################################
class CRNNFunc(nn.Module):
    def __init__(self, t_ar, T_ar, w_in, w_b, w_out):
        super(CRNNFunc, self).__init__()
        self.t_ar = t_ar
        self.T_ar = T_ar
        self.w_in = w_in
        self.w_b = w_b
        self.w_out= w_out

    def forward(self, t, u):
        Tq = linear_interpolation(self.t_ar, self.T_ar)(t)
        Y  = clamp(u, lb, ub)
        logX= torch.log(Y)
        w_v = torch.cat([logX, 
                         torch.tensor([-1/(R_kcal*Tq), torch.log(Tq)], device=device)])
        inter = self.w_in.T @ w_v + self.w_b
        inter = clamp(inter, intermediate_min, intermediate_max)
        du = self.w_out @ torch.exp(inter)
        return clamp(du, ll_du, ul_du)

def crnn_predict(t_ar, T_ar, u0, time_array, w_in, w_b, w_out):
    sol = odeint(CRNNFunc(t_ar, T_ar, w_in, w_b, w_out),
                 u0, time_array, method='dopri5', atol=1e-6, rtol=1e-6)
    return clamp(sol.transpose(0,1), lb, ub)

################################################################################
# 4) Plot two models in one plot
################################################################################
def plot_sol_3_two_models(
    i_exp_list,
    solutions_MODEL1, references_MODEL1, times_MODEL1,
    solutions_MODEL2, references_MODEL2, times_MODEL2,
    cond_info,
    species_name=["H2","CH4","C2H4","C2H6","C3H6","C4H8-1","NC6H14"],
    ref_sparsity=40
):
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    fig, axes= plt.subplots(3,3, figsize=(16,12))
    axes= axes.flatten()

    cond_colors = ["red", "blue", "green"]
    MODEL1_linestyle= 'dashed'
    MODEL1_marker   = 'o'
    MODEL2_linestyle= 'solid'
    MODEL2_marker   = '^'

    info_lines= []

    for idx, i_exp in enumerate(i_exp_list):
        c = cond_colors[idx]

        # MODEL1
        t_MODEL1 = times_MODEL1[idx].cpu().numpy()
        y_MODEL1_pred= solutions_MODEL1[idx].cpu().numpy()
        y_MODEL1_ref = references_MODEL1[idx].cpu().numpy()

        # MODEL2
        t_MODEL2 = times_MODEL2[idx].cpu().numpy()
        y_MODEL2_pred= solutions_MODEL2[idx].cpu().numpy()
        y_MODEL2_ref = references_MODEL2[idx].cpu().numpy()

        info_lines.append(f"Cond {idx+1}: {cond_info[idx]}")

        for i_sp, sp_name in enumerate(species_name):
            ax= axes[i_sp]
            # MODEL1
            ax.plot(t_MODEL1, y_MODEL1_pred[i_sp], color=c, linestyle=MODEL1_linestyle, linewidth=3)
            ax.scatter(t_MODEL1[::ref_sparsity], y_MODEL1_ref[i_sp,::ref_sparsity],
                       color=c, marker=MODEL1_marker, facecolors='none', s=140)
            # MODEL2
            ax.plot(t_MODEL2, y_MODEL2_pred[i_sp], color=c, linestyle=MODEL2_linestyle, linewidth=3)
            ax.scatter(t_MODEL2[::ref_sparsity], y_MODEL2_ref[i_sp,::ref_sparsity],
                       color=c, marker=MODEL2_marker, facecolors='none', s=140)

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

    # legend in subplot(7)
    axes[7].axis('off')
    import matplotlib.lines as mlines
    cond1_line = mlines.Line2D([], [], color=cond_colors[0], linewidth=3, label="Cond1")
    cond2_line = mlines.Line2D([], [], color=cond_colors[1], linewidth=3, label="Cond2")
    cond3_line = mlines.Line2D([], [], color=cond_colors[2], linewidth=3, label="Cond3")

    JetSurf_line  = mlines.Line2D([], [], color='black', linestyle='none', marker='o',
                                  markersize=20, markerfacecolor='none', markeredgecolor='black',label='JetSurF')
    llnl_line  = mlines.Line2D([], [], color='black', linestyle='none', marker='^',
                               markersize=20, markerfacecolor='none', markeredgecolor='black', label='LLNL')

    axes[7].legend(handles=[cond1_line, cond2_line, cond3_line, JetSurf_line, llnl_line],
                   loc='center', fontsize=26)

    # subplot(8): conditions info
    axes[8].axis('off')
    info_text = "\n\n".join(info_lines)
    axes[8].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=22)

    fig.tight_layout()
    fig.savefig("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_FIGURE/JetSurf_LLNL_on_comparison_revised.png")
    plt.close(fig)
    print("Saved JetSurf_LLNL_two_model_3cond_plot.png")

################################################################################
# 5) main()
################################################################################
def main():
    print("=== Start JetSurf & LLNL dual-model script ===")

    # ----------------------------------------------
    # MODEL1 Dataset load
    # ----------------------------------------------
    n_exp=810
    ns=9
    file_paths_MODEL1 = [f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/JetSurf_Eon_{i+1}.txt" for i in range(n_exp)]

    class RawDataDataset_MODEL1:
        def __init__(self, fpaths, ns):
            self.fpaths = fpaths
            self.ns = ns
        def __getitem__(self, idx):
            arr = np.loadtxt(self.fpaths[idx]).T
            arr_t = torch.tensor(arr, dtype=torch.float32, device=device)
            Y = arr_t[3:3+self.ns,:]*1e3
            T = arr_t[1,:]
            P = arr_t[2,:]
            tim= arr_t[0,:]
            return T,P,Y,tim
        def __len__(self):
            return len(self.fpaths)

    ds_MODEL1 = RawDataDataset_MODEL1(file_paths_MODEL1, ns)
    Tlist_MODEL1 = []
    Plist_MODEL1 = []
    ylabel_MODEL1 = []
    tsteps_MODEL1= []
    for i in range(n_exp):
        T_,P_,Y_,tim_ = ds_MODEL1[i]
        Tlist_MODEL1.append(T_)
        Plist_MODEL1.append(P_)
        ylabel_MODEL1.append(Y_)
        tsteps_MODEL1.append(tim_)

    Tlist_MODEL1 = torch.stack(Tlist_MODEL1, dim=0)
    Plist_MODEL1 = torch.stack(Plist_MODEL1, dim=0)
    ylabel_MODEL1= torch.stack(ylabel_MODEL1, dim=0)
    tsteps_MODEL1= torch.stack(tsteps_MODEL1, dim=0)

    T_ini_MODEL1 = Tlist_MODEL1[:,0]
    P_ini_MODEL1 = Plist_MODEL1[:,0]

    # ----------------------------------------------
    # MODEL2 Dataset load
    # ----------------------------------------------
    file_paths_MODEL2= [f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/LLNL_Eon_{i+1}.txt" for i in range(n_exp)]
    class RawDataDataset_MODEL2:
        def __init__(self, fpaths, ns):
            self.fpaths = fpaths
            self.ns = ns
        def __getitem__(self, idx):
            arr = np.loadtxt(self.fpaths[idx]).T
            arr_t= torch.tensor(arr, dtype=torch.float32, device=device)
            Y= arr_t[3:3+self.ns,:]*1e3
            T= arr_t[1,:]
            P= arr_t[2,:]
            tim= arr_t[0,:]
            return T,P,Y,tim
        def __len__(self):
            return len(self.fpaths)

    dsL = RawDataDataset_MODEL2(file_paths_MODEL2, ns)
    Tlist_MODEL2 = []
    Plist_MODEL2 = []
    ylabel_MODEL2= []
    tsteps_MODEL2= []
    for i in range(n_exp):
        T_,P_,Y_,tim_ = dsL[i]
        Tlist_MODEL2.append(T_)
        Plist_MODEL2.append(P_)
        ylabel_MODEL2.append(Y_)
        tsteps_MODEL2.append(tim_)

    Tlist_MODEL2= torch.stack(Tlist_MODEL2, dim=0)
    Plist_MODEL2= torch.stack(Plist_MODEL2, dim=0)
    ylabel_MODEL2= torch.stack(ylabel_MODEL2, dim=0)
    tsteps_MODEL2= torch.stack(tsteps_MODEL2, dim=0)

    T_ini_MODEL2= Tlist_MODEL2[:,0]
    P_ini_MODEL2= Plist_MODEL2[:,0]

    # Geometry
    loaded_geom= np.loadtxt("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_DATASET_CONTAINER/4D_valid/reactor_geometry.txt")
    geom_t= torch.tensor(loaded_geom, dtype=torch.float32, device=device)
    L_ini_list= geom_t[:,0]
    u0_ini_list= geom_t[:,1]

    # ----------------------------------------------
    # MODEL1 MLP (Temp/Time) & CRNN param
    # ----------------------------------------------
    # MODEL1
    # MLP_temp, MLP_time load
    model_temp_MODEL1 = MLP_Temp(2, ntotal-1).to(device)
    model_time_MODEL1 = MLP_Time(4, ntotal-1).to(device)

    from_surr= "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE"
    model_temp_MODEL1.load_state_dict(torch.load(os.path.join(from_surr,"TEMP_PRED_MODEL_PARAMETER_CONTAINER","mlp_weights_JetSurf_2D.pth"), map_location=device))
    model_time_MODEL1.load_state_dict(torch.load(os.path.join(from_surr,"TIME_PRED_MODEL_PARAMETER_CONTAINER","mlp_weights_JetSurf_4D_time_on.pth"), map_location=device))
    model_temp_MODEL1.eval()
    model_time_MODEL1.eval()

    # CRNN npz param
    data_MODEL1= np.load("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_JetSurf_Eon.npz",allow_pickle=True)
    param_MODEL1= data_MODEL1['parameters'][-1]
    w_in_MODEL1= torch.tensor(param_MODEL1['w_in'],  dtype=torch.float32, device=device)
    w_b_MODEL1 = torch.tensor(param_MODEL1['w_b'],   dtype=torch.float32, device=device)
    w_out_MODEL1= torch.tensor(param_MODEL1['w_out'],dtype=torch.float32, device=device)

    # MODEL1 scale
    import pickle
    input_scale_temp_MODEL1 = np.array([[870, 1e5],[1150, 3e5]])
    with open(os.path.join(from_surr,"TEMP_PRED_MODEL_PARAMETER_CONTAINER","min_max_values_mlp_JetSurf_2D.pkl"), 'rb') as fT:
        out_temp_MODEL1 = pickle.load(fT)
    min_val_temp_MODEL1, max_val_temp_MODEL1= out_temp_MODEL1['min'], out_temp_MODEL1['max']

    input_scale_time_MODEL1 = np.array([[870,1e5,0.5,2.5],[1150,3e5,1.0,5.0]])
    with open(os.path.join(from_surr,"TIME_PRED_MODEL_PARAMETER_CONTAINER","min_max_values_mlp_JetSurf_4D_time_on.pkl"), 'rb') as fTime:
        out_time_MODEL1= pickle.load(fTime)
    min_val_time_MODEL1, max_val_time_MODEL1= out_time_MODEL1['min'], out_time_MODEL1['max']

    def scale_temp_inp_MODEL1(T_, P_):
        x= torch.zeros((1,2), device=device)
        x[0,0]= (T_ - input_scale_temp_MODEL1[0,0])/(input_scale_temp_MODEL1[1,0]-input_scale_temp_MODEL1[0,0])
        x[0,1]= (P_ - input_scale_temp_MODEL1[0,1])/(input_scale_temp_MODEL1[1,1]-input_scale_temp_MODEL1[0,1])
        return x

    def scale_time_inp_MODEL1(T_,P_,L_,u_):
        x= torch.zeros((1,4), device=device)
        x[0,0]= (T_ - input_scale_time_MODEL1[0,0])/(input_scale_time_MODEL1[1,0]- input_scale_time_MODEL1[0,0])
        x[0,1]= (P_ - input_scale_time_MODEL1[0,1])/(input_scale_time_MODEL1[1,1]- input_scale_time_MODEL1[0,1])
        x[0,2]= (L_ - input_scale_time_MODEL1[0,2])/(input_scale_time_MODEL1[1,2]- input_scale_time_MODEL1[0,2])
        x[0,3]= (u_ - input_scale_time_MODEL1[0,3])/(input_scale_time_MODEL1[1,3]- input_scale_time_MODEL1[0,3])
        return x

    def predict_temp_profile_MODEL1(T_, P_):
        with torch.no_grad():
            inp= scale_temp_inp_MODEL1(T_,P_)
            out_s= model_temp_MODEL1(inp).squeeze(0)
        out_real= out_s*(max_val_temp_MODEL1 - min_val_temp_MODEL1)+ min_val_temp_MODEL1
        T0= torch.tensor([T_], device=device)
        return torch.cat([T0, out_real], dim=0)

    def predict_time_profile_MODEL1(T_,P_,L_,u_):
        with torch.no_grad():
            inp= scale_time_inp_MODEL1(T_,P_,L_,u_)
            out_s= model_time_MODEL1(inp).squeeze(0)
        out_real= out_s*(max_val_time_MODEL1 - min_val_time_MODEL1)+ min_val_time_MODEL1
        t0= torch.zeros((1,), device=device)
        t_all= torch.cat([t0, out_real], dim=0).cpu().numpy()
        t_all= enforce_strict(t_all)
        return torch.tensor(t_all, device=device)

    # ----------------------------------------------------------
    # Directly calculate y0_ with initial T, P, Cantera database
    # ----------------------------------------------------------

    spec_conc_0_list = build_spec_conc_0_list(
    T_ini_list=T_ini_MODEL1.cpu().numpy(),  # T_ini_list 준비
    P_ini_list=P_ini_MODEL1.cpu().numpy(),      # P_ini_list 준비
    n_exp=n_exp,
    ns=ns,
    device=device,
    steam_dilution_ratio=steam_dilution_ratio,
    R_J=R_J,
    mw_NC6H14=mw_NC6H14,
    mw_H2O=mw_H2O)

    # ----------------------------------------------
    # MODEL1 Dictionary (T,P)->(time_full, species_full)
    # ----------------------------------------------
    dict_MODEL1={}
    print(f"spec_conc_0_list ready! Shape: {spec_conc_0_list.shape}")

    def compute_crnn_full_MODEL1(T_, P_):
        # idx => y0
        idx_ = None
        for i_ in range(n_exp):
            if abs(T_ini_MODEL1[i_].item()-T_)<1e-7 and abs(P_ini_MODEL1[i_].item()-P_)<1e-7:
                idx_ = i_; break
        if idx_ is None:
            y0_ = torch.zeros(ns, device=device)
        else:
            y0_ = ylabel_MODEL1[idx_,:,0]
            # Overlap the y0_
            y0_ = spec_conc_0_list[idx_]
        Tlist_full= predict_temp_profile_MODEL1(T_,P_)
        time_full= predict_time_profile_MODEL1(T_,P_, 1.0, 2.5)

        # CRNN
        species_full= crnn_predict(time_full, Tlist_full, y0_, time_full, w_in_MODEL1,w_b_MODEL1,w_out_MODEL1)
        return time_full, species_full

    for i_ in range(n_exp):
        TT= float(T_ini_MODEL1[i_].item())
        PP= float(P_ini_MODEL1[i_].item())
        dict_MODEL1[(TT,PP)] = None

    for key_ in dict_MODEL1.keys():
        T_,P_ = key_
        tf_, sp_ = compute_crnn_full_MODEL1(T_,P_)
        dict_MODEL1[(T_,P_)] = (tf_, sp_)

    print("MODEL1 dictionary building done.")

    #----------------------------------------------
    # MODEL2 MLP (Temp/Time) & CRNN param
    #----------------------------------------------
    # MODEL2
    model_temp_MODEL2= MLP_Temp(2, ntotal-1).to(device)
    model_time_MODEL2= MLP_Time(4, ntotal-1).to(device)

    model_temp_MODEL2.load_state_dict(torch.load(
        os.path.join(from_surr,"TEMP_PRED_MODEL_PARAMETER_CONTAINER","mlp_weights_LLNL_2D.pth"), map_location=device))
    model_time_MODEL2.load_state_dict(torch.load(
        os.path.join(from_surr,"TIME_PRED_MODEL_PARAMETER_CONTAINER","mlp_weights_LLNL_4D_time_on.pth"), map_location=device))
    model_temp_MODEL2.eval()
    model_time_MODEL2.eval()

    data_MODEL2= np.load("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_LLNL_Eon.npz",
                       allow_pickle=True)
    param_llnl= data_MODEL2['parameters'][-1]
    w_in_MODEL2 = torch.tensor(param_llnl['w_in'], dtype=torch.float32, device=device)
    w_b_MODEL2  = torch.tensor(param_llnl['w_b'],  dtype=torch.float32, device=device)
    w_out_MODEL2= torch.tensor(param_llnl['w_out'],dtype=torch.float32, device=device)

    # MODEL2 scale
    input_scale_temp_MODEL2= np.array([[870,1e5],[1150,3e5]])
    with open(os.path.join(from_surr,"TEMP_PRED_MODEL_PARAMETER_CONTAINER","min_max_values_mlp_LLNL_2D.pkl"),'rb') as ftL:
        out_temp_MODEL2= pickle.load(ftL)
    min_val_temp_MODEL2, max_val_temp_MODEL2= out_temp_MODEL2['min'], out_temp_MODEL2['max']

    input_scale_time_MODEL2= np.array([[870,1e5,0.5,2.5],[1150,3e5,1.0,5.0]])
    with open(os.path.join(from_surr,"TIME_PRED_MODEL_PARAMETER_CONTAINER","min_max_values_mlp_LLNL_4D_time_on.pkl"),'rb') as ftmL:
        out_time_MODEL2= pickle.load(ftmL)
    min_val_time_MODEL2, max_val_time_MODEL2= out_time_MODEL2['min'], out_time_MODEL2['max']

    def scale_temp_inp_MODEL2(T_,P_):
        x= torch.zeros((1,2), device=device)
        x[0,0]= (T_- input_scale_temp_MODEL2[0,0])/(input_scale_temp_MODEL2[1,0]- input_scale_temp_MODEL2[0,0])
        x[0,1]= (P_- input_scale_temp_MODEL2[0,1])/(input_scale_temp_MODEL2[1,1]- input_scale_temp_MODEL2[0,1])
        return x

    def scale_time_inp_MODEL2(T_,P_,L_,u_):
        x= torch.zeros((1,4), device=device)
        x[0,0]= (T_- input_scale_time_MODEL2[0,0])/(input_scale_time_MODEL2[1,0]- input_scale_time_MODEL2[0,0])
        x[0,1]= (P_- input_scale_time_MODEL2[0,1])/(input_scale_time_MODEL2[1,1]- input_scale_time_MODEL2[0,1])
        x[0,2]= (L_- input_scale_time_MODEL2[0,2])/(input_scale_time_MODEL2[1,2]- input_scale_time_MODEL2[0,2])
        x[0,3]= (u_- input_scale_time_MODEL2[0,3])/(input_scale_time_MODEL2[1,3]- input_scale_time_MODEL2[0,3])
        return x

    def predict_temp_profile_MODEL2(T_,P_):
        with torch.no_grad():
            inp= scale_temp_inp_MODEL2(T_,P_)
            out_s= model_temp_MODEL2(inp).squeeze(0)
        out_real= out_s*(max_val_temp_MODEL2-min_val_temp_MODEL2)+ min_val_temp_MODEL2
        T0= torch.tensor([T_], device=device)
        return torch.cat([T0, out_real], dim=0)

    def predict_time_profile_MODEL2(T_,P_,L_,u_):
        with torch.no_grad():
            inp= scale_time_inp_MODEL2(T_,P_,L_,u_)
            out_s= model_time_MODEL2(inp).squeeze(0)
        out_real= out_s*(max_val_time_MODEL2-min_val_time_MODEL2)+ min_val_time_MODEL2
        t0= torch.zeros((1,), device=device)
        t_all= torch.cat([t0, out_real], dim=0).cpu().numpy()
        t_all= enforce_strict(t_all)
        return torch.tensor(t_all, device=device)

    # Dictionary MODEL2
    dict_MODEL2={}
    for i_ in range(n_exp):
        TT= float(T_ini_MODEL2[i_].item())
        PP= float(P_ini_MODEL2[i_].item())
        dict_MODEL2[(TT,PP)] = None

    def compute_crnn_full_MODEL2(T_, P_):
        idx_= None
        for i_ in range(n_exp):
            if abs(T_ini_MODEL2[i_].item()- T_)<1e-7 and abs(P_ini_MODEL2[i_].item()-P_)<1e-7:
                idx_= i_; break
        if idx_ is None:
            y0_= torch.zeros(ns, device=device)
        else:
            y0_= ylabel_MODEL2[idx_,:,0]
            # Overlap the y0_
            y0_ = spec_conc_0_list[idx_]
        Tlist_full= predict_temp_profile_MODEL2(T_,P_)
        time_full= predict_time_profile_MODEL2(T_,P_, 1.0,2.5)

        # CRNN
        species_full= crnn_predict(time_full, Tlist_full, y0_, time_full, w_in_MODEL2,w_b_MODEL2,w_out_MODEL2)
        return time_full, species_full

    for key_ in dict_MODEL2.keys():
        T_,P_= key_
        tf_, sp_ = compute_crnn_full_MODEL2(T_, P_)
        dict_MODEL2[(T_,P_)] = (tf_, sp_)

    print("MODEL2 dictionary building done.")

    #----------------------------------------------
    # Test set -> 3 conditions -> partial trim -> 2-model plot
    #----------------------------------------------
    all_idx= np.arange(n_exp)
    train_idx, temp_idx= train_test_split(all_idx, test_size=0.2, random_state=42)
    valid_idx, test_idx= train_test_split(temp_idx, test_size=0.5, random_state=42)

    # 3 conditions
    sorted_test_idx= sorted(all_idx, key=lambda i: T_ini_MODEL1[i].item())  # JetSurf basis
    n_test= len(sorted_test_idx)
    i_exp1= sorted_test_idx[200]
    i_exp2= sorted_test_idx[410]
    i_exp3= sorted_test_idx[800]
    i_exp_list= [i_exp1, i_exp2, i_exp3]

    solutions_MODEL1, references_MODEL1, times_MODEL1= [], [], []
    solutions_MODEL2, references_MODEL2, times_MODEL2= [], [], []
    cond_info= []

    for iE in i_exp_list:
        T_MODEL1= float(T_ini_MODEL1[iE].item())
        P_MODEL1= float(P_ini_MODEL1[iE].item())
        L_ = float(L_ini_list[iE].item())
        u_ = float(u0_ini_list[iE].item())

        # MODEL1 dictionary
        timeF_MODEL1, spF_MODEL1= dict_MODEL1[(T_MODEL1,P_MODEL1)]
        # end_time => MLP_time for MODEL1
        time_short_MODEL1= predict_time_profile_MODEL1(T_MODEL1, P_MODEL1, L_, u_)
        end_time_MODEL1 = time_short_MODEL1[-1].item()
        arr_MODEL1= timeF_MODEL1.cpu().numpy()
        idx_cut_MODEL1= np.argmin(np.abs(arr_MODEL1 - end_time_MODEL1))
        time_trim_MODEL1 = timeF_MODEL1[:idx_cut_MODEL1+1]
        species_trim_MODEL1= spF_MODEL1[:,:idx_cut_MODEL1+1]

        # reference MODEL1
        raw_t_MODEL1= tsteps_MODEL1[iE].cpu().numpy()
        raw_y_MODEL1= ylabel_MODEL1[iE].cpu().numpy()
        ref_MODEL1_list= []
        for sp_i in range(ns):
            row_sp=[]
            for t_ in time_trim_MODEL1.cpu().numpy():
                idx_ref= np.argmin(np.abs(raw_t_MODEL1 - t_))
                row_sp.append(raw_y_MODEL1[sp_i, idx_ref])
            ref_MODEL1_list.append(row_sp)
        ref_MODEL1_np= torch.tensor(ref_MODEL1_list, device=device, dtype=torch.float32)

        solutions_MODEL1.append(species_trim_MODEL1)
        references_MODEL1.append(ref_MODEL1_np)
        times_MODEL1.append(time_trim_MODEL1)

        # MODEL2 dictionary
        T_MODEL2= float(T_ini_MODEL2[iE].item())
        P_MODEL2= float(P_ini_MODEL2[iE].item())
        timeF_MODEL2, spF_MODEL2= dict_MODEL2[(T_MODEL2,P_MODEL2)]
        time_short_MODEL2= predict_time_profile_MODEL2(T_MODEL2, P_MODEL2, L_, u_)
        end_time_MODEL2= time_short_MODEL2[-1].item()
        arr_MODEL2= timeF_MODEL2.cpu().numpy()
        idx_cut_MODEL2= np.argmin(np.abs(arr_MODEL2 - end_time_MODEL2))
        time_trim_MODEL2= timeF_MODEL2[:idx_cut_MODEL2+1]
        species_trim_MODEL2= spF_MODEL2[:,:idx_cut_MODEL2+1]

        # reference LLNL
        raw_t_MODEL2= tsteps_MODEL2[iE].cpu().numpy()
        raw_y_MODEL2= ylabel_MODEL2[iE].cpu().numpy()
        ref_MODEL2_list= []
        for sp_i in range(ns):
            row_sp=[]
            for t_ in time_trim_MODEL2.cpu().numpy():
                idx_ref= np.argmin(np.abs(raw_t_MODEL2 - t_))
                row_sp.append(raw_y_MODEL2[sp_i, idx_ref])
            ref_MODEL2_list.append(row_sp)
        ref_MODEL2_np= torch.tensor(ref_MODEL2_list, device=device, dtype=torch.float32)

        solutions_MODEL2.append(species_trim_MODEL2)
        references_MODEL2.append(ref_MODEL2_np)
        times_MODEL2.append(time_trim_MODEL2)

        cond_info.append(f"T={T_MODEL1:.2f} K, P={(P_MODEL1/1e3):.2f} kPa\nL={L_:.2f} m, u0={u_:.2f} m/s")

    # Plot
    plot_sol_3_two_models(
        i_exp_list,
        solutions_MODEL1, references_MODEL1, times_MODEL1,
        solutions_MODEL2, references_MODEL2, times_MODEL2,
        cond_info
    )

    print("=== Done: two_model_3cond_plot.png ===")

# ----------------------------------
# 7) main()
# ----------------------------------
if __name__ == "__main__":
    main()


