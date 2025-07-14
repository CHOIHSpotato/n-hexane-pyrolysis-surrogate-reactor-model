import os
import numpy as np
import cantera as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchdiffeq import odeint
import pickle
import time

###############################################################################
# Two models are used in this code now
####### MODEL1: LLNL ###########

# Base detailed kinetic model for surrogate model training 
# If you want to analyze other models, change the name of pkl, pth, npz file for the models
################################################################################

# Running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# 4) main()
################################################################################
def main():
    print("=== Start surrogate model script ===")

    start_time = time.time()
    # ----------------------------------------------
    # MODEL1 Dataset load
    # ----------------------------------------------
    geometry_df = pd.read_csv("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/sampling_case_4D.csv", header=None)
    n_exp = geometry_df.shape[0] # The number of cases
    print("the number of cases")
    print(n_exp)
    ns=9
    file_paths_MODEL1 = [f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_CONTAINER/4D_on/LLNL_Eon_{i+1}.txt" for i in range(n_exp)]
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

    T_ini_list_tensor = torch.tensor(geometry_df.iloc[:, 0].values, dtype=torch.float32, device=device)
    P_ini_list_tensor = torch.tensor(geometry_df.iloc[:, 1].values*1.0e+5, dtype=torch.float32, device=device)
    L_ini_list = torch.tensor(geometry_df.iloc[:, 2].values, dtype=torch.float32, device=device)
    u0_ini_list = torch.tensor(geometry_df.iloc[:, 3].values, dtype=torch.float32, device=device)

    # ----------------------------------------------
    # MODEL1 MLP (Temp/Time) & CRNN param
    # ----------------------------------------------
    # MODEL1
    # MLP_temp, MLP_time load
    model_temp_MODEL1 = MLP_Temp(2, ntotal-1).to(device)
    model_time_MODEL1 = MLP_Time(4, ntotal-1).to(device)

    from_surr= "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE"
    model_temp_MODEL1.load_state_dict(torch.load(os.path.join(from_surr,"TEMP_PRED_MODEL_PARAMETER_CONTAINER","mlp_weights_LLNL_2D.pth"), map_location=device))
    model_time_MODEL1.load_state_dict(torch.load(os.path.join(from_surr,"TIME_PRED_MODEL_PARAMETER_CONTAINER","mlp_weights_LLNL_4D_time_on.pth"), map_location=device))
    model_temp_MODEL1.eval()
    model_time_MODEL1.eval()

    # CRNN npz param
    data_MODEL1= np.load("C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_LLNL_Eon.npz",allow_pickle=True)
    param_MODEL1= data_MODEL1['parameters'][-1]
    w_in_MODEL1= torch.tensor(param_MODEL1['w_in'],  dtype=torch.float32, device=device)
    w_b_MODEL1 = torch.tensor(param_MODEL1['w_b'],   dtype=torch.float32, device=device)
    w_out_MODEL1= torch.tensor(param_MODEL1['w_out'],dtype=torch.float32, device=device)

    # MODEL1 scale
    input_scale_temp_MODEL1 = np.array([[870, 1e5],[1150, 3e5]])
    with open(os.path.join(from_surr,"TEMP_PRED_MODEL_PARAMETER_CONTAINER","min_max_values_mlp_LLNL_2D.pkl"), 'rb') as fT:
        out_temp_MODEL1 = pickle.load(fT)
    min_val_temp_MODEL1, max_val_temp_MODEL1= out_temp_MODEL1['min'], out_temp_MODEL1['max']

    input_scale_time_MODEL1 = np.array([[870,1e5,0.5,2.5],[1150,3e5,1.0,5.0]])
    with open(os.path.join(from_surr,"TIME_PRED_MODEL_PARAMETER_CONTAINER","min_max_values_mlp_LLNL_4D_time_on.pkl"), 'rb') as fTime:
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
    T_ini_list=T_ini_list_tensor.cpu().numpy(),  # T_ini_list 준비
    P_ini_list=P_ini_list_tensor.cpu().numpy(),      # P_ini_list 준비
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
            if abs(T_ini_list_tensor[i_].item()-T_)<1e-7 and abs(P_ini_list_tensor[i_].item()-P_)<1e-7:
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
        return time_full, species_full, Tlist_full

    for i_ in range(n_exp):
        TT= float(T_ini_list_tensor[i_].item())
        PP= float(P_ini_list_tensor[i_].item())
        dict_MODEL1[(TT,PP)] = None

    for key_ in dict_MODEL1.keys():
        T_,P_ = key_
        tf_, sp_, Temp_full  = compute_crnn_full_MODEL1(T_,P_)
        dict_MODEL1[(T_,P_)] = (tf_, sp_, Temp_full)

    print("MODEL1 dictionary building done.")

    # --------------------------------------------
    # Save prediction results
    # --------------------------------------------

    all_idx= np.arange(n_exp)
    save_dir = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/VALIDATION_FIGURE/LLNL_Eon"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start prediction save for {len(all_idx)} conditions ...")
    i_obs_MODEL1 = torch.arange(0, ns-2, device=device)

    for idx, iE in enumerate(all_idx, 1):
        T_ = float(T_ini_list_tensor[iE].item())
        P_ = float(P_ini_list_tensor[iE].item())

        L_ = float(L_ini_list[iE].item())
        u_ = float(u0_ini_list[iE].item())

        # MODEL1: trimmed prediction
        timeF_MODEL1, spF_MODEL1, Temp_full_MODEL1 = dict_MODEL1[(T_, P_)]
        time_short_MODEL1 = predict_time_profile_MODEL1(T_, P_, L_, u_)
        end_time = time_short_MODEL1[-1].item()
        arr_MODEL1 = timeF_MODEL1.cpu().numpy()
        idx_cut = np.argmin(np.abs(arr_MODEL1 - end_time))
        time_trim = timeF_MODEL1[:idx_cut + 1]
        Temp_trim = Temp_full_MODEL1[:idx_cut + 1]
        species_trim = spF_MODEL1[:, :idx_cut + 1][i_obs_MODEL1, :]
        species_trim[:-1,0] = 0.0
        P_trim = torch.full_like(time_trim, P_)
        L_trim = torch.full_like(time_trim, L_)
        u0_trim = torch.full_like(time_trim, u_)
    
        time_np = time_trim.cpu().numpy()
        temp_np = Temp_trim.cpu().numpy()
        species_np = species_trim.cpu().numpy()
        P_np = P_trim.cpu().numpy()
        L_np = L_trim.cpu().numpy()
        u0_np = u0_trim.cpu().numpy()

        output_array = np.vstack([time_np, temp_np, P_np, L_np, u0_np, *species_np]).T
        save_path = os.path.join(save_dir, f"pred_LLNLon_{idx}.txt")
        np.savetxt(save_path, output_array, fmt="%.6e")
        print(f"[{idx}/{len(all_idx)}] Saved: {save_path}")

    print("All MODEL1 test predictions saved.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.3f} seconds")

    # --------------------------------------------
    # RMSE / RelError / NRMSE / FCD for Eon
    # --------------------------------------------

    result_rows = []

    print("Start accuracy analysis...")

    species_name = ["H2", "CH4", "C2H4", "C2H6", "C3H6", "C4H8-1", "NC6H14"]

    epsilon_rel = 1.0e-5

    for idx, iE in enumerate(all_idx, 1):
        T_ini = T_ini_list_tensor[iE].item()
        P_ini = P_ini_list_tensor[iE].item()
        L_ini = L_ini_list[iE].item()
        u0_ini = u0_ini_list[iE].item()

        # Model prediction (trimmed)
        timeF_MODEL1, spF_MODEL1, Temp_full_MODEL1 = dict_MODEL1[(T_ini, P_ini)]
        time_short_MODEL1 = predict_time_profile_MODEL1(T_ini, P_ini, L_ini, u0_ini)
        end_time = time_short_MODEL1[-1].item()
        arr_MODEL1 = timeF_MODEL1.cpu().numpy()
        idx_cut = np.argmin(np.abs(arr_MODEL1 - end_time))
        species_trim = spF_MODEL1[:, :idx_cut + 1]

        pred = species_trim.cpu().numpy()

        # Reference
        raw_t = tsteps_MODEL1[iE].cpu().numpy()
        raw_y = ylabel_MODEL1[iE].cpu().numpy()

        ref_list = []
        for sp_i in range(ns):
            row = []
            for t_ in arr_MODEL1[:idx_cut + 1]:
                idx_ref = np.argmin(np.abs(raw_t - t_))
                row.append(raw_y[sp_i, idx_ref])
            ref_list.append(row)

        ref = np.array(ref_list)

        for sp_idx in range(len(species_name)):
            pred_sp = pred[sp_idx, 1:]
            ref_sp = ref[sp_idx, 1:]
            pred_final = pred_sp[-1]
            ref_final = ref_sp[-1]

            rmse_final = np.sqrt((pred_final - ref_final) ** 2)
            rel_final = np.abs(pred_final - ref_final) / (np.abs(ref_final) + epsilon_rel) * 100
            nrmse_final = rmse_final / (np.max(ref_sp) - np.min(ref_sp) + epsilon_rel)
            rmse_time_avg = np.sqrt(np.mean((pred_sp - ref_sp) ** 2))
            rel_time_avg = np.mean(np.abs(pred_sp - ref_sp) / (np.abs(ref_sp) + epsilon_rel)) * 100
            nrmse_time_avg = rmse_time_avg / (np.max(ref_sp) - np.min(ref_sp) + epsilon_rel)
    
            mu_pred = np.mean(pred_sp)
            mu_ref = np.mean(ref_sp)
            sigma_pred = np.std(pred_sp)
            sigma_ref = np.std(ref_sp)
            fcd = np.sqrt((mu_pred - mu_ref) ** 2 + (sigma_pred - sigma_ref) ** 2)

            max_diff = np.max(np.abs(pred_sp - ref_sp))
            max_norm = max_diff / (np.max(np.abs(ref_sp)) + epsilon_rel)

            result_rows.append([
                idx, species_name[sp_idx],
                T_ini, P_ini, L_ini, u0_ini,
                rmse_final, nrmse_final, rel_final, 
                rmse_time_avg, nrmse_time_avg, rel_time_avg, 
                fcd, max_norm
                ])

        print(f"[{idx}/{len(all_idx)}] done")

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
    csv_path = os.path.join(save_dir, "final_species_rmse_relerror.csv")
    result_df.to_csv(csv_path, index=False, float_format="%.6e")
    print(f"Saved: {csv_path}")

# ----------------------------------
# 7) main()
# ----------------------------------
if __name__ == "__main__":
    main()


