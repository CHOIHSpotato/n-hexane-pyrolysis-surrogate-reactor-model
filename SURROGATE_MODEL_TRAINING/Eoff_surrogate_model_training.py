import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# Running device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(False) # Activate 'True' only when you debugging the code
# 'True' slows down the code running

# Hyperparameters
num_epochs = 300
n_exp, ns, nr, ntotal = 800, 9, 9, 801
grad_clip_value_set = 2.0 * 1.0e+2
learning_rate = 5.0 * 1.0e-3
opt_step_size = 10
opt_gamma = 0.6

# Optimized parameters associated with 1st neuron arrhenius parameter
# for each detailed mechanisms which are used for generating datasets
A_fit = torch.tensor(18.42068, dtype=torch.float32, device=device, requires_grad=False)

# LLNL
b_fit = torch.tensor(2.3263, dtype=torch.float32, device=device, requires_grad=False)
Ea_fit = torch.tensor(67.933, dtype=torch.float32, device=device, requires_grad=False)

# NUIG
#b_fit = torch.tensor(1.858, dtype=torch.float32, device=device, requires_grad=False)
#Ea_fit = torch.tensor(58.397, dtype=torch.float32, device=device, requires_grad=False)

# JetSurf
#b_fit = torch.tensor(2.1133, dtype=torch.float32, device=device, requires_grad=False)
#Ea_fit = torch.tensor(61.713, dtype=torch.float32, device=device, requires_grad=False)

lb = torch.tensor(1.0e-5, dtype=torch.float32, device=device, requires_grad=False)
ub = torch.tensor(6.0e+1, dtype=torch.float32, device=device, requires_grad=False)
intermediate_min = torch.tensor(-3.0e+1, dtype=torch.float32, device=device, requires_grad=False)
intermediate_max = torch.tensor(3.0e+1, dtype=torch.float32, device=device, requires_grad=False)
R_kcal = torch.tensor(1.9872036e-3, dtype=torch.float32, device=device, requires_grad=False)

# Updating parameter definition and Element balacne check
#####################################################################################
# Define bounds
ll_wout, ul_wout = torch.tensor(-2.0, dtype=torch.float32, device=device), torch.tensor(2.0, dtype=torch.float32, device=device)
ll_win, ul_win = torch.tensor(0.0, dtype=torch.float32, device=device), torch.tensor(2.0, dtype=torch.float32, device=device)
ll_Ea, ul_Ea = torch.tensor(10.0, dtype=torch.float32, device=device), torch.tensor(200.0, dtype=torch.float32, device=device)
ll_b, ul_b = torch.tensor(-3.0, dtype=torch.float32, device=device), torch.tensor(3.0, dtype=torch.float32, device=device)
ll_A, ul_A = torch.tensor(3.0, dtype=torch.float32, device=device), torch.tensor(21.0, dtype=torch.float32, device=device)
ll_du, ul_du = torch.tensor(-100000.0, dtype=torch.float32, device=device), torch.tensor(100000.0, dtype=torch.float32, device=device)
ll_sf, ul_sf = torch.tensor(0.1, device=device), torch.tensor(2.0, device=device)
T_eff_fit = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=False)

bounds = (ll_wout, ul_wout, ll_win, ul_win, ll_Ea, ul_Ea, ll_b, ul_b, ll_A, ul_A)

# Training, Validation, Test set
train_idx, temp_idx = train_test_split(np.arange(n_exp), test_size=0.2, random_state=42)
valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# Which species you want to estimate
# Emit the predict possibility of some heavy species (ex. C5+) for better convergence
# Excluding resovoir species prediction
i_obs = torch.arange(0, ns - 2, device=device)

# Upper and lower limit
def clamp(tensor, min_val, max_val):
    return torch.clamp(tensor, min_val, max_val)

# Lodaing training datasets
class RawDataDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, ns, device):
        file_paths = file_paths
        ns = ns
        device = device

    def __len__(self):
        return len(file_paths)

    def __getitem__(self, idx):
        filepath = file_paths[idx]
        rawdata = np.loadtxt(filepath).T
        rawdata = torch.tensor(rawdata, dtype=torch.float32, device=device)
        ylabel = rawdata[3:3 + ns, :] * 1.0e+3  # 1000 muliplied to convert [kmol/m3] to [mol/m3] 
        Tlist = rawdata[1, :]
        Plist = rawdata[2, :]
        u0 = ylabel[:, 0]
        tsteps = rawdata[0, :]
        return Tlist, Plist, ylabel, u0, tsteps

# LLNL
file_paths = [f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/CRNN_TEMP_PRED_MODEL_TRAINING_DATASET/fix_input_off/LLNL_Eoff_{i + 1}.txt" for i in range(n_exp)]

# NUIG
#file_paths = [f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/CRNN_TEMP_PRED_MODEL_TRAINING_DATASET/fix_input_off/NUIG_Eoff_{i + 1}.txt" for i in range(n_exp)]

# JetSurf
#file_paths = [f"C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/CRNN_TEMP_PRED_MODEL_TRAINING_DATASET/fix_input_off/JetSurf_Eoff_{i + 1}.txt" for i in range(n_exp)]

dataset = RawDataDataset(file_paths, ns, device)

tsteps = torch.stack([dataset[i][4] for i in range(n_exp)], dim=0)
ylabel = torch.stack([dataset[i][2] for i in range(n_exp)], dim=0)
Tlist = torch.stack([dataset[i][0] for i in range(n_exp)], dim=0)
Plist = torch.stack([dataset[i][1] for i in range(n_exp)], dim=0)
u0_list = ylabel[:, :, 0]
yscale = clamp(ylabel.amax(dim=2) - ylabel.amin(dim=2), 1e-6, torch.inf)
print(f"yscale: {yscale}, {yscale.size()}, {type(yscale)}")
print(f"tsteps: {tsteps}, {tsteps.size()}, {type(tsteps)}")

# Determine the total number of elements
size_p = nr * (3 + 2 * ns)

# Sample absolute values uniformly from [0.1, 1)
abs_values = torch.rand(size_p, device=device) * (1 - (0.1*1/T_eff_fit)) + 0.1*1/T_eff_fit
abs_values = abs_values * T_eff_fit

# Randomly choose a sign for each element (either -1 or 1)
signs = torch.randint(0, 2, (size_p,), device=device) * 2 - 1
# Construct the final tensor
p = abs_values * signs
#p = p.double()
p.requires_grad_(True)

# Modify specific indices using out-of-place operations
p = torch.cat([
    torch.tensor([1.0], device=device),  # p[0] = 1.0
    p[1:nr].clone(),  # Remaining part of the first nr segment
    torch.tensor([1.0], device=device),  # p[nr] = 1.0
    p[nr+1:nr*2].clone(),  # Remaining part of the second nr segment
    torch.tensor([1.0], device=device), # p[nr*2] = 1.0
    p[nr*2+1:nr * (ns * 2 + 3)].clone()  # Remaining elements before the last three # p[nr * (ns + 3)+1:nr * (ns + 3)+3] = [1.0, 1.0, 1.0]
    ]).detach().requires_grad_(True)  # Ensure the final tensor is a leaf tensor

print(f"p: {p}, {p.size()}, {type(p)}")
print(f"Is p a leaf tensor? {p.is_leaf}")  # Should be True
print(f"Requires grad: {p.requires_grad}")  # Should be True
#####################################################################################

# Initialize species matrix (E_)
varnames = ["H2", "CH4", "C2H4", "C2H6", "C3H6", "C4H8-1", "NC6H14", "C4H10", "C5H10-1"]
E_H = torch.tensor([2, 4, 4, 6, 6, 8, 14, 10, 10], dtype=torch.float32, device=device)
E_C = torch.tensor([0, 1, 2, 2, 3, 4, 6, 4, 5], dtype=torch.float32, device=device)
E_ = torch.stack([E_H, E_C], dim=1)[:ns, :]  # Size: (ns, ne)
_, _, Vh = torch.linalg.svd(E_.T, full_matrices=True)
E_null = Vh[E_.size(1):].T  # Nullspace of E_.T
#E_null = E_null.double()

# Prepare initial conditions
bbb = torch.zeros(E_.size(1), 1, dtype=torch.float32, device=device)  # B vector
w_out_1_ini = torch.zeros(ns, 1, dtype=torch.float32, device=device)  # Initialize w_out_1
w_out_1_ini[6, 0] = -1.0  # Fix X[6, 0] = -1.0

# Adjust B to account for the fixed value of X
fixed_X_contribution = E_.T[:, 6:7] @ w_out_1_ini[6:7, :]
B_adjusted = bbb - fixed_X_contribution

# Solve the reduced system
A_reduced = torch.cat([E_.T[:, :6], E_.T[:, 7:]], dim=1)
if torch.linalg.matrix_rank(A_reduced) < A_reduced.size(1):
    X_reduced_solution = torch.linalg.lstsq(A_reduced, B_adjusted).solution
else:
    X_reduced_solution = torch.linalg.solve(A_reduced, B_adjusted)

# Combine the solution for X_reduced with the fixed value X[6, 0]
w_out_1_ini[:6, :] = X_reduced_solution[:6, :]  # Fill the first 6 elements
w_out_1_ini[7:, :] = X_reduced_solution[6:, :]  # Fill the last 2 elements
residual = E_.T @ w_out_1_ini - bbb
print("Residual (AX - B):\n", residual)

# Combine the solution for X_reduced with the fixed value X[6, 0]
w_out_1_ini[:6, :] = X_reduced_solution[:6, :]
w_out_1_ini[7:, :] = X_reduced_solution[6:, :]

# Initial guess of w_out of reaction 1 matched with w_out_1_ini
decomp_fit = torch.tensor(0.4, device=device)
decomp_mul_fit = torch.tensor(0.5, device=device)

with torch.no_grad():
    p[nr * 3:nr * (ns + 3)] += decomp_fit
    p[nr * 3:nr * (ns + 3)] *= decomp_mul_fit
    p[nr*3] = w_out_1_ini[0]
    p[nr*4] = w_out_1_ini[1]
    p[nr*5] = w_out_1_ini[2]
    p[nr*6] = w_out_1_ini[3]
    p[nr*7] = w_out_1_ini[4]
    p[nr*8] = w_out_1_ini[5]
    p[nr*9] = w_out_1_ini[6]
    p[nr*10] = w_out_1_ini[7]
    p[nr*11] = w_out_1_ini[8]
    p[:nr] = torch.abs(p[:nr])
    p[nr*2:nr*3] = torch.abs(p[:nr])
    p[0] = p[0] * (1/(A_fit/(A_fit+ns+nr)))
    p[nr] = p[nr] * (1/((A_fit+b_fit+nr)/(A_fit+b_fit+nr+ns)))
    p[nr*2] = p[nr*2] * (1/((Ea_fit+A_fit+b_fit+ns+nr)/(Ea_fit-b_fit-ns-nr)))

print(f"p: {p}, {p.size()}, {type(p)}")
print(f"Is p a leaf tensor? {p.is_leaf}")  # Should be True
print(f"Requires grad: {p.requires_grad}")  # Should be True

def ParameterConverter(p):
        slope_factor = p[-(3*nr):]
        slope_factor = clamp(slope_factor, ll_sf, ul_sf)

        slope_A = A_fit * (A_fit/(A_fit+ns+nr))
        slope_b = b_fit * ((A_fit+b_fit+nr)/(A_fit+b_fit+nr+ns))
        slope_Ea = Ea_fit * ((Ea_fit+A_fit+b_fit+ns+nr)/(Ea_fit-b_fit-ns-nr))

        w_b = torch.abs(p[:nr]) * slope_A 
        w_in_b = p[nr:nr * 2] * slope_b
        w_in_Ea = torch.abs(p[nr * 2:nr * 3] * slope_Ea)
        w_out = p[nr * 3:nr * (ns + 3)].view(ns, nr)
        w_in_only = p[nr * (ns + 3):nr * (ns * 2 + 3)].view(ns, nr)

        # Adjust the first column of w_out with the computed w_out_1_ini
        w_out_adjusted = w_out.clone()
        #w_out_adjusted[:, 0] = w_out_1_ini.squeeze()

        # Adjust remaining columns using nullspace
        for i in range(0, nr):
            Xabcd = E_null
            eps = 1e-4
            abcd = torch.linalg.solve(Xabcd.T @ Xabcd + eps * torch.eye(Xabcd.shape[1], device=device, dtype=torch.float32), 
                                        Xabcd.T @ w_out_adjusted[:, i]).to(torch.float32)
            #abcd = torch.linalg.solve(Xabcd.T @ Xabcd, Xabcd.T @ w_out_adjusted[:, i])
            w_out_adjusted[:, i] = Xabcd @ abcd
            #element_balance = E_.T @ w_out_adjusted[:, i] - bbb
            #print("element_balance:\n", element_balance)
        
        # Clamp adjusted w_out within bounds
        w_out_adjusted = clamp(w_out_adjusted, ll_wout, ul_wout)

        # Clamp other parameters
        w_in_only = clamp(-w_out_adjusted, ll_win, ul_win)
        w_in_Ea = clamp(w_in_Ea, ll_Ea, ul_Ea)
        w_in_b = clamp(w_in_b, ll_b, ul_b)
        w_b = clamp(w_b, ll_A, ul_A)

        # Combine w_in components
        w_in = torch.cat([w_in_only, w_in_Ea.unsqueeze(0), w_in_b.unsqueeze(0)], dim=0)
        return w_in, w_b, w_out_adjusted

#####################################################################################    
w_in_c, w_b_c, w_out_c = ParameterConverter(p)

print(f"w_in: {w_in_c}, {w_in_c.size()}, {type(w_in_c)}")
print(f"Requires grad: {w_in_c.requires_grad}")  # Should be True
print(w_in_c.grad_fn)  # Should be form like <ClampBackward0 object at 0x000001E70F7CAF20>
print(f"w_b: {w_b_c}, {w_b_c.size()}, {type(w_b_c)}")
print(f"Requires grad: {w_b_c.requires_grad}")  # Should be True
print(w_b_c.grad_fn)  # Should be form like <ClampBackward0 object at 0x000001E70F7CAF20>
print(f"w_out: {w_out_c}, {w_out_c.size()}, {type(w_out_c)}")
print(f"Requires grad: {w_out_c.requires_grad}")  # Should be True
print(w_out_c.grad_fn)  # Should be form like <ClampBackward0 object at 0x000001E70F7CAF20>

del w_in_c, w_b_c, w_out_c # Clear the memory
#####################################################################################

# Linear interpolation
# When daptive time stepping ode solver cannot find proper steps => Interpolation
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

# CRNN
class CRNN(nn.Module):
    def __init__(self, tsteps):
        super(CRNN, self).__init__()
        self.tsteps = tsteps

    def wrapper(self, itpT, itpP, w_in, w_b, w_out, yscale):
        class CRNNFunc(nn.Module):
            def __init__(self, itpT, itpP, w_in, w_b, w_out, yscale):
                super(CRNNFunc, self).__init__()
                self.itpT = itpT
                self.itpP = itpP
                self.w_in = w_in
                self.w_b = w_b
                self.w_out = w_out
                self.yscale = yscale

            def forward(self, t, u):
                T = self.itpT(t)
                P = self.itpP(t)
                Y = torch.clamp(u, lb, ub)
                logX = torch.log(Y)
                w_v = torch.cat([logX, torch.tensor([-1 / (R_kcal * T), torch.log(T)], device=device)])
                w_in_x = torch.matmul(self.w_in.T, w_v)
                intermediate = w_in_x + self.w_b
                intermediate = torch.clamp(intermediate, min=intermediate_min, max=intermediate_max)
                assert torch.isfinite(intermediate).all(), "Intermediate value has NaN or Inf"
                du = torch.matmul(self.w_out, torch.exp(intermediate))
                du_clamped = torch.clamp(du, ll_du, ul_du)
                assert torch.isfinite(du).all(), "du contains NaN or Inf"
                # Apply different clamps based on the sign of du.
                #du_clamped = torch.where(
                #    du < 0,
                #    torch.clamp(du, min=ll_du_neg, max=ul_du_neg),   # Negative branch
                #    torch.clamp(du, min=ll_du_pos,  max=ul_du_pos)        # Positive branch
                #)

                # Fix zero values if any (here we assign them to 0.001)
                #du_clamped = torch.where(du == 0, torch.full_like(du, 0.001), du_clamped)
                return du_clamped

        return CRNNFunc(itpT, itpP, w_in, w_b, w_out, yscale)

# Helper function for information subplot
def create_info_subplot(ax, info):
    ax.axis('off')  # Remove axes
    ax.text(0.5, 0.5, info, ha='center', va='center', fontsize=12, wrap=True)

# Main plotting function
def plot_sol(i_exp, solution, reference, Tlist, Plist, tsteps):
    species_name = ["H2", "CH4", "C2H4", "C2H6", "C3H6", "C4H8-1", "NC6H14"]

    temp_iexp = Tlist[i_exp, 0].item()
    pres_iexp = Plist[i_exp, 0].item() / 1.0e3
    time_iexp = tsteps[i_exp, :]

    #x = time_iexp.detach().numpy()
    time_cal = time_iexp.cpu().detach().numpy()
    x = np.array([time_cal])
    x = x.reshape(-1)

    # Predictions
    y_pred = solution.cpu().detach().numpy()

    # Reference
    y_ref = reference.cpu().detach().numpy()

    # Calculate the loss at the end of the reaction
    losses = np.abs((y_pred[:, -1] - y_ref[:, -1]) * 100 / (y_ref[:, -1] + 1e-6))

    # Create subplots for each species
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(len(species_name)):
        ax = axes[i]
        ax.plot(x, y_pred[i], label="Predicted", linewidth=3, color="red")
        ax.scatter(x, y_ref[i], label="Reference", s=10, color="blue")
        ax.set_title(species_name[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Concentration [mol/m3]")
        ax.legend()

    # Information strings
    info1 = f"Temperature: {temp_iexp:.2f} K\nPressure: {pres_iexp:.2f} kPa"
    info2 = "Final product loss:\n" + "\n".join([f"{species_name[i]}: {losses[i]:.2f} %" for i in range(len(species_name))])

    # Add information subplots to the grid
    create_info_subplot(axes[7], info1)
    create_info_subplot(axes[8], info2)

    fig.tight_layout()

    return fig

# Trainer class
class Trainer:
    def __init__(self, tsteps, crnn, ylabel, yscale, u0_list, Tlist, Plist, i_obs):
        self.tsteps = tsteps
        #print(f"tsteps: {type(self.tsteps)}")
        self.crnn = crnn
        self.ylabel = ylabel
        self.yscale = yscale
        self.u0_list = u0_list
        #print(f"u0_list: {type(self.u0_list)}")
        self.Tlist = Tlist
        self.Plist = Plist
        self.i_obs = i_obs

    def predict_n_ode(self, p, i_exp):
        u0 = self.u0_list[i_exp]
        Tlist_exp = self.Tlist[i_exp, :]
        Plist_exp = self.Plist[i_exp, :]
        timelist_exp = self.tsteps[i_exp, :]

        itpT = linear_interpolation(timelist_exp, Tlist_exp)
        itpP = linear_interpolation(timelist_exp, Plist_exp)

        w_in, w_b, w_out = ParameterConverter(p)

        crnn_func = self.crnn.wrapper(itpT, itpP, w_in, w_b, w_out, yscale)
        
        sol = odeint(crnn_func, u0, timelist_exp, method='dopri5', atol=1e-3, rtol=1e-2) # Adaptive time solver

        return torch.clamp(sol.T, lb, ub)

    def loss_n_ode(self, p, i_exp):
        ref = self.ylabel[i_exp, self.i_obs, :]
        pred = self.predict_n_ode(p, i_exp)[self.i_obs, :]

        pred_normalized = pred / self.yscale[i_exp,self.i_obs].unsqueeze(1)
        ref_normalized = ref / self.yscale[i_exp,self.i_obs].unsqueeze(1)

        #loss = nn.functional.l1_loss(pred_normalized, ref_normalized)
        loss = nn.functional.mse_loss(pred_normalized, ref_normalized)
        return loss

    def train(self, p, train_idx, valid_idx, epochs, optimizer, save_path, grad_clip_value=None):
        history = {'train_loss': [], 'valid_loss': [], 'parameters': []}
        progress_bar = tqdm(range(epochs), desc="Training Progress")

        for epoch in progress_bar:
            print(f"Requires grad: {p.requires_grad}")
            # Training loop

            # Shuffle train_idx for each epoch
            random.shuffle(train_idx)

            # Training loop with mini-batches
            total_train_loss = 0.0
            # Training loop with mini-batches
            total_train_loss = 0.0
            for i_exp in train_idx:
                loss = self.loss_n_ode(p, i_exp)
                optimizer.zero_grad()
                loss.backward()

                #print(p.grad)
                if grad_clip_value:
                    torch.nn.utils.clip_grad_norm_([p], grad_clip_value)
                #print("updated p:", p)
                optimizer.step()
                total_train_loss += loss.item()

            train_loss = total_train_loss / len(train_idx)
            history['train_loss'].append(train_loss)

            # Validation loop
            total_valid_loss = 0.0
            with torch.no_grad():
                for i_exp in valid_idx:
                    loss = self.loss_n_ode(p, i_exp)
                    total_valid_loss += loss.item()

            valid_loss = total_valid_loss / len(valid_idx)
            # Adjust learning rate if validation loss doesn't decrease
            scheduler.step(valid_loss)

            # Manually print LR since `verbose=True` is deprecated
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, LR: {current_lr:.6e}")
            history['valid_loss'].append(valid_loss)

            # Save parameters
            w_in, w_b, w_out = ParameterConverter(p)
            history['parameters'].append({'w_in': w_in.detach().cpu().numpy(),
                                           'w_b': w_b.detach().cpu().numpy(),
                                           'w_out': w_out.detach().cpu().numpy()})

            # Update progress bar
            progress_bar.set_postfix({'Train Loss': train_loss, 'Valid Loss': valid_loss})

            # Save history periodically
            np.savez(save_path, **history)


        # Save final parameters and p after all epochs
        with torch.no_grad():
            w_in_up, w_b_up, w_out_up = ParameterConverter(p)
            final_data = {
                **history,
                'final_parameters': {
                    'w_in': w_in_up.detach().cpu().numpy(),
                    'w_b': w_b_up.detach().cpu().numpy(),
                    'w_out': w_out_up.detach().cpu().numpy(),
                },
                'updated_p': p.detach().cpu().numpy()  # Save the updated p tensor
            }

            # Save all data to the same file
            np.savez(save_path, **final_data)

        print(f"Training complete. Final data saved to {save_path}")

        
        return final_data

    def test(self, p, test_idx):
        total_test_loss = 0.0
        with torch.no_grad():

            for i_exp in test_idx:
                loss = self.loss_n_ode(p, i_exp)
                total_test_loss += loss.item()

                # Generate predictions and plot results
                solution_pred = self.predict_n_ode(p, i_exp)
                solution_ref = self.ylabel[i_exp, :, :]
                fig = plot_sol(i_exp, solution_pred, solution_ref, self.Tlist, self.Plist, self.tsteps)
                #fig.savefig(f"C:/CRNN/CRNN_train_server/result/local_output/LLNL_testset/i_exp_{i_exp}.png")
                plt.close(fig)

        avg_test_loss = total_test_loss / len(test_idx)
        print(f"Test Loss: {avg_test_loss:.6f}")

        return avg_test_loss
    
# Training
import torch.optim.lr_scheduler as lr_scheduler
optimizer = optim.AdamW([p], lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, threshold=1e-4, threshold_mode='rel')
crnn = CRNN(tsteps)

trainer = Trainer(tsteps, crnn, ylabel, yscale, u0_list, Tlist, Plist, i_obs)
epochs = num_epochs
save_path = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/SURROGATE_MODEL_PARAMETER_CONTAINER/training_history_LLNL_Eoff.npz"

history = trainer.train(p, train_idx, valid_idx, epochs, optimizer, save_path, grad_clip_value=grad_clip_value_set)
#history = trainer.train(p, train_idx, valid_idx, epochs, optimizer, save_path, grad_clip_value=None)

# Test the model with test datasets which were not used at the training session
test_loss = trainer.test(p, test_idx)
print(test_loss)

torch.cuda.empty_cache()  # Clear GPU memory
