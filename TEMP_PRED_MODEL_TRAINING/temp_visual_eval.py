import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.ticker import FormatStrFormatter

# -----------------------------------------------------------
work_dir = "C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE"
weights_path = work_dir + "/TEMP_PRED_MODEL_PARAMETER_CONTAINER/mlp_weights_LLNL_2D.pth"
pkl_path = work_dir + "/TEMP_PRED_MODEL_PARAMETER_CONTAINER/min_max_values_mlp_LLNL_2D.pkl"

# -----------------------------------------------------------
input_node = 2
neurons = 512
nsteps = 801
output_node = nsteps - 1

# -----------------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    def __init__(self, output_node):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_node, neurons)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(neurons, neurons)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(neurons, neurons)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(neurons, output_node)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        out = self.fc4(x)
        return out

# -----------------------------------------------------------
class TemperatureDataset(torch.utils.data.Dataset):
    def __init__(self, dir=work_dir):
        #filename = dir + '/INDEPENDENT_DATASET_CONTAINER/sampling_case_4D.csv'
        filename = dir+'/INDEPENDENT_DATASET_CONTAINER/sampling_case_2D.csv'
        input_params = np.genfromtxt(filename, skip_header=0, delimiter=',', dtype=float)

        #input_params = input_params[:, :-2]

        num_data = len(input_params)

        self.input_scale = np.array([[870, 1.0],
                                     [1150, 3.0]])

        output_temp = np.zeros((num_data, output_node))
        for i in range(num_data):
            #filename = dir + f'/INDEPENDENT_DATASET_CONTAINER/4D_on/LLNL_Eon_{i+1}.txt'
            filename = dir+f'/INDEPENDENT_DATASET_CONTAINER/2D_on/LLNL_Eon_{i+1}.txt'
            rawdata = np.loadtxt(filename).T
            output_temp[i, :] = rawdata[1, 1:]

        with open(dir + '/TEMP_PRED_MODEL_PARAMETER_CONTAINER/min_max_values_mlp_LLNL_2D.pkl', 'rb') as f:
            scale_data = pickle.load(f)
        self.output_temp_scale = np.array([scale_data['min'], scale_data['max']])

        input_params[:, 0] = (input_params[:, 0] - self.input_scale[0, 0]) / (self.input_scale[1, 0] - self.input_scale[0, 0])
        input_params[:, 1] = (input_params[:, 1] - self.input_scale[0, 1]) / (self.input_scale[1, 1] - self.input_scale[0, 1])

        output_temp = (output_temp - self.output_temp_scale[0]) / (self.output_temp_scale[1] - self.output_temp_scale[0])

        self.inputs = input_params
        self.outputs = output_temp
        self.N = len(self.inputs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.outputs[idx]).float()
        return x, y

    def get_minmax(self):
        return self.input_scale, self.output_temp_scale


# -----------------------------------------------------------
dataset = TemperatureDataset()
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False)

# -----------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiLayerPerceptron(output_node).to(device)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

# -----------------------------------------------------------
input_scale, output_temp_scale = dataset.get_minmax()

relerror_list = []
Y_true_list = []
Y_pred_list = []
rmse_list = []
relerror_list = []

with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        outputs_np = outputs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        outputs_np = outputs_np * (output_temp_scale[1] - output_temp_scale[0]) + output_temp_scale[0]
        labels_np = labels_np * (output_temp_scale[1] - output_temp_scale[0]) + output_temp_scale[0]

        Y_true_list.append(labels_np.flatten())
        Y_pred_list.append(outputs_np.flatten())

        # Case-wise RMSE & RelError
        for pred, true in zip(outputs_np, labels_np):
            rmse = np.sqrt(np.mean((pred - true)**2))
            rmse_list.append(rmse)

            epsilon = 1e-8
            rel_error = np.mean(np.abs(pred - true) / (np.abs(true) + epsilon)) * 100
            relerror_list.append(rel_error)

Y_true = np.concatenate(Y_true_list)
Y_pred = np.concatenate(Y_pred_list)

# -----------------------------------------------------------
# Parity Plot
plt.figure(figsize=(8,8))
plt.scatter(Y_true, Y_pred, alpha=0.3, color='blue', s=10)
lims = [min(Y_true.min(), Y_pred.min()), max(Y_true.max(), Y_pred.max())]
plt.plot(lims, lims, 'r--', lw=2)
plt.xlabel("True [K]")
plt.ylabel("Predicted [K]")
plt.title("Parity Plot")
r2 = r2_score(Y_true, Y_pred)
plt.text(0.05, 0.95, f"$R^2$ = {r2:.4f}", transform=plt.gca().transAxes, fontsize=16, bbox=dict(boxstyle="round", facecolor="white"))
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# Residual plot
residuals = Y_true - Y_pred

plt.figure(figsize=(8, 6))
plt.scatter(Y_true, residuals, alpha=0.3, color='darkgreen', s=30, edgecolors='none')
plt.axhline(0, color='red', linestyle='--', linewidth=2.0)

plt.xlabel("True [K]", fontsize=22, labelpad=6)
plt.ylabel("Residuals [K]", fontsize=22, labelpad=6)
plt.tick_params(axis='both', direction='in', length=8, width=1.2, labelsize=18)

plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.tight_layout()
plt.savefig("C:/Users/KOREA/Desktop/ResidualPlot_TempPrediction_LLNLEON.png", dpi=400)
plt.show()

# -----------------------------
# RMSE Histogram
plt.figure(figsize=(8, 6))
plt.hist(rmse_list, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("RMSE per Case [s]", fontsize=16)
plt.ylabel("Number of Cases", fontsize=16)
plt.title("Case-wise RMSE")
plt.grid(True)
plt.show()

# -----------------------------------------------------------
# Relative Error Histogram
plt.figure(figsize=(8, 6))
plt.hist(relerror_list, bins=30, color='gold', edgecolor='black', alpha=0.9)

plt.xlabel("Average Relative Error [%] per Case", fontsize=20, labelpad=6)
plt.ylabel("Number of Cases", fontsize=20, labelpad=6)
plt.tick_params(axis='both', direction='in', length=8, width=1.2, labelsize=16)

plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()
plt.savefig("C:/Users/KOREA/Desktop/RelError_Histogram_TempPrediction_LLNLEON.png", dpi=400)
plt.show()
print(f"Average RMSE: {np.mean(rmse_list):.6f} s")
print(f"Average Relative Error: {np.mean(relerror_list):.6f} %")

print(f"Average RelError [%]: {np.mean(relerror_list):.6f}")
print(f"RelError Std [%]: {np.std(relerror_list):.6f}")
