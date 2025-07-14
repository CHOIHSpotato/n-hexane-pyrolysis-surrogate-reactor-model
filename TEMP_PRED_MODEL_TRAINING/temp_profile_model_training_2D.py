import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils as torch_utils  # Importing utils for gradient clipping
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Working directory
work_dir = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE'

# Running device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
input_node = 2
nsteps = 801
output_node = nsteps-1 # model predict temperature profile except initial temperature
batch_size = 32
learning_rate = 1.0e-3
num_epochs = 20
neurons = 512

class TemperatureDataset(torch.utils.data.Dataset):
    def __init__(self, dir=work_dir, opt='training'):
        super(TemperatureDataset, self).__init__()

        # Loading input variables (inlet temperature and pressure) from the sampling result
        filename = dir+'/CRNN_TEMP_PRED_MODEL_TRAINING_DATASET_CONTATINER/sampling_case_2D.csv'
        input_params = np.genfromtxt(filename, skip_header=0,delimiter = ',' , dtype = float)[:,:]
        #print(f"input_params: {input_params}, {type(input_params)}")

        num_data = len(input_params)
        self.input_scale = np.asarray([[870, 1.0],
                                        [1150, 3.0]])  # Initial temperature and pressure

        # Loading Cantera simulation result (output)
        output_temp= np.zeros((num_data, output_node))
        for i in range(num_data):
            filename = dir+f'/CRNN_TEMP_PRED_MODEL_TRAINING_DATASET_CONTATINER/fix_input_on/JetSurf_Eon_{i+1}.txt'
            rawdata = np.loadtxt(filename).T
            output_temp[i,:] = rawdata[1, 1:]

        self.output_temp_scale = np.asarray([np.min(output_temp), np.max(output_temp)])

        # Regularize input variables between 0 and 1
        input_params[:,0] = (input_params[:,0]  - self.input_scale[0,0])/(self.input_scale[1,0] - self.input_scale[0,0]) # initial temperature
        input_params[:,1] = (input_params[:,1]  - self.input_scale[0,1])/(self.input_scale[1,1] - self.input_scale[0,1]) # initial pressure

        output_temp = (output_temp - self.output_temp_scale[0])/(self.output_temp_scale[1] - self.output_temp_scale[0])

        # training dataset, validation dataset, test datset
        input_train, input_test, output_train, output_test = train_test_split(input_params, output_temp, test_size=0.2, random_state=2024)
        input_valid, input_test, output_valid, output_test = train_test_split(input_test, output_test, test_size=0.5, random_state=2024)

        if opt == 'training':
            self.inputs = input_train
            self.outputs = output_train
            self.N = len(self.inputs)

            # Save Min, Max value in .pkl file
            scale_data = {'min': self.output_temp_scale[0], 'max': self.output_temp_scale[1]}
            with open(dir+'/TEMP_PRED_MODEL_PARAMETER_CONTAINER/min_max_values_mlp_JetSurf_2D_.pkl', 'wb') as f:
              pickle.dump(scale_data, f)

            print(f'>> number of train data : {self.N }')
            print(f'\t input shape : {self.inputs.shape}')
            print(f'\t output shape : {self.outputs.shape}')

        elif opt == 'valid':
            self.inputs = input_valid
            self.outputs = output_valid
            self.N = len(self.inputs)

            print(f'>> number of valid data : {self.N }')
            print(f'\t input shape : {self.inputs.shape}')
            print(f'\t output shape : {self.outputs.shape}')

        elif opt == 'test':
            self.inputs = input_test
            self.outputs = output_test
            self.N = len(self.inputs)

            print(f'>> number of test data : {self.N }')
            print(f'\t input shape : {self.inputs.shape}')
            print(f'\t output shape : {self.outputs.shape}')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.outputs[idx]).float()

        return x, y

    def get_minmax(self):
        return self.input_scale, self.output_temp_scale

    def num_data(self):
        return self.N

train_dataset = TemperatureDataset(dir=work_dir, opt='training')
valid_dataset = TemperatureDataset(dir=work_dir, opt='valid')
test_dataset = TemperatureDataset(dir=work_dir, opt='test')

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset,batch_size = batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = True)

# Network structure
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

model = MultiLayerPerceptron(output_node).to(device)

# Loss function
criterion =  nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)

total_step = len(train_loader)

#grad_clip_value = 1.0

history_train = []
history_valid = []

for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    history_train.append(running_loss/total_step)

    # Step the scheduler AFTER each epoch
    scheduler.step()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, disable=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)

            running_loss += val_loss.item()
    history_valid.append(running_loss/total_step)

    print('Epoch [{}/{}], Loss: {:.6f}, Val_Loss: {:.6f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))


# Visulize the loss
plt.figure(figsize=(6,4))
plt.xlabel('epoch'); plt.ylabel('loss')
plt.plot(history_train, 'b', label='training')
plt.plot(history_valid, 'r', label='validation')
plt.yscale('log')
plt.legend()
plt.show()

# Save the model parameters in .pth file
torch.save(model.state_dict(), work_dir+'/TEMP_PRED_MODEL_PARAMETER_CONTAINER/mlp_weights_JetSurf_2D_.pth')

num_test = test_dataset.num_data()
input_params = np.zeros((num_test, input_node)) # input variables
output_nn = np.zeros((num_test, output_node)) # predicted output
output_sim = np.zeros((num_test, output_node))   # true value

# Test loop
model.eval()
with torch.no_grad():
    idx=0
    for images, labels in tqdm(test_loader, disable=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        input_params[idx:idx+len(images)] = images.cpu().numpy()
        output_nn[idx:idx+len(images)] = outputs.cpu().numpy()
        output_sim[idx:idx+len(images)] = labels.cpu().numpy()
        idx = idx+len(images)

# Reverse regularization => real value
input_scale, output_temp_scale = train_dataset.get_minmax()

input_params[:,0] = input_params[:,0]*(input_scale[1,0]-input_scale[0,0]) + input_scale[0,0]
input_params[:,1] = input_params[:,1]*(input_scale[1,1]-input_scale[0,1]) + input_scale[0,1]

output_nn = output_nn*(output_temp_scale[1]-output_temp_scale[0])+output_temp_scale[0]
output_sim= output_sim*(output_temp_scale[1]-output_temp_scale[0])+output_temp_scale[0]

# Accuracy evaluation
mape = np.abs(output_nn-output_sim)/np.abs(output_sim)
acc = (1-mape)*100
print(acc)
print(np.average(acc))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# output_sim: shape (N_cases, N_time)
# output_nn:  shape (N_cases, N_time)

y_true = output_sim.flatten()
y_pred = output_nn.flatten()

# R² score
r2 = r2_score(y_true, y_pred)
print(f"R² (True vs Prediction): {r2:.6f}")

# Parity plot (real value and predicted value)
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, label="Prediction vs True", alpha=0.6, color='blue', s=20)

# Complete prediction basis line; y = x
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, color='red', linewidth=2, label='Ideal: y = x')

# Equation and R² in text
eq_text = f"$R^2$ = {r2:.6f}"
plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.4", edgecolor='gray', facecolor='white'))

plt.xlabel("Simulation Output [K]", fontsize=22)
plt.ylabel("Prediction Output [K]", fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tick_params(axis='both', direction='in', length=8, width=1.2)
plt.legend(fontsize=20)
plt.grid(False)
plt.show()

# RMSE per case
# output_sim: shape (N_cases, N_time)
# output_nn: shape (N_cases, N_time)

rmse_list = []
for i in range(output_sim.shape[0]):
    case_true = output_sim[i, :]
    case_pred = output_nn[i, :]
    rmse = np.sqrt(mean_squared_error(case_true, case_pred))
    rmse_list.append(rmse)

rmse_array = np.array(rmse_list)

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(rmse_array, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("RMSE per Case", fontsize=16)
plt.ylabel("Number of Cases", fontsize=16)
#plt.title("Distribution of Case-wise RMSE", fontsize=18)
plt.grid(True)
plt.show()

print(f"Average RMSE: {rmse_array.mean():.6f}")
print(f"RMSE Standard deviation : {rmse_array.std():.6f}")

# MAE per case
mae_list = []
for i in range(output_sim.shape[0]):
    case_true = output_sim[i, :]
    case_pred = output_nn[i, :]
    mae = mean_absolute_error(case_true, case_pred)
    mae_list.append(mae)

mae_array = np.array(mae_list)

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(mae_array, bins=30, color='lightcoral', edgecolor='black')
plt.xlabel("MAE per Case", fontsize=16)
plt.ylabel("Number of Cases", fontsize=16)
#plt.title("Distribution of Case-wise MAE", fontsize=18)
plt.grid(True)
plt.show()

print(f"Average MAE: {mae_array.mean():.6f}")
print(f"MAE Standard deviation : {mae_array.std():.6f}")

# Relative Error per case
rel_error_list = []
for i in range(output_sim.shape[0]):
    case_true = output_sim[i, :]
    case_pred = output_nn[i, :]

    rel_error = np.mean(
        np.abs(case_pred - case_true) / (np.abs(case_true) + 1e-12)
    ) * 100  # percent
    rel_error_list.append(rel_error)

rel_error_array = np.array(rel_error_list)

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(rel_error_array, bins=30, color='orange', edgecolor='black')
plt.xlabel("Relative Error [%] per Case", fontsize=16)
plt.ylabel("Number of Cases", fontsize=16)
plt.grid(True)
plt.show()

print(f"Average Relative Error: {rel_error_array.mean():.6f} %")
print(f"Relative Error Standard Deviation: {rel_error_array.std():.6f} %")

torch.cuda.empty_cache()