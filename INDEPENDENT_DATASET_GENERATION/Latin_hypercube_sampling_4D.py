import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from scipy.stats import qmc

# Working directory
work_dir = 'C:/Users/KOREA/Desktop/Naphtha_ML_SOURCE_CODE/INDEPENDENT_DATASET_GENERATION'

# Latin Hypercube Sampling function definition
def get_latin_hypercube_samples(bounds, num_samples, seed=None):
    sampler = qmc.LatinHypercube(d=len(bounds),
                                 optimization="random-cd",
                                 seed=seed)
    samples = sampler.random(num_samples)
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    return pd.DataFrame(samples, columns=list(bounds.keys()))

# 4-dimensional variables type and range
bounds = {"Temperature": [870.0,1150.0],
          "Pressure": [1.0,3.0],
          "Length": [0.5,1.0],
          "velocity": [2.5,5.0]}

# The number of sampling cases
num_samples = 400

# Sampling
sampling_case = get_latin_hypercube_samples(bounds=bounds,
                                            num_samples=num_samples,
                                            seed=13895)

# Save sampling results
sampling_case.to_csv(work_dir + "/sampling_case_4D.csv", index=False, header=False)
sampling_case

# Visualizing the sampling results
df = sampling_case
fig = make_subplots(rows=4, cols=4)
for i in range(4):
    for j in range(4):
        if i == j:
            fig.add_trace(
                go.Histogram(x=df.iloc[:, i],
                             nbinsx=25,
                             marker_color="cadetblue"),
                row=i+1, col=j+1
            )
        elif j < i:
            fig.add_trace(
                go.Scatter(x=df.iloc[:, j],
                           y=df.iloc[:, i],
                           mode="markers",
                           marker=dict(color="cadetblue")),
                row=i+1, col=j+1
            )
fig.update_layout(height=800,
                  width=800,
                  showlegend=False)
for i in range(1, 5):
    for j in range(1, 5):
        fig.update_xaxes(title_text=df.columns[j-1], row=i, col=j)
        fig.update_yaxes(title_text=df.columns[i-1], row=i, col=j)

fig.show()