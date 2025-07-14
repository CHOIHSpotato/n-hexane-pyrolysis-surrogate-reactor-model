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

# 2-dimensional variables type and range
bounds = {"Temperature": [870.0,1150.0],
          "Pressure": [1.0,3.0]}

# The number of sampling cases
num_samples = 400

# Sampling
sampling_case = get_latin_hypercube_samples(bounds=bounds,
                                            num_samples=num_samples,
                                            seed=12984)

# Save sampling results
sampling_case.to_csv(work_dir + "/sampling_case_2D.csv", index=False, header=False)
sampling_case

# Visualizing the sampling results
df = sampling_case
fig = make_subplots(rows=2, cols=2)
for i in range(2):
    for j in range(2):
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

label_map = {
    "Temperature": "Temperature [K]",
    "Pressure": "Pressure [bar]"
}
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text=label_map[df.columns[j-1]], row=i, col=j)
        fig.update_xaxes(title_font=dict(size=22), tickfont=dict(size=18))
        fig.update_xaxes(title_standoff=5)
        
        if i == j:
            fig.update_yaxes(title_text="Number of points", row=i, col=j)  
            fig.update_yaxes(title_font=dict(size=22), tickfont=dict(size=18))
            fig.update_yaxes(title_standoff=5)
        else:
            fig.update_yaxes(title_text=label_map[df.columns[i-1]], row=i, col=j) 
            fig.update_yaxes(title_font=dict(size=22), tickfont=dict(size=18))
            fig.update_yaxes(title_standoff=5)

fig.show()
