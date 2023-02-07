"""
Generate data for the non-linear pendulum.

To add extra terms, make b != 0 and drag != 0.
"""

import numpy as np
import scipy as sci
from scipy.integrate import solve_ivp
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def pendulum(t, x, b=0, k=0, drag=0.0):
    theta, velocity = x
    theta_dot = velocity
    velocity_dot = drag * np.cos(k * t) - np.sin(theta) - b * velocity
    return theta_dot, velocity_dot


# Set coefficient terms: Make these non-zero to add "extra" terms
b = 0.2
k = 0.0
drag = 0.0

# Generate data with sampled data points
dt_train = 0.03
dt = dt_train
tmax = 1000 * dt
grid = np.arange(0, tmax, dt)

# Numerical integration
x0 = np.array([3 / 4 * np.pi, 0.0])
sol = solve_ivp(pendulum, (0, tmax),
                y0=x0,
                args=(b, k, drag),
                t_eval=grid,
                rtol=1e-5,
                atol=1e-9)

t_set = np.reshape(grid, (-1, 1))
theta = np.reshape(sol.y[0], (-1, 1))

data_train = np.concatenate((t_set, theta), axis=1)

data_test = np.concatenate((t_set, theta), axis=1)

fig = plt.figure(figsize=(30, 17), dpi=100, facecolor="w", edgecolor="k")
makerSize = 2.5
plt.scatter(
    data_test[:, 0].reshape(-1),
    data_test[:, 1].reshape(-1),
)

plt.scatter(
    data_train[:, 0].reshape(-1),
    data_train[:, 1].reshape(-1),
)
plt.title("theta'' = - sin(theta)")
plt.legend(["Test points", "Training points"], prop={"size": 40})
plt.savefig("./pendulum/data.jpg")

df_train = pd.DataFrame(data_train, columns=["t", "theta"])
df_test = pd.DataFrame(data_test, columns=["t", "theta"])

N = t_set.shape[0]
train, test = train_test_split(df_train, test_size=0.1, train_size=20 / N)
train.to_csv("train.csv", encoding="utf-8", index=False)
train, test = train_test_split(df_test, test_size=800 / N, train_size=0.1)
test.to_csv("test.csv", encoding="utf-8", index=False)




