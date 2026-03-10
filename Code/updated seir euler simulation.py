import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the data
data = pd.read_csv(r'C:\Users\yancy\OneDrive\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#2.csv')
data.columns = ['day', 'date', 'active_cases']
cases = data['active_cases'].values
days_data = data['day'].values

# Model parameters and initial conditions
N = 10000
S0, E0, I0, R0 = 9990, 5, 5, 0
num_days = 100
dt = 0.1

# SEIR simulation function
def simulate_seir(params):
    beta, sigma, gamma = params
    S, E, I, R = [S0], [E0], [I0], [R0]
    for _ in range(int(num_days/dt)):
        s, e, i, r = S[-1], E[-1], I[-1], R[-1]
        ds = -beta * s * i / N
        de = beta * s * i / N - sigma * e
        di = sigma * e - gamma * i
        dr = gamma * i
        S.append(max(s + ds*dt,0))
        E.append(max(e + de*dt,0))
        I.append(max(i + di*dt,0))
        R.append(max(r + dr*dt,0))
    return np.array(I)

#  Objective function: least squares difference between simulated infectious and actual cases
def objective(params):
    I_sim = simulate_seir(params)
    t_sim = np.linspace(0, num_days, len(I_sim))
    I_interp = np.interp(days_data, t_sim, I_sim)
    return np.sum((I_interp - cases)**2)

# Fit parameters
res = minimize(objective, [0.27, 1/5, 1/7], bounds=[(0,1),(0,1),(0,1)])
beta_fit, sigma_fit, gamma_fit = res.x
print(f"Fitted parameters: beta={beta_fit:.4f}, sigma={sigma_fit:.4f}, gamma={gamma_fit:.4f}")

# Run simulation with fitted parameters
I_fit = simulate_seir([beta_fit, sigma_fit, gamma_fit])

# Create time array for plotting the fit
t = np.linspace(0, num_days, len(I_fit))

# Plot SEIR simulation fitted to active cases
plt.figure(figsize=(10,6))
plt.plot(t, I_fit, label='Infectious (SEIR Fit)')
plt.scatter(days_data, cases, color='red', label='Actual Active Cases', s=15)
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Simulation Fitted to Active Cases')
plt.legend()
plt.grid(True)
plt.show()
