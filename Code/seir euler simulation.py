

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv(r'C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE_1.csv', parse_dates=['date'], header=0, index_col=None)
data = pd.read_csv(r'C:\Users\yancy\OneDrive\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#2.csv')
data.columns = ['day', 'date', 'active_cases']
cases = data['active_cases']
days_data = data['day']


#model parameters
beta = 0.27   # Infection rate estimates from R0
sigma = 1/5   # Rate exposed people became infectious
gamma = 1/7   # Recovery rate estimates from R0

N = 10000      # Total population size
num_days = 100    # Number of days to simulate
dt = 0.1       # Time step for the simulation (1 day)

# initial populations
S = [9990]  # Initial susceptible population
E = [5]    # Initial exposed population
I = [5]    # Initial infectious population
R = [0]    # Initial recovered population

# Euler method simulation

# for day in range(num_days):
num_steps = int(num_days / dt)
for _ in range(num_steps):
    #Get the most recent values
    s = S[-1]
    e = E[-1]
    i = I[-1]
    r = R[-1]
    # Calculate the changes
    ds = -beta * s * i / N   # Change in susceptible population
    de = beta * s * i / N - sigma * e  # Change in exposed population
    di = sigma * e - gamma * i  # Change in infectious population
    dr = gamma * i  # Change in recovered population

#     #Euler update step: next value = current value + change * time step
    S.append(s + ds * dt)
    E.append(e + de * dt)
    I.append(i + di * dt)
    R.append(r + dr * dt)

    # Avoid negative populations
    S[-1] = max(S[-1], 0)
    E[-1] = max(E[-1], 0)
    I[-1] = max(I[-1], 0)
    R[-1] = max(R[-1], 0)

# Time array for plotting 
t = np.linspace(0, num_days, len(S))

# Plot SEIR simulation 
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infectious')
plt.plot(t, R, label='Recovered')

# Overlay actual data 
plt.scatter(days_data, cases, color='red', label='Actual Active Cases', s=15)

plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Simulation vs Actual Active Cases')
plt.legend()
plt.grid(True)
plt.show()


