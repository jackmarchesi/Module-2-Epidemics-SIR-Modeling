import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the csv file data
data = pd.read_csv(r'C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE_1.csv', parse_dates=['date'], header=0, index_col=None)
#data = pd.read_csv(
#    r'C:\Users\yancy\OneDrive\BME2315\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#2.csv'
#)
data.columns = ['day', 'date', 'active_cases']  # rename columns
cases = data['active_cases'].values  # number of infected people each day
days_data = data['day'].values       # day numbers

# Initial populations & parameters

N = 10000        # Total population
S0 = 9990        # Initially susceptible
E0 = 5           # Initially exposed
I0 = 5           # Initially infectious
R0 = 0           # Initially recovered
num_days = 100   # How many days to simulate
dt = 0.1         # Time step (fraction of a day)


# SEIR simulation function

def simulate_seir(params):
    """
    Simulates the SEIR model given beta, sigma, gamma.
    Returns arrays for S, E, I, R over time.
    """
    beta, sigma, gamma = params
    
    # Initialize lists to store each compartment
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    
    # Loop over time steps
    for _ in range(int(num_days/dt)):
        s, e, i, r = S[-1], E[-1], I[-1], R[-1]  # current values
        
        # SEIR equations
        ds = -beta * s * i / N           # change in susceptible
        de = beta * s * i / N - sigma*e  # change in exposed
        di = sigma*e - gamma*i           # change in infectious
        dr = gamma*i                     # change in recovered
        
        # Update values using Euler method and prevent negative numbers
        S.append(max(s + ds*dt, 0))
        E.append(max(e + de*dt, 0))
        I.append(max(i + di*dt, 0))
        R.append(max(r + dr*dt, 0))
    
    # Convert lists to arrays for easier handling
    return np.array(S), np.array(E), np.array(I), np.array(R)

# Objective function for fitting

def objective(params):
    """
    Computes the difference between the model-predicted infections (I)
    and the actual active cases. This is what we want to minimize.
    """
    _, _, I_sim, _ = simulate_seir(params)
    
    # Create time array for the simulation
    t_sim = np.linspace(0, num_days, len(I_sim))
    
    # Interpolate the simulation to match the actual data days
    I_interp = np.interp(days_data, t_sim, I_sim)
    
    # Return sum of squared differences (least squares)
    return np.sum((I_interp - cases)**2)

# Fit the SEIR model to the data

# Start with initial guesses for beta, sigma, gamma
initial_guess = [0.27, 1/5, 1/7]

# Use scipy.optimize.minimize to find the best-fitting parameters
res = minimize(objective, initial_guess, bounds=[(0,1),(0,1),(0,1)])

# Extract the fitted parameters
beta_fit, sigma_fit, gamma_fit = res.x
print(f"Fitted parameters: beta={beta_fit:.4f}, sigma={sigma_fit:.4f}, gamma={gamma_fit:.4f}")


# Run SEIR simulation with fitted parameters

S_fit, E_fit, I_fit, R_fit = simulate_seir([beta_fit, sigma_fit, gamma_fit])
t = np.linspace(0, num_days, len(S_fit))  # time array for plotting

# Plot the results

plt.figure(figsize=(12,6))

# Plot SEIR compartments
plt.plot(t, S_fit, label='Susceptible', color='blue')
plt.plot(t, E_fit, label='Exposed', color='orange')
plt.plot(t, I_fit, label='Infectious (Model)', color='green')
plt.plot(t, R_fit, label='Recovered', color='purple')

# Overlay actual active cases as red dots
plt.scatter(days_data, cases, color='red', label='Actual Active Cases', s=25)

# Add labels and title
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model vs Actual Active Cases')

# Show legend and grid
plt.legend()
plt.grid(True)
plt.show()
