

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load UVA data 

data = pd.read_csv(r'C:\\Users\\yancy\\OneDrive\\BME2315\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv',
parse_dates=['date'])

data.columns = ['day','date','active_cases']

cases = data['active_cases'].values
days_data = data['day'].values


#Input same initial parameters and conditions as UVA for fitting the model to their data

N = 10000
S0 = 9990
E0 = 5
I0 = 5
R0 = 0

num_days = 120
dt = 0.1


# Simulate the SEIR model using the Euler method with the given parameters and initial conditions, and return the time series of each compartment as numpy arrays

def simulate_seir(params):  # function to simulate the SEIR model using the Euler method

    beta, sigma, gamma = params  # extract the parameters from the input list

    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]

    for step in range(int(num_days/dt)):  

        s = S[-1]
        e = E[-1]
        i = I[-1]
        r = R[-1]

        ds = -beta*s*i/N  # change in susceptible population
        de = beta*s*i/N - sigma*e  # change in exposed population
        di = sigma*e - gamma*i  # change in infectious population
        dr = gamma*i   # change in recovered population

        S.append(max(s+ds*dt,0))  # update the compartment values using the Euler method and ensure they don't go negative
        E.append(max(e+de*dt,0))
        I.append(max(i+di*dt,0))
        R.append(max(r+dr*dt,0))

    return np.array(S), np.array(E), np.array(I), np.array(R)  # return the time series of each compartment as numpy arrays


# Define an objective function to fit the model parameters to the UVA data using least squares on the infectious curve, which will be minimized by the optimization algorithm

def objective(params):  # objective function to minimize: sum of squared differences between model-predicted infectious and actual cases

    S_sim,E_sim,I_sim,R_sim = simulate_seir(params)  # run the SEIR simulation with the given parameters to get the predicted number of infectious individuals over time

    t_sim = np.linspace(0,num_days,len(I_sim))  # create a time array corresponding to the length of the simulation output

    I_interp = np.interp(days_data,t_sim,I_sim)  # interpolate the simulated infectious curve to match the actual data days, so we can compare them directly

    return np.sum((I_interp-cases)**2)  # return the sum of squared differences between the interpolated simulated infectious curve and the actual active cases, which is what we want to minimize


# Fit the model parameters by minimizing the objective function using an optimization algorithm, starting from an initial guess for the parameters

initial_guess = [0.27,1/5,1/7]  # initial guesses for beta, sigma, gamma without the interventions

result = minimize(objective,initial_guess,bounds=[(0,1),(0,1),(0,1)])   # minimize the objective function with respect to the parameters, with bounds to ensure they stay within a reasonable range

beta_fit,sigma_fit,gamma_fit = result.x  # extract the fitted parameters from the optimization result

print("Fitted parameters:")
print("beta =",beta_fit)
print("sigma =",sigma_fit)
print("gamma =",gamma_fit)


#Implement VTech parameters and initial conditions for the SEIR model, and simulate the baseline outbreak without any interventions using the fitted parameters to get the predicted number of infectious individuals over time

N = 40000  # Total population size for Virginia Tech (larger than UVA's 10,000 to reflect the larger campus population)
S0 = N - 1
E0 = 5
I0 = 1
R0 = 0


#Baseline simulation for Virginia Tech with fitted parameters

S_vt,E_vt,I_vt,R_vt = simulate_seir([beta_fit,sigma_fit,gamma_fit])  # run the SEIR simulation for Virginia Tech using the fitted parameters to get the predicted number of susceptible, exposed, infectious, and recovered individuals over time

t = np.linspace(0,num_days,len(S_vt))  # create a time array corresponding to the length of the simulation output for plotting


# Now implement the school closure intervention in the SEIR model 

def simulate_school_closure(params): # function to simulate the SEIR model with a school closure intervention

    beta,sigma,gamma = params

    closure_start = 70  #intial school closure day
    closure_end = 84   # 2 weeks later

    S=[S0]
    E=[E0]
    I=[I0]
    R=[R0]

    for step in range(int(num_days/dt)):

        s=S[-1]
        e=E[-1]
        i=I[-1]
        r=R[-1]

        current_day = step*dt

        # reduce contacts during closure
        if closure_start <= current_day <= closure_end:
            beta_effective = beta * 0.2  # assume 80% reduction in transmission during school closure
        else:
            beta_effective = beta  # normal transmission rate outside of closure period

        ds = -beta_effective*s*i/N
        de = beta_effective*s*i/N - sigma*e
        di = sigma*e - gamma*i
        dr = gamma*i

        S.append(max(s+ds*dt,0))
        E.append(max(e+de*dt,0))
        I.append(max(i+di*dt,0))
        R.append(max(r+dr*dt,0))

    return np.array(S),np.array(E),np.array(I),np.array(R)  # return the arrays for susceptible, exposed, infectious, and recovered individuals over time from the modified SEIR simulation with school closure intervention


S_vt_close,E_vt_close,I_vt_close,R_vt_close = simulate_school_closure(
[beta_fit,sigma_fit,gamma_fit])


#Plotting the results of the baseline scenario (no intervention) and the school closure intervention scenario for Virginia Tech

plt.figure(figsize=(10,6))

plt.plot(t,I_vt,label="No Intervention",linewidth=3)

plt.plot(t,I_vt_close,label="2 Week School Closure",linewidth=3)

plt.axvline(x=70,color='red',linestyle='--',label="School Closure Begins")

plt.xlabel("Days")
plt.ylabel("Active Infectious Individuals")

plt.title("Virus Spread at Virginia Tech w/ School Closure Intervention")   

plt.legend()
plt.grid(True)

plt.show()