#Objective: implement testing + quarantine parameters starting from day 70 into the SEIR model and analyze their impact on the outbreak dynamics compared to the baseline scenario with no interventions.It will reduce the infectious period by 2 days

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


data = pd.read_csv(r'C:\\Users\\yancy\\OneDrive\\BME2315\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv')
data.columns = ['day','date','active_cases']

cases = data['active_cases'].values # number of infected people each day
days_data = data['day'].values # day numbers


#initial UVA parameters and conditions

N = 10000 # total population
S0 = 9990 # initially susceptible
E0 = 5 # initially exposed
I0 = 5 # initially infectious
R0 = 0 # initially recovered

num_days = 120 # number of days to simulate
dt = 0.1

#SEIR simulation function with optional testing + quarantine intervention starting from day 70
 
def simulate_seir(params):

    beta, sigma, gamma = params # extract the parameters from the input list

    S=[S0]
    E=[E0]
    I=[I0]
    R=[R0]

    for step in range(int(num_days/dt)):   # loop over time steps

        s = S[-1] 
        e = E[-1]
        i = I[-1]
        r = R[-1]

        ds = -beta*s*i/N  # change in susceptible
        de = beta*s*i/N - sigma*e  # change in exposed
        di = sigma*e - gamma*i  # change in infectious
        dr = gamma*i  # change in recovered

        S.append(max(s+ds*dt,0))  # update the compartment values using the Euler method and ensure they don't go negative
        E.append(max(e+de*dt,0))  
        I.append(max(i+di*dt,0))
        R.append(max(r+dr*dt,0))

    return np.array(S),np.array(E),np.array(I),np.array(R)


#Objective function to fit the model parameters to the UVA data using least squares on the infectious curve

def objective(params):  # objective function to minimize: sum of squared differences between model-predicted infectious and actual cases

    S_sim,E_sim,I_sim,R_sim = simulate_seir(params)  # run the SEIR simulation with the given parameters to get the predicted number of infectious individuals over time

    t_sim = np.linspace(0,num_days,len(I_sim))  # create a time array corresponding to the length of the simulation output

    I_interp = np.interp(days_data,t_sim,I_sim)  # interpolate the simulated infectious curve to match the actual data days, so we can compare them directly

    return np.sum((I_interp-cases)**2)  # return the sum of squared differences between the interpolated simulated infectious curve and the actual active cases, which is what we want to minimize

#fit the model parameters

initial_guess = [0.27,1/5,1/7]  # initial guesses for beta, sigma, gamma without the interventions 
result = minimize(objective,initial_guess,bounds=[(0,1),(0,1),(0,1)]) #

beta_fit,sigma_fit,gamma_fit = result.x # extract the fitted parameters from the optimization result

print("Fitted parameters:")  # print the fitted parameters for beta, sigma, and gamma
print("beta =",beta_fit)
print("sigma =",sigma_fit)
print("gamma =",gamma_fit)


#run UVA model with fitted parameters to get the predicted number of infectious individuals over time

S_fit,E_fit,I_fit,R_fit = simulate_seir([beta_fit,sigma_fit,gamma_fit])

t = np.linspace(0,num_days,len(S_fit))


#implement VT parameters and initial conditions

N = 40000
S0 = N - 1
E0 = 5
I0 = 1
R0 = 0


#Testing + quarantine intervention function that modifies the SEIR simulation to reduce the infectious period by 2 days starting from day 70

def simulate_testing_intervention(params):

    beta,sigma,gamma = params

    gamma_quarantine = 1/5   # infectious period reduced by 2 days due to testing + quarantine

    S=[S0]
    E=[E0]
    I=[I0]
    R=[R0]

    for step in range(int(num_days/dt)): 

        s = S[-1]
        e = E[-1]
        i = I[-1]
        r = R[-1]

        current_day = step*dt

        if current_day >= 70:
            gamma_effective = gamma_quarantine
        else:
            gamma_effective = gamma

        ds = -beta*s*i/N
        de = beta*s*i/N - sigma*e
        di = sigma*e - gamma_effective*i
        dr = gamma_effective*i

        S.append(max(s+ds*dt,0))
        E.append(max(e+de*dt,0))
        I.append(max(i+di*dt,0))
        R.append(max(r+dr*dt,0))

    return np.array(S),np.array(E),np.array(I),np.array(R)  # return the arrays for susceptible, exposed, infectious, and recovered individuals over time from the modified SEIR simulation with testing + quarantine intervention

#Run the SEIR simulation for both the baseline scenario (no intervention) and the testing + quarantine intervention scenario using the fitted parameters for VTech

S_vt,E_vt,I_vt,R_vt = simulate_seir([beta_fit,sigma_fit,gamma_fit])

S_vt_test,E_vt_test,I_vt_test,R_vt_test = simulate_testing_intervention(
[beta_fit,sigma_fit,gamma_fit])


#Plotting the results 

t = np.linspace(0,num_days,len(S_vt))

plt.figure(figsize=(12,6))

plt.plot(t,I_vt,label="VT No Intervention",linewidth=3)

plt.plot(t,I_vt_test,label="VT Testing + Quarantine",linewidth=3)

plt.axvline(x=70,color='red',linestyle='--',label="Intervention begins")

plt.xlabel("Days")
plt.ylabel("Active Infectious Individuals")

plt.title("Virus Spread at Virginia Tech w/ Testing + Quarantine Intervention")

plt.legend()
plt.grid(True)

plt.show()