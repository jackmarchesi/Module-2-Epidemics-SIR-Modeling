import numpy as np
import matplotlib.pyplot as plt

#model parameters
beta = 0.27   # Infection rate estimares from R0
sigma = 1/5   # Rate exposed people became infectious
gamma = 1/7   # Recovery rate estimates from R0
N = 1000      # Total population size
days = 160    # Number of days to simulate
dt = 0.1       # Time step for the simulation (1 day)

# initial populations
S = [9990]  # Initial susceptible population
E = [5]    # Initial exposed population
I = [5]    # Initial infectious population
R = [0]    # Initial recovered population

# Euler method simulation
for day in range(days):
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

    #Euler update step: next value = current value + change * time step
    S.append(s + ds * dt)
    E.append(e + de * dt)
    I.append(i + di * dt)
    R.append(r + dr * dt)

# Plotting the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infectious')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Epidemic Simulation (Euler Method)')
plt.legend()
plt.show()


