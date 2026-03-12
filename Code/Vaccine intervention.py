import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the data
data = pd.read_csv(r'C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-2-Epidemics-SIR-Modeling\Data\mystery_virus_daily_active_counts_RELEASE#3.csv', parse_dates=['date'], header=0, index_col=None)
data.columns = ['day', 'date', 'active_cases']
cases = data['active_cases'].values
days_data = data['day'].values

# Model parameters and initial conditions
N = 10000
S0 = 9990
E0 = 5
I0 = 5
R0 = 0
num_days = 120
dt = 0.1

# SEIR simulation function
def simulate_seir(params):
    beta, sigma, gamma = params
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    for _ in range(int(num_days/dt)):
        s, e, i, r = S[-1], E[-1], I[-1], R[-1]
        ds = -beta * s * i / N
        de = beta * s * i / N - sigma*e
        di = sigma*e - gamma*i
        dr = gamma*i
        S.append(max(s + ds*dt, 0))
        E.append(max(e + de*dt, 0))
        I.append(max(i + di*dt, 0))
        R.append(max(r + dr*dt, 0))
    return np.array(S), np.array(E), np.array(I), np.array(R)

# Objective function to minimize
def objective(params):
    _, _, I_sim, _ = simulate_seir(params)
    t_sim = np.linspace(0, num_days, len(I_sim))
    I_interp = np.interp(days_data, t_sim, I_sim)
    return np.sum((I_interp - cases)**2)

# Fit parameters
initial_guess = [0.27, 1/5, 1/7]
res = minimize(objective, initial_guess, bounds=[(0,1),(0,1),(0,1)])
beta_fit, sigma_fit, gamma_fit = res.x
print(f"Fitted parameters: beta={beta_fit:.4f}, sigma={sigma_fit:.4f}, gamma={gamma_fit:.4f}")

# Run fitted SEIR
S_fit, E_fit, I_fit, R_fit = simulate_seir([beta_fit, sigma_fit, gamma_fit])
t = np.linspace(0, num_days, len(S_fit))

peak_infections = np.max(I_fit)
peak_index = np.argmax(I_fit)
peak_day = t[peak_index]
print(f"Peak infections: {peak_infections:.0f}")
print(f"Day of peak infections: {peak_day:.1f}")

# Plot SEIR simulation fitted to active cases with peak annotation
plt.figure(figsize=(12,6))
plt.plot(t, S_fit, label='Susceptible', color='blue')
plt.plot(t, E_fit, label='Exposed', color='orange')
plt.plot(t, I_fit, label='Infectious (Model)', color='green')
plt.plot(t, R_fit, label='Recovered', color='purple')
plt.scatter(days_data, cases, color='red', label='Actual Active Cases', s=25)
plt.scatter(peak_day, peak_infections, color='black', s=50, label='Peak Infection')
plt.annotate(
    f'Peak: {int(peak_infections)}\nDay: {int(peak_day)}',
    xy=(peak_day, peak_infections),
    xytext=(peak_day+5, peak_infections+200),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10
)
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model vs Actual Active Cases with Peak Infection')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# VACCINE INTERVENTIONS - Slide 8 (added below, original code untouched above)
# =============================================================================

# VT population parameters (from slide instructions)
N_vt = 10000
I0_vt = 1
R0_vt = 0
E0_vt = E0  # same E0 as UVA
S0_vt = N_vt - I0_vt - E0_vt - R0_vt

# Run VT baseline: reuse simulate_seir logic but with VT initial conditions
def simulate_seir_ic(params, N_pop, S_init, E_init, I_init, R_init, n_days):
    """Same Euler method as simulate_seir, but with custom initial conditions."""
    beta, sigma, gamma = params
    S = [S_init]; E = [E_init]; I = [I_init]; R = [R_init]
    for _ in range(int(n_days/dt)):
        s, e, i, r = S[-1], E[-1], I[-1], R[-1]
        ds = -beta * s * i / N_pop
        de =  beta * s * i / N_pop - sigma*e
        di =  sigma*e - gamma*i
        dr =  gamma*i
        S.append(max(s + ds*dt, 0))
        E.append(max(e + de*dt, 0))
        I.append(max(i + di*dt, 0))
        R.append(max(r + dr*dt, 0))
    return np.array(S), np.array(E), np.array(I), np.array(R)

# Baseline VT simulation days 0-120
S_vt, E_vt, I_vt, R_vt = simulate_seir_ic(
    [beta_fit, sigma_fit, gamma_fit], N_vt, S0_vt, E0_vt, I0_vt, R0_vt, n_days=120
)
t_vt = np.linspace(0, 120, len(I_vt))

# Snapshot SEIR state at day 70
idx_70 = int(70 / dt)
S70 = S_vt[idx_70]; E70 = E_vt[idx_70]; I70 = I_vt[idx_70]; R70 = R_vt[idx_70]

# --- Scenario 1: Single vaccine event on day 70 (n=2000, 90% efficacy)
S70_s1 = max(S70 - 0.90 * 2000, 0)
R70_s1 = R70 + 0.90 * 2000
_, _, I_single, _ = simulate_seir_ic(
    [beta_fit, sigma_fit, gamma_fit], N_vt, S70_s1, E70, I70, R70_s1, n_days=50
)
t_single = np.linspace(70, 120, len(I_single))

# --- Scenario 2: Rollout — 1000 on day 70, day 80, day 90 (90% efficacy each)
# Segment 1: day 70 dose, simulate to day 80
S70_r = max(S70 - 0.90 * 1000, 0)
R70_r = R70 + 0.90 * 1000
S_seg1, E_seg1, I_seg1, R_seg1 = simulate_seir_ic(
    [beta_fit, sigma_fit, gamma_fit], N_vt, S70_r, E70, I70, R70_r, n_days=10
)

# Segment 2: day 80 dose, simulate to day 90
S80_r = max(S_seg1[-1] - 0.90 * 1000, 0)
R80_r = R_seg1[-1] + 0.90 * 1000
S_seg2, E_seg2, I_seg2, R_seg2 = simulate_seir_ic(
    [beta_fit, sigma_fit, gamma_fit], N_vt, S80_r, E_seg1[-1], I_seg1[-1], R80_r, n_days=10
)

# Segment 3: day 90 dose, simulate to day 120
S90_r = max(S_seg2[-1] - 0.90 * 1000, 0)
R90_r = R_seg2[-1] + 0.90 * 1000
S_seg3, E_seg3, I_seg3, R_seg3 = simulate_seir_ic(
    [beta_fit, sigma_fit, gamma_fit], N_vt, S90_r, E_seg2[-1], I_seg2[-1], R90_r, n_days=30
)

# Stitch rollout segments together
I_rollout = np.concatenate([I_seg1, I_seg2[1:], I_seg3[1:]])
t_rollout = np.linspace(70, 120, len(I_rollout))

# Print summary
print(f"\n--- VT Vaccine Intervention Summary ---")
mask_70 = t_vt >= 70
print(f"Baseline peak I:     {np.max(I_vt[mask_70]):.0f} at day {t_vt[mask_70][np.argmax(I_vt[mask_70])]:.1f}")
print(f"Single-event peak I: {np.max(I_single):.0f} at day {t_single[np.argmax(I_single)]:.1f}")
print(f"Rollout peak I:      {np.max(I_rollout):.0f} at day {t_rollout[np.argmax(I_rollout)]:.1f}")

# Plot vaccine interventions
plt.figure(figsize=(12,6))
plt.plot(t_vt, I_vt, color='steelblue', linewidth=2, label='Baseline - no intervention (VT)')
plt.plot(t_single,  I_single,  color='darkorange', linewidth=2, label='Single vaccine event (day 70, n=2000, 90% efficacy)')
plt.plot(t_rollout, I_rollout, color='green',      linewidth=2, label='Vaccine rollout (day 70/80/90, n=1000 each, 90% efficacy)')
plt.axvline(70, color='gray', linestyle='--', linewidth=1, label='Day 70 (intervention start)')

# Annotate peaks
for I_arr, t_arr, color, dy in [
    (I_vt[mask_70], t_vt[mask_70], 'steelblue',  500),
    (I_single,      t_single,      'darkorange', -800),
    (I_rollout,     t_rollout,     'green',       500),
]:
    pk_val = np.max(I_arr)
    pk_day = t_arr[np.argmax(I_arr)]
    plt.scatter(pk_day, pk_val, color=color, s=60, zorder=5)
    plt.annotate(
        f'Peak: {int(pk_val)}\nDay: {int(pk_day)}',
        xy=(pk_day, pk_val),
        xytext=(pk_day+3, pk_val+dy),
        arrowprops=dict(facecolor=color, arrowstyle='->'),
        fontsize=9
    )

plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model - Vaccine Interventions at VT (Day 70)')
plt.legend()
plt.grid(True)
plt.show()